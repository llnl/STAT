# SPDX-FileCopyrightText: Copyright (c) 2007-2020, Lawrence Livermore National Security, LLC.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# LLNL-CODE-750488.
# SPDX-License-Identifier: BSD-3-Clause

"""Control a remote STAT sampling session from the desktop GUI.

RemoteGUI supplies the attach, sample, and detach callbacks for the classic
STAT window. It translates the selected job or PID and the GUI settings into
requests to stat-gui-server over SSH. RemoteSession tracks the attachment and
keeps local copies of graphs and source files for the viewer.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import re
from pathlib import Path
import shutil
import tempfile
import time

from remote.stat_client_connection import StatRemoteClient


def add_remote_arguments(parser):
    """Register SSH connection, scheduler, and remote sampling options."""
    parser.add_argument('--remote-login', help='SSH login, e.g. user@cluster')
    parser.add_argument('--remote-prefix', help='STAT installation on the SSH server')
    parser.add_argument('--remote-server-host', help='explicit second SSH hop')
    parser.add_argument('--ssh-option', action='append', default=[], help='extra OpenSSH argument (repeatable)')

    parser.add_argument('--scheduler', choices=['auto', 'slurm', 'flux', 'lsf', 'alps'], default='auto')
    parser.add_argument('--flux-allocation', metavar='ALLOCATION_ID', help='outer Flux allocation containing the application')
    parser.add_argument('--remote-job', metavar='JOB_ID', help='scheduler job to attach to (e.g. Slurm job ID, inner Flux application job ID)')

    parser.add_argument('--backend', choices=['dyninst', 'pystack', 'gdb'])
    parser.add_argument('--native', action='store_true', help='include native PyStack frames')


def add_gui_arguments(gui_parser, version="4.3.0", have_pystack=True):
    """Register options for STAT GUI startup and for standalone help without GTK."""
    trace_group = gui_parser.add_mutually_exclusive_group()
    trace_group.add_argument("-P", "--withpc", help="sample program counter in addition to function name", action="store_true")
    trace_group.add_argument("-m", "--withmoduleoffset", help="sample module offset only", action="store_true")
    trace_group.add_argument("-i", "--withline", help="sample source line number in addition to function name", action="store_true")
    gui_parser.add_argument("-s", "--sleep", help="sleep for the specified number of seconds before attaching", metavar="SLEEPTIME", type=int)
    gui_parser.add_argument("-o", "--withopenmp", help="translate OpenMP stacks to logical application view", action="store_true")
    gui_parser.add_argument("-w", "--withthreads", help="sample helper threads in addition to the main thread", action="store_true")
    gui_parser.add_argument("-y", "--pythontrace", help="gather Python script level stack traces", action="store_true")

    gui_parser.set_defaults(pystack=False)
    if have_pystack:
        gui_parser.add_argument("-K", "--pystack", help="gather Python application stack traces using PyStack", action="store_true")

    gui_parser.add_argument("--with-children", help="sample additionally child processes using PyStack", action="store_true")
    gui_parser.add_argument("-U", "--countrep", help="only gather count and a single representative", action="store_true")

    gui_parser.add_argument("-d", "--debugdaemons", help="launch the daemons under the debugger", action="store_true")
    gui_parser.add_argument("-j", "--jobid", help="append specified job ID to output directory name")
    gui_parser.add_argument("-L", "--logdir", help="logging output directory")
    gui_parser.add_argument("-l", "--log", help="enable debug logging", choices=['FE', 'BE', 'CP'], action="append")

    gui_parser.add_argument("-G", "--gdb", help="use the GDB backend", action="store_true")
    gui_parser.add_argument("-Q", "--cudaquick", help="gather less comprehensive, but faster cuda traces", action="store_true")

    trace_group.add_argument("-M", "--mrnetprintf", help="use MRNet's print for logging", action="store_true")
    gui_parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=version))

    attach_group = gui_parser.add_mutually_exclusive_group()
    attach_group.add_argument("-a", "--attach", help="attach to the specified [hostname:]PID", metavar='LAUNCHERPID')
    attach_group.add_argument("-C", "--create", help="launch the application under STAT's control. All args after -C are used to launch the app", nargs=argparse.REMAINDER)
    attach_group.add_argument("-I", "--serial", help="attach to the specified [<exe>@<host>:<pid>]+. All args after -I are interpreted as serial processes to attach to", nargs=argparse.REMAINDER)

    add_remote_arguments(gui_parser)


def gui_help():
    parser = argparse.ArgumentParser(description='STAT GUI: attach locally or over SSH; browse saved graphs with stat-view.')
    add_gui_arguments(parser)
    parser.print_help()


class RemoteSession:
    """Own a connection and keep ownership until detach is acknowledged."""

    def __init__(self, client):
        """Prepare session state and temporary storage for downloaded files."""
        self.client = client
        self.session_id = None
        self.state = 'new'
        self.health = {}
        self.files = tempfile.TemporaryDirectory(prefix='stat-remote-')
        self.sources = {}

    def connect(self):
        """Start the server connection and read its capabilities."""
        self.client.ensure_remote_server()
        self.health = self.client.remote_health
        return self.health

    def request(self, method, params=None, on_progress=None):
        """Send a request and retain the session ID and state reported by the server."""
        params = dict(params or {})
        if self.session_id is not None:
            params['session_id'] = self.session_id
        try:
            result = self.client.request(method, params, on_progress=on_progress)
        except Exception as exc:
            # a failed attach can still leave the server responsible for a paused target.
            context = getattr(exc, 'error', {}).get('context') or {}
            self.session_id = context.get('session_id', self.session_id)
            self.state = context.get('state', self.state)
            raise

        self.session_id = result.get('session_id', self.session_id)
        self.state = result.get('state', self.state)
        return result

    def attach(self, params):
        """Attach to a target using a backend advertised by the server."""
        if self.session_id is not None:
            raise RuntimeError('Detach the current application first')
        if params['backend'] not in self.health.get('supported_backends', []):
            raise RuntimeError('The server does not provide %s' % params['backend'])
        return self.request('sessions.attach', dict(params, initial_sample=False))

    def detach(self):
        """Release the target and clear the session ID after server acknowledgement."""
        if self.session_id is not None:
            result = self.request('sessions.detach')
            if result.get('state') != 'detached':
                raise RuntimeError('Server did not acknowledge detach')
            self.session_id = None

    def save_dot(self, result):
        """Save a returned DOT graph for display in a local viewer tab."""
        path = Path(self.files.name) / os.path.basename(result.get('dot_filename') or 'remote.dot')
        path.write_text(result['dot_text'])
        return str(path), result['dot_text']

    def source(self, source, ranks, session_id):
        """Return a cached file or fetch it through the ranks represented by the selected node."""
        key = (session_id, source)
        if key in self.sources and os.path.isfile(self.sources[key]):
            return self.sources[key]
        if session_id != self.session_id or session_id is None:
            raise RuntimeError('Source is unavailable after detaching this session')

        result = self.request('sources.get', {'source': source, 'ranks': ranks[:16]})

        identity = hashlib.sha256(('%s\0%s' % key).encode()).hexdigest()
        path = Path(self.files.name) / (identity + '-' + (os.path.basename(source) or 'source'))
        path.write_bytes(base64.b64decode(result['content_b64'], validate=True))
        self.sources[key] = str(path)
        return str(path)

    def close(self):
        """Detach and release the SSH connection and temporary files."""
        try:
            self.detach()
        finally:
            self.client.close(permanent=True)
            self.files.cleanup()


class RemoteGUI:
    """Handle attach and sample actions in the STAT GUI by sending requests over SSH and displaying the returned call graphs."""

    supports_cuda_quick = False
    remote = None
    remote_busy = False
    closing = False

    def __init__(self, args):
        """Check remote startup options before opening the shared STAT window."""
        if args.create is not None:
            raise SystemExit('--create is unavailable in remote mode. Attach to an existing process or job.')

        self.status_rows = []
        self.remote = None
        self.remote_busy = False
        self.closing = False

        for selected, backend in ((args.gdb, 'gdb'), (args.pystack, 'pystack')):
            if selected:
                if args.backend and args.backend != backend:
                    raise SystemExit('Select only one sampling backend.')
                args.backend = backend

        if args.cudaquick and (args.backend != 'gdb' or not args.withthreads):
            raise SystemExit('--cudaquick requires the GDB backend and --withthreads.')

        super().__init__(args)
        self.setup_remote_window()

    def pack_graph_notebook(self, box):
        """Group the graph tabs and their connection status in a vertical column."""
        from STATview import gtk

        self.graph_column = gtk.VBox()
        super().pack_graph_notebook(self.graph_column)
        box.pack_start(self.graph_column, True, True, 0)

    def setup_remote_window(self):
        """Install the main status row, refresh timer, and window-close handler once."""
        from gi.repository import GLib
        if not self.status_rows:
            self.add_status_row(self.graph_column)
            self.connect('delete-event', self._delete)
            timer = GLib.timeout_add_seconds(1, self.update_remote_status)
            self.connect('destroy', lambda widget: GLib.source_remove(timer))

    def add_status_row(self, container):
        """Add a spinner and connection label to the main window or Attach dialog for Remote Connect."""
        from STATview import gtk

        row = gtk.HBox(spacing=6)
        row.set_border_width(4)
        spinner = gtk.Spinner()
        label = gtk.Label()
        row.pack_start(spinner, False, False, 0)
        row.pack_start(label, False, False, 0)
        container.pack_start(row, False, False, 0)

        widgets = (spinner, label)
        self.status_rows.append(widgets)
        row.connect('destroy', lambda widget: self.status_rows.remove(widgets))
        row.show_all()
        self.update_remote_status()

    def update_remote_status(self, operation=None):
        """Refresh both footers on the GTK thread. Inspect the local SSH process only."""
        if self.remote_busy and operation is None:
            return True

        proc = self.remote.client.proc if self.remote else None
        state = operation or ('Ready' if proc is not None and proc.poll() is None else 'Disconnected')
        text = ' · '.join(filter(None, ('Remote', self.remote_args.remote_login, state)))

        for spinner, label in self.status_rows:
            if label.get_text() != text:
                label.set_text(text)
            if spinner.get_property('active') != bool(operation):
                spinner.set_property('active', bool(operation))
        return True

    def set_action_sensitivity(self, state):
        """Enable session actions according to the state of the remote attachment."""
        super().set_action_sensitivity(state)
        if self.remote and self.remote.session_id and state != 'busy':
            # Failed attach cleanup can leave a session owned before its first
            # sample. Always allow acknowledged detach in that state.
            for action in self.actiongroup.list_actions():
                if action.get_name() in ('Attach', 'ReAttach'):
                    action.set_sensitive(False)
                elif action.get_name() == 'Detach':
                    action.set_sensitive(True)

    def _run(self, operation, function, *args):
        """Keep GTK responsive without reentering a transport request."""
        from STATview import run_gtk_main_loop

        if self.remote_busy:
            raise RuntimeError('A remote operation is already in progress')

        self.remote_busy = True
        self.set_action_sensitivity('busy')
        self.update_remote_status(operation)
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(function, *args)
                # process GTK events on the main thread while the worker waits for SSH.
                while not future.done():
                    run_gtk_main_loop()
                    time.sleep(.02)
                return future.result()
        finally:
            self.remote_busy = False
            self.update_remote_status()
            state = self.remote.state if self.remote else 'new'
            self.set_action_sensitivity({'attached': 'paused', 'detached': 'detached'}.get(state, state))

    def _error(self, exc):
        """Dismiss the wait dialog and display the remote error."""
        from STATview import show_error_dialog, stat_wait_dialog

        stat_wait_dialog.destroy()
        show_error_dialog('Remote STAT: %s' % exc, self)
        return False

    def _delete(self, *args):
        """Keep the window open while a remote request is in progress."""
        return self.remote_busy

    def on_attach(self, action):
        """Connect to the selected server and open the scheduler attach dialog."""
        self.setup_remote_window()
        self.show_all()
        args = self.remote_args
        try:
            if self.remote and self.remote.state == 'detached':
                self._run('Closing connection…', self.remote.close)
                self.remote = None
            self._connect_from_args()
        except Exception as exc:
            return self._error(exc)

        scheduler = args.scheduler if args.scheduler != 'auto' else 'slurm'
        self.types['Resource Manager'] = [scheduler.title()]
        self.options['Resource Manager'] = scheduler.title()
        return self.show_attach_dialog(remote=True)

    def _connect_from_args(self):
        """Open the requested SSH route and initialize the server's sampling options."""
        self.setup_remote_window()
        if self.remote is None:
            args = self.remote_args
            if not getattr(args, 'remote_login', None):
                raise ValueError(
                    'To attach to a remote job, restart the GUI from a terminal:\n\n'
                    'stat-gui --remote-login user@login-host\n\n'
                    'Run stat-gui --help for SSH and scheduler options.')
            prefix = getattr(args, 'remote_prefix', None)
            command = StatRemoteClient.command_for_prefix(prefix) if prefix else 'stat-gui-server'
            client = StatRemoteClient.ssh(
                args.remote_login, command, ssh_options=args.ssh_option, remote_prefix=prefix,
                remote_server_host=args.remote_server_host, flux_allocation=args.flux_allocation)

            # for a Slurm job, discover the launcher host unless the user chose the SSH route.
            if args.remote_job and args.scheduler in ('auto', 'slurm') and not args.remote_server_host and not args.flux_allocation:
                from remote.stat_client_slurm import resolve_on_login
                client = self._run('Selecting server host…', resolve_on_login, client, args.remote_job)
            self.connect_client(client)

            # a blank entry uses the server's debugger, independent of desktop settings.
            self.options['GDB Path'] = ''
            supported = self.remote.health['supported_backends']
            self.types['Sampling Backend'] = supported
            backend = args.backend or supported[0]

            self.options['Use PyStack'] = backend == 'pystack'
            self.options['GDB BE'] = backend == 'gdb'
            self.options['With CUDA Quick'] = args.cudaquick

            if args.native:
                self.options['PyStack Frames'] = 'Python + C/C++'

    def connect_client(self, client):
        """Check a new server connection before replacing the current one."""
        session = RemoteSession(client)
        try:
            self._run('Connecting…', session.connect)
            requested = self.remote_args.backend
            if requested and requested not in session.health['supported_backends']:
                raise ValueError('The server does not provide the requested %s backend' % requested)
            if self.remote_args.cudaquick and not session.health.get('cuda_quick'):
                raise ValueError('The server does not support CUDA Quick sampling')
            if self.remote:
                self._run('Closing connection…', self.remote.close)
        except Exception:
            session.close()
            raise

        self.remote = session
        self.supports_cuda_quick = session.health.get('cuda_quick', False)
        self.update_remote_status()

    def setup_job_picker(self, box, frame, dialog, attach_button):
        """Connect the job ID field and running-job selector to launcher discovery."""
        from STATview import gtk

        self.add_status_row(dialog.vbox)
        self.attach_button = attach_button

        self.running_jobs = gtk.ComboBoxText()
        row = gtk.HBox()
        row.pack_start(gtk.Label('Running Jobs'), False, False, 5)
        row.pack_start(self.running_jobs, True, True, 5)
        refresh = gtk.Button('Refresh Jobs')
        row.pack_start(refresh, False, False, 5)
        box.pack_start(row, False, False, 0)
        box.reorder_child(row, 0)
        row.show_all()

        self.filter_entry.set_text('')
        dialog.set_destroy_with_parent(True)
        self.jobid_entry.set_text(self.remote_args.remote_job or '')

        def edit_job(entry):
            """Clear the previous launcher when the job ID changes."""
            self.clear_launcher(frame)
            if self.running_jobs.get_active_id() != entry.get_text().strip():
                self.running_jobs.set_active(-1)

        self.jobid_entry.connect('changed', edit_job)

        def select_job(combo):
            """Resolve launchers for the job selected in the dropdown."""
            if combo.get_active_id():
                self.jobid_entry.set_text(combo.get_active_id())
                self.on_update_process_listing(None, frame, dialog)

        self.running_jobs.connect('changed', select_job)
        refresh.connect('clicked', lambda w: self.refresh_jobs(frame, dialog))

        if self.options['Resource Manager'] == 'Slurm':
            self.refresh_jobs(frame, dialog, initial=True)
        else:
            row.set_sensitive(False)
            self.on_update_process_listing(None, frame, dialog)

    def refresh_jobs(self, frame, dialog, initial=False):
        """Reload the running Slurm jobs and restore the selected job if possible."""
        previous = self.jobid_entry.get_text().strip()
        self.clear_launcher(frame)
        self.running_jobs.remove_all()
        dialog.set_sensitive(False)
        try:
            rows = self._run('Refreshing jobs…', self.remote.request, 'jobs.list', {'scheduler': 'slurm'})['jobs']
            rows = sorted(rows, key=lambda row: not row.get('is_batch', True))
            for row in rows:
                self.running_jobs.append(row['jobid'], '  '.join(
                    str(row.get(field, '')) for field in ('jobid', 'name', 'nodelist')))
        except Exception as exc:
            self.show_job_status(frame, str(exc))
            return
        finally:
            dialog.set_sensitive(True)

        # keep the current job when it is still running.
        if previous and self.running_jobs.set_active_id(previous):
            return
        if initial and previous:
            self.on_update_process_listing(None, frame, dialog)
        elif rows:
            self.running_jobs.set_active(0)
        else:
            self.jobid_entry.set_text('')
            self.show_job_status(frame, 'No running Slurm jobs.')

    def on_update_process_listing(self, widget, frame, dialog, listing_filter=None):
        """Show matching launchers or explain how to attach to the selected Flux job."""
        from STATview import gtk, gtk_wrap_new_with_label_from_widget

        self.clear_launcher(frame)
        jobid = self.jobid_entry.get_text().strip()
        if not jobid:
            return

        scheduler = self.options['Resource Manager'].lower()

        # for flux, no process or launcher to attach to.
        if scheduler == 'flux':
            self.show_job_status(frame, 'Click Attach to start the Flux debug launcher for job %s.' % jobid)
            self.attach_button.set_sensitive(True)
            return
        dialog.set_sensitive(False)

        try:
            # for Slurm, find the launcher host and PIDs matching the job ID.
            result = self._run(
                'Finding launcher for job %s…' % jobid, self.remote.request,
                'jobs.resolve', {'scheduler': scheduler, 'jobid': jobid},
            )
            host = result['remote_host']
            details = {p['pid']: p['command_line'] for p in result['processes']}
            pattern = re.compile(self.filter_entry.get_text())
            processes = []

            for pid in result['pids']:
                command = details.get(pid) or result['launcher_exe']
                if pattern.search(command if self.options['Filter Full Command Line'] else command.split()[0]):
                    processes.append((pid, command))

            if not processes:
                self.show_job_status(frame, 'No matching launcher for job %s.' % jobid)
                return

            self.options['Remote Host'] = host
            self.rh_entry.set_text(host)
            frame.set_label('Current Process List (%s)' % host)

            box = gtk.VBox()
            group = None
            if len(processes) > 1:
                # keep all launchers unselected until the user picks one.
                group = gtk_wrap_new_with_label_from_widget(None, '')
                group.set_no_show_all(True)
                box.pack_start(group, False, False, 0)
            self.pack_process_list(box, processes, True, group)
            self.attach_button.set_sensitive(self.options['PID'] is not None)

            scroll = gtk.ScrolledWindow()
            scroll.add_with_viewport(box)
            frame.add(scroll)
            scroll.show_all()
        except Exception as exc:
            if getattr(exc, 'error', {}).get('code') == 'job_resolve_failed':
                self.show_job_status(frame, 'Could not find a running launcher for job %s. Refresh Jobs or select another job.' % jobid)
            else:
                self.show_job_status(frame, str(exc))
        finally:
            dialog.set_sensitive(True)

    def pid_toggle_cb(self, button, pid, command):
        """Update the selected PID and enable Attach when a launcher is selected."""
        super().pid_toggle_cb(button, pid, command)
        self.attach_button.set_sensitive(self.options['PID'] is not None)

    def clear_launcher(self, frame):
        """Clear the launcher selection and disable Attach until a new target is ready."""
        self.options['PID'] = None
        self.options['Remote Host'] = ''
        self.rh_entry.set_text('')
        frame.set_label('Current Process List')
        self.attach_button.set_sensitive(False)
        if frame.get_child():
            frame.remove(frame.get_child())

    def show_job_status(self, frame, message):
        """Display job discovery instructions or an error in the process list."""
        from STATview import gtk
        label = gtk.Label(message)
        label.set_selectable(True)
        label.set_line_wrap(True)
        frame.add(label)
        label.show()

    def attach_cb(self, attach_dialog, launch, serial, application_option=None):
        """Build an attach target from the Flux job ID or selected launcher PID."""
        jobid = self.jobid_entry.get_text().strip() if attach_dialog is not None else None
        if jobid and self.remote_args.scheduler == 'flux':
            target = dict(mode='jobid', scheduler='flux', jobid=jobid)
        else:
            target = dict(mode='attach', remote_host=self.options['Remote Host'], pid=self.options['PID'])
        return self.attach_target(attach_dialog, target, jobid)

    def attach_serial_processes_cb(self, entry=None, attach_dialog=None, in_processes=None):
        """Attach to the process list supplied on the command line or in the dialog."""
        processes = in_processes if in_processes is not None else entry.get_text().split()
        return self.attach_target(attach_dialog, dict(mode='serial', processes=processes))

    def attach_target(self, dialog, target, jobid=None):
        """Apply dialog settings, verify the target, and start the remote attachment."""
        try:
            self._connect_from_args()
            if dialog is not None:
                dialog.set_sensitive(False)
                self.update_attach_options()
                self.update_sample_options()

            if jobid and self.options['Resource Manager'] == 'Slurm':
                self.prepare_slurm_launcher(jobid, target['remote_host'], target['pid'])
            params = self.get_remote_params()
            if params['backend'] == 'gdb' and self.options['GDB Path'].strip():
                params['gdb_path'] = self.options['GDB Path'].strip()
            params.update(target)
            if params.get('remote_host') == self.remote.health['hostname']:
                params['remote_host'] = 'localhost'
            self._attach_remote(params)
            return 0
        except Exception as exc:
            return self._error(exc)
        finally:
            if dialog is not None:
                dialog.set_sensitive(True)
                # An attached session needs the main window's Detach action even if sampling failed.
                if self.remote and self.remote.session_id:
                    dialog.response(0)

    def prepare_slurm_launcher(self, jobid, host, pid):
        """Recheck the selected launcher through the route used for this attach."""
        from remote.stat_client_slurm import check_launcher, resolve_on_login
        args = self.remote_args
        if not args.remote_server_host and not args.flux_allocation and self.remote.health['hostname'] != host:
            client = self._run('Selecting server host…', resolve_on_login, self.remote.client, jobid, (host, pid))
            self.connect_client(client)
        else:
            job = self._run('Checking launcher…', self.remote.request, 'jobs.resolve',
                            {'scheduler': 'slurm', 'jobid': jobid})
            check_launcher(job, jobid, host, pid)

    def _attach_remote(self, params):
        """Attach to the target, remember it for reattach, and collect the first sample."""
        self.last_remote_target = dict(params)
        self._run('Attaching…', self.remote.attach, params)
        self.attached = True
        self.reattach = True
        if self.sample() is False:
            raise RuntimeError('Attached, but the first sample failed. Detach or retry sampling.')

    def get_remote_params(self):
        """Translate the current GUI settings into server request parameters."""
        options = self.options
        params = dict(
            backend=self.get_sampling_backend(),
            cuda_quick=options.get('With CUDA Quick', False),
            with_threads=options.get('With Threads', False),
            with_children=options.get('With Children', getattr(self.remote_args, 'with_children', False)),
            native=options['PyStack Frames'] == 'Python + C/C++',
            sample_type=options['Sample Type'],
            edge_type=options['Edge Type'],
            num_retries=options['Num Retries'],
            retry_freq=options['Retry Frequency (us)'],
            max_threads_per_daemon=options['Max Threads Per Daemon'],
            clear_on_sample=options['Clear On Sample'],
        )
        if params['backend'] != 'pystack':
            params['native'] = params['with_children'] = False
        # keys must match remote API parameters; values must match the option names in STATGUI.py.
        fields = {
            'with_openmp': 'With OpenMP', 'python_trace': 'Gather Python Traces',
            'topology_type': 'Topology Type', 'topology': 'Topology',
            'communication_nodes': 'Communication Nodes', 'cp_policy': 'CP Policy',
            'comm_procs_per_node': 'Communication Processes per Node',
            'daemons_per_node': 'Daemons per Node', 'output_job_id': 'Output Job ID',
            'verbosity': 'Verbosity Type', 'debug_backends': 'Debug Backends',
            'check_node_access': 'Check Node Access', 'log_frontend': 'Log Frontend',
            'log_backend': 'Log Backend', 'log_cp': 'Log CP', 'log_sw': 'Log SW',
            'log_swerr': 'Log SWERR', 'use_mrnet_printf': 'Use MRNet Printf',
        }
        params.update((field, options[option]) for field, option in fields.items() if option in options)
        if options.get('Log Dir'):
            params['log_dir'] = options['Log Dir']
        params['var_spec'] = getattr(self, 'var_spec', [])
        return params

    def launch_application_cb(self, *args):
        """Explain that remote GUI sampling requires an existing target."""
        return self._error(RuntimeError('Select an existing PID or scheduler job to attach'))

    def on_reattach(self, action):
        """Attach again using the last target and its sampling parameters."""
        try:
            self._attach_remote(self.last_remote_target)
            return 0
        except Exception as exc:
            return self._error(exc)

    def _control(self, operation, method):
        """Run a session control request and report errors in the GUI."""
        try:
            self._run(operation, self.remote.request, method)
            return 0
        except Exception as exc:
            return self._error(exc)

    def on_pause(self, action=None):
        """Ask the server to pause the attached application."""
        return self._control('Pausing application…', 'sessions.pause')

    def on_resume(self, action=None):
        """Ask the server to continue the attached application."""
        return self._control('Resuming application…', 'sessions.resume')

    def sample(self, keep_var_spec=False):
        """Request one snapshot and display its call graph."""
        return self._sample(False)

    def sample_multiple(self):
        """Request repeated sampling and display the merged graph."""
        return self._sample(True)

    def _collect_multiple(self, params):
        """Display the server's progress while one SSH request collects all snapshots."""
        from gi.repository import GLib
        from STATview import stat_wait_dialog

        def collect():
            dialog = stat_wait_dialog.wait_dialog
            dialog.set_modal(True)
            dialog.set_deletable(False)
            bar = stat_wait_dialog.progress_bar
            bar.set_show_text(True)
            bar.set_text('Waiting for the first sample…')

            def display(event):
                if stat_wait_dialog.wait_dialog is dialog and dialog.get_visible():
                    fraction = max(bar.get_fraction(), event['fraction'])
                    if 'trace_index' in event:
                        bar.set_text('Sample %d of %d' % (event['trace_index'], event['trace_count']))
                    else:
                        bar.set_text('Gathering merged graph…')
                    if event.get('task_complete'):
                        stat_wait_dialog.update(fraction)
                    else:
                        bar.set_fraction(fraction)
                return False

            # the SSH worker posts progress to the GTK thread.
            def on_progress(event):
                GLib.idle_add(display, event)

            return self._run('Sampling stacks…', self.remote.request, 'sessions.sample_multiple', params, on_progress)

        # the server owns the sampling loop and cannot cancel an in-flight request.
        return stat_wait_dialog.show_wait_dialog_and_run(
            collect, (), ['Sample Stack Traces', 'Gather Stack Traces'], self, cancelable=False)

    def _sample(self, multiple):
        """Send the sampling request and populate tabs with the requested graphs."""
        try:
            self.update_sample_options()
            params = self.get_remote_params()
            if multiple:
                params.update(
                    num_traces=self.options['Num Traces'],
                    trace_frequency_ms=self.options['Trace Frequency (ms)'],
                )
                result = self._collect_multiple(params)
                if result is False:
                    return False  # the wait dialog already displayed the request error
            else:
                result = self._run('Sampling stacks…', self.remote.request, 'sessions.sample', params)

            # give each snapshot its own tab and leave the final tab for the merged graph.
            if multiple and self.options['Gather Individual Samples']:
                for snapshot in result.get('samples', []):
                    self._show_dot(snapshot)
                    self.create_new_tab()
                    self.notebook.set_current_page(len(self.tabs) - 1)

            self._show_dot(result)
            return 0
        except Exception as exc:
            return self._error(exc)

    def _show_dot(self, result):
        """Load a returned graph and bind its source lookup to the session that sampled it."""
        from STAThelper import get_task_list
        filename, dot = self.remote.save_dot(result)
        page = self.notebook.get_current_page()
        self.tabs[page].history_view.get_buffer().set_text('')
        self.set_dotcode(dot, filename, page)
        self.tabs[page].label.set_text(os.path.basename(filename))

        session_id = self.remote.session_id
        owner = self.remote

        def resolve(source, node):
            """Fetch the selected source using rank information from this graph."""
            try:
                ranks = get_task_list(getattr(node, 'edge_label', ''))
            except (ValueError, IndexError):
                ranks = []
            return self._run('Fetching source…', owner.source, source, ranks, session_id)

        self.tabs[page].widget.graph.source_resolver = resolve
        self.update_history()

    def update_prefs_and_sample_cb(self, sample_function, widget, dialog, action):
        """Apply the initial run time before starting the selected sample."""
        from STATview import stat_wait_dialog

        def wait_before_sample(seconds):
            wait_dialog = stat_wait_dialog.wait_dialog
            wait_dialog.set_title('Run Time Before Sample')
            wait_dialog.set_modal(True)
            wait_dialog.set_deletable(False)
            self.sleep(seconds)

        dialog.hide()
        try:
            self.update_sample_options()
            seconds = self.options['Run Time Before Sample (sec)']
            if seconds > 0:
                if self.remote.state != 'running' and self.on_resume() is False:
                    return False
                if stat_wait_dialog.show_wait_dialog_and_run(wait_before_sample, (seconds,), parent=self) is False:
                    return False
            return sample_function()
        finally:
            dialog.destroy()

    def _derive_graph(self, function, *args):
        """Carry the source resolver over when a viewer action replaces the graph."""
        resolver = self.get_current_graph().source_resolver
        result = function(*args)
        if result:
            self.get_current_graph().source_resolver = resolver
        return result

    def on_reset_layout(self, invert=False):
        """Reset the layout while preserving remote source lookup."""
        return self._derive_graph(super().on_reset_layout, invert)

    def on_translate(self):
        """Translate the graph while preserving remote source lookup."""
        return self._derive_graph(super().on_translate)

    def on_cache_sources(self, action):
        """Copy remote source files into the viewer cache for this graph."""
        from STATview import get_cache_file_path, dot_file_path_split

        graph = self.get_current_graph()
        if graph.source_resolver is None:
            return super().on_cache_sources(action)

        copied = set()
        try:
            directory, _ = dot_file_path_split(graph)
            # merged nodes can list several source paths; use the first node for each path to identify ranks to query.
            for node in graph.nodes:
                for source in node.attrs.get('source', '').split('\\n'):
                    if source in ('', '?', '(null)') or source in copied:
                        continue

                    local = graph.source_resolver(source, node)
                    destination = Path(get_cache_file_path(directory, source))
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(local, destination)
                    copied.add(source)
        except Exception as exc:
            return self._error(exc)

    def get_full_edge_label(self, widget, button_clicked, node):
        """Explain how to obtain all ranks for an edge sampled in representative mode."""
        from STATview import show_error_dialog

        show_error_dialog(
            'Resample with the full list edge option to retrieve all ranks.', self)

    def on_properties(self, action):
        """Show the server's current application status."""
        from STATview import gtk
        import json
        try:
            result = self._run('Getting application status…', self.remote.request, 'sessions.status')
            dialog = gtk.Dialog('Remote application', self)
            dialog.vbox.pack_start(gtk.Label(json.dumps(result, indent=2)), True, True, 5)
            dialog.add_button('Close', gtk.ResponseType.CLOSE)
            dialog.show_all()
            dialog.run()
            dialog.destroy()
        except Exception as exc:
            self._error(exc)

    def on_about(self, action):
        """Display version information for the remote GUI."""
        from STATview import gtk

        dialog = gtk.AboutDialog(transient_for=self)
        dialog.set_program_name('STAT remote GUI')
        dialog.set_version('4.3.0')
        dialog.run()
        dialog.destroy()

    def on_detach(self, widget=None, **kwargs):
        """Wait for remote detach before enabling another attachment."""
        try:
            self._run('Detaching…', self.remote.detach)
            self.attached = False
            self.set_action_sensitivity('detached')
            return 0
        except Exception as exc:
            return self._error(exc)

    def on_fatal_error(self):
        """Keep detach available if the server still owns the attachment."""
        if self.remote and self.remote.session_id:
            self.set_action_sensitivity('paused')

    def on_destroy(self, action):
        """Close the remote session before leaving the GTK main loop."""
        from STATview import gtk

        if self.remote_busy or self.closing:
            return
        self.closing = True
        if self.remote:
            try:
                self._run('Closing connection…', self.remote.close)
            except Exception as exc:
                import sys
                sys.stderr.write('Remote detach unconfirmed: %s\n' % exc)
        gtk.main_quit()
