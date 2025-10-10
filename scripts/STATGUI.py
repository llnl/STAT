#!/bin/env python

"""@package STATGUI
A GUI for driving the Stack Trace Analysis Tool."""

__copyright__ = """Copyright (c) 2007-2020, Lawrence Livermore National Security, LLC."""
__license__ = """Produced at the Lawrence Livermore National Laboratory
Written by Gregory Lee <lee218@llnl.gov>, Dorian Arnold, Matthew LeGendre, Dong Ahn, Bronis de Supinski, Barton Miller, Martin Schulz, Niklas Nielson, Nicklas Bo Jensen, Jesper Nielson, and Sven Karlsson.
LLNL-CODE-750488.
All rights reserved.

This file is part of STAT. For details, see http://www.github.com/LLNL/STAT. Please also read STAT/LICENSE.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

        Redistributions of source code must retain the above copyright notice, this list of conditions and the disclaimer below.
        Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the disclaimer (as noted below) in the documentation and/or other materials provided with the distribution.
        Neither the name of the LLNS/LLNL nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
__author__ = ["Gregory Lee <lee218@llnl.gov>", "Dorian Arnold", "Matthew LeGendre", "Dong Ahn", "Bronis de Supinski", "Barton Miller", "Martin Schulz", "Niklas Nielson", "Nicklas Bo Jensen", "Jesper Nielson"]
__version_major__ = 4
__version_minor__ = 2
__version_revision__ = 2
__version__ = "%d.%d.%d" %(__version_major__, __version_minor__, __version_revision__)

import STAThelper
from STAThelper import var_spec_to_string, get_task_list, get_proctab, HAVE_PYGMENTS, exec_and_exit
if HAVE_PYGMENTS:
    import pygments
    import pango
    from pygments.lexers import CLexer
    from pygments.lexers import CppLexer
    from pygments.lexers import FortranLexer
    from STAThelper import STATviewFormatter
import STATview
from STATview import STATDotWindow, stat_wait_dialog, show_error_dialog, search_paths, STAT_LOGO, run_gtk_main_loop

import sys
import os
import ctypes

HAVE_DLOPEN = hasattr(sys, 'getdlopenflags')
if HAVE_DLOPEN is True:
    dlflags = sys.getdlopenflags()
    new_dlflags = ctypes.RTLD_GLOBAL | dlflags
    sys.setdlopenflags(new_dlflags)

from STAT import STAT_FrontEnd, intArray, STAT_LOG_NONE, STAT_LOG_FE, STAT_LOG_BE, STAT_LOG_CP, STAT_LOG_MRN, STAT_LOG_SW, STAT_LOG_SWERR, STAT_OK, STAT_APPLICATION_EXITED, STAT_VERBOSE_ERROR, STAT_VERBOSE_FULL, STAT_VERBOSE_STDOUT, STAT_TOPOLOGY_AUTO, STAT_TOPOLOGY_DEPTH, STAT_TOPOLOGY_FANOUT, STAT_TOPOLOGY_USER, STAT_PENDING_ACK, STAT_LAUNCH, STAT_ATTACH, STAT_SERIAL_ATTACH, STAT_GDB_ATTACH, STAT_SERIAL_GDB_ATTACH, STAT_SAMPLE_FUNCTION_ONLY, STAT_SAMPLE_LINE, STAT_SAMPLE_PC, STAT_SAMPLE_COUNT_REP, STAT_SAMPLE_THREADS, STAT_SAMPLE_CLEAR_ON_SAMPLE, STAT_SAMPLE_PYTHON, STAT_SAMPLE_MODULE_OFFSET, STAT_CP_NONE, STAT_CP_SHAREAPPNODES, STAT_CP_EXCLUSIVE
HAVE_OPENMP_SUPPORT = True
try:
    from STAT import STAT_SAMPLE_OPENMP
except:
    HAVE_OPENMP_SUPPORT = False
HAVE_GDB_SUPPORT = True
try:
    from STAT import STAT_SAMPLE_CUDA_QUICK
except:
    HAVE_GDB_SUPPORT = False
HAVE_DYSECT = True
try:
    from STAT import DysectAPI_OK, DysectAPI_Error, DysectAPI_InvalidSystemState, DysectAPI_LibraryNotLoaded, DysectAPI_SymbolNotFound, DysectAPI_SessionCont, DysectAPI_SessionQuit, DysectAPI_DomainNotFound, DysectAPI_NetworkError, DysectAPI_DomainExpressionError, DysectAPI_StreamError, DysectAPI_OptimizedOut
    import DysectView
    from DysectView import DysectDotWindow
except:
    HAVE_DYSECT = False
HAVE_ATTACH_HELPER = True
try:
    import attach_helper
except:
    HAVE_ATTACH_HELPER = False
HAVE_PSUTIL = True
try:
    import psutil
except:
    HAVE_PSUTIL = False
import subprocess
HAVE_COMMANDS=True
try:
    import commands
except:
    HAVE_COMMANDS=False
    commands = subprocess
import time
import string
import re
import argparse

try:
    import xdot
except:
    raise Exception('STATview requires xdot\nxdot can be downloaded from https://github.com/jrfonseca/xdot.py\nWhen installing STAT via Spack, please be sure to `spack activate py-xdot`')
try:
    # xdot 0.9: compatibility wrapper
    xdot_ui_actions = xdot.ui.actions
    xdot_ui_elements = xdot.ui.elements
    xdot_ui_window = xdot.ui.window
    xdot_dot_parser = xdot.dot.parser
    xdot_dot_lexer = xdot.dot.lexer
except:
    # STAT-shipped xdot.py wrapper
    xdot_ui_actions = xdot
    xdot_ui_elements = xdot
    xdot_ui_window = xdot
    xdot_dot_parser = xdot
    xdot_dot_lexer = xdot

has_gtk = False
import_error = ''
try:
    import gtk
    import gobject
    import pango
    gtk_wrap_new_with_label_from_widget = gtk.RadioButton
    gtk_wrap_new_from_widget = gtk.RadioButton
    has_gtk = True
except ImportError as e:
    import_error = e
    pass #raise Exception('%s\nSTATview requires gtk and gobject\n' % repr(e))
except RuntimeError as e:
    import_error = e
    pass #raise Exception('%s\nThere was a problem loading the gtk and gobject module.\nIs X11 forwarding enabled?\n' % repr(e))
except Exception as e:
    import_error = e
    pass #raise Exception('%s\nThere was a problem loading the gtk module.\n' % repr(e))

if has_gtk == False:
    try:
        import gi
        gi.require_version('Gtk', '3.0')
        from gi.repository import Gtk as gtk
        from gi.repository import Gdk as gdk
        from gi.repository import GObject as gobject
        from gi.repository import GdkPixbuf
        gtk.POLICY_AUTOMATIC = gtk.PolicyType.AUTOMATIC
        gtk.POLICY_NEVER = gtk.PolicyType.NEVER
        gtk.Orientation.VERTICAL = gtk.Orientation.VERTICAL
        gtk.POS_TOP = gtk.PositionType.TOP
        gtk.gdk = gdk
        gtk.ORIENTATION_VERTICAL = gtk.Orientation.VERTICAL
        gtk.gdk.pixbuf_new_from_file = GdkPixbuf.Pixbuf.new_from_file
        gtk_wrap_new_with_label_from_widget = gtk.RadioButton.new_with_label_from_widget
        gtk_wrap_new_from_widget = gtk.RadioButton.new_from_widget
    except Exception as e2:
        raise Exception('%s\n%s\nThere was a problem loading the gtk module.\n' % (repr(import_error), repr(e2)))


## The STATGUI window adds STAT operations to the STATview window.
class STATGUI(STATDotWindow):
    """The STATGUI window adds STAT operations to the STATview window."""

    ui2 = ''
    ui2 += '<ui>\n'
    ui2 += '    <toolbar name="STAT_Actions">\n'
    ui2 += '        <toolitem action="Attach"/>\n'
    ui2 += '        <toolitem action="ReAttach"/>\n'
    ui2 += '        <toolitem action="Detach"/>\n'
    ui2 += '        <toolitem action="Pause"/>\n'
    ui2 += '        <toolitem action="Resume"/>\n'
#    ui2 += '        <toolitem action="Kill"/>\n'
    ui2 += '        <toolitem action="Sample"/>\n'
    ui2 += '        <toolitem action="SampleMultiple"/>\n'
    ui2 += '    </toolbar>\n'
    ui2 += '</ui>\n'

    def __init__(self, args):
        """The constructor.

        Initializes all instance varaibles and creates the GUI window.
        """
        if not os.path.exists('%s/.STAT' % os.environ['HOME']):
            try:
                os.mkdir('%s/.STAT' % os.environ['HOME'])
            except:
                pass
        self.STAT = STAT_FrontEnd()
        self.properties_window = None
        self.proctab_file_path = None
        self.proctab = None
        self.marked_line_number = None
        self.attached = False
        self.reattach = False
        self.serial_attach = False
        self.visualizer_window = None
        self.sample_task_list = ['Sample Stack Traces', 'Gather Stack Traces', 'Render Stack Traces']
        self.attach_task_list = ['Launch Daemons', 'Connect to Daemons', 'Attach to Application']
        self.attach_task_list += self.sample_task_list
        types = {'Topology Type':     ['automatic', 'depth', 'max fanout', 'custom'],
                 'Verbosity Type':    ['error', 'stdout', 'full'],
                 'Sample Type':       ['function only', 'function and pc', 'module offset', 'function and line'],
                 'Edge Type':         ['full list', 'count and representative'],
                 'Remote Host Shell': ['rsh', 'ssh'],
                 'Serial Remote Host Shell': ['rsh', 'ssh'],
                 'Resource Manager':  ['Auto', 'Alps', 'Slurm'],
                 'CP Policy':         ['none', 'share app nodes', 'exclusive']}
        if not hasattr(self, "types"):
            self.types = {}
        for opt in types:
            self.types[opt] = types[opt]
        try:
            gdb_path = os.environ['STAT_GDB']
        except:
            gdb_path = 'gdb'
        options = {'Remote Host':                      "localhost",
                   'Remote Host Shell':                "ssh",
                   'Serial Remote Host Shell':         "ssh",
                   'Resource Manager':                 "Auto",
                   'PID':                              None,
                   'Launcher Exe':                     '',
                   'Serial PID':                       None,
                   'Serial Exe':                       '',
                   'Serial Process List':              '',
                   'Topology Type':                    'automatic',
                   'Topology':                         '1',
                   'Check Node Access':                False,
                   'CP Policy':                        'share app nodes',
                   'Daemons per Node':                 1,
                   'Tool Daemon Path':                 self.STAT.getToolDaemonExe(),
                   'Filter Path':                      self.STAT.getFilterPath(),
                   'Job Launcher':                     'mpirun|srun|sattach|orterun|aprun|runjob|wreckrun|mpiexec|jsrun|job',
                   'Job ID':                           '',
                   'Filter Ranks':                     '',
                   'Filter Hosts':                     '',
                   'Filter Full Command Line':         False,
                   'Filter Full Command Line ':        False, # this option is for serial attach and adds a space at the end to avoid conflict
                   'Log Dir':                          os.environ['HOME'],
                   'Output Job ID':                    '',
                   'Log Frontend':                     False,
                   'Log Backend':                      False,
                   'Log CP':                           False,
                   'Log SW':                           False,
                   'Log SWERR':                        False,
                   'Use MRNet Printf':                 False,
                   'GDB BE':                           False,
                   'GDB Path':                         gdb_path,
                   'With CUDA Quick':                  False,
                   'Verbosity Type':                   'error',
                   'Communication Nodes':              '',
                   'Communication Processes per Node': 8,
                   'Num Traces':                       10,
                   'Trace Frequency (ms)':             1000,
                   'Num Retries':                      5,
                   'Retry Frequency (us)':             10,
                   'Max Threads Per Daemon':           512,
                   'With Threads':                     False,
                   'With OpenMP':                      False,
                   'Gather Python Traces':             False,
                   'Clear On Sample':                  True,
                   'Gather Individual Samples':        False,
                   'Run Time Before Sample (sec)':     0,
                   'Enable DySectAPI':                 False,
                   'DySectAPI Session':                '',
                   'DySectAPI Session Args':           '',
                   'DySectAPI Show Default Probes':    False,
                   'Sample Type':                      'function only',
                   'Edge Type':                        'full list',
                   'DDT Path':                         STAThelper.which('ddt'),
                   'DDT LaunchMON Prefix':             '/usr/local',
                   'TotalView Path':                   STAThelper.which('totalview'),
                   'Additional Debugger Args':         ''}
        if not hasattr(self, "options"):
            self.options = {}
        for option in options:
            self.options[option] = options[option]
        if 'STAT_CHECK_NODE_ACCESS' in os.environ:
            self.options['Check Node Access'] = True
        if 'STAT_LMON_DEBUG_BES' in os.environ:
            self.options['Debug Backends'] = True
        else:
            self.options['Debug Backends'] = False

        # Check for site default options then for user default options
        site_options_path = '%s/etc/STAT/STAT.conf' % self.STAT.getInstallPrefix()
        user_options_path = '%s/.STATrc' % (os.environ.get('HOME'))
        for path in [site_options_path, user_options_path]:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        for line in f:
                            if line[0] == '#' or line.strip() == '':
                                continue
                            split_line = line.split('=')
                            if len(split_line) != 2:
                                sys.stderr.write('invalid preference specification %s in file %s\n' % (line.strip('\n'), path))
                                continue
                            option = split_line[0].strip()
                            value = split_line[1].strip()
                            if option in self.options.keys():
                                if type(self.options[option]) == int:
                                    value = int(value)
                                elif type(self.options[option]) == bool:
                                    if value.lower() == 'true':
                                        value = True
                                    elif value.lower() == 'false':
                                        value = False
                                    else:
                                        sys.stderr.write('invalid value %s for option %s as specified in file %s.  Expecting either "true" or "false".\n' % (value, option, path))
                                        continue
                                self.options[option] = value
                            elif option == 'Source Search Path':
                                if os.path.exists(value):
                                    search_paths['source'].append(value)
                                else:
                                    sys.stderr.write('search path %s specified in %s is not accessible\n' % (value, path))
                            elif option == 'Include Search Path':
                                if os.path.exists(value):
                                    search_paths['include'].append(value)
                                else:
                                    sys.stderr.write('search path %s specified in %s is not accessible\n' % (value, path))
                            else:
                                sys.stderr.write('invalid option %s in file %s\n' % (option, path))
                except IOError as e:
                    sys.stderr.write('%s\nfailed to open preferences file %s\n' % (repr(e), path))
                    continue
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    sys.stderr.write('%s\nfailed to process preferences file %s\n' % (repr(e), path))
                    continue

        if args is not None:
            if args.withmoduleoffset is True:
                self.options['Sample Type'] = 'module offset'
            elif args.withpc is True:
                self.options['Sample Type'] = 'function and pc'
            elif args.withline is True:
                self.options['Sample Type'] = 'function and line'
            if args.countrep is True:
                self.options['Edge Type'] = 'count and representative'
            if args.pythontrace is True:
                self.options['Gather Python Traces'] = True
            if args.withthreads is True:
                self.options['With Threads'] = True
            if args.withopenmp is True:
                self.options['With Threads'] = True
                self.options['With OpenMP'] = True
            if args.debugdaemons is True:
                self.options['Debug Backends'] = True
            if args.logdir is not None:
                self.options['Log Dir'] = args.logdir
            if HAVE_GDB_SUPPORT:
                if args.gdb is True:
                    self.options['GDB BE'] = True
                if args.cudaquick is True:
                    self.options['With CUDA Quick'] = True
            if args.log is not None:
                if 'CP' in args.log:
                    self.options['Log CP'] = True
                if 'FE' in args.log:
                    self.options['Log Frontend'] = True
                if 'BE' in args.log:
                    self.options['Log Backend'] = True
            if args.mrnetprintf is True:
                self.options['Use MRNet Printf'] = True
            if args.jobid is not None:
                self.options['Output Job ID'] = args.jobid
            if args.sleep is not None:
                sys.stdout.write('sleeping for %d seconds\n' % (args.sleep))
                time.sleep(args.sleep)
        self.var_spec = []
        self.filter_entry = None
        self.serial_filter_entry = None
        uimanager = gtk.UIManager()
        self.actiongroup = gtk.ActionGroup('Commands')
        actions = []
        actions.append(('Attach', gtk.STOCK_CONNECT, 'Attach', None, 'Attach to an application', self.on_attach))
        actions.append(('ReAttach', gtk.STOCK_CONNECT, 'ReAttach', None, 'Attach to the last application STAT detached from', lambda a: stat_wait_dialog.show_wait_dialog_and_run(self.on_reattach, (a,), self.attach_task_list, self)))
        actions.append(('Detach', gtk.STOCK_DISCONNECT, 'Detach', None, 'Detach from the application', lambda a: stat_wait_dialog.show_wait_dialog_and_run(self.on_detach, (a,), ["Detach From Application"], self)))
        actions.append(('Pause', gtk.STOCK_MEDIA_PAUSE, 'Pause', None, 'Pause the application', lambda a: stat_wait_dialog.show_wait_dialog_and_run(self.on_pause, (a,), ["Pause Application"], self)))
        actions.append(('Resume', gtk.STOCK_MEDIA_PLAY, 'Resume', None, 'Resume the application', lambda a: stat_wait_dialog.show_wait_dialog_and_run(self.on_resume, (a,), ['Resume Application'], self)))
#        actions.append(('Kill', gtk.STOCK_STOP, 'Kill', None, 'Kill the application', lambda a: stat_wait_dialog.show_wait_dialog_and_run(self.on_kill, (a,), ['Terminating Application'], self)))
        actions.append(('Sample', gtk.STOCK_FIND, 'Sample', None, 'Gather a stack trace sample from the application', lambda a: self.on_sample(a, False)))
        actions.append(('SampleMultiple', gtk.STOCK_ZOOM_IN, 'Sample\nMultiple', None, 'Gather the merged stack traces accumulated over time', lambda a: self.on_sample(a, True)))
        self.actiongroup.add_actions(actions)
        uimanager.insert_action_group(self.actiongroup, 0)
        self.set_action_sensitivity('new')
        uimanager.add_ui_from_string(self.ui2)
        toolbar = uimanager.get_widget('/STAT_Actions')
        toolbar.set_orientation(gtk.ORIENTATION_VERTICAL)
        self.hbox = gtk.HBox()
        self.hbox.pack_start(toolbar, False, False, 0)
        lines = STATDotWindow.ui.split('\n')
        count = 0
        for line in lines:
            count += 1
            if line.find('menuitem action="SaveAs') != -1:
                break
        lines.insert(count, '            <menuitem action="Properties"/>')
        lines.insert(count, '            <separator/>')
        STATDotWindow.ui = ''
        for line in lines:
            STATDotWindow.ui += line + '\n'
        menu_actions = []
        menu_actions.append(('Properties', gtk.STOCK_PROPERTIES, 'P_roperties', '<control>R', 'View application properties', self.on_properties))
        menu_actions.append(('About', gtk.STOCK_ABOUT, None, None, None, self.on_about))
        STATDotWindow.__init__(self, menu_actions)
        self.set_title('STAT')
        help_text = """Search for processes on a
specified host, range of hosts, or
semi-colon separated list of hosts.
Example specifications:
host1
host1;host2
host[1-10]
host[1-10,12,15-20]
host[1-10,12,15-20];otherhost[30]
"""
        self.search_types.append(('hosts', self.search_hosts, help_text))

        if args is not None:
            if args.attach is not None:
                self.options['PID'] = int(args.attach.split(':')[-1])
                if args.attach.find(':') != -1:
                    self.options['Remote Host'] = args.attach.split(':')[0]
                if HAVE_GDB_SUPPORT:
                    if args.gdb is not None:
                        self.options['GDB BE'] = True
                stat_wait_dialog.show_wait_dialog_and_run(self.attach_cb, (None, False, False, STAT_ATTACH), self.attach_task_list)
                return
            elif args.serial is not None:
                self.serial_attach = True
                stat_wait_dialog.show_wait_dialog_and_run(self.attach_serial_processes_cb, (None, None, args.serial), self.attach_task_list)
                return
            elif args.create is not None:
                stat_wait_dialog.show_wait_dialog_and_run(self.launch_application_cb, (None, None, args.create), self.attach_task_list)
                return
        self.on_attach(None)

    def search_hosts(self, text, match_case_check_box):
        """Callback to handle activation of focus task text entry."""
        if match_case_check_box.get_active() is False:
            text = text.lower()
        temp_host_list = text.replace(' ', '').split(';')
        host_list = []
        for host in temp_host_list:
            if host.find('[') != -1:
                prefix = host[0:host.find('[')]
                host_range = host[host.find('['):host.find(']') + 1]
                node_list = get_task_list(host_range)
                for node in node_list:
                    node_name = '%s%d' % (prefix, node)
                    host_list.append(node_name)
            else:
                host_list.append(host)
        ret = self.set_proctab()
        if ret is False:
            return False
        tasks = ''
        for rank, host, pid, exe_index in self.proctab.process_list:
            if match_case_check_box.get_active() is False:
                host = host.lower()
            if host in host_list:
                tasks += str(rank) + ','
        tasks = tasks[0:-1]
        self.get_current_graph().focus_tasks(tasks)
        return True

    def on_about(self, action):
        """Display info about STAT."""
        about_dialog = gtk.AboutDialog()
        about_dialog.set_name('STAT')
        about_dialog.set_authors(__author__)
        about_dialog.set_copyright(__copyright__)
        about_dialog.set_license(__license__)
        about_dialog.set_wrap_license(80)
        version = intArray(3)
        if self.STAT is None:
            self.STAT = STAT_FrontEnd()
        self.STAT.getVersion(version)
        about_dialog.set_version('%d.%d.%d' % (version[0], version[1], version[2]))
        try:
            pixbuf = gtk.gdk.pixbuf_new_from_file(STAT_LOGO)
            about_dialog.set_logo(pixbuf)
        except Exception as e:
            pass
        about_dialog.set_website('https://github.com/lee218llnl/STAT')
        about_dialog.show_all()
        about_dialog.run()
        about_dialog.destroy()

    def set_action_sensitivity(self, state):
        """Set the sensitivity of the buttons."""
        for action in self.actiongroup.list_actions():
            action.set_sensitive(True)
            if action.get_name() in ['ReAttach', 'Detach', 'Pause', 'Resume', 'Kill', 'Sample', 'SampleMultiple', 'SampleMultiple'] and state == 'new':
                action.set_sensitive(False)
            elif action.get_name() in ['Detach', 'Pause', 'Resume', 'Kill', 'Sample', 'SampleMultiple', 'SampleMultiple'] and state == 'detached':
                action.set_sensitive(False)
            elif action.get_name() in ['Attach', 'ReAttach', 'Pause'] and state == 'paused':
                action.set_sensitive(False)
            elif action.get_name() in ['Attach', 'ReAttach', 'Resume'] and state == 'running':
                action.set_sensitive(False)
            elif state == 'busy':
                action.set_sensitive(False)

    def on_properties(self, action):
        """Display a window with application properties."""
        if self.attached is False:
            show_error_dialog('Application properties only available after attaching\n', self)
            return False
        if self.properties_window is not None:
            self.properties_window.present()
            return True
        self.options['Filter Ranks'] = ''
        self.options['Filter Hosts'] = ''
        self.ptab_sw = None

        # gather application properties
        num_nodes = self.STAT.getNumApplNodes()
        num_procs = self.STAT.getNumApplProcs()
        ret = self.set_proctab()
        if ret is False:
            show_error_dialog('Failed to set process table file\n', self)
            return False

        job_launcher = "%s:%d" % (self.proctab.launcher_host, self.proctab.launcher_pid)
        entries = range(len(self.proctab.process_list))
        entries = list(entries)
        for rank, host, pid, exe_index in self.proctab.process_list:
            entries[rank] = '%d %s:%d %d\n' % (rank, host, pid, exe_index)

        self.properties_window = gtk.Window()
        self.properties_window.set_title('Properties')
        self.properties_window.connect('destroy', self.on_properties_destroy)
        vbox = gtk.VBox()

        frame = gtk.Frame(label='Application Executable(s) (index:path)')
        text_view = gtk.TextView()
        text_buffer = gtk.TextBuffer()
        exes = ''
        for i in range(len(self.proctab.executable_paths)):
            exes += ('%d:%s\n' % (i, self.proctab.executable_paths[i]))
        text_buffer.set_text(exes)
        text_view.set_buffer(text_buffer)
        text_view.set_wrap_mode(False)
        text_view.set_editable(False)
        text_view.set_cursor_visible(False)
        frame.add(text_view)
        vbox.pack_start(frame, False, False, 0)

        frame = gtk.Frame(label='Number of application nodes')
        text_view = gtk.TextView()
        text_buffer = gtk.TextBuffer()
        text_buffer.set_text('%d' % num_nodes)
        text_view.set_buffer(text_buffer)
        text_view.set_wrap_mode(False)
        text_view.set_editable(False)
        text_view.set_cursor_visible(False)
        frame.add(text_view)
        vbox.pack_start(frame, False, False, 0)

        frame = gtk.Frame(label='Number of application processes')
        text_view = gtk.TextView()
        text_buffer = gtk.TextBuffer()
        text_buffer.set_text('%d' % num_procs)
        text_view.set_buffer(text_buffer)
        text_view.set_wrap_mode(False)
        text_view.set_editable(False)
        text_view.set_cursor_visible(False)
        frame.add(text_view)
        vbox.pack_start(frame, False, False, 0)

        if self.STAT.getApplicationOption() != STAT_SERIAL_ATTACH and self.STAT.getApplicationOption() != STAT_SERIAL_GDB_ATTACH:
            frame = gtk.Frame(label='Job Launcher (host:PID)')
            text_view = gtk.TextView()
            text_buffer = gtk.TextBuffer()
            text_buffer.set_text(job_launcher)
            text_view.set_buffer(text_buffer)
            text_view.set_wrap_mode(False)
            text_view.set_editable(False)
            text_view.set_cursor_visible(False)
            frame.add(text_view)
            vbox.pack_start(frame, False, False, 0)

        proctab_frame = gtk.Frame(label='Process Table (rank host:PID exe_index)')
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label('Filter Ranks'), False, False, 5)
        self.rank_filter_entry = self.pack_entry_and_button(self.options['Filter Ranks'], self.on_update_filter_ranks, proctab_frame, self.properties_window, "Filter Ranks", hbox, True, True, 0)
        vbox.pack_start(hbox, False, False, 0)
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label('Filter Hosts'), False, False, 5)
        self.host_filter_entry = self.pack_entry_and_button(self.options['Filter Hosts'], self.on_update_filter_hosts, proctab_frame, self.properties_window, "Filter Hosts", hbox, True, True, 0)
        vbox.pack_start(hbox, False, False, 0)
        self.on_update_proctab(None, proctab_frame, self.properties_window)
        vbox.pack_start(proctab_frame, True, True, 0)

        #frame = gtk.Frame(label='Process Table (rank host:PID exe_index)')
        #text_view = gtk.TextView()
        #text_view.set_size_request(400, 200)
        #text_buffer = gtk.TextBuffer()
        #text_buffer.set_text(process_table)
        #text_view.set_buffer(text_buffer)
        #text_view.set_wrap_mode(False)
        #text_view.set_editable(False)
        #text_view.set_cursor_visible(False)
        #sw = gtk.ScrolledWindow()
        #sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        #sw.add(text_view)
        #frame.add(sw)
        #vbox.pack_start(frame, True, True, 0)

        self.properties_window.add(vbox)
        self.properties_window.show_all()

    def on_properties_destroy(self, action):
        """Clean up the properties window."""
        self.properties_window = None

    def pid_toggle_cb(self, action, pid, command):
        """A callback to modify the PID option."""
        self.options['PID'] = pid
        self.options['Launcher Exe'] = command

    def serial_pid_toggle_cb(self, action, pid, command):
        """A callback to modify the PID option."""
        self.options['Serial PID'] = pid
        self.options['Serial Exe'] = command

    def on_add_serial_process(self, widget, attach_dialog, serial_process_list_entry):
        text = serial_process_list_entry.get_text()
        exe = self.options['Serial Exe']
        if (exe.find(' \\_') != -1):
            exe = exe.split()[1]
        else:
            exe = exe.split()[0]
        text += ' %s@%s:%s ' % (exe, self.options['Remote Host'], self.options['Serial PID'])
        text = serial_process_list_entry.set_text(text)
        attach_dialog.show_all()

    def on_update_filter_hosts(self, w, proctab_frame, parent, entry):
        self.options['Filter Hosts'] = entry.get_text()
        entry.set_text(self.options['Filter Hosts'])
        self.on_update_proctab(w, proctab_frame, parent)

    def on_update_filter_ranks(self, w, proctab_frame, parent, entry):
        self.options['Filter Ranks'] = entry.get_text()
        entry.set_text(self.options['Filter Ranks'])
        self.on_update_proctab(w, proctab_frame, parent)

    def on_update_proctab(self, widget, proctab_frame, proctab_window):
        try:
            if self.ptab_sw is not None:
                proctab_frame.remove(self.ptab_sw)
        except:
            pass
        rank_filter = get_task_list('[%s]' %self.options['Filter Ranks'])
        host_filter = []
        temp_host_list = self.options['Filter Hosts'].replace(' ', '').split(';')
        for host in temp_host_list:
            if host.find('[') != -1:
                prefix = host[0:host.find('[')]
                host_range = host[host.find('['):host.find(']') + 1]
                node_list = get_task_list(host_range)
                for node in node_list:
                    node_name = '%s%d' % (prefix, node)
                    host_filter.append(node_name)
            elif host != '':
                host_filter.append(host)
        self.ptab_sw = gtk.ScrolledWindow()
        self.ptab_sw.set_shadow_type(gtk.SHADOW_ETCHED_IN)
        self.ptab_sw.set_policy(gtk.POLICY_NEVER, gtk.POLICY_AUTOMATIC)
        (PTAB_INDEX_RANK, PTAB_INDEX_HOST, PTAB_INDEX_PID, PTAB_INDEX_EXE) = range(4)
        list_store = gtk.ListStore(gobject.TYPE_UINT, gobject.TYPE_STRING, gobject.TYPE_UINT, gobject.TYPE_UINT)
        for rank, host, pid, exe_index in self.proctab.process_list:
            if rank_filter != [] and rank not in rank_filter:
                continue
            if host_filter != [] and host not in host_filter:
                continue
            iterator = list_store.append()
            item = (rank, host, pid, exe_index)
            list_store.set(iterator, PTAB_INDEX_RANK, item[PTAB_INDEX_RANK], PTAB_INDEX_HOST, item[PTAB_INDEX_HOST], PTAB_INDEX_PID, item[PTAB_INDEX_PID], PTAB_INDEX_EXE, item[PTAB_INDEX_EXE])
        treeview = gtk.TreeView(list_store)
        treeview.set_rules_hint(True)
        treeview.set_search_column(PTAB_INDEX_RANK)
        treeview.set_size_request(400, 400)
        self.ptab_sw.add(treeview)
        proctab_frame.add(self.ptab_sw)

        column = gtk.TreeViewColumn('Rank', gtk.CellRendererText(), text=PTAB_INDEX_RANK)
        column.set_sort_column_id(PTAB_INDEX_RANK)
        treeview.append_column(column)
        column = gtk.TreeViewColumn('Host', gtk.CellRendererText(), text=PTAB_INDEX_HOST)
        column.set_sort_column_id(PTAB_INDEX_HOST)
        treeview.append_column(column)
        column = gtk.TreeViewColumn('PID', gtk.CellRendererText(), text=PTAB_INDEX_PID)
        column.set_sort_column_id(PTAB_INDEX_PID)
        treeview.append_column(column)
        column = gtk.TreeViewColumn('EXE', gtk.CellRendererText(), text=PTAB_INDEX_EXE)
        column.set_sort_column_id(PTAB_INDEX_EXE)
        treeview.append_column(column)
        self.properties_window.show_all()

    def on_update_process_listing(self, widget, frame, attach_dialog, listing_filter=None):
        """Generate a wait dialog and search for user processes."""
        stat_wait_dialog.show_wait_dialog_and_run(self._on_update_process_listing, (widget, frame, attach_dialog, listing_filter), [], attach_dialog)

    def on_update_serial_process_listing(self, widget, frame, attach_dialog, listing_filter=None):
        """Generate a wait dialog and search for user processes."""
        stat_wait_dialog.show_wait_dialog_and_run(self._on_update_serial_process_listing, (widget, frame, attach_dialog, listing_filter), [], attach_dialog)

    def _on_update_process_listing2(self, attach_dialog, listing_filter, vbox, is_parallel):
        """Search for user processes."""
        self.options['Remote Host Shell'] = self.types['Remote Host Shell'][self.combo_boxes['Remote Host Shell'].get_active()]
        self.options['Serial Remote Host Shell'] = self.types['Serial Remote Host Shell'][self.combo_boxes['Serial Remote Host Shell'].get_active()]
        if is_parallel is True:
            my_rsh = self.options['Remote Host Shell']
        else:
            my_rsh = self.options['Serial Remote Host Shell']
        pid_list = ''
        if listing_filter is not None:
            filter_compiled_re = re.compile(listing_filter.get_text())
        if HAVE_ATTACH_HELPER and self.options['Job ID'] != '':
            self.options['Remote Host'], rm_exe, pids = attach_helper.jobid_to_hostname_pid(self.options['Resource Manager'], self.options['Job ID'], my_rsh)
            self.options['Job Launcher'] = rm_exe
            if self.options['Remote Host'] == None:
                show_error_dialog('Failed to find host for job ID %s' % self.options['Job ID'], attach_dialog)
                self.options['Job ID'] = ''
                return False
            self.options['Job ID'] = ''
            for i, pid in enumerate(pids):
                if i != 0:
                    pid_list += ','
                pid_list += str(pid)
            self.filter_entry.set_text(self.options['Job Launcher'])
            self.rh_entry.set_text(self.options['Remote Host'])
        psutil_success = False
        process_list = [] # format of item = (pid, cmd_line)
        if HAVE_PSUTIL is True and (self.options['Remote Host'] == 'localhost' or self.options['Remote Host'] == ''):
            my_procs = []
            pid_depth = {}
            for pid in psutil.pids():
                try:
                    proc = psutil.Process(pid)
                    if proc.username() == os.getlogin():
                        my_procs.append(proc)
                except:
                    pass
            handled_procs = []
            for proc in my_procs:
                if proc in handled_procs:
                    continue
                cmd_line =  ' '.join(proc.cmdline())
                pid = proc.pid
                try:
                    exe = os.path.realpath(proc.exe())
                except:
                    continue
                if listing_filter is not None:
                    if listing_filter.get_text() != '':
                        if (not self.options["Filter Full Command Line"] and is_parallel) or (not self.options["Filter Full Command Line "] and not is_parallel):
                            my_filter = filter_compiled_re.search(exe)
                        else:
                            my_filter = filter_compiled_re.search(cmd_line)
                        if my_filter is None:
                            continue
                try:
                    pid_depth[pid] = pid_depth[proc.parent().pid] + 1
                except Exception as e:
                    pid_depth[pid] = 0
                handled_procs.append(proc)
                if pid_depth[pid] > 0:
                    cmd_line = ' \\_ '.rjust(pid_depth[pid] * 3 + 3) + cmd_line
                output = commands.getoutput('file %s' % (exe))
                process_list.append((pid, cmd_line))
            psutil_success = True
        if psutil_success == False:
            if self.options['Remote Host'] == 'localhost' or self.options['Remote Host'] == '':
                output = commands.getoutput('ps xww')
            else:
                if pid_list != '':
                    output = commands.getoutput('%s %s ps ww -p %s' % (my_rsh, self.options['Remote Host'], pid_list))
                else:
                    output = commands.getoutput('%s %s ps xww' % (my_rsh, self.options['Remote Host']))
                if output.find('Hostname not found') != -1 or output.find('PID') == -1:
                    show_error_dialog('Failed to get process listing for %s:\n\t %s' % (self.options['Remote Host'], output), attach_dialog)
                    return False
            output = output.split('\n')
            pid_index = 0
            for line in output:
                if line.find('PID') != -1:
                    break
                pid_index += 1
            output = output[pid_index:]
            line = output[0].split()
            pid_index = 0
            command_index = 0
            for counter, token in enumerate(line):
                if token == 'PID':
                    pid_index = counter
                elif token == 'COMMAND':
                    command_index = counter
            for line in output[1:]:
                try:
                    cmd_line = ''
                    for token in line.split()[command_index:]:
                        cmd_line += ' %s' % token
                except:
                    continue
                if listing_filter is not None:
                    if listing_filter.get_text() != '':
                        if (not self.options["Filter Full Command Line"] and is_parallel) or (not self.options["Filter Full Command Line "] and not is_parallel):
                            my_filter = filter_compiled_re.search(cmd_line.split()[0])
                        else:
                            my_filter = filter_compiled_re.search(cmd_line)
                        if my_filter is None:
                            continue
                process_list.append((int(line.split()[pid_index]), cmd_line))

        started = False
        for (pid, cmd_line) in process_list:
            text = '% 5d %s' %(pid, cmd_line)
            if started is False:
                started = True
                first_radio_button = gtk_wrap_new_with_label_from_widget(None, text)
                first_radio_button.set_active(True)
                radio_button = first_radio_button
                if is_parallel:
                    self.options['PID'] = pid
                    self.options['Launcher Exe'] = cmd_line
                else:
                    self.options['Serial PID'] = pid
                    self.options['Serial Exe'] = cmd_line
            else:
                radio_button = gtk_wrap_new_with_label_from_widget(first_radio_button, text)
                radio_button.set_active(False)
            if is_parallel:
                radio_button.connect("toggled", self.pid_toggle_cb, pid, cmd_line)
            else:
                radio_button.connect("toggled", self.serial_pid_toggle_cb, pid, cmd_line)
            vbox.pack_start(radio_button, False, False, 0)

    def _on_update_process_listing(self, widget, frame, attach_dialog, listing_filter=None):
        vbox = gtk.VBox()
        self._on_update_process_listing2(attach_dialog, listing_filter, vbox, True)
        try:
            if self.sw is not None:
                frame.remove(self.sw)
        except:
            pass
        self.sw = gtk.ScrolledWindow()
        self.sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        self.sw.add_with_viewport(vbox)
        frame.add(self.sw)
        attach_dialog.show_all()

    def _on_update_serial_process_listing(self, widget, frame, attach_dialog, listing_filter=None):
        vbox = gtk.VBox()
        self._on_update_process_listing2(attach_dialog, listing_filter, vbox, False)
        try:
            if self.serial_sw is not None:
                frame.remove(self.serial_sw)
        except:
            pass
        self.serial_sw = gtk.ScrolledWindow()
        self.serial_sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        self.serial_sw.add_with_viewport(vbox)
        frame.add(self.serial_sw)
        attach_dialog.show_all()

    def on_update_job_id(self, w, frame, attach_dialog, entry):
        """Callback to search a job id for resource manager processes."""
        self.options['Job ID'] = entry.get_text()
        entry.set_text(self.options['Job ID'])
        self.options['Resource Manager'] = self.types['Resource Manager'][self.combo_boxes['Resource Manager'].get_active()]
        self.on_update_process_listing(w, frame, attach_dialog, None)


    def on_update_remote_host(self, w, frame, attach_dialog, entry):
        """Callback to search a remote host for processes."""
        self.options['Remote Host'] = entry.get_text()
        entry.set_text(self.options['Remote Host'])
        self.on_update_process_listing(w, frame, attach_dialog, self.filter_entry)

    def on_update_serial_remote_host(self, w, frame, attach_dialog, entry):
        """Callback to search a remote host for processes."""
        self.options['Remote Host'] = entry.get_text()
        entry.set_text(self.options['Remote Host'])
        self.on_update_serial_process_listing(w, frame, attach_dialog, self.serial_filter_entry)

    def on_cancel_attach(self, widget, dialog):
        """Callback to handle canceling of attach dialog."""
        dialog.hide()

    def manipulate_cb(self, widget, data, node):
        """Overloaded manipulate callback to handle variable search."""
        STATDotWindow.manipulate_cb(self, widget, data, node)
        if data == 'Temporally Order Children':
            self.check_for_loop_variable()

    def on_to_traverse_least_progress(self, action):
        """Overloaded traverse callback to handle variable search."""
        STATDotWindow.on_to_traverse_least_progress(self, action)
        self.check_for_loop_variable()

    def on_to_traverse_most_progress(self, action):
        """Overloaded traverse callback to handle variable search."""
        STATDotWindow.on_to_traverse_most_progress(self, action)
        self.check_for_loop_variable()

    def clear_var_spec_and_destroy_dialog(self, dialog):
        """Handle case where we don't want to gather loop variable."""
        self.var_spec = []
        dialog.hide()

    def check_for_loop_variable(self):
        """Searches for loop ordering variables.

        If found, prompt user to gather stack trace with variable values."""
        found = False
        for node in self.get_current_graph().nodes:
            if node.attrs.get("lex_string") is not None:
                if node.attrs["lex_string"].find('$') != -1 and node.attrs["lex_string"].find('=') == -1:
                    source = node.attrs["source"]
                    line = int(node.attrs["line"].strip(":"))
                    temp = node.attrs["lex_string"][node.attrs["lex_string"].find('$') + 1:]
                    while 1:
                        var = temp[:temp.find('(')]
                        if var == 'iter#':
                            break
                        found = True
                        self.var_spec.append((source, line, node.depth, var))
                        if temp.find('$') != -1:
                            temp = temp[temp.find('$') + 1:]
                        else:
                            break
        if found is True:
            dialog = gtk.Dialog('Gather Loop Order Variable(s)?', self)
            label = gtk.Label('Temporal ordering requires STAT\nto fetch loop ordering variables.\nDo you want STAT to gather these variables?\nNote, this generates a new call prefix tree.')
            dialog.vbox.pack_start(label, False, False, 0)
            hbox = gtk.HButtonBox()
            button = gtk.Button(stock=gtk.STOCK_YES)

            #button.connect("clicked", lambda w: self.run_and_destroy_dialog(stat_wait_dialog.show_wait_dialog_and_run, (self.sample, (True,), self), self.sample_task_list, dialog))
            button.connect("clicked", lambda w: stat_wait_dialog.show_wait_dialog_and_run(self.run_and_destroy_dialog, (self.hacky_sample_keep_state, (), dialog), self.sample_task_list))

            hbox.pack_start(button, True, True, 0)
            button = gtk.Button('Open in\nnew tab')

            #button.connect("clicked", lambda w: self.run_and_destroy_dialog(self.new_tab_and_sample, (), dialog))
            button.connect("clicked", lambda w: stat_wait_dialog.show_wait_dialog_and_run(self.run_and_destroy_dialog, (self.hacky_sample_keep_state, (True,), dialog), self.sample_task_list))

            hbox.pack_start(button, True, True, 0)
            button = gtk.Button(stock=gtk.STOCK_NO)
            button.connect("clicked", lambda w: self.clear_var_spec_and_destroy_dialog(dialog))
            hbox.pack_start(button, True, True, 0)
            dialog.vbox.pack_start(hbox, False, False, 0)
            dialog.show_all()
            dialog.run()

    def new_tab_and_sample(self):
        """Create a new tab and gather a stack sample"""
        page = self.notebook.get_current_page()
        self.create_new_tab(page + 1)
        self.notebook.set_current_page(page + 1)
        self.sample(True)

    def hacky_sample_keep_state(self, new_tab=False):
        """Gather a new stack trace.

        Also perform TO operations from previous trace.
        """
        actions = self.get_current_graph().action_history
        if new_tab is True:
            page = self.notebook.get_current_page()
            self.create_new_tab(page + 1)
            self.notebook.set_current_page(page + 1)
        self.sample(True)
        for action in actions:
            if action.find('Traverse Most Progress') != -1:
                self.on_to_traverse_most_progress(None)
            if action.find('Traverse Least Progress') != -1:
                self.on_to_traverse_least_progress(None)
        self.get_current_graph().adjust_dims()
        self.get_current_widget().zoom_to_fit()

    def run_and_destroy_dialog(self, function, args, dialog):
        """Run the specified function and destroys the specified dialog."""
        function(*args)
        dialog.hide()

    def on_reattach(self, action):
        """Attach to the same job as the previous session.

        Also gather a stack trace.
        """
        if self.serial_attach is True:
            if self.STAT is None:
                self.STAT = STAT_FrontEnd()
            for process in self.process_list.split():
                self.STAT.addSerialProcess(process)
            if self.options['GDB BE'] is True:
                self.attach_cb(None, False, True, STAT_SERIAL_GDB_ATTACH)
            else:
                self.attach_cb(None, False, True, STAT_SERIAL_ATTACH)
        else:
            self.attach_cb(None, False, False, STAT_ATTACH)
        return True

    def on_choose_dysect_session(self, attach_dialog, widget):
        """Callback to generate an open file dialog."""
        chooser = gtk.FileChooserDialog(title="Select a DySectAPI session",
                                        action=gtk.FILE_CHOOSER_ACTION_OPEN,
                                        buttons=(gtk.STOCK_CANCEL,
                                                 gtk.RESPONSE_CANCEL,
                                                 gtk.STOCK_OPEN,
                                                 gtk.RESPONSE_OK))
        chooser.set_default_response(gtk.RESPONSE_OK)
        chooser.set_current_folder(os.getcwd())
        chooser.set_select_multiple(False)
        file_filter = gtk.FileFilter()
        file_filter.set_name("DySectAPI session source file")
        file_filter.add_pattern("*.cpp")
        file_filter.add_pattern("*.cxx")
        file_filter.add_pattern("*.CXX")
        file_filter.add_pattern("*.c")
        file_filter.add_pattern("*.C")
        chooser.add_filter(file_filter)
        file_filter = gtk.FileFilter()
        file_filter.set_name("All files")
        file_filter.add_pattern("*")
        chooser.add_filter(file_filter)
        if chooser.run() == gtk.RESPONSE_OK:
            filenames = chooser.get_filenames()
            chooser.destroy()
            self.options['DySectAPI Session'] = filenames[0]
            vbox = gtk.VBox()
            children = widget.get_children()
            for child in children:
                if type(child) == type(gtk.CheckButton()):
                    child.set_active(True)
                if type(child) == type(gtk.Label()):
                    child.set_text('Session: %s' % self.options['DySectAPI Session'])
            attach_dialog.show_all()
            self.options['Enable DySectAPI'] = True
        else:
            chooser.destroy()

    def on_attach(self, action):
        """Generate a dialog to attach to a new job."""
        if self.reattach is True and self.STAT is not None:
            self.STAT.shutDown()
            self.STAT = None
        if self.STAT is None:
            self.STAT = STAT_FrontEnd()
        self.options['PID'] = None
        self.options['Launcher Exe'] = ''
        self.options['Serial PID'] = None
        self.options['Serial Exe'] = ''
        attach_dialog = gtk.Dialog('Attach', self)
        attach_dialog.set_default_size(400, 400)
        notebook = gtk.Notebook()
        notebook.set_tab_pos(gtk.POS_TOP)
        notebook.set_scrollable(False)

        # parallel attach
        process_frame = gtk.Frame(label='Current Process List')
        vbox = gtk.VBox()

        self.rm_expander = gtk.Expander(label="Search for job by Resource Manager job ID")
        hbox = gtk.HBox()
        self.jobid_entry = self.pack_entry_and_button(self.options['Job ID'], self.on_update_job_id, process_frame, attach_dialog, "Search for Job", hbox, True, True, 0)
        hbox.pack_start(gtk.VSeparator(), False, False, 5)
        self.pack_combo_box(hbox, 'Resource Manager')
        self.rm_expander.add(hbox)
        vbox.pack_start(self.rm_expander, False, False, 0)

        self.hn_expander = gtk.Expander(label="Search for job by hostname")
        hbox = gtk.HBox()
        self.rh_entry = self.pack_entry_and_button(self.options['Remote Host'], self.on_update_remote_host, process_frame, attach_dialog, "Search Remote Host", hbox)
        self.hn_expander.add(hbox)
        self.hn_expander.set_expanded(True)
        vbox.pack_start(self.hn_expander, False, False, 0)

        self.rm_expander.connect("activate", lambda w: self.hn_expander.set_expanded(self.rm_expander.get_expanded()))
        self.hn_expander.connect("activate", lambda w: self.rm_expander.set_expanded(self.hn_expander.get_expanded()))

        vbox.pack_start(gtk.HSeparator(), False, False, 0)
        hbox = gtk.HBox()
        self.pack_combo_box(hbox, 'Remote Host Shell')
        vbox.pack_start(hbox, False, False, 0)
        vbox.pack_start(process_frame, True, True, 0)
        self.pack_check_button(vbox, 'Filter Full Command Line', False, False, 5)
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label('Filter Process List'), False, False, 5)
        self.filter_entry = self.pack_entry_and_button(self.options['Job Launcher'], self.on_update_process_listing, process_frame, attach_dialog, "Filter", hbox, True, True, 0)
        vbox.pack_start(hbox, False, False, 0)
        hbox = gtk.HBox()
        button = gtk.Button("Refresh Process List")
        button.connect("clicked", lambda w: self.on_update_process_listing(w, process_frame, attach_dialog, self.filter_entry))
        hbox.pack_start(button, True, True, 10)
        vbox.pack_start(hbox, False, False, 0)
        vbox.pack_start(gtk.HSeparator(), False, False, 10)
        hbox = gtk.HBox()
        button = gtk.Button(stock=gtk.STOCK_CANCEL)
        button.connect("clicked", lambda w: self.on_cancel_attach(w, attach_dialog))
        hbox.pack_start(button, False, False, 0)
        button = gtk.Button("Attach")
        button.connect("clicked", lambda w: stat_wait_dialog.show_wait_dialog_and_run(self.attach_cb, (attach_dialog, False, False, STAT_ATTACH), self.attach_task_list, attach_dialog))
        hbox.pack_start(button, True, True, 0)
        vbox.pack_start(hbox, False, False, 0)
        label = gtk.Label('Attach')
        notebook.append_page(vbox, label)

        # launch
        vbox = gtk.VBox()
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label('Enter application launch string:'), False, False, 0)
        vbox.pack_start(hbox, False, False, 10)
        entry = gtk.Entry()
        entry.set_max_length(8192)
        entry.connect("activate", lambda w: stat_wait_dialog.show_wait_dialog_and_run(self.launch_application_cb, (entry, attach_dialog), self.attach_task_list, attach_dialog))
        vbox.pack_start(entry, False, False, 0)
        hbox = gtk.HButtonBox()
        button = gtk.Button(stock=gtk.STOCK_CANCEL)
        button.connect("clicked", lambda w: self.on_cancel_attach(w, attach_dialog))
        hbox.pack_end(button, False, False, 0)
        button = gtk.Button("Launch")
        button.connect("clicked", lambda w: stat_wait_dialog.show_wait_dialog_and_run(self.launch_application_cb, (entry, attach_dialog), self.attach_task_list, attach_dialog))
        hbox.pack_start(button, False, False, 0)
        vbox.pack_end(hbox, False, False, 0)
        label = gtk.Label('Launch')
        notebook.append_page(vbox, label)

        # serial attach tab
        serial_process_frame = gtk.Frame(label='Current Process List')
        vbox = gtk.VBox()
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label('Search Host'), False, False, 5)
        self.pack_entry_and_button(self.options['Remote Host'], self.on_update_serial_remote_host, serial_process_frame, attach_dialog, "Search", hbox)
        hbox.pack_start(gtk.VSeparator(), False, False, 5)
        self.pack_combo_box(hbox, 'Serial Remote Host Shell')
        vbox.pack_start(hbox, False, False, 0)
        vbox.pack_start(serial_process_frame, True, True, 0)
        self.pack_check_button(vbox, 'Filter Full Command Line ', False, False, 5)
        hbox = gtk.HBox()
        hbox.pack_start(gtk.Label('Filter Process List'), False, False, 5)
        self.serial_filter_entry = self.pack_entry_and_button("", self.on_update_serial_process_listing, serial_process_frame, attach_dialog, "Filter", hbox, True, True, 0)
        vbox.pack_start(hbox, False, False, 0)
        hbox = gtk.HBox()
        button = gtk.Button("Refresh Process List")
        button.connect("clicked", lambda w: self.on_update_serial_process_listing(w, serial_process_frame, attach_dialog, self.serial_filter_entry))
        hbox.pack_start(button, False, False, 10)
        button = gtk.Button("Add Process to Attach List")
        hbox.pack_start(button, True, True, 10)
        self.serial_process_list_entry = gtk.Entry()
        self.serial_process_list_entry.set_max_length(8192)
        self.serial_process_list_entry.set_text(self.options['Serial Process List'])
        button.connect("clicked", lambda w: self.on_add_serial_process(w, attach_dialog, self.serial_process_list_entry))
        vbox.pack_start(hbox, False, False, 10)
        hbox = gtk.HBox()
        label = gtk.Label("Attach List")
        hbox.pack_start(label, False, False, 10)
        self.serial_process_list_entry.connect("activate", lambda w: stat_wait_dialog.show_wait_dialog_and_run(self.attach_serial_processes_cb, (self.serial_process_list_entry, attach_dialog), self.attach_task_list, attach_dialog))
        hbox.pack_start(self.serial_process_list_entry, True, True, 0)
        vbox.pack_start(hbox, False, False, 0)
        vbox.pack_start(gtk.HSeparator(), False, False, 10)
        hbox = gtk.HBox()
        button = gtk.Button(stock=gtk.STOCK_CANCEL)
        button.connect("clicked", lambda w: self.on_cancel_attach(w, attach_dialog))
        hbox.pack_start(button, False, False, 10)
        button = gtk.Button("Attach to Attach List")
        button.connect("clicked", lambda w: stat_wait_dialog.show_wait_dialog_and_run(self.attach_serial_processes_cb, (self.serial_process_list_entry, attach_dialog), self.attach_task_list, attach_dialog))
        hbox.pack_start(button, True, True, 0)
        vbox.pack_start(hbox, False, False, 0)
        label = gtk.Label('Serial Attach')
        notebook.append_page(vbox, label)

        # DySect options
        if HAVE_DYSECT:
            vbox = gtk.VBox()
            frame = gtk.Frame(label='DySectAPI Options')
            dysect_vbox = gtk.VBox()
            self.pack_check_button(dysect_vbox, 'Enable DySectAPI', False, False, 5)
            label = gtk.Label('Session: %s' % self.options['DySectAPI Session'])
            label.set_alignment(0, .5)
            dysect_vbox.pack_start(label, True, True, 0)
            button = gtk.Button('Select DySectAPI Session')
            button.connect("clicked", lambda w: self.on_choose_dysect_session(attach_dialog, dysect_vbox))
            dysect_vbox.pack_start(button, False, False, 0)
            self.pack_string_option(dysect_vbox, 'DySectAPI Session Args', attach_dialog)
            frame.add(dysect_vbox)
            vbox.pack_start(frame, False, False, 0)
            hbox = gtk.HBox()
            button = gtk.Button(stock=gtk.STOCK_CANCEL)
            button.connect("clicked", lambda w: self.on_cancel_attach(w, attach_dialog))
            hbox.pack_start(button, False, False, 0)
            vbox.pack_end(hbox, False, False, 0)
            label = gtk.Label('DySect')
            notebook.append_page(vbox, label)

        # sample options
        vbox = gtk.VBox()
        self.pack_sample_options(vbox, False, True)
        hbox = gtk.HBox()
        button = gtk.Button(stock=gtk.STOCK_CANCEL)
        button.connect("clicked", lambda w: self.on_cancel_attach(w, attach_dialog))
        hbox.pack_start(button, False, False, 0)
        vbox.pack_end(hbox, False, False, 0)
        label = gtk.Label('Sample Options')
        notebook.append_page(vbox, label)

        # topology options
        vbox = gtk.VBox()
        frame = gtk.Frame(label='Topology Options')
        vbox2 = gtk.VBox()
        self.pack_combo_box(vbox2, 'Topology Type')
        self.pack_string_option(vbox2, 'Topology', attach_dialog)
        self.pack_string_option(vbox2, 'Communication Nodes', attach_dialog)
        self.pack_check_button(vbox2, 'Check Node Access', False, False, 5)
        self.pack_combo_box(vbox2, 'CP Policy')
        self.pack_spinbutton(vbox2, 'Communication Processes per Node')
        self.pack_spinbutton(vbox2, 'Daemons per Node')
        frame.add(vbox2)
        vbox.pack_start(frame, False, False, 0)
        hbox = gtk.HBox()
        button = gtk.Button(stock=gtk.STOCK_CANCEL)
        button.connect("clicked", lambda w: self.on_cancel_attach(w, attach_dialog))
        hbox.pack_start(button, False, False, 0)
        vbox.pack_end(hbox, False, False, 0)
        label = gtk.Label('Topology')
        notebook.append_page(vbox, label)

        # misc options
        vbox = gtk.VBox()
        frame = gtk.Frame(label='Tool Paths')
        vbox2 = gtk.VBox()
        self.pack_string_option(vbox2, 'Tool Daemon Path', attach_dialog)
        self.pack_string_option(vbox2, 'Filter Path', attach_dialog)
        if HAVE_GDB_SUPPORT:
            self.pack_check_button(vbox2, 'GDB BE')
            self.pack_string_option(vbox2, 'GDB Path', attach_dialog)
        frame.add(vbox2)
        vbox.pack_start(frame, False, False, 5)
        frame = gtk.Frame(label='Debug Logs')
        vbox2 = gtk.VBox()
        self.pack_check_button(vbox2, 'Log Frontend')
        self.pack_check_button(vbox2, 'Log Backend')
        self.pack_check_button(vbox2, 'Log CP')
        self.pack_check_button(vbox2, 'Log SW')
        self.pack_check_button(vbox2, 'Log SWERR')
        self.pack_string_option(vbox2, 'Log Dir', attach_dialog)
        self.pack_check_button(vbox2, 'Use MRNet Printf')
        frame.add(vbox2)
        vbox.pack_start(frame, False, False, 5)
        frame = gtk.Frame(label='Misc')
        vbox2 = gtk.VBox()
        self.pack_combo_box(vbox2, 'Verbosity Type')
        self.pack_check_button(vbox2, 'Debug Backends')
        frame.add(vbox2)
        vbox.pack_start(frame, False, False, 5)
        hbox = gtk.HBox()
        button = gtk.Button(stock=gtk.STOCK_CANCEL)
        button.connect("clicked", lambda w: self.on_cancel_attach(w, attach_dialog))
        hbox.pack_start(button, False, False, 0)
        vbox.pack_end(hbox, False, False, 0)
        label = gtk.Label('Advanced')
        notebook.append_page(vbox, label)

        self.on_update_process_listing(None, process_frame, attach_dialog, self.filter_entry)
        self.on_update_serial_process_listing(None, serial_process_frame, attach_dialog, self.serial_filter_entry)
        attach_dialog.vbox.pack_start(notebook, True, True, 0)
        attach_dialog.show_all()
        attach_dialog.run()
        self.sw = None
        self.serial_sw = None
        return True

    def attach_serial_processes_cb(self, entry=None, attach_dialog=None, in_processes=None):
        """Callback to attach to serial processes."""
        processes = in_processes
        if processes == None:
            processes = entry.get_text()
            processes = processes.split()
        self.process_list = ' '.join(processes)
        self.serial_attach = True
        if len(processes) <= 0:
            show_error_dialog('No processes selected.  Please select processes\nto attach to by selecting the desired radio\nbuttons and clicking "Add Serial Process" or by\nmanually entering host:pid pairs in the text entry.', self)
            return False
        if self.STAT is None:
            self.STAT = STAT_FrontEnd()
        for process in processes:
            self.STAT.addSerialProcess(process)
        if self.options['GDB BE'] is True:
            self.attach_cb(attach_dialog, False, True, STAT_SERIAL_GDB_ATTACH)
        else:
            self.attach_cb(attach_dialog, False, True, STAT_SERIAL_ATTACH)
        return True

    def launch_application_cb(self, entry=None, attach_dialog=None, in_args=None):
        """Callback to launch an application under STAT's control."""
        args = in_args
        if args == None:
            args = entry.get_text()
            args = args.split()
        if len(args) <= 0:
            show_error_dialog('No launch string specified.  Please enter\nthe application launch string.\n', self)
            return False
        self.options['Launcher Exe'] = args[0]
        for arg in args:
            self.STAT.addLauncherArgv(arg)
        self.attach_cb(attach_dialog, True, False, STAT_LAUNCH)
        return True

    def destroy_dysect_dialog(self, dialog):
        self.dysect_dialog = None
        self.close_visualizer(None)
        self.on_detach(dialog)

    def update_dysect_active_bar(self):
        """Register activity on the active bar."""
        if self.dysect_timer != None:
            self.dysect_progress_bar.pulse()
        return True

    def close_visualizer(self, action):
        if self.visualizer_window:
            self.visualizer_window.destroy()
            self.visualizer_window = None

    def open_visualizer(self):
        """ Open a visualization window """
        DysectView.show_default_probes = self.options["DySectAPI Show Default Probes"]
        if self.visualizer_window == None:
            if hasattr(self, "out_dir"):
                self.visualizer_window = DysectDotWindow(os.path.join(self.out_dir, 'dysect_session.dot'), self)
            elif self.STAT != None:
                self.visualizer_window = DysectDotWindow(os.path.join(self.STAT.getOutDir(), 'dysect_session.dot'), self)
            else:
                sys.stderr.write('could not determine dysect session dot file path\n')
                return False
            self.visualizer_window.connect('destroy', self.close_visualizer)
        else:
            self.visualizer_window.present()

    def scroll_probe_window(self, filename, line):
        if line == None:
            return
        (session_file_directory, session_filename) = os.path.split(self.options['DySectAPI Session'])
        if filename != session_filename and filename != self.options['DySectAPI Session']:
            return
        text_buffer = self.probe_text_view.get_buffer()
        if text_buffer == None:
            return
        if self.marked_line_number != None:
            (marked_line_start, marked_line_end) = self.marked_line_number
            text_buffer.remove_tag_by_name("bold_tag", marked_line_start, marked_line_end)
        probe_line_start = text_buffer.get_iter_at_line(int(line) - 1)
        self.probe_text_view.scroll_to_iter(probe_line_start, 0.0, True, 0.0, 0.7)
        probe_line_end = text_buffer.get_iter_at_line_offset(int(line) - 1, self.line_number_width + 2)
        text_buffer.apply_tag_by_name("bold_tag", probe_line_start, probe_line_end)
        self.marked_line_number = (probe_line_start, probe_line_end)

    def stop_dysect_session(self, button):
        self.dysect_session = False
        button.set_sensitive(False)
        if self.STAT:
            self.STAT.dysectStop()
        self.on_detach(button)

    def attach_cb(self, attach_dialog, launch, serial, application_option=STAT_ATTACH):
        """Callback to attach to a job and gather a stack trace."""
        if serial is False:
            self.serial_attach = False
            self.process_list = None
        else:
            self.serial_attach = True
        if attach_dialog is not None:
            run_gtk_main_loop()
            attach_dialog.hide()
        stat_wait_dialog.update_progress_bar(0.01)
        try:
            self.options['Topology Type'] = self.types['Topology Type'][self.combo_boxes['Topology Type'].get_active()]
            self.options['CP Policy'] = self.types['CP Policy'][self.combo_boxes['CP Policy'].get_active()]
            self.options['Verbosity Type'] = self.types['Verbosity Type'][self.combo_boxes['Verbosity Type'].get_active()]
            self.options['Communication Processes per Node'] = int(self.spinners['Communication Processes per Node'].get_value())
            self.options['Daemons per Node'] = int(self.spinners['Daemons per Node'].get_value())
        except Exception as e:
            pass
        if self.options['Check Node Access'] == True:
            os.environ['STAT_CHECK_NODE_ACCESS'] = '1'
        if self.STAT is None:
            self.STAT = STAT_FrontEnd()
        if launch is False and serial is False:
            if self.options['PID'] is None:
                show_error_dialog('No job selected.  Please select a valid\njob launcher process to attach to.\n', self)
                return False

        self.STAT.setToolDaemonExe(self.options['Tool Daemon Path'])
        self.STAT.setFilterPath(self.options['Filter Path'])
        os.environ['STAT_GDB'] = self.options['GDB Path']
        log_type = STAT_LOG_NONE
        if self.options['GDB BE'] and application_option != STAT_SERIAL_GDB_ATTACH:
            application_option = STAT_GDB_ATTACH
        if self.options['Log Frontend']:
            log_type |= STAT_LOG_FE
        if self.options['Log Backend']:
            log_type |= STAT_LOG_BE
        if self.options['Log CP']:
            log_type |= STAT_LOG_CP
        if self.options['Log SW']:
            log_type |= STAT_LOG_SW
        if self.options['Log SWERR']:
            log_type |= STAT_LOG_SWERR
        if self.options['Use MRNet Printf']:
            log_type |= STAT_LOG_MRN
        if log_type != STAT_LOG_NONE:
            ret = self.STAT.startLog(log_type, self.options['Log Dir'])
            if ret != STAT_OK:
                show_error_dialog('Failed to Start Log:\n%s' % self.STAT.getLastErrorMessage(), self)
                self.on_fatal_error()
                return False
        if self.options['Verbosity Type'] == 'error':
            self.STAT.setVerbose(STAT_VERBOSE_ERROR)
        elif self.options['Verbosity Type'] == 'stdout':
            self.STAT.setVerbose(STAT_VERBOSE_STDOUT)
        elif self.options['Verbosity Type'] == 'full':
            self.STAT.setVerbose(STAT_VERBOSE_FULL)
        if self.options['Debug Backends']:
            os.environ['LMON_DEBUG_BES'] = "1"
        else:
            if 'LMON_DEBUG_BES' in os.environ:
                del os.environ['LMON_DEBUG_BES']
        if HAVE_DYSECT is True and self.options['Enable DySectAPI'] is True:
            os.environ['STAT_GROUP_OPS'] = "1"
        self.STAT.setProcsPerNode(self.options['Communication Processes per Node'])
        self.STAT.setNDaemonsPerNode(self.options['Daemons per Node'])
        if self.options['Output Job ID'] != '':
            self.STAT.setJobId(self.options['Output Job ID'])
        stat_wait_dialog.update_progress_bar(0.05)

        self.STAT.setApplicationOption(application_option)
        if launch is False:
            if serial is False:
                if self.options['Remote Host'] != "localhost":
                    ret = self.STAT.attachAndSpawnDaemons(self.options['PID'], self.options['Remote Host'])
                else:
                    ret = self.STAT.attachAndSpawnDaemons(self.options['PID'])
            else:
                ret = self.STAT.setupForSerialAttach()
        else:
            if self.options['Remote Host'] != "localhost":
                ret = self.STAT.launchAndSpawnDaemons(self.options['Remote Host'])
            else:
                ret = self.STAT.launchAndSpawnDaemons()
            if self.options['PID'] is None:
                self.set_proctab()
                self.options['PID'] = self.proctab.launcher_pid
        if 'LMON_DEBUG_BES' in os.environ:
            del os.environ['LMON_DEBUG_BES']
        if ret != STAT_OK:
            show_error_dialog('Failed to Launch Daemons:\n%s' % self.STAT.getLastErrorMessage(), self)
            self.on_fatal_error()
            return False
        stat_wait_dialog.update(0.10)

        topology_type = STAT_TOPOLOGY_AUTO
        if self.options['Topology Type'] == 'automatic':
            topology_type = STAT_TOPOLOGY_AUTO
        elif self.options['Topology Type'] == 'depth':
            topology_type = STAT_TOPOLOGY_DEPTH
        elif self.options['Topology Type'] == 'max fanout':
            topology_type = STAT_TOPOLOGY_FANOUT
        elif self.options['Topology Type'] == 'custom':
            topology_type = STAT_TOPOLOGY_USER

        cp_policy = STAT_CP_NONE
        if self.options['CP Policy'] == 'share app nodes':
            cp_policy = STAT_CP_SHAREAPPNODES
        elif self.options['CP Policy'] == 'exclusive':
            cp_policy = STAT_CP_EXCLUSIVE

        if self.options['Communication Nodes'] != '':
            ret = self.STAT.launchMrnetTree(topology_type, self.options['Topology'], self.options['Communication Nodes'], False, cp_policy)
        else:
            ret = self.STAT.launchMrnetTree(topology_type, self.options['Topology'], '', False, cp_policy)
        if ret != STAT_OK:
            show_error_dialog('Failed to Launch MRNet Tree:\n%s' % self.STAT.getLastErrorMessage(), self)
            self.on_fatal_error()
            return False

        if application_option != STAT_SERIAL_ATTACH and application_option != STAT_SERIAL_GDB_ATTACH:
            while 1:
                run_gtk_main_loop()
                ret = self.STAT.connectMrnetTree(False)
                if ret == STAT_OK:
                    break
                elif ret != STAT_PENDING_ACK:
                    show_error_dialog('Failed to connect to MRNet tree:\n%s' % self.STAT.getLastErrorMessage(), self)
                    self.on_fatal_error()
                    return ret
        ret = self.STAT.setupConnectedMrnetTree()
        if ret != STAT_OK:
            show_error_dialog('Failed to setup STAT:\n%s' % self.STAT.getLastErrorMessage(), self)
            self.on_fatal_error()
            return False

        stat_wait_dialog.update(0.15)
        ret = self.STAT.attachApplication(False)
        if ret != STAT_OK:
            show_error_dialog('Failed to attach to application:\n%s' % self.STAT.getLastErrorMessage(), self)
            self.on_fatal_error()
            return False
        while 1:
            run_gtk_main_loop()
            ret = self.STAT.receiveAck(False)
            if ret == STAT_OK:
                break
            elif ret != STAT_PENDING_ACK:
                show_error_dialog('Failed to attach:\n%s' % self.STAT.getLastErrorMessage(), self)
                self.on_fatal_error()
                return ret
        stat_wait_dialog.update(0.20)
        self.attached = True

        if HAVE_DYSECT is True and self.options['Enable DySectAPI'] is True:
            stat_wait_dialog.destroy()
            self.run_dysect_session()
        else:
            ret = self.sample()
            if ret != STAT_OK:
                return ret
            self.set_action_sensitivity('paused')
            stat_wait_dialog.update_progress_bar(1.0)

    def on_detach(self, widget, stop_list=intArray(0), stop_list_len=0):
        """Determine the process state and detach from the current job."""
        self.proctab_file_path = None
        self.proctab = None
        if self.attached is False:
            self.STAT = None
            self.reattach = False
            return True
        self.var_spec = []
        self.show_all()

        ret = self.STAT.detachApplication(stop_list, stop_list_len, False)
        if ret != STAT_OK:
            show_error_dialog('Failed to detach from application:\n%s' % self.STAT.getLastErrorMessage(), self)
            self.on_fatal_error()
            return ret
        while 1:
            run_gtk_main_loop()
            ret = self.STAT.receiveAck(False)
            if ret == STAT_OK:
                break
            elif ret != STAT_PENDING_ACK:
                show_error_dialog('Failed to sample stack trace:\n%s' % self.STAT.getLastErrorMessage(), self)
                self.on_fatal_error()
                return ret
        if ret != STAT_OK:
            show_error_dialog('Failed to detach from application:\n%s' % self.STAT.getLastErrorMessage(), self)
            self.on_fatal_error()
            return False
        self.STAT.shutDown()
        self.STAT = None
        self.reattach = True
        self.attached = False
        self.set_action_sensitivity('detached')

    def on_destroy(self, action):
        """Callback to quit."""
        if self.STAT is not None:
            if self.attached:
                ret = self.STAT.detachApplication(True)
                if ret != STAT_OK:
                    sys.stderr.write('Failed to detach from application:\n%s\n' % self.STAT.getLastErrorMessage())
                self.STAT.shutDown()
            self.STAT = None
        gtk.main_quit()

    def on_pause(self, action=None):
        """Callback to pause the job."""
        self.set_action_sensitivity('paused')
        ret = self.STAT.pause()
        if ret != STAT_OK:
            show_error_dialog('Failed to pause application:\n%s' % self.STAT.getLastErrorMessage(), self)
            self.on_fatal_error()
            return ret
        while 1:
            run_gtk_main_loop()
            ret = self.STAT.receiveAck(False)
            if ret == STAT_OK:
                break
            elif ret != STAT_PENDING_ACK:
                show_error_dialog('Failed to pause application:\n%s' % self.STAT.getLastErrorMessage(), self)
                self.on_fatal_error()
                return ret
        return ret

    def on_resume(self, action=None):
        """Callback to resume the job."""
        self.set_action_sensitivity('running')
        ret = self.STAT.resume()
        if ret != STAT_OK:
            show_error_dialog('Failed to resume application:\n%s' % self.STAT.getLastErrorMessage(), self)
            self.on_fatal_error()
            return ret
        while 1:
            run_gtk_main_loop()
            ret = self.STAT.receiveAck(False)
            if ret == STAT_OK:
                break
            elif ret != STAT_PENDING_ACK:
                show_error_dialog('Failed to resume application:\n%s' % self.STAT.getLastErrorMessage(), self)
                self.on_fatal_error()
                return ret
        return ret

    def on_kill(self, action):
        """Callback to kill the job."""
        self.set_action_sensitivity('new')

    def update_sample_options(self):
        """Update sample options."""
        for option in ['Num Retries', 'Retry Frequency (us)', 'Max Threads Per Daemon', 'Run Time Before Sample (sec)', 'Num Traces', 'Trace Frequency (ms)']:
            try:
                self.options[option] = int(self.spinners[option].get_value())
            except:
                pass

    def sleep(self, seconds):
        """Sleep for specified time with a progress bar."""
        for i in range(int(seconds * 100)):
            stat_wait_dialog.update_progress_bar(float(i) / seconds / 100)
            time.sleep(.01)

    def update_prefs_and_sample_cb(self, sample_function, widget, dialog, action):
        """Callback to update sample preferences and sample traces."""
        dialog.hide()
        self.update_sample_options()
        if self.options['Run Time Before Sample (sec)'] != 0:
            if self.STAT.isRunning() is False:
                ret = self.on_resume()
                if ret != STAT_OK:
                    return False
            stat_wait_dialog.show_wait_dialog_and_run(self.sleep, (self.options['Run Time Before Sample (sec)'],), [], self)
        if sample_function == self.sample_multiple:
            stat_wait_dialog.show_wait_dialog_and_run(sample_function, (), self.sample_task_list, self, True)
        else:
            stat_wait_dialog.show_wait_dialog_and_run(sample_function, (), self.sample_task_list, self)

    def on_sample(self, action, multiple):
        """Callback to sample stack traces."""
        dialog = gtk.Dialog('Stack Sample Preferences', self)
        vbox = gtk.VBox()
        self.pack_sample_options(vbox, multiple)
        dialog.vbox.pack_start(vbox, False, False, 0)
        hbox = gtk.HButtonBox()
        button = gtk.Button(stock=gtk.STOCK_CANCEL)
        button.connect("clicked", lambda w: dialog.destroy())
        hbox.pack_start(button, False, True, 0)
        button = gtk.Button(stock=gtk.STOCK_OK)
        if multiple is True:
            button.connect("clicked", lambda w: self.update_prefs_and_sample_cb(self.sample_multiple, w, dialog, 'OK'))
        else:
            button.connect("clicked", lambda w: self.update_prefs_and_sample_cb(self.sample, w, dialog, 'OK'))
        hbox.pack_start(button, False, True, 0)
        dialog.vbox.pack_start(hbox, False, False, 0)
        dialog.show_all()
        dialog.run()

    def sample(self, keep_var_spec=False):
        """Sample stack traces."""
        ret_val = STAT_OK
        if keep_var_spec is False:
            self.var_spec = []
            # below is a test for local variables:
            #self.var_spec.append(("/g/g0/lee218/src/STAT/examples/src/to_test.c", 38, 5, "i"))
            #self.var_spec.append(("/g/g0/lee218/src/STAT/examples/src/to_test.c", 38, 4, "i"))
        self.update_sample_options()
        ret = self.STAT.pause()
        stat_wait_dialog.update_progress_bar(0.35)
        self.set_action_sensitivity('paused')
        if ret != STAT_OK:
            show_error_dialog('Failed to pause application during sample stack trace operation:\n%s' % self.STAT.getLastErrorMessage(), self)
            self.on_fatal_error()
            return ret

        sample_type = 0
        if self.options['Sample Type'] == 'function and pc':
            sample_type += STAT_SAMPLE_PC
        if self.options['Sample Type'] == 'module offset':
            sample_type += STAT_SAMPLE_MODULE_OFFSET
        if self.options['Sample Type'] == 'function and line':
            sample_type += STAT_SAMPLE_LINE
        if self.options['Edge Type'] != 'full list':
            sample_type += STAT_SAMPLE_COUNT_REP
        if self.options['With Threads']:
            sample_type += STAT_SAMPLE_THREADS
        if self.options['With OpenMP'] and HAVE_OPENMP_SUPPORT:
            sample_type += STAT_SAMPLE_OPENMP
        if self.options['With CUDA Quick'] and HAVE_GDB_SUPPORT:
            sample_type += STAT_SAMPLE_CUDA_QUICK
        if self.options['Gather Python Traces']:
            sample_type += STAT_SAMPLE_PYTHON
        if self.options['Clear On Sample']:
            sample_type += STAT_SAMPLE_CLEAR_ON_SAMPLE
        ret = self.STAT.sampleStackTraces(sample_type, 1, 1, self.options['Num Retries'], self.options['Retry Frequency (us)'], self.options['Max Threads Per Daemon'], False, var_spec_to_string(self.var_spec))

        if ret != STAT_OK:
            show_error_dialog('Failed to sample stack trace:\n%s' % self.STAT.getLastErrorMessage(), self)
            self.on_fatal_error()
            return ret
        while 1:
            run_gtk_main_loop()
            ret = self.STAT.receiveAck(False)
            if ret == STAT_OK:
                break
            elif ret != STAT_PENDING_ACK:
                show_error_dialog('Failed to sample stack trace:\n%s' % self.STAT.getLastErrorMessage(), self)
                self.on_fatal_error()
                return ret
        stat_wait_dialog.update(0.60)
        ret = self.STAT.gatherLastTrace(False)
        if ret != STAT_OK:
            show_error_dialog('Failed to gather stack trace:\n%s' % self.STAT.getLastErrorMessage(), self)
            self.on_fatal_error()
            return ret
        while 1:
            run_gtk_main_loop()
            ret = self.STAT.receiveAck(False)
            if ret == STAT_OK:
                break
            elif ret != STAT_PENDING_ACK:
                show_error_dialog('Failed to sample stack trace:\n%s' % self.STAT.getLastErrorMessage(), self)
                self.on_fatal_error()
                return ret
        stat_wait_dialog.update(0.70)
        filename = self.STAT.getLastDotFilename()
        try:
            with open(filename, 'r') as f:
                self.tabs[self.notebook.get_current_page()].history_view.get_buffer().set_text('')
                self.set_dotcode(f.read(), filename, self.notebook.get_current_page())
        except IOError as e:
            show_error_dialog('%s\nFailed to open file %s' % (repr(e), filename), self)
            return ret
        except Exception as e:
            show_error_dialog('%s\nFailed to process file %s' % (repr(e), filename), self)
            return ret
        stat_wait_dialog.update(1.0)
        if ret_val != STAT_OK:
            show_error_dialog('An error was detected:\n%s' % self.STAT.getLastErrorMessage(), self)
            self.on_fatal_error()
        return ret_val

    def sample_multiple(self):
        """Sample multiple stack traces."""

        ret_val = STAT_OK
        self.update_sample_options()
        stat_wait_dialog.update_progress_bar(0.01)

        previous_clear = self.options['Clear On Sample']
        for i in range(self.options['Num Traces']):
            if stat_wait_dialog.cancelled is True:
                stat_wait_dialog.cancelled = False
                break
            if i == 1:
                self.options['Clear On Sample'] = False
            ret = self.STAT.pause()
            if ret == STAT_APPLICATION_EXITED:
                ret_val = STAT_APPLICATION_EXITED
                break
            if ret != STAT_OK:
                show_error_dialog('Failed to pause application during sample multiple operation:\n%s' % self.STAT.getLastErrorMessage(), self)
                self.on_fatal_error()
                return ret
            self.set_action_sensitivity('paused')

            sample_type = 0
            if self.options['Sample Type'] == 'function and pc':
                sample_type += STAT_SAMPLE_PC
            if self.options['Sample Type'] == 'module offset':
                sample_type += STAT_SAMPLE_MODULE_OFFSET
            if self.options['Sample Type'] == 'function and line':
                sample_type += STAT_SAMPLE_LINE
            if self.options['Edge Type'] != 'full list':
                sample_type += STAT_SAMPLE_COUNT_REP
            if self.options['With Threads']:
                sample_type += STAT_SAMPLE_THREADS
            if self.options['With OpenMP'] and HAVE_OPENMP_SUPPORT:
                sample_type += STAT_SAMPLE_OPENMP
            if self.options['With CUDA Quick'] and HAVE_GDB_SUPPORT:
                sample_type += STAT_SAMPLE_CUDA_QUICK
            if self.options['Gather Python Traces']:
                sample_type += STAT_SAMPLE_PYTHON
            if self.options['Clear On Sample']:
                sample_type += STAT_SAMPLE_CLEAR_ON_SAMPLE
            ret = self.STAT.sampleStackTraces(sample_type, 1, 0, self.options['Num Retries'], self.options['Retry Frequency (us)'], self.options['Max Threads Per Daemon'], False, var_spec_to_string(self.var_spec))
            if ret != STAT_OK:
                if ret == STAT_APPLICATION_EXITED:
                    ret_val = STAT_APPLICATION_EXITED
                    break
                show_error_dialog('Failed to sample stack trace:\n%s' % self.STAT.getLastErrorMessage(), self)
                self.on_fatal_error()
                return ret
            while 1:
                run_gtk_main_loop()
                ret = self.STAT.receiveAck(False)
                if ret == STAT_OK:
                    break
                elif ret == STAT_APPLICATION_EXITED:
                    ret_val = STAT_APPLICATION_EXITED
                    break
                elif ret != STAT_PENDING_ACK:
                    show_error_dialog('Failed to receive ack from sample operation:\n%s' % self.STAT.getLastErrorMessage(), self)
                    self.on_fatal_error()
                    return ret
            if ret == STAT_APPLICATION_EXITED:
                ret_val = STAT_APPLICATION_EXITED
                break

            if self.options['Gather Individual Samples'] is True:
                ret = self.STAT.gatherLastTrace(False)
                if ret != STAT_OK:
                    show_error_dialog('Failed to gather last stack trace:\n%s' % self.STAT.getLastErrorMessage(), self)
                    self.on_fatal_error()
                    return ret
                while 1:
                    run_gtk_main_loop()
                    ret = self.STAT.receiveAck(False)
                    if ret == STAT_OK:
                        break
                    elif ret != STAT_PENDING_ACK:
                        show_error_dialog('Failed to receive last stack trace:\n%s' % self.STAT.getLastErrorMessage(), self)
                        self.on_fatal_error()
                        return ret
                filename = self.STAT.getLastDotFilename()
                try:
                    with open(filename, 'r') as f:
                        page = self.notebook.get_current_page()
                        self.create_new_tab(page + 1)
                        self.notebook.set_current_page(page + 1)
                        self.tabs[self.notebook.get_current_page()].history_view.get_buffer().set_text('')
                        self.set_dotcode(f.read(), filename, self.notebook.get_current_page())
                except IOError as e:
                    show_error_dialog('%s\nFailed to open file %s' % (repr(e), filename), self)
                    return False
                except Exception as e:
                    show_error_dialog('%s\nFailed to process file %s' % (repr(e), filename), self)
                    return False

            stat_wait_dialog.update_progress_bar(float(i) / self.options['Num Traces'])
            if ret_val != STAT_OK:
                break
            if i != self.options['Num Traces'] - 1:
                ret = self.STAT.resume()
                if ret == STAT_APPLICATION_EXITED:
                    ret_val = STAT_APPLICATION_EXITED
                    break
                elif ret != STAT_OK:
                    return ret
                self.set_action_sensitivity('running')
                for i in range(int(self.options['Trace Frequency (ms)'] / 10)):
                    time.sleep(.01)
                    run_gtk_main_loop()
        self.options['Clear On Sample'] = previous_clear
        stat_wait_dialog.update(0.91)
        ret = self.STAT.gatherTraces(False)
        if ret != STAT_OK:
            show_error_dialog('Failed to gather stack traces:\n%s' % self.STAT.getLastErrorMessage(), self)
            self.on_fatal_error()
            return ret
        while 1:
            run_gtk_main_loop()
            ret = self.STAT.receiveAck(False)
            if ret == STAT_OK:
                break
            elif ret != STAT_PENDING_ACK:
                show_error_dialog('Failed to gather stack traces:\n%s' % self.STAT.getLastErrorMessage(), self)
                self.on_fatal_error()
                return ret
        stat_wait_dialog.update(0.95)
        filename = self.STAT.getLastDotFilename()
        try:
            with open(filename, 'r') as f:
                if self.options['Gather Individual Samples'] is True:
                    page = self.notebook.get_current_page()
                    self.create_new_tab(page + 1)
                    self.notebook.set_current_page(page + 1)
                else:
                    self.tabs[self.notebook.get_current_page()].history_view.get_buffer().set_text('')
                self.set_dotcode(f.read(), filename, self.notebook.get_current_page())
        except IOError as e:
            show_error_dialog('%s\nFailed to open file %s' % (repr(e), filename), self)
            return False
        except Exception as e:
            show_error_dialog('%s\nFailed to process file %s' % (repr(e), filename), self)
            return False
        stat_wait_dialog.update(1.0)
        if ret_val != STAT_OK:
            show_error_dialog('An error was detected:\n%s' % self.STAT.getLastErrorMessage(), self)
            self.on_fatal_error()

        return ret_val

    def run_dysect_session(self):
        """Run the dysect session."""

        self.set_action_sensitivity('busy')
        self.STAT.resume()
        dysect_c = '%s/bin/dysectc' % self.STAT.getInstallPrefix()
        if not os.path.exists(dysect_c):
            show_error_dialog('DySect compiler %s not found' % dysect_c, self)
            self.on_fatal_error()
            return -1
        proc = subprocess.Popen([dysect_c, self.options['DySectAPI Session']], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout_output, stderr_output = proc.communicate()
        if stderr_output != '':
            sys.stderr.write('dysectc outputted error message: %s\n' % stderr_output)
        if proc.returncode != 0:
            show_error_dialog('Failed to compile session %s:\n\n%s' % (self.options['DySectAPI Session'], stderr_output))
            self.on_fatal_error()
            return proc.returncode
        session_so = os.path.join(os.getcwd(), 'lib' + os.path.basename(self.options['DySectAPI Session']).strip('.cpp') + '.so')
        if not os.path.exists(session_so):
            show_error_dialog('Compiled session %s not found' % session_so)
            self.on_fatal_error()
            return -1
        dysect_args = self.options["DySectAPI Session Args"].split()
        ret = self.STAT.dysectSetup(session_so, -1, len(dysect_args), dysect_args)
        if ret != STAT_OK:
            show_error_dialog('Failed to setup DySect Session %s:\n%s' % (session_so, self.STAT.getLastErrorMessage()), self)
            self.on_fatal_error()
            return ret
        self.dysect_session = True

        self.dysect_dialog = gtk.Dialog("DySectAPI Console", self)
        self.dysect_dialog.set_default_size(1048, 630)
        self.dysect_dialog.set_destroy_with_parent(True)
        self.dysect_dialog.connect('destroy', lambda w: self.destroy_dysect_dialog(w))
        vpaned = gtk.VPaned()
        my_frame = gtk.Frame(label="Session %s:" % self.options['DySectAPI Session'])
        sw = gtk.ScrolledWindow()
        sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        text_view = gtk.TextView()
        self.probe_text_view = text_view
        text_view.set_editable(False)
        text_view.set_cursor_visible(False)
        text_view_buffer = text_view.get_buffer()
        text_view.set_wrap_mode(gtk.WRAP_NONE)
        sw.add(text_view)
        my_frame.add(sw)
        with open(self.options['DySectAPI Session'], "r") as dysect_session_file:
            if HAVE_PYGMENTS:
                text_view_buffer.create_tag("monospace", family="monospace")
                pygments.highlight(dysect_session_file.read(), CppLexer(), STATviewFormatter())
                text_view_buffer.create_tag('bold_tag', weight=pango.WEIGHT_BOLD)
                text_view_buffer.create_tag('italics_tag', style=pango.STYLE_ITALIC)
                text_view_buffer.create_tag('underline_tag', underline=pango.UNDERLINE_SINGLE)
                lines = STAThelper.pygments_lines
                iterator = text_view_buffer.get_iter_at_offset(0)
                width = len(str(len(lines)))
                self.line_number_width = width
                for i, line in enumerate(lines):
                    source_string = "%0*d| " % (width, i + 1)
                    text_view_buffer.insert_with_tags_by_name(iterator, source_string, "monospace")
                    for item in line:
                        source_string, format_tuple = item
                        pygments_color, bold, italics, underline = format_tuple
                        foreground = gtk.gdk.color_parse(pygments_color)
                        fore_color_tag = "color_fore%s" % (pygments_color)
                        try:
                            text_view_buffer.create_tag(fore_color_tag, foreground_gdk=foreground)
                        except:
                            pass
                        args = [iterator, source_string, fore_color_tag, "monospace"]
                        if bold:
                            args.append('bold_tag')
                        if italics:
                            args.append('italics_tag')
                        if underline:
                            args.append('underline_tag')
                        text_view_buffer.insert_with_tags_by_name(*tuple(args))
            else:
                text_view_buffer.set_text(dysect_session_file.read())
        vpaned.pack1(my_frame, True, True)
        self.separator = gtk.HSeparator()
        self.dysect_dialog.vbox.pack_start(self.separator, False, True, 5)

        self.out_dir = self.STAT.getOutDir()
        dysect_outfile = os.path.join(self.out_dir, 'dysect_output.txt')
        my_frame = gtk.Frame(label="Output %s:" % dysect_outfile)
        sw = gtk.ScrolledWindow()
        sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        text_view = gtk.TextView()
        text_view.set_editable(False)
        text_view.set_cursor_visible(False)
        text_view_buffer = text_view.get_buffer()
        iterator = text_view_buffer.get_iter_at_offset(0)
        text_view_buffer.create_tag("monospace", family="monospace")
        text_view.set_wrap_mode(gtk.WRAP_NONE)
        sw.add(text_view)
        my_frame.add(sw)
        vpaned.pack2(my_frame, True, True)
        self.dysect_dialog.vbox.pack_start(vpaned, True, True, 5)
        self.separator = gtk.HSeparator()
        self.dysect_dialog.vbox.pack_start(self.separator, False, True, 5)

        # Visualization button
        vis_box = gtk.HButtonBox()
        vis_button = gtk.Button("_Visualize Probe Tree")
        vis_button.connect("clicked", lambda w: self.open_visualizer())
        vis_box.pack_end(vis_button, False, True, 5)

        box2 = gtk.HButtonBox()
        stop_button = gtk.Button(stock=gtk.STOCK_STOP)
        stop_button.connect("clicked", lambda w: self.stop_dysect_session(w))
        box2.pack_end(stop_button, False, True, 5)
        self.dysect_timer = gobject.timeout_add(75, self.update_dysect_active_bar)
        self.dysect_progress_bar = gtk.ProgressBar()
        self.dysect_progress_bar.set_fraction(0.0)
        self.dysect_progress_bar.set_text('session running')
        box2.pack_start(self.dysect_progress_bar, True, True, 5)
        dyysect_ok_button = gtk.Button(stock=gtk.STOCK_OK)
        dyysect_ok_button.connect("clicked", lambda w: self.dysect_dialog.destroy())
        dyysect_ok_button.set_sensitive(False)
        box2.pack_end(dyysect_ok_button, False, True, 5)
        self.dysect_dialog.vbox.pack_end(box2, False, False, 0)
        self.dysect_dialog.vbox.pack_end(vis_box, False, False, 5)
        self.dysect_dialog.show_all()

        if not os.path.exists(dysect_outfile):
            sys.stderr.write('DySect session output file %s not found\n' % dysect_outfile)
            self.on_fatal_error()
            return -1
        with open(dysect_outfile, "r") as dysect_output_file:
            prev = ''
            dot_file_list = []
            break_next_iter = False
            while 1 and self.dysect_session is True and self.STAT != None:
                filename = self.STAT.getLastDotFilename()
                if filename != "NULL" and filename != prev:
                    prev = filename
                    for file in os.listdir(self.out_dir):
                        if file.find('.dot') == -1 or file in dot_file_list or file == "dysect_session.dot":
                            continue
                        dot_file_list.append(file)
                        filename = os.path.join(self.out_dir, file)
                        if len(dot_file_list) > 1:
                            page = self.notebook.get_current_page()
                            self.create_new_tab(page + 1)
                            self.notebook.set_current_page(page + 1)
                        try:
                            with open(filename, 'r') as dot_file:
                                self.tabs[self.notebook.get_current_page()].history_view.get_buffer().set_text('')
                                self.set_dotcode(dot_file.read(), filename, self.notebook.get_current_page())
                        except IOError as e:
                            show_error_dialog('%s\nFailed to open file %s' % (repr(e), filename), self)
                        except Exception as e:
                            show_error_dialog('%s\nFailed to process file %s' % (repr(e), filename), self)
                text = dysect_output_file.read()
                if text != '' and self.dysect_dialog != None:
                    text_view_buffer.insert_with_tags_by_name(iterator, text, "monospace")
                    self.dysect_dialog.show_all()

                run_gtk_main_loop()
                if break_next_iter is True:
                    break
                if self.dysect_session and self.STAT:
                    ret = self.STAT.dysectListen(False)
                    if ret != DysectAPI_SessionCont and ret != STAT_PENDING_ACK:
                        break_next_iter = True
        dyysect_ok_button.set_sensitive(True)
        if self.dysect_timer != None:
            gobject.source_remove(self.dysect_timer)
            self.dysect_timer = None
        self.dysect_progress_bar.set_fraction(1.0)
        self.dysect_progress_bar.set_text('session complete')
        if self.dysect_dialog != None:
            self.dysect_dialog.show_all()
        run_gtk_main_loop()
        if self.STAT:
            self.STAT.dysectStop()
        stop_button.set_sensitive(False)
        return STAT_OK

    def on_identify_num_eq_classes(self, action):
        """Callback to identify equivalence classes."""
        if self.get_current_graph().cur_filename == '':
            return False
        num_eq_classes = self.get_current_graph().identify_num_eq_classes(self.get_current_widget())
        self.eq_state = {}
        self.dont_recurse = False
        eq_dialog = gtk.Dialog("Equivalence Classes", self)
        eq_dialog.set_default_size(400, 400)
        my_frame = gtk.Frame(label="%d Equivalence Classes:" % (len(num_eq_classes)))
        sw = gtk.ScrolledWindow()
        sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        classes = []
        for eq_class in num_eq_classes:
            current_class = '['
            prev = -2
            in_range = False
            task_count = 0
            task_list, fill_color_string, font_color_string = eq_class
            for num in task_list:
                task_count += 1
                if num == prev + 1:
                    in_range = True
                else:
                    if in_range is True:
                        current_class += '-%d' % prev
                    if prev != -2:
                        current_class += ', '
                    current_class += '%d' % num
                    in_range = False
                prev = num
            if in_range is True:
                current_class += '-%d' % num
            current_class += ']\n'
            item = {}
            item['class'] = task_list
            item['fill_color_string'] = fill_color_string
            item['font_color_string'] = font_color_string
            item['string'] = current_class.strip('\n').replace(',', ', ')
            classes.append(item)
        hbox = gtk.HBox()
        label = gtk.Label(' Rep  All  None  tasks')
        hbox.pack_start(label, False, False, 0)
        vbox = gtk.VBox()
        vbox.pack_start(hbox, False, False, 0)
        hbox = gtk.HBox()
        rep_button = gtk.CheckButton()
        all_button = gtk.CheckButton()
        none_button = gtk.CheckButton()
        rep_button.set_active(True)
        hbox.pack_start(rep_button, False, False, 0)
        hbox.pack_start(all_button, False, False, 7)
        hbox.pack_start(none_button, False, False, 9)
        label = gtk.Label(' select all')
        hbox.pack_start(label, False, False, 0)
        vbox.pack_start(hbox, False, False, 0)
        self.eq_state['rep_button'] = rep_button
        self.eq_state['all_button'] = all_button
        self.eq_state['none_button'] = none_button
        rep_button.connect('toggled', lambda w: self.on_toggle_eq(w, 'all', 'rep_button'))
        all_button.connect('toggled', lambda w: self.on_toggle_eq(w, 'all', 'all_button'))
        none_button.connect('toggled', lambda w: self.on_toggle_eq(w, 'all', 'none_button'))
        for item in classes:
            hbox = gtk.HBox()
            rep_button = gtk_wrap_new_from_widget(None)
            all_button = gtk_wrap_new_from_widget(rep_button)
            none_button = gtk_wrap_new_from_widget(rep_button)
            rep_button.connect('toggled', lambda w: self.on_toggle_eq(w, classes.index(item), 'rep_button'))
            all_button.connect('toggled', lambda w: self.on_toggle_eq(w, classes.index(item), 'all_button'))
            none_button.connect('toggled', lambda w: self.on_toggle_eq(w, classes.index(item), 'none_button'))
            rep_button.set_active(True)
            item['rep_button'] = rep_button
            item['all_button'] = all_button
            item['none_button'] = none_button
            hbox.pack_start(rep_button, False, False, 0)
            hbox.pack_start(all_button, False, False, 7)
            hbox.pack_start(none_button, False, False, 9)
            list_view = gtk.TextView()
            list_view.set_wrap_mode(True)
            list_view.set_editable(False)
            list_view.set_cursor_visible(False)
            iterator = list_view.get_buffer().get_iter_at_offset(0)
            list_view.get_buffer().create_tag("monospace", family="monospace")
            foreground = gtk.gdk.color_parse(item['font_color_string'])
            background = gtk.gdk.color_parse(item['fill_color_string'])
            fore_color_tag = 'fore%s%s' % (item['font_color_string'], item['fill_color_string'])
            back_color_tag = 'back%s%s' % (item['font_color_string'], item['fill_color_string'])
            list_view.get_buffer().create_tag(fore_color_tag, foreground_gdk=foreground)
            list_view.get_buffer().create_tag(back_color_tag, background_gdk=background)
            list_view.get_buffer().insert_with_tags_by_name(iterator, item['string'], fore_color_tag, back_color_tag, "monospace")
            hbox.pack_start(list_view, True, True, 5)
            vbox.pack_start(gtk.HSeparator(), False, False, 0)
            vbox.pack_start(hbox, False, False, 0)
        self.eq_state['classes'] = classes
        sw.add_with_viewport(vbox)
        my_frame.add(sw)
        eq_dialog.vbox.pack_start(my_frame, True, True, 5)
        my_frame = gtk.Frame(label="Manually Specify Additional Tasks:")
        entry = gtk.Entry()
        entry.set_max_length(65536)
        my_frame.add(entry)
        self.eq_state['entry'] = entry
        eq_dialog.vbox.pack_start(my_frame, False, False, 5)
        self.separator = gtk.HSeparator()
        eq_dialog.vbox.pack_start(self.separator, False, True, 5)
        box2 = gtk.HButtonBox()

        debuggers = ['TotalView', 'DDT bulk attach', 'DDT host:PID attach']
        for debugger in debuggers:
            button = gtk.Button(" Attach %s \n to Subset " % debugger)
            button.connect("clicked", self.launch_debugger_cb, (debugger, eq_dialog))
            box2.pack_start(button, False, True, 5)
        button = gtk.Button(" Debugger \n Options ")
        button.connect("clicked", self.set_debugger_options_cb, eq_dialog)
        box2.pack_start(button, False, True, 5)
        button = gtk.Button(stock=gtk.STOCK_CANCEL)
        button.connect("clicked", lambda w, d: eq_dialog.destroy(), "ok")
        box2.pack_end(button, False, True, 5)
        eq_dialog.vbox.pack_end(box2, False, False, 0)
        eq_dialog.show_all()
        return True

    def set_debugger_options_cb(self, w, parent):
        """Callback to update debugger options."""
        dialog = gtk.Dialog('Debugger Options', parent)
        frame = gtk.Frame(label='Options')
        vbox = gtk.VBox()
        self.pack_string_option(vbox, 'DDT Path', dialog)
        self.pack_string_option(vbox, 'DDT LaunchMON Prefix', dialog)
        self.pack_string_option(vbox, 'TotalView Path', dialog)
        self.pack_string_option(vbox, 'Additional Debugger Args', dialog)
        frame.add(vbox)
        dialog.vbox.pack_start(frame, True, True, 0)
        hbox = gtk.HButtonBox()
        button = gtk.Button(stock=gtk.STOCK_CANCEL)
        button.connect("clicked", lambda w: dialog.destroy())
        hbox.pack_start(button, True, True, 0)
        button = gtk.Button(stock=gtk.STOCK_OK)
        button.connect("clicked", lambda w: dialog.destroy())
        hbox.pack_start(button, True, True, 0)
        dialog.vbox.pack_start(hbox, False, False, 10)
        dialog.show_all()
        dialog.run()

    def on_toggle_eq(self, w, button, button_type):
        """Callback to toggle equivalence class representatives."""
        if self.dont_recurse is True:
            return True
        self.dont_recurse = True
        self.eq_state['rep_button'].set_active(False)
        self.eq_state['all_button'].set_active(False)
        self.eq_state['none_button'].set_active(False)
        if button == 'all':
            for item in self.eq_state['classes']:
                item[button_type].set_active(True)
            self.eq_state[button_type].set_active(True)
        self.dont_recurse = False
        return True

    def set_proctab(self):
        if self.proctab_file_path is not None and self.proctab is not None:
            return True
        if self.STAT is None:
            return False

        self.proctab_file_path = ''
        out_dir = self.STAT.getOutDir()
        file_prefix = self.STAT.getFilePrefix()
        self.proctab_file_path = out_dir + '/' + file_prefix + '.ptab'
        if not os.path.exists(self.proctab_file_path):
            failed_ptab_path = self.proctab_file_path
            directory = os.path.dirname(os.path.abspath(self.get_current_graph().cur_filename))
            self.proctab_file_path = ''
            for filename in os.listdir(directory):
                if filename.find('.ptab') != -1:
                    self.proctab_file_path = directory + '/' + filename
                    break
        if self.proctab_file_path == '':
            show_error_dialog('Failed to find process table file %s or .ptab file in %s.' % (failed_ptab_path, directory), self)
            return False

        try:
            self.proctab = get_proctab(self.proctab_file_path)
        except IOError as e:
            show_error_dialog('%s\nfailed to open process table file:\n\n%s\n\nPlease be sure that it is a valid process table file outputted from STAT.' % (repr(e), self.proctab_file_path), self)
            return False
        except Exception as e:
            show_error_dialog('%s\nfailed to process process table file:\n\n%s\n\nPlease be sure that it is a valid process table file outputted from STAT.' % (repr(e), self.proctab_file_path), self)
            return False
        return True

    def launch_debugger_cb(self, widget, args):
        """Callback to launch full-featured debugger on a subset of tasks."""
        if self.attached is False:
            show_error_dialog('Subset attach is only available when STAT is attached to the application.  Please (re)attach to the application.', self)
            return False
        debugger, eq_dialog = args
        subset_list = []
        for item in self.eq_state['classes']:
            if item['rep_button'].get_active() is True and len(item['class']) > 0:
                subset_list.append(item['class'][0])
            elif item['all_button'].get_active() is True and len(item['class']) > 0:
                subset_list += item['class']
        additional_tasks = '[' + self.eq_state['entry'].get_text().strip(' ').strip('[').strip(']') + ']'
        task_list_set = set(get_task_list(additional_tasks) + subset_list)
        subset_list = list(task_list_set)
        if subset_list == []:
            show_error_dialog('No tasks selected.  Please choose a subset of tasks to debug', self)
            return False
        eq_dialog.destroy()
        # use current STAT session to determine job launcher PID
        ret = self.set_proctab()
        if ret is False:
            return False
        executable = self.proctab.executable_path
        self.executable_path = executable
        self.cancel = False
        while os.path.exists(self.executable_path) is False and self.cancel is False:
            # Open dialog prompting for base path
            dialog = gtk.Dialog('Please enter the full path for the executable')
            label = gtk.Label('Failed to find executable:\n%s\nPlease enter the executable full path' % self.executable_path)
            dialog.vbox.pack_start(label, True, True, 0)
            entry = gtk.Entry()
            entry.set_max_length(8192)

            def activate_entry(entry, dialog):
                self.executable_path = entry.get_text()
                dialog.destroy()
            entry.connect("activate", lambda w: activate_entry(entry, dialog))
            dialog.vbox.pack_start(entry, True, True, 0)
            hbox = gtk.HButtonBox()
            button = gtk.Button(stock=gtk.STOCK_CANCEL)

            def cancel_entry(dialog):
                entry.set_text('cancel_operation')
                self.cancel = True
                dialog.destroy()
            button.connect("clicked", lambda w: cancel_entry(dialog))
            hbox.pack_start(button, False, False, 0)
            button = gtk.Button(stock=gtk.STOCK_OK)
            button.connect("clicked", lambda w: activate_entry(entry, dialog))
            hbox.pack_start(button, False, False, 0)
            dialog.vbox.pack_start(hbox, False, False, 0)
            dialog.show_all()
            dialog.run()
            if entry.get_text() == 'cancel_operation':
                dialog.destroy()
                return False
        if self.cancel is True:
            return False
        pids = []
        hosts = []
        exes = []
        for rank, host, pid, exe_index in self.proctab.process_list:
            if rank in subset_list:
                pids.append(pid)
                hosts.append(host)
                exes.append(self.proctab.executable_paths[exe_index])
        arg_list = []
        if debugger == 'TotalView':
            filepath = self.options['TotalView Path']
            if not filepath or not os.access(filepath, os.X_OK):
                show_error_dialog('Failed to locate executable totalview\ndefault: %s\n' % filepath, self)
                return
            arg_list.append(filepath)
            cli_attach = False
            if self.STAT is not None:
                if self.STAT.getApplicationOption() == STAT_SERIAL_ATTACH or self.STAT.getApplicationOption() != STAT_SERIAL_GDB_ATTACH:
                    cli_attach = True
            if cli_attach is True:
                arg_list.append('-e')
                for counter, rank in enumerate(subset_list):
                    if counter == 0:
                        cli_command = "set new_pid [dattach -r %s -e %s %d] ;" % (hosts[counter], exes[counter], pids[counter])
                    else:
                        cli_command += " dattach -r %s -g $CGROUP($new_pid) -e %s %d ;" % (hosts[counter], exes[counter], pids[counter])
                arg_list.append(cli_command)
            else:
                arg_list.append('-parallel_attach')
                arg_list.append('yes')
                if self.options['Remote Host'] not in ['localhost', '']:
                    arg_list.append('-remote')
                    arg_list.append(self.options['Remote Host'])
                arg_list.append('-pid')
                arg_list.append(str(self.proctab.launcher_pid))
                str_list = ''
                for rank in subset_list:
                    str_list += '%s ' % (rank)
                arg_list.append('-default_parallel_attach_subset=%s' % (str_list))
                exe = self.options['Launcher Exe']
                if (exe.find(' \\_') != -1):
                    exe = exe.split()[1]
                else:
                    exe = exe.split()[0]
                arg_list.append(exe)
        elif debugger == 'DDT bulk attach' or debugger == 'DDT host:PID attach':
            filepath = self.options['DDT Path']
            if not filepath or not os.access(filepath, os.X_OK):
                show_error_dialog('Failed to locate executable ddt\ndefault: %s\n' % filepath, self)
                return

            arg_list.append(filepath)
            if debugger == 'DDT bulk attach':
                arg_list.append("--attach-mpi")
                arg_list.append(str(self.proctab.launcher_pid))
                arg_list.append("--subset")
                rank_list_arg = ''
                for rank in subset_list:
                    if rank == subset_list[0]:
                        rank_list_arg += '%d' % rank
                    else:
                        rank_list_arg += ',%d' % rank
                arg_list.append(rank_list_arg)
            else:
                arg_list.append("--attach")
                hplist = self.proctab.process_list
                attach_list_arg = ''
                for rank in subset_list:
                    attach_list_arg += hplist[rank][1]+':'+'%d' % hplist[rank][2]+','
                arg_list.append(attach_list_arg[:-1])
            arg_list.append(self.executable_path)

        for arg in self.options['Additional Debugger Args'].split():
            arg_list.insert(-1, arg)

        # First Detach STAT!!!
        if self.STAT is not None:
            stop_list = intArray(len(subset_list))
            i = -1
            for rank in subset_list:
                i += 1
                stop_list[i] = rank
            self.on_detach(None, stop_list, len(subset_list))

        sys.stdout.write('fork exec %s %s\n' % (debugger, arg_list))
        if os.fork() == 0:
            exec_and_exit(arg_list)

    def get_full_edge_label(self, widget, button_clicked, node):
        edge_label = self.STAT.getNodeInEdge(int(node.node_name))
        old_label = node.edge_label
        node.edge_label = edge_label
        task_list = get_task_list(edge_label)
        node.num_leaf_tasks = -1
        if node.edge_label_id in STATview.task_label_id_to_list:
            STATview.task_label_id_to_list[node.edge_label_id] = task_list
        if old_label in STATview.task_label_to_list:
            del STATview.task_label_to_list[old_label]
        for inode in self.get_current_graph().nodes:
            if inode.edge_label == old_label:
                inode.edge_label = edge_label
                inode.num_leaf_tasks = -1
                if inode.edge_label_id in STATview.task_label_id_to_list:
                    STATview.task_label_id_to_list[inode.edge_label_id] = task_list
        try:
            self.my_dialog.destroy()
        except:
            pass
        self.on_node_clicked(widget, button_clicked, node)

    def on_fatal_error(self):
        """Handle fatal error."""
        self.set_action_sensitivity('new')
        self.STAT.shutDown()
        self.STAT = None
        self.reattach = False
        self.attached = False
        self.show_all()

    def pack_sample_options(self, vbox, multiple, attach=False):
        """Pack the sample options into the specified vbox."""
        frame = gtk.Frame(label='Per Sample Options')
        vbox2 = gtk.VBox()
        self.pack_check_button(vbox2, 'With Threads', False, False, 0)
        self.pack_spinbutton(vbox2, 'Max Threads Per Daemon')
        if HAVE_OPENMP_SUPPORT:
            self.pack_check_button(vbox2, 'With OpenMP', False, False, 0)
        if HAVE_GDB_SUPPORT:
            self.pack_check_button(vbox2, 'With CUDA Quick', False, False, 0)
        self.pack_check_button(vbox2, 'Gather Python Traces', False, False, 5)
        frame2 = gtk.Frame(label='Stack Frame (node) Sample Options')
        vbox3 = gtk.VBox()
        self.pack_radio_buttons(vbox3, 'Sample Type')
        frame2.add(vbox3)
        vbox2.pack_start(frame2, False, False, 5)
        frame2 = gtk.Frame(label='Process Set (edge) Sample Options')
        vbox3 = gtk.VBox()
        self.pack_radio_buttons(vbox3, 'Edge Type')
        frame2.add(vbox3)
        vbox2.pack_start(frame2, False, False, 5)

        if attach is False:
            self.pack_spinbutton(vbox2, 'Run Time Before Sample (sec)')
        expander = gtk.Expander(label="Advanced")
        hbox = gtk.HBox()
        self.pack_spinbutton(hbox, 'Num Retries')
        self.pack_spinbutton(hbox, 'Retry Frequency (us)')
        expander.add(hbox)
        vbox2.pack_start(expander, False, False, 5)
        frame.add(vbox2)
        vbox.pack_start(frame, False, False, 5)
        if multiple:
            frame = gtk.Frame(label='Multiple Sample Options')
            vbox2 = gtk.VBox()
            hbox = gtk.HBox()
            self.pack_spinbutton(hbox, 'Num Traces')
            self.pack_spinbutton(hbox, 'Trace Frequency (ms)')
            vbox2.pack_start(hbox, False, False, 5)
            self.pack_check_button(vbox2, 'Gather Individual Samples')
            self.pack_check_button(vbox2, 'Clear On Sample')
            frame.add(vbox2)
            vbox.pack_start(frame, False, False, 5)


def STATGUI_main(args):
    """The STATGUI main."""
    window = STATGUI(args)
    STATview.window = window
    window.connect('destroy', window.on_destroy)
    #gtk.gdk.threads_init()
    gtk.main()

if __name__ == '__main__':
    sys.stderr.write('WARNING: STATGUI.py should not be directly invoked. This has been replaced by STATmain.py and the "view" subcommand.\n')
    sys.exit(1)
