#!/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Manage remote STAT sessions over SSH.

This module maps remote API requests to STAT's core API through the Python
bindings for STAT_FrontEnd. It manages attachment, process control, sampling,
DOT graph export, source retrieval, and detach.

StatSessionController owns at most one active session per SSH connection.
StatSession keeps its native frontend alive so successive requests can pause,
sample, and resume the same application. Running stat-cl for each request
would create a separate frontend and lose that attachment between commands.

Request flow::

    StatSessionController.dispatch()
    |
    +-- sessions.attach
    |   `-- create_session() reserves the session
    |       `-- attach() returns its session_id
    |
    `-- requests using that session_id
        |
        +-- sessions.pause / sessions.resume
        |   `-- pause() / resume() waits for acknowledgement
        |
        +-- sessions.sample
        |   `-- sample() pauses the job and returns one 2D graph
        |
        +-- sessions.sample_multiple
        |   +-- sample() collects each 2D snapshot
        |   `-- gatherTraces() produces the merged 3D graph
        |
        `-- sessions.detach
            `-- detach() waits for acknowledgement and shuts down the frontend
                `-- the controller releases the session

Sampling leaves the job paused after the final snapshot. If the SSH connection
closes while the job is attached, the server also attempts to detach.
"""

from __future__ import print_function

import os
import pwd
import socket
import sys
import time
import uuid
import ctypes

from stat_build_info import STAT_VERSION, GIT_COMMIT, GIT_DIRTY

from remote import stat_server_helpers as attach_helper
from remote.stat_server_flux import FluxDebugHelper
from STAThelper import get_proctab, var_spec_to_string

if hasattr(sys, "getdlopenflags"):
    sys.setdlopenflags(ctypes.RTLD_GLOBAL | sys.getdlopenflags())

from STAT import (
    STAT_FrontEnd,
    STAT_LOG_NONE,
    STAT_LOG_FE,
    STAT_LOG_BE,
    STAT_LOG_CP,
    STAT_LOG_MRN,
    STAT_LOG_SW,
    STAT_LOG_SWERR,
    STAT_OK,
    STAT_VERBOSE_ERROR,
    STAT_VERBOSE_FULL,
    STAT_VERBOSE_STDOUT,
    STAT_TOPOLOGY_AUTO,
    STAT_TOPOLOGY_DEPTH,
    STAT_TOPOLOGY_FANOUT,
    STAT_TOPOLOGY_USER,
    STAT_PENDING_ACK,
    STAT_LAUNCH,
    STAT_ATTACH,
    STAT_SERIAL_ATTACH,
    STAT_GDB_ATTACH,
    STAT_SERIAL_GDB_ATTACH,
    STAT_SERIAL_PYSTACK_ATTACH,
    STAT_DAEMON_BACKEND_PYSTACK,
    STAT_DAEMON_BACKEND_GDB,
    STAT_PYSTACK_BACKEND_ENABLED,
    STAT_SAMPLE_PYSTACK,
    STAT_SAMPLE_CHILDREN,
    STAT_SAMPLE_LINE,
    STAT_SAMPLE_PC,
    STAT_SAMPLE_THREADS,
    STAT_SAMPLE_CLEAR_ON_SAMPLE,
    STAT_SAMPLE_PYTHON,
    STAT_SAMPLE_MODULE_OFFSET,
    STAT_CP_NONE,
    STAT_CP_SHAREAPPNODES,
    STAT_CP_EXCLUSIVE,
)

try:
    from STAT import STAT_SAMPLE_OPENMP
except Exception:
    STAT_SAMPLE_OPENMP = None

SUPPORTED_BACKENDS = ("dyninst", "pystack") if STAT_PYSTACK_BACKEND_ENABLED else ("dyninst",)
# The bindings export this sample flag only in GDB-enabled builds.
try:
    from STAT import STAT_SAMPLE_CUDA_QUICK
except ImportError:
    STAT_SAMPLE_CUDA_QUICK = None
else:
    SUPPORTED_BACKENDS += ("gdb",)
DEFAULT_BACKEND = "dyninst"

# defaults for daemon and sampling options in remote requests.
DEFAULT_COMM_PROCS_PER_NODE = 8
DEFAULT_DAEMONS_PER_NODE = 1
DEFAULT_NUM_RETRIES = 5
DEFAULT_RETRY_FREQ_US = 10
DEFAULT_MAX_THREADS_PER_DAEMON = 512
DEFAULT_NUM_TRACES = 2
DEFAULT_TRACE_FREQUENCY_MS = 1000

# polling for native acknowledgements and MRNet connection setup.
NATIVE_WAIT_TIMEOUT_SECONDS = 600
NATIVE_POLL_INTERVAL_SECONDS = 0.05


class StatRemoteError(Exception):
    """Describe a remote operation failure with actionable context.

    The server turns this error into a response; session cleanup remains the
    responsibility of the caller.
    """

    def __init__(self, code, message, phase=None, context=None):
        Exception.__init__(self, message)
        self.code = code
        self.message = message
        self.phase = phase
        self.context = context or {}

    def to_dict(self):
        """Return the error fields consumed by the JSON protocol layer."""
        return {
            "code": self.code,
            "message": self.message,
            "phase": self.phase,
            "context": self.context,
        }


def _bool(params, key, default=False):
    """Read a boolean option, accepting the existing textual true values."""
    value = params.get(key, default)
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return bool(value)


def _int(params, key, default):
    """Read an integer option or return a structured parameter error."""
    try:
        return int(params.get(key, default))
    except (TypeError, ValueError, OverflowError):
        raise StatRemoteError("invalid_params", "%s must be an integer" % key)


def _str(params, key, default=""):
    """Read a string option, treating None as the default."""
    value = params.get(key, default)
    if value is None:
        return default
    return str(value)


def _scheduler(params):
    """Translate scheduler selection failures into protocol parameter errors."""
    try:
        return attach_helper.select_scheduler(_str(params, "scheduler", "auto"))
    except ValueError as exc:
        raise StatRemoteError("invalid_scheduler", str(exc), "resolve") from exc


def _stat_error(stat, code, phase, message, context=None):
    """Combine operation context with the native frontend diagnostic."""
    detail = ""
    try:
        detail = stat.getLastErrorMessage()
    except Exception:
        detail = ""
    if detail:
        message = "%s: %s" % (message, detail)
    return StatRemoteError(code, message, phase, context)


def _emit_progress(callback, fraction, task_complete=False, **details):
    """Forward a bounded progress fraction when a caller requested updates."""
    if callback is None:
        return
    event = {
        "fraction": max(0.0, min(1.0, float(fraction))),
        "task_complete": bool(task_complete),
    }
    event.update(details)
    callback(event)


class StatSession(object):
    """Keep one application attached for interactive use through the remote-connect API.

    The CLI and future GUI reuse the same native frontend to pause, sample and
    resume. The session owns its resources until detach succeeds.
    """

    def __init__(self, session_id=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.stat = STAT_FrontEnd()
        self.state = "new"
        self.backend = DEFAULT_BACKEND
        self.mode = None
        self.remote_host = None
        self.pid = None
        self.flux_helper = None
        self.job = None
        self.rm = None
        self.remote_shell = "ssh"
        self.last_dot_filename = None
        self.out_dir = None
        self.created_at = time.time()

    def _sample_type(self, params):
        """Translate frame, thread and accumulation options into native sample flags."""
        # TODO: Test PC and module-offset samples and the OpenMP sampling flag over SSH.
        sample_type = 0
        stack_type = _str(params, "sample_type", _str(params, "stack_type", "function only"))

        if stack_type == "function and pc":
            sample_type |= STAT_SAMPLE_PC
        elif stack_type == "module offset":
            sample_type |= STAT_SAMPLE_MODULE_OFFSET
        elif stack_type == "function and line":
            sample_type |= STAT_SAMPLE_LINE

        with_threads = _bool(params, "with_threads", False)

        if _bool(params, "cuda_quick"):
            if self.backend != "gdb" or not with_threads or STAT_SAMPLE_CUDA_QUICK is None:
                raise StatRemoteError("invalid_params", "cuda_quick requires GDB with thread sampling", "sample")
            sample_type |= STAT_SAMPLE_CUDA_QUICK

        if with_threads:
            sample_type |= STAT_SAMPLE_THREADS
        if _bool(params, "with_openmp", False) and STAT_SAMPLE_OPENMP is not None:
            sample_type |= STAT_SAMPLE_OPENMP
        if self.backend == "pystack":
            sample_type |= STAT_SAMPLE_PYSTACK
            if _bool(params, "with_children"):
                sample_type |= STAT_SAMPLE_CHILDREN
        elif _bool(params, "native") or _bool(params, "with_children"):
            raise StatRemoteError("invalid_params", "native and with_children require the pystack backend", "sample")
        elif _bool(params, "python_trace", False):
            sample_type |= STAT_SAMPLE_PYTHON

        if _bool(params, "clear_on_sample", True):
            sample_type |= STAT_SAMPLE_CLEAR_ON_SAMPLE
        return sample_type

    def _topology(self, params):
        """Translate MRNet topology and communication-process placement options."""
        # TODO: Test topology options over remote connect.
        topology_type = _str(params, "topology_type", "automatic")
        if topology_type == "depth":
            stat_topology = STAT_TOPOLOGY_DEPTH
        elif topology_type == "max fanout":
            stat_topology = STAT_TOPOLOGY_FANOUT
        elif topology_type == "custom":
            stat_topology = STAT_TOPOLOGY_USER
        else:
            stat_topology = STAT_TOPOLOGY_AUTO

        cp_policy = _str(params, "cp_policy", "share app nodes")
        if cp_policy == "exclusive":
            stat_cp_policy = STAT_CP_EXCLUSIVE
        elif cp_policy == "none":
            stat_cp_policy = STAT_CP_NONE
        else:
            stat_cp_policy = STAT_CP_SHAREAPPNODES
        return stat_topology, _str(params, "topology", "1"), stat_cp_policy

    def _attach_mode(self, serial=False):
        """Select the native attachment mode independently of daemon startup."""
        if self.backend == "pystack":
            return STAT_SERIAL_PYSTACK_ATTACH if serial else STAT_ATTACH
        if self.backend == "gdb":
            return STAT_SERIAL_GDB_ATTACH if serial else STAT_GDB_ATTACH
        return STAT_SERIAL_ATTACH if serial else STAT_ATTACH

    def _wait_ack(self, phase, context=None, timeout_sec=NATIVE_WAIT_TIMEOUT_SECONDS):
        """Wait for one asynchronous native operation to acknowledge completion.

        Poll without blocking so pending acknowledgements have a bounded wait.
        A timeout reports uncertainty; it does not cancel the native operation.
        """
        deadline = time.monotonic() + timeout_sec
        while True:
            ret = self.stat.receiveAck(False)
            if ret == STAT_OK:
                return
            if ret != STAT_PENDING_ACK:
                raise _stat_error(self.stat, "stat_%s_failed" % phase, phase, "STAT operation failed", context)
            if time.monotonic() > deadline:
                raise StatRemoteError(
                    "stat_%s_timeout" % phase, "Timed out waiting for STAT %s acknowledgement" % phase,
                    phase, context)
            time.sleep(NATIVE_POLL_INTERVAL_SECONDS)

    def _configure_common(self, params, application_option):
        """Configure daemon paths, diagnostics and frontend options before attachment."""
        # TODO: Test daemon and filter path overrides, diagnostic logging, and nondefault daemon counts over remote connect.
        tool_daemon_path = params.get("tool_daemon_path")
        filter_path = params.get("filter_path")
        if tool_daemon_path:
            self.stat.setToolDaemonExe(str(tool_daemon_path))
        if filter_path:
            self.stat.setFilterPath(str(filter_path))

        log_type = STAT_LOG_NONE
        if _bool(params, "log_frontend", False):
            log_type |= STAT_LOG_FE
        if _bool(params, "log_backend", False):
            log_type |= STAT_LOG_BE
        if _bool(params, "log_cp", False):
            log_type |= STAT_LOG_CP
        if _bool(params, "log_sw", False):
            log_type |= STAT_LOG_SW
        if _bool(params, "log_swerr", False):
            log_type |= STAT_LOG_SWERR
        if _bool(params, "use_mrnet_printf", False):
            log_type |= STAT_LOG_MRN
        if log_type != STAT_LOG_NONE:
            ret = self.stat.startLog(log_type, _str(params, "log_dir", os.environ.get("HOME", ".")))
            if ret != STAT_OK:
                raise _stat_error(self.stat, "stat_log_failed", "attach", "Failed to start STAT log")

        verbosity = _str(params, "verbosity", "error")
        if verbosity == "stdout":
            self.stat.setVerbose(STAT_VERBOSE_STDOUT)
        elif verbosity == "full":
            self.stat.setVerbose(STAT_VERBOSE_FULL)
        else:
            self.stat.setVerbose(STAT_VERBOSE_ERROR)

        if _bool(params, "debug_backends", False):
            os.environ["LMON_DEBUG_BES"] = "1"
        elif "LMON_DEBUG_BES" in os.environ:
            del os.environ["LMON_DEBUG_BES"]

        if _bool(params, "check_node_access", False):
            os.environ["STAT_CHECK_NODE_ACCESS"] = "1"
        elif "STAT_CHECK_NODE_ACCESS" in os.environ:
            del os.environ["STAT_CHECK_NODE_ACCESS"]

        self.stat.setProcsPerNode(_int(params, "comm_procs_per_node", DEFAULT_COMM_PROCS_PER_NODE))
        self.stat.setNDaemonsPerNode(_int(params, "daemons_per_node", DEFAULT_DAEMONS_PER_NODE))

        output_job_id = params.get("output_job_id")

        if output_job_id:
            self.stat.setJobId(str(output_job_id))
        self.stat.setApplicationOption(application_option)

        if self.backend == "pystack":
            self.stat.setDaemonBackend(STAT_DAEMON_BACKEND_PYSTACK)
        elif self.backend == "gdb":
            self.stat.setDaemonBackend(STAT_DAEMON_BACKEND_GDB)

    def attach(self, params, progress=None):
        """Attach to an application and retain the frontend for interactive requests.

        Targets may already be controlled when attachment fails, so the controller
        retains ownership and handles cleanup.
        """
        self.backend = _str(params, "backend", DEFAULT_BACKEND)
        if _bool(params, "gdb_backend", False):
            # the legacy flag overrides Dyninst (the default), but cannot select GDB alongside PyStack.
            if self.backend not in ("dyninst", "gdb"):
                raise StatRemoteError("invalid_params", "gdb_backend conflicts with the requested backend", "attach")
            self.backend = "gdb"
        if self.backend not in SUPPORTED_BACKENDS:
            raise StatRemoteError("invalid_backend", "Unsupported backend %s" % self.backend, "attach")

        # attach: use an existing MPI launcher PID and its registered ranks.
        # jobid: resolve a scheduler job to a launcher, then follow attach.
        # serial: use application processes directly, without an MPI launcher.
        # launch: start a new application through the supplied launcher command.
        mode = _str(params, "mode", "attach")
        serial = mode == "serial"
        application_option = self._attach_mode(serial=serial)

        self._configure_common(params, application_option)

        if mode == "jobid":
            jobid = _str(params, "jobid")
            if not jobid:
                raise StatRemoteError("invalid_params", "jobid is required for mode=jobid", "attach")

            # only job discovery depends on the scheduler; sampling uses native STAT.
            self.rm = _scheduler(params)
            if self.rm == "flux":
                self.flux_helper = FluxDebugHelper(jobid)
                try:
                    self.flux_helper.start()
                except (OSError, RuntimeError) as exc:
                    raise StatRemoteError("flux_attach_failed", str(exc), "attach", {"jobid": jobid}) from exc
                remote_host, launcher_exe, pids = "localhost", "flux-job", [self.flux_helper.pid]
            else:
                remote_host, launcher_exe, pids = attach_helper.jobid_to_hostname_pid(
                    self.rm, jobid, self.remote_shell)

            if remote_host is None or not pids:
                raise StatRemoteError(
                    "job_resolve_failed", "Failed to resolve job id %s" % jobid,
                    "attach", {"jobid": jobid})

            if len(pids) != 1:
                raise StatRemoteError(
                    "ambiguous_launcher", "Multiple launchers found; attach by launcher PID instead",
                    "attach", {"jobid": jobid, "scheduler": self.rm, "pids": pids})

            self.remote_host = remote_host
            self.pid = int(pids[0])
            self.job = {"jobid": jobid, "launcher_exe": launcher_exe, "pids": pids}
            mode = "attach"

        elif mode == "attach":
            self.remote_host = _str(params, "remote_host", "localhost")
            self.pid = _int(params, "pid", 0)
            if self.pid <= 0:
                raise StatRemoteError("invalid_params", "pid is required for mode=attach", "attach")

        elif mode == "launch":
            argv = params.get("argv") or []
            if not argv:
                raise StatRemoteError("invalid_params", "argv is required for mode=launch", "attach")

            for arg in argv:
                self.stat.addLauncherArgv(str(arg))

            application_option = STAT_LAUNCH
            self.stat.setApplicationOption(application_option)
            self.remote_host = _str(params, "remote_host", "localhost")

        elif mode == "serial":
            processes = params.get("processes") or []
            if not processes:
                raise StatRemoteError("invalid_params", "processes is required for mode=serial", "attach")

            for process in processes:
                self.stat.addSerialProcess(str(process))

        else:
            raise StatRemoteError("invalid_params", "Unsupported attach mode %s" % mode, "attach")

        # start the appropriate native daemon-launch path for the selected target.
        self.mode = mode
        if mode == "attach":
            if self.remote_host and self.remote_host != "localhost":
                ret = self.stat.attachAndSpawnDaemons(self.pid, self.remote_host)
            else:
                ret = self.stat.attachAndSpawnDaemons(self.pid)

        elif mode == "launch":
            if self.remote_host and self.remote_host != "localhost":
                ret = self.stat.launchAndSpawnDaemons(self.remote_host)
            else:
                ret = self.stat.launchAndSpawnDaemons()

        else:
            ret = self.stat.setupForSerialAttach()

        if "LMON_DEBUG_BES" in os.environ:
            del os.environ["LMON_DEBUG_BES"]

        if ret != STAT_OK:
            raise _stat_error(
                self.stat, "stat_attach_failed", "attach",
                "Failed to launch STAT daemons", self.context())

        _emit_progress(progress, 0.10, task_complete=True)

        # build the communication tree before attaching the application.
        # TODO: Test topology options over remote connect.
        topology_type, topology, cp_policy = self._topology(params)
        comm_nodes = _str(params, "communication_nodes", "")
        ret = self.stat.launchMrnetTree(topology_type, topology, comm_nodes, False, cp_policy)

        if ret != STAT_OK:
            raise _stat_error(
                self.stat, "stat_mrnet_launch_failed", "attach",
                "Failed to launch MRNet tree", self.context())

        if mode != "serial":
            deadline = time.monotonic() + NATIVE_WAIT_TIMEOUT_SECONDS
            while True:
                if time.monotonic() >= deadline:
                    raise StatRemoteError(
                        "stat_mrnet_connect_timeout", "Timed out connecting MRNet tree",
                        "attach", self.context())

                ret = self.stat.connectMrnetTree(False)
                if ret == STAT_OK:
                    break
                if ret != STAT_PENDING_ACK:
                    raise _stat_error(
                        self.stat, "stat_mrnet_connect_failed", "attach",
                        "Failed to connect MRNet tree", self.context())

                time.sleep(NATIVE_POLL_INTERVAL_SECONDS)

        ret = self.stat.setupConnectedMrnetTree()

        if ret != STAT_OK:
            raise _stat_error(
                self.stat, "stat_setup_failed", "attach",
                "Failed to setup connected MRNet tree", self.context())

        _emit_progress(progress, 0.15, task_complete=True)

        ret = self.stat.attachApplication(False)

        if ret != STAT_OK:
            raise _stat_error(
                self.stat, "stat_application_attach_failed", "attach",
                "Failed to attach application", self.context())

        # attachment may already control targets even if its acknowledgement fails.
        self.state = "attaching"
        self._wait_ack("attach", self.context())

        _emit_progress(progress, 0.20, task_complete=True)

        self.state = "attached"

        try:
            self.out_dir = self.stat.getOutDir()
        except Exception:
            self.out_dir = None

        result = self.status()
        if _bool(params, "initial_sample", False):
            sample_params = dict(params)
            sample_params.setdefault("clear_on_sample", True)
            result["initial_sample"] = self.sample(sample_params, progress)

        return result

    def pause(self):
        """Pause attached targets and update state after acknowledgement."""
        if self.state in ("new", "detached"):
            raise StatRemoteError("invalid_state", "Session is not attached", "pause", self.context())
        ret = self.stat.pause(False)
        if ret != STAT_OK:
            raise _stat_error(self.stat, "stat_pause_failed", "pause", "Failed to pause application", self.context())
        self._wait_ack("pause", self.context())
        self.state = "paused"
        return self.status()

    def resume(self):
        """Resume attached targets and update state after acknowledgement."""
        if self.state in ("new", "detached"):
            raise StatRemoteError("invalid_state", "Session is not attached", "resume", self.context())
        ret = self.stat.resume(False)
        if ret != STAT_OK:
            raise _stat_error(self.stat, "stat_resume_failed", "resume", "Failed to resume application", self.context())
        self._wait_ack("resume", self.context())
        self.state = "running"
        return self.status()

    def _read_last_dot(self):
        """Read the graph most recently exported by the native frontend."""
        filename = self.stat.getLastDotFilename()
        self.last_dot_filename = filename
        try:
            with open(filename, "r") as stream:
                dot_text = stream.read()
        except Exception as exc:
            raise StatRemoteError(
                "artifact_read_failed", "Failed to read DOT file %s: %s" % (filename, exc),
                "artifact", self.context())
        return {
            "dot_filename": filename,
            "dot_text": dot_text,
            "out_dir": self.out_dir,
        }

    def sample(self, params, progress=None):
        """Pause the application, collect one snapshot and return its 2D DOT graph.

        Wait for pause, sampling and gather acknowledgements in order. Retries
        use retry_freq in microseconds. Leave targets paused so a GUI can inspect
        the result before explicitly resuming or detaching.
        """
        if self.state in ("new", "detached"):
            raise StatRemoteError("invalid_state", "Session is not attached", "sample", self.context())
        if _str(params, "backend", self.backend) != self.backend:
            raise StatRemoteError("invalid_backend", "Changing backend requires a new attachment", "sample")

        sample_type = self._sample_type(params)
        # validate native arguments before pausing a running application.
        num_retries = _int(params, "num_retries", DEFAULT_NUM_RETRIES)
        retry_freq = _int(params, "retry_freq", DEFAULT_RETRY_FREQ_US)
        max_threads = _int(params, "max_threads_per_daemon", DEFAULT_MAX_THREADS_PER_DAEMON)

        if not all(0 <= value <= 0xffffffff for value in (num_retries, retry_freq, max_threads)):
            raise StatRemoteError("invalid_params", "Retry and thread options must fit an unsigned 32-bit integer", "sample")
        try:
            # TODO: Test remote sampling with a nonempty variable specification.
            var_spec = var_spec_to_string(params.get("var_spec", []))
        except (TypeError, ValueError, OverflowError) as exc:
            raise StatRemoteError("invalid_params", "Invalid variable specification", "sample") from exc
        native_mode = "Python + Native" if _bool(params, "native") else "Python Only"

        _emit_progress(progress, 0.01)
        ret = self.stat.pause(False)

        if ret != STAT_OK:
            raise _stat_error(
                self.stat, "stat_pause_failed", "sample",
                "Failed to pause application before sample", self.context())

        self._wait_ack("pause", self.context())
        self.state = "paused"

        _emit_progress(progress, 0.35)

        ret = self.stat.sampleStackTraces(
            sample_type,
            1,
            1,
            num_retries, retry_freq, max_threads, False, var_spec, native_mode,
        )
        if ret != STAT_OK:
            raise _stat_error(
                self.stat, "stat_sample_failed", "sample",
                "Failed to sample stack traces", self.context())
        self._wait_ack("sample", self.context())
        _emit_progress(progress, 0.60, task_complete=True)

        ret = self.stat.gatherLastTrace(False)
        if ret != STAT_OK:
            raise _stat_error(
                self.stat, "stat_gather_failed", "sample",
                "Failed to gather stack traces", self.context())
        self._wait_ack("gather", self.context())
        _emit_progress(progress, 0.70, task_complete=True)

        self.state = "paused"
        result = self.status()
        result.update(self._read_last_dot())
        return result

    def sample_multiple(self, params, progress=None):
        """Collect snapshots and export their accumulated 3D graph.

        Clear previous traces for the first sample, then retain later traces.
        A positive interval resumes the application between snapshots; zero
        keeps it paused. Each snapshot exports a 2D graph. Finally gather all
        traces, wait for acknowledgement and return the 3D graph plus snapshots.
        """
        params = dict(params)
        params["clear_on_sample"] = True
        n_traces = max(1, _int(params, "num_traces", DEFAULT_NUM_TRACES))
        trace_frequency_ms = max(0, _int(params, "trace_frequency_ms", DEFAULT_TRACE_FREQUENCY_MS))

        samples = []
        for index in range(n_traces):
            if index > 0:
                params["clear_on_sample"] = False

            def trace_progress(event, trace_index=index):
                event = dict(event)
                event["trace_index"] = trace_index + 1
                event["trace_count"] = n_traces
                event["fraction"] = (trace_index + event.get("fraction", 0.0)) / float(n_traces)
                if progress is not None and not event.get("task_complete"):
                    progress(event)

            samples.append(self.sample(params, trace_progress))
            if index != n_traces - 1 and trace_frequency_ms > 0:
                if self.state != "running":
                    self.resume()
                time.sleep(float(trace_frequency_ms) / 1000.0)

        # gatherLastTrace exports each 2D snapshot; gatherTraces exports the accumulated 3D graph.
        _emit_progress(progress, 0.91, task_complete=True)
        ret = self.stat.gatherTraces(False)

        if ret != STAT_OK:
            raise _stat_error(
                self.stat, "stat_gather_failed", "sample",
                "Failed to gather merged stack traces", self.context())
        self._wait_ack("gather", self.context())

        result = self.status()
        result.update(self._read_last_dot())
        result["samples"] = samples
        _emit_progress(progress, 0.95, task_complete=True)
        return result

    def detach(self):
        """Release the application before shutting down the native frontend.

        Already-detached sessions are harmless to close again. Mark completion
        only after detach acknowledgement and shutdown; errors leave ownership
        with the controller for a retry.
        """
        if self.state == "detached":
            return self.status()
        if self.state != "new":
            ret = self.stat.detachApplication(False)
            if ret != STAT_OK:
                raise _stat_error(self.stat, "stat_detach_failed", "detach", "Failed to detach application", self.context())
            self._wait_ack("detach", self.context())

        self.stat.shutDown()
        if self.flux_helper:
            self.flux_helper.close()
        self.state = "detached"
        return self.status()

    def source(self, params):
        """Validate the session and retrieve source through its recorded processes.

        Load STAT's process table, then delegate rank selection and bounded
        namespace reads to the source helper. Translate retrieval errors into
        protocol errors; never fall back to files on the client machine.
        """
        if self.state in ("new", "detached"):
            raise StatRemoteError("invalid_state", "Session is not attached", "source", self.context())
        source = _str(params, "source")

        if not source or "\x00" in source:
            raise StatRemoteError("invalid_source_path", "A valid source path is required", "source", self.context())

        if not self.out_dir:
            raise StatRemoteError(
                "process_table_not_found", "STAT output directory is unavailable",
                "source", self.context())

        # resolve host/PID from STAT's recorded ranks.
        try:
            file_prefix = self.stat.getFilePrefix()
            proctab_path = os.path.join(self.out_dir, "%s.ptab" % file_prefix)
            proctab = get_proctab(proctab_path)
        except Exception as exc:
            raise StatRemoteError(
                "process_table_not_found", "Failed to read the STAT process table: %s" % exc,
                "source", self.context())

        result = attach_helper.read_source_from_ranks(
            self.remote_shell, proctab.process_list, source, params.get("ranks"))

        if not result.get("ok"):
            raise StatRemoteError(
                result["code"], result["message"], "source", result.get("context"))
        return result

    def context(self):
        """Return attachment identity and state for responses and diagnostics."""
        return {
            "session_id": self.session_id,
            "state": self.state,
            "backend": self.backend,
            "mode": self.mode,
            "remote_host": self.remote_host,
            "pid": self.pid,
            "job": self.job,
            "rm": self.rm,
            "remote_shell": self.remote_shell,
        }

    def status(self):
        """Add output locations and creation time to the session context."""
        data = self.context()
        data.update({
            "created_at": self.created_at,
            "last_dot_filename": self.last_dot_filename,
            "out_dir": self.out_dir,
        })
        return data


class StatSessionController(object):
    """Route remote requests and manage the server's application session.

    Only one live attachment is allowed because native STAT frontends share
    global state. Failed cleanup retains ownership so detach can be retried.
    """

    def __init__(self):
        self.session = None

    def health(self):
        # native frontends share graph tables; even a temporary one can invalidate
        # an attached session when destroyed. Health must not construct one.
        """Report server capabilities without constructing a native frontend."""
        return {
            "hostname": socket.gethostname(),
            "user": os.environ.get("USER"),
            "stat_version": STAT_VERSION,
            "git_commit": GIT_COMMIT,
            "git_dirty": GIT_DIRTY,
            "cuda_quick": STAT_SAMPLE_CUDA_QUICK is not None,
            "supported_backends": list(SUPPORTED_BACKENDS),
        }

    def jobs_list(self, params):
        """List jobs for interactive selection; listing currently supports Slurm only."""
        if _scheduler(params) != "slurm":
            raise StatRemoteError("unsupported_operation", "Job listing currently supports Slurm only", "list")
        user = pwd.getpwuid(os.geteuid()).pw_name
        try:
            jobs = attach_helper.list_jobs("Slurm", user)
        except RuntimeError as exc:
            raise StatRemoteError("job_list_failed", str(exc), "list") from exc
        return {"jobs": jobs}

    def jobs_resolve(self, params):
        """Resolve a scheduler job into launcher host, PIDs and command lines."""
        jobid = _str(params, "jobid")
        if not jobid:
            raise StatRemoteError("invalid_params", "jobid is required", "resolve")
        scheduler = _scheduler(params)
        if scheduler == "flux":
            raise StatRemoteError("unsupported_operation", "Use sessions.attach with a Flux job ID", "resolve")
        remote_host, launcher_exe, pids = attach_helper.jobid_to_hostname_pid(
            scheduler,
            jobid,
            "ssh",
        )

        if remote_host is None or not pids:
            raise StatRemoteError(
                "job_resolve_failed", "Failed to resolve job id %s" % jobid,
                "resolve", {"jobid": jobid})
        processes = attach_helper.get_remote_process_details(
            "ssh",
            remote_host,
            pids,
        )
        return {
            "remote_host": remote_host,
            "launcher_exe": launcher_exe,
            "pids": pids,
            "processes": processes,
        }

    def create_session(self, params, progress=None):
        # a frontend owns process-global native state. Keep failed cleanup owned
        # too, until detach succeeds; another session needs a separate server.
        """Create a frontend and reserve its attachment slot before attaching targets.

        Keep failed cleanup registered: another frontend must not overwrite its
        native state. Release the slot only when rollback or explicit detach
        completes successfully.
        """
        if self.session is not None:
            raise StatRemoteError(
                "session_busy", "Detach the existing session before attaching another application",
                "attach", {"session_id": self.session.session_id})

        gdb_path = ""
        if params.get("backend") == "gdb" or _bool(params, "gdb_backend"):
            gdb_path = _str(params, "gdb_path").strip()

        session = StatSession()
        self.session = session

        # native STAT reads STAT_GDB when building the daemon arguments for this attach.
        previous_gdb = os.environ.get("STAT_GDB")
        try:
            if gdb_path:
                os.environ["STAT_GDB"] = gdb_path
            return session.attach(params, progress)
        except Exception as attach_error:
            try:
                session.detach()
            except Exception as cleanup_error:
                raise StatRemoteError(
                    "attach_cleanup_failed",
                    "Attachment failed: %s. Cleanup failed: %s" % (attach_error, cleanup_error),
                    "attach", {"session_id": session.session_id,
                               "cleanup_confirmed": False}) from attach_error
            self.session = None
            raise
        finally:
            if gdb_path:
                if previous_gdb is None:
                    os.environ.pop("STAT_GDB", None)
                else:
                    os.environ["STAT_GDB"] = previous_gdb

    def _session(self, params, phase):
        """Find the requested session or report its missing identifier."""
        session_id = _str(params, "session_id")
        session = self.session
        if session is None or session.session_id != session_id:
            raise StatRemoteError(
                "unknown_session", "Unknown session %s" % session_id,
                phase, {"session_id": session_id})
        return session

    def dispatch(self, method, params, progress=None):
        """Map protocol operations to controller or session methods.

        Attach creates ownership; subsequent operations require session_id.
        Remove a session only after detach succeeds. Explicit shutdown uses
        the same cleanup path as connection closure.
        """
        params = params or {}
        if method == 'server.shutdown':
            self.close_session()
            return {'shutdown': True}
        if method == "health":
            return self.health()
        if method == "jobs.list":
            return self.jobs_list(params)
        if method == "jobs.resolve":
            return self.jobs_resolve(params)
        if method == "sessions.attach":
            return self.create_session(params, progress)
        if method == "sessions.pause":
            return self._session(params, "pause").pause()
        if method == "sessions.resume":
            return self._session(params, "resume").resume()
        if method == "sessions.sample":
            return self._session(params, "sample").sample(params, progress)
        if method == "sessions.sample_multiple":
            return self._session(params, "sample").sample_multiple(params, progress)
        if method == "sessions.detach":
            session = self._session(params, "detach")
            result = session.detach()
            self.session = None
            return result
        if method == "sessions.status":
            return self._session(params, "status").status()
        if method == "sources.get":
            return self._session(params, "source").source(params)
        raise StatRemoteError("unknown_method", "Unknown method %s" % method)

    def close_session(self):
        """Release the session, retaining ownership if detach needs a retry."""
        if self.session is None:
            return
        try:
            self.session.detach()
        except Exception as exc:
            failure = {"session_id": self.session.session_id, "error": str(exc)}
            raise StatRemoteError(
                "shutdown_cleanup_failed", "Failed to detach remote session; cleanup is unconfirmed",
                "shutdown", {"failures": [failure]}) from exc
        self.session = None
