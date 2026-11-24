#!/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Python client for the STAT remote-connect interface over SSH.

This module is imported by stat_client_cli.py and the standalone GUI client.
It starts the server, exchanges newline-delimited JSON requests and progress
updates, and closes the connection.

SSH uses the user's SSH configuration and authenticates noninteractively.
The caller chooses the server host. For Flux, flux_allocation specifies
the allocation ID to proxy into before starting the server. The server
handles job discovery and attachment.

Requests time out after DEFAULT_REQUEST_TIMEOUT_SECONDS by default. On close,
the client allows up to SSH_OPERATION_TIMEOUT_SECONDS for server cleanup
before terminating local SSH. If a request times out or the connection drops,
verify that the application was released before attempting another attach.

Typical connection lifecycle::

    client = StatRemoteClient.ssh("user@cluster")
    try:
        client.ensure_remote_server()
        status = client.remote_health
        # attach with sessions.attach, then use the returned session_id.
        # use sessions.sample_multiple for snapshots plus a merged 3D graph.
        # finish with sessions.detach and wait for its acknowledgement.
    finally:
        client.close()
"""

from __future__ import print_function

import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time

from remote.stat_ssh import remote_shell_argv


# review and adjust these timeouts based on testing at scale (TODO).

# default deadline for a complete server request, including progress messages.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600

# limit for one-off SSH commands and for SSH to exit after a server shutdown request.
SSH_OPERATION_TIMEOUT_SECONDS = 180

# short wait to collect final stderr or reap the local SSH process after signaling it.
SSH_CLEANUP_WAIT_SECONDS = 2

# interval for checking the request deadline and serving the GUI idle callback.
RESPONSE_POLL_INTERVAL_SECONDS = 0.05


class StatRemoteClientError(Exception):
    """A structured error returned by the server, retaining its diagnostic fields."""

    def __init__(self, error):
        self.error = error
        Exception.__init__(self, error.get("message", "Remote STAT error"))


class StatRemoteArchitectureMismatch(RuntimeError):
    """The selected SSH host cannot load the configured STAT installation."""

    def __init__(self, host, host_arch, stat_arch):
        self.host = host
        self.host_arch = host_arch
        self.stat_arch = stat_arch
        RuntimeError.__init__(
            self, "STAT installation is %s, but server host %s is %s" %
            (stat_arch, host, host_arch))


class StatRemoteClient(object):
    """Own one server transport and allow one request at a time on that connection."""

    def __init__(self, argv, remote_login=None, ssh_options=None, remote_prefix=None,
                 remote_server_host=None):
        self.argv = argv
        self.proc = None
        self.next_id = 1
        self.request_timeout = DEFAULT_REQUEST_TIMEOUT_SECONDS
        # serialize protocol requests separately from transport start/close operations.
        self.lock = threading.Lock()
        self.lifecycle_lock = threading.RLock()
        self.closed = False
        self.remote_login = remote_login
        self.ssh_options = ssh_options or []
        self.remote_prefix = remote_prefix
        self.remote_server_host = remote_server_host
        self.remote_command = argv[-1] if argv else None
        self.remote_ready = False
        self.architecture_verified_host = None
        self.remote_health = None
        self.stderr_lines = []
        self.stderr_reader = None

    @classmethod
    def ssh(cls, remote_login, remote_command="stat-gui-server", ssh_options=None,
            remote_prefix=None, remote_server_host=None, flux_allocation=None):
        """Configure SSH; the server starts lazily on the first request."""
        client = cls([], remote_login, ssh_options, remote_prefix,
                     remote_server_host)
        if flux_allocation is not None:
            # the server wrapper loads its runtime environment before calling flux proxy.
            remote_command = "%s --flux-allocation %s" % (remote_command, shlex.quote(flux_allocation))
        client.remote_command = remote_command
        client._rebuild_ssh_argv()
        return client

    @staticmethod
    def _server_command(command, remote_server_host=None):
        """Wrap a command in a second SSH hop when the server runs elsewhere."""
        if remote_server_host is None:
            return command
        argv = remote_shell_argv(
            remote_server_host, command,
            options=["-o", "GSSAPIDelegateCredentials=yes"])
        return "exec " + " ".join(shlex.quote(arg) for arg in argv)

    def _rebuild_ssh_argv(self):
        """Build the persistent server command using the selected SSH route."""
        self.argv = remote_shell_argv(
            self.remote_login,
            self._server_command(self.remote_command, self.remote_server_host),
            options=self.ssh_options)

    def _run_ssh(self, command, server_host=None, timeout=SSH_OPERATION_TIMEOUT_SECONDS):
        """Run a bounded setup probe independently of the persistent JSON connection."""
        argv = remote_shell_argv(
            self.remote_login, self._server_command(command, server_host),
            options=self.ssh_options)
        return subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, check=False, timeout=timeout)

    @staticmethod
    def command_for_prefix(remote_prefix):
        """Return the default remote server command for a STAT prefix."""
        return "%s/bin/stat-gui-server" % shlex.quote(remote_prefix)

    def check_remote_prefix_architecture(self, remote_prefix, server_host=None):
        """Verify that the selected STAT installation matches the server architecture.

        Inspect the remote shell and STATD ELF headers through the same SSH route
        used to start the server. Cache a successful host check to avoid repeating
        it before each request. Missing tools/files or incompatible architectures
        raise an error before any application is attached.
        """
        if server_host is None:
            server_host = self.remote_server_host
        host = server_host or self.remote_login
        statd = "%s/bin/STATD" % remote_prefix.rstrip("/")
        command = "uname -m; readelf -h /proc/self/exe %s" % shlex.quote(statd)
        result = self._run_ssh(command, server_host)
        lines = result.stdout.splitlines()
        machines = re.findall(r"^\s*Machine:\s*(.+)$", result.stdout, re.MULTILINE)

        if result.returncode != 0 or not lines or len(machines) != 2:
            message = "Could not inspect %s on %s." % (statd, host)
            detail = result.stderr.strip()
            if detail:
                message += " " + detail

            # additional message for a few common error patterns seen
            if "pam_slurm_adopt" in detail and "no active jobs" in detail.lower():
                message += " If the job just started, retry after the prolog finishes."
            elif "permission denied (publickey" in detail.lower():
                if server_host:
                    message += (" If the job just started, retry after the Slurm prolog finishes."
                                " Also, check passwordless access to nodes.")
                else:
                    message += " Check passwordless SSH access to the login host."
            else:
                message += " Check --remote-login, --remote-server-host, and --remote-prefix."

            raise RuntimeError(message)

        host_arch = lines[0]
        host_machine, statd_arch = machines
        if host_machine != statd_arch:
            raise StatRemoteArchitectureMismatch(host, host_arch, statd_arch)

        self.architecture_verified_host = host
        return {"host": host, "host_arch": host_arch,
                "stat_arch": statd_arch}

    def ensure_remote_server(self):
        """Check the configured remote server once before using it."""
        if self.remote_ready:
            return
        server_host = self.remote_server_host or self.remote_login
        if (self.remote_prefix is not None and
                self.architecture_verified_host != server_host):
            self.check_remote_prefix_architecture(self.remote_prefix)
        self.remote_health = self.request("health")
        self.remote_ready = True

    def start(self):
        """Start the owned process and drain stderr separately from JSON stdout."""
        with self.lifecycle_lock:
            if self.closed:
                raise RuntimeError("Remote STAT client has been shut down")
            if self.proc is not None:
                return
            self.stderr_lines = []
            self.proc = subprocess.Popen(
                self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, universal_newlines=True, bufsize=1)
            self.stderr_reader = threading.Thread(
                target=self._read_stderr, args=(self.proc.stderr,))
            self.stderr_reader.daemon = True
            self.stderr_reader.start()

    def _read_stderr(self, stream):
        """Forward server diagnostics and retain a small startup error tail."""
        try:
            for line in stream:
                sys.stderr.write(line)
                sys.stderr.flush()
                self.stderr_lines.append(line)
                if len(self.stderr_lines) > 20:
                    self.stderr_lines.pop(0)
        except (ValueError, OSError):
            # close() may release the pipe while a launched job still holds it.
            if not stream.closed:
                raise

    def close(self, permanent=False):
        """Request server cleanup, then close and reap the local transport.

        Send shutdown and close stdin so the server can detach and exit. Wait up
        to 180 seconds before terminating, then killing, the local transport if
        necessary. This bounds local cleanup; it does not prove remote detach.

        Broken pipes are expected after connection loss, so still attempt to reap
        the process and release its streams. permanent=True also prevents restart.
        """
        with self.lifecycle_lock:
            if permanent:
                self.closed = True
            proc = self.proc
            if proc is None:
                return
            self.remote_ready = False

            # request graceful server cleanup before closing its protocol input.
            try:
                request = {'id': self.next_id, 'method': 'server.shutdown', 'params': {}}
                self.next_id += 1
                proc.stdin.write(json.dumps(request, sort_keys=True) + '\n')
                proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.stdin.close()
            except Exception:
                pass

            # escalation affects our local transport, not arbitrary remote processes.
            try:
                proc.wait(timeout=SSH_OPERATION_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                sys.stderr.write("STAT: Remote cleanup timed out; remote daemon cleanup "
                                 "is unconfirmed. Closing SSH.\n")
                proc.terminate()
                try:
                    proc.wait(timeout=SSH_CLEANUP_WAIT_SECONDS)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=SSH_CLEANUP_WAIT_SECONDS)

            # clear ownership only after the process has actually been reaped.
            self.proc = None
            if self.stderr_reader is not None:
                self.stderr_reader.join(SSH_CLEANUP_WAIT_SECONDS)
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()

    def request(self, method, params=None, on_progress=None, on_idle=None):
        """Send one operation and wait for its result, forwarding progress callbacks.

        Parameters pass unchanged to the server, including retry_freq in microseconds.
        Repeated sampling belongs to sessions.sample_multiple, not a transport loop.
        on_idle lets an interactive caller service its UI while waiting for a reply.
        """
        if not self.lock.acquire(False):
            raise RuntimeError("remote STAT request already in progress")
        try:
            return self._request(method, params, on_progress, on_idle)
        except (ConnectionError, EOFError) as exc:
            self.close()
            raise RuntimeError(
                "%s\nRemote STAT connection lost; process cleanup is unconfirmed." % exc
            ) from exc
        finally:
            self.lock.release()

    def request_in_progress(self):
        """Report whether this connection already has an outstanding request."""
        return self.lock.locked()

    def _request(self, method, params, on_progress, on_idle):
        """Write one request and consume its progress messages and final reply.

        Start the transport and write under the lifecycle lock, then release it
        while waiting so close() can interrupt the connection. request() holds
        the separate request lock to prevent competing readers on stdout.

        Read each line in a worker thread so the caller can service on_idle and
        enforce one deadline across all progress messages. Ignore non-JSON lines
        and other request IDs; return the matching result or raise its error.
        """
        with self.lifecycle_lock:
            self.start()
            proc = self.proc
            if proc is None or self.closed:
                raise RuntimeError("Remote STAT connection closed")
            request_id = self.next_id
            self.next_id += 1
            request = {"id": request_id, "method": method, "params": params or {}}
            self._debug_request(request)
            proc.stdin.write(json.dumps(request, sort_keys=True) + "\n")
            proc.stdin.flush()

        deadline = time.monotonic() + self.request_timeout

        # one request may emit several progress messages before its final response.
        # this loop reads that response stream; it does not initiate more samples.
        while True:
            lines = []
            errors = []

            def read_line():
                try:
                    lines.append(proc.stdout.readline())
                except Exception as exc:
                    errors.append(exc)

            reader = threading.Thread(target=read_line)
            reader.daemon = True
            reader.start()
            while reader.is_alive():
                reader.join(RESPONSE_POLL_INTERVAL_SECONDS)
                if time.monotonic() >= deadline:
                    self.close()
                    raise RuntimeError("Timed out waiting for remote STAT %s; "
                                       "remote cleanup is unconfirmed" % method)
                if on_idle is not None:
                    on_idle()

            if errors:
                raise errors[0]

            line = lines[0]
            if line == "":
                if self.stderr_reader is not None:
                    self.stderr_reader.join(SSH_CLEANUP_WAIT_SECONDS)
                message = "Remote STAT server exited before replying to %s" % method
                details = "".join(self.stderr_lines).strip()
                if details:
                    message += ":\n%s" % details
                raise EOFError(message)

            # only messages for this request can complete the operation.
            try:
                response = json.loads(line)
            except ValueError:
                continue
            if response.get("id") != request_id:
                continue

            if response.get("event") == "progress":
                if on_progress is not None:
                    on_progress(response.get("progress", {}))
                continue

            if not response.get("ok"):
                error = response.get("error", {})
                if method == "sessions.attach" and error.get("code") == "stat_attach_failed":
                    message = error["message"] + "\nIf the workload just started, retry after it is ready."
                    log_dir = error.get("context", {}).get("log_dir")

                    if log_dir:
                        message += " If it persists, report stderr and logs from %s." % log_dir
                    else:
                        message += (" If it persists, rerun with --remote-log-dir <absolute-shared-path>"
                                    " and report stderr and logs.")
                    error["message"] = message

                raise StatRemoteClientError(error)
            return response.get("result")

    @staticmethod
    def _debug_request(request):
        """Write the complete JSON request to stderr when explicitly enabled."""
        enabled = os.environ.get("STAT_REMOTE_GUI_DEBUG_REQUESTS", "")
        if enabled.lower() not in ("1", "true", "yes", "on"):
            return
        sys.stderr.write("STAT remote request: %s\n" % json.dumps(request, sort_keys=True))
        sys.stderr.flush()
