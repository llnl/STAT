# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Discover application launchers and retrieve source for remote STAT sessions.

The session module delegates scheduler and process-filesystem queries here so
native attachment and sampling remain separate from command execution.

Job listing currently supports Slurm. Source reads use /proc/PID/root or cwd to
find the file seen by the application, including inside containers (needed for
for future remote GUI support).
"""

import json
import shlex
import shutil
import socket
import subprocess

import attach_helper
from remote.stat_ssh import remote_shell_argv


# cap raw file content at 4 MiB when copying source from the container (future remote GUI support).
REMOTE_SOURCE_MAX_BYTES = 4 * 1024 * 1024

# try at most four rank processes to copy source from the container (future remote GUI support).
REMOTE_SOURCE_MAX_ATTEMPTS = 4

# seconds per source-read command on the process host.
REMOTE_SOURCE_NODE_TIMEOUT = 5

auto_detect_rm = attach_helper.auto_detect_rm


def select_scheduler(scheduler="auto"):
    """Select a supported job-ID resolver, using launcher presence for auto mode.

    Auto mode requires exactly one supported launcher on PATH. Explicit selection
    is checked against STAT's resolver names, without probing scheduler access.
    """
    scheduler = str(scheduler).lower()
    if scheduler == "auto":
        available = [name for name, launcher in attach_helper.rms.items()
                     if shutil.which(launcher)]
        if len(available) != 1:
            raise ValueError("Specify scheduler explicitly; detected: %s" %
                             (", ".join(available) or "none"))
        return available[0]
    if scheduler == "flux":
        return scheduler
    if scheduler not in attach_helper.rms:
        raise ValueError("Unsupported job-ID scheduler %s; supported: %s" %
                         (scheduler, ", ".join(sorted([*attach_helper.rms, "flux"]))))
    return scheduler


def list_jobs(rm=None, user=None):
    """Dispatch job listing; currently only Slurm has an implementation.

    Return an empty list for an undetected or unsupported scheduler. This does
    not restrict native attachment through launchers supported by LaunchMON.

    TODO: Implement other schedulers.
    """
    if rm == None or rm.lower() == 'auto':
        rm = auto_detect_rm()
    if rm == None:
        return []

    rm = rm.lower()
    if rm == 'slurm':
        from remote.stat_server_slurm import list_slurm_jobs
        return list_slurm_jobs(user)

    return []


def jobid_to_hostname_pid(rm, jobid, remoteshell):
    """Resolve a job to its launcher host, executable and candidate PIDs.

    Slurm uses the discovery module; other schedulers reuse STAT's existing
    resolver. Return an empty result when no launcher is found.
    Forward errors from Slurm and launcher process queries.
    """
    pids = []
    if rm == None or rm.lower() == 'auto':
        rm = auto_detect_rm()
    if rm == None:
        return None, None, pids

    rm = rm.lower()
    if rm == 'slurm':
        from remote.stat_server_slurm import SlurmJobDiscovery

        def run_command(command, host=None):
            argv = (["bash", "-c", command] if host is None or is_local_host(host) else
                    remote_shell_argv(host, command, executable=remoteshell))
            return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, check=False, timeout=30)

        try:
            resolved = SlurmJobDiscovery(run_command).resolve(jobid)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Slurm launcher lookup timed out after %s seconds" % exc.timeout) from exc
        except OSError as exc:
            raise RuntimeError(str(exc)) from exc
        if not resolved['pids']:
            return None, None, []
        return resolved['remote_host'], resolved['launcher_exe'], resolved['pids']
    return attach_helper.jobid_to_hostname_pid(rm, jobid, remoteshell)


def get_remote_process_details(remoteshell, remotehost, pids):
    """Query only the supplied launcher PIDs, preserving their input order.

    Return PID, parent PID and full command line for display. Missing processes
    have empty details; a failed command returns an empty list.
    """
    if remotehost in [None, ''] or not pids:
        return []

    pid_list = ','.join(str(pid) for pid in pids)
    try:
        command = "ps ww -o pid=,ppid=,args= -p %s" % shlex.quote(pid_list)
        rc, stdout, _stderr = _remote_command(remoteshell, remotehost, command, timeout=30)
        if rc != 0:
            return []
    except (OSError, subprocess.TimeoutExpired):
        return []

    command_lines = {}
    parent_pids = {}

    for line in stdout.splitlines():
        fields = line.strip().split(None, 2)
        if not fields:
            continue
        try:
            pid = int(fields[0])
            parent_pid = int(fields[1])
        except ValueError:
            continue
        parent_pids[pid] = parent_pid
        command_lines[pid] = fields[2] if len(fields) > 2 else ''

    return [
        {
            'pid': int(pid),
            'ppid': parent_pids.get(int(pid)),
            'command_line': command_lines.get(int(pid), ''),
        }
        for pid in pids
    ]


# the source file may exist only inside the application's container or working directory.
# read_process_source() passes this script to _run_remote_python(), which invokes
# python3 -c locally or over SSH on the process host, with PID, path and byte limit.
# it reads through /proc/PID/root or cwd and prints one JSON result.
_REMOTE_SOURCE_FILE_SCRIPT = r'''
import base64
import json
import os
import sys

pid = int(sys.argv[1])
source = sys.argv[2]
max_bytes = int(sys.argv[3])
if "\x00" in source:
    print(json.dumps({"ok": False, "code": "invalid_source_path",
                      "message": "source path contains a NUL byte"}))
    raise SystemExit(0)

# resolve paths through the target process, which may use a container filesystem.
if os.path.isabs(source):
    resolved = "/proc/%d/root%s" % (pid, source)
else:
    resolved = "/proc/%d/cwd/%s" % (pid, source)

try:
    if not os.path.isfile(resolved):
        raise IOError("file does not exist")
    with open(resolved, "rb") as stream:
        # one extra byte distinguishes an oversized file from an exact-size file.
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        print(json.dumps({
            "ok": False,
            "code": "source_too_large",
            "message": "source file exceeds %d bytes" % max_bytes,
            "resolved_path": resolved,
        }, sort_keys=True))
    else:
        print(json.dumps({
            "ok": True,
            "content_b64": base64.b64encode(content).decode("ascii"),
            "resolved_path": resolved,
            "size": len(content),
        }, sort_keys=True))
except Exception as exc:
    print(json.dumps({
        "ok": False,
        "code": "source_not_found",
        "message": str(exc),
        "resolved_path": resolved,
    }, sort_keys=True))
'''


def _decode(data):
    """Normalize subprocess output while preserving undecodable diagnostics."""
    if isinstance(data, bytes):
        return data.decode(errors='replace')
    return data


def _run_command(argv, timeout=30):
    """Return (exit status, stdout, stderr), decoding output as text.

    Timeouts are in seconds; None disables the deadline. On timeout, kill and
    reap the local command and return status 124 with a diagnostic. For SSH,
    this does not guarantee termination of a remote process.
    """
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        stderr = _decode(stderr) + "\ncommand timed out after %s seconds" % timeout
        return 124, _decode(stdout), stderr
    return proc.returncode, _decode(stdout), _decode(stderr)


def is_local_host(host):
    """Check whether host is the machine running the STAT server."""
    hostname = socket.gethostname()
    return host in ("localhost", hostname, hostname.split(".", 1)[0], socket.getfqdn())


def _remote_command(remoteshell, remotehost, command, timeout=None):
    """Run on the server for local hostnames, otherwise through the remote shell."""
    if is_local_host(remotehost):
        return _run_command(["sh", "-c", command], timeout=timeout)
    return _run_command(
        remote_shell_argv(remotehost, command, executable=remoteshell), timeout=timeout)


def _quote(value):
    """Quote one argument for execution by the remote shell."""
    return shlex.quote(str(value))


def _run_remote_python(remoteshell, remotehost, script, args, timeout=None):
    """Execute a Python reader on the process host and decode its JSON result.

    Quote the script and each argument because SSH passes a shell command.
    Command failures raise an error; reader-level failures remain result data."""
    command = "python3 -c %s %s" % (
        _quote(script),
        " ".join(_quote(arg) for arg in args),
    )
    rc, stdout, stderr = _remote_command(remoteshell, remotehost, command, timeout=timeout)
    if rc != 0:
        raise RuntimeError("remote command failed on %s: %s" % (remotehost, stderr.strip()))
    return json.loads(stdout)


def read_process_source(remoteshell, remotehost, pid, source, max_bytes,
                        timeout=None):
    """Read one file through one target process on its host.

    max_bytes bounds raw content; timeout is in seconds. Success returns
    base64 content, size and resolved path. File errors return ok=False with
    a code and message; command or malformed-response failures raise.
    """
    return _run_remote_python(
        remoteshell, remotehost, _REMOTE_SOURCE_FILE_SCRIPT,
        [pid, source, max_bytes], timeout=timeout)


def read_source_from_ranks(remote_shell, process_list, source, preferred_ranks=None):
    """Read source from bounded candidates in a session's recorded process table.

    process_list contains (rank, host, PID, executable index) entries from STAT.
    Prefer GUI-selected ranks, then try other ranks in order. Processes may have
    exited or may see different filesystems, so retain failed-attempt details.
    Read through each process namespace with a 4 MiB limit, at most four attempts
    and a five-second timeout per attempt.
    """
    # prefer ranks selected in the GUI; fall back if a process has exited or lacks the file.
    selected_ranks = []
    for rank in preferred_ranks or []:
        try:
            selected_ranks.append(int(rank))
        except Exception:
            pass
    rank_order = dict(
        (rank, index) for index, rank in enumerate(selected_ranks))
    processes = list(process_list)
    processes.sort(key=lambda item: (
        item[0] not in rank_order,
        rank_order.get(item[0], item[0]),
    ))

    # bound remote work and retain per-attempt context if no candidate succeeds.
    attempts = []
    for rank, host, pid, _exe_index in processes[:REMOTE_SOURCE_MAX_ATTEMPTS]:
        try:
            result = read_process_source(
                remote_shell, host, pid, source,
                REMOTE_SOURCE_MAX_BYTES,
                timeout=REMOTE_SOURCE_NODE_TIMEOUT)
        except Exception as exc:
            attempts.append({"rank": rank, "host": host, "pid": pid,
                             "error": str(exc)})
            continue

        if result.get("ok"):
            result.update({
                "source": source,
                "rank": rank,
                "host": host,
                "pid": pid,
            })
            return result

        # an oversized file is a terminal error, not a reason to try more ranks.
        if result.get("code") == "source_too_large":
            result["context"] = {"source": source, "rank": rank,
                                 "host": host, "pid": pid}
            return result
        attempts.append({"rank": rank, "host": host, "pid": pid,
                         "error": result.get("message", "not found")})

    return {"ok": False, "code": "source_not_found",
            "message": "Failed to read %s from the sampled process filesystem" % source,
            "context": {"source": source, "attempts": attempts}}
