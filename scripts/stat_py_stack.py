#!/bin/env python
"""Collect Python and optional native stacks for STAT through the PyStack CLI.

This adapter selects sampling targets, runs collectors and adds process context.
stat_pystack_process_tree.py supplies procfs information and ancestry; stat_pystack_parser.py
converts collector output into frame records. Public entry points return traces
and metadata to STAT_BackEnd, which controls attachment and merges the graphs.

The adapter samples individual processes or process trees and returns their stacks
with process, optional thread, and rank context. Targets remain stopped by default;
STAT_BackEnd manages pause/resume and detachment. Failures are reported while
successful stacks are retained.

TODO: Current tested workloads complete in reasonable time. In case of a larger
process tree, if collection time is a bottleneck, reducing repeated collector startup
and /proc reads might benefit.
"""

import os
import math
import re
import errno
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import stat_pystack_process_tree as process_tree
import stat_pystack_parser as stack_parser

# Regex for determining whether an executable basename is a conventional Python interpreter name.
# Used for wrapper discovery and labels; matches python, pythonX.YY, et.
_PYTHON_NAME_RE = re.compile(r"python(?:[0-9]+(?:\.[0-9]+)*)?[dmtu]*")

# Skip placeholder thread names when choosing a descriptive process label.
_GENERIC_THREAD_NAMES = {"", "unknown", "thread"}

# Skip common entry-point and bootstrap functions when choosing a process label from its stack.
# These functions remain in the sampled stack; only label selection ignores them.
_GENERIC_FUNCTION_NAMES = {
    "",
    "<module>",
    "<unknown>",
    "bootstrap",
    "bootstrap_inner",
    "main",
    "module",
    "run",
    "spawn_main",
    "unknown",
}


# --- Environment settings ---

def _enabled(name, default):
    """Read an environment toggle; 0, false, no and off disable it, ignoring case and whitespace."""
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


# --- Optional target resume and state restoration ---

def _resume_stopped_target(pid):
    """Continue the selected PID for an explicitly requested resume-around-sampling call."""
    try:
        os.kill(int(pid), signal.SIGCONT)
    except OSError as exc:
        raise RuntimeError("Failed to resume PyStack target pid %s: %s" % (pid, exc))


def _stop_resumed_target(pid):
    """Stop a temporarily resumed target again; tolerate a target that has already exited."""
    try:
        os.kill(int(pid), signal.SIGSTOP)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return
        raise RuntimeError("Failed to stop PyStack target pid %s after sampling: %s" % (pid, exc))


def _run_with_optional_resume(pid, resume_stopped, callback):
    """Run collection, optionally continuing this PID and stopping it again afterward.

    The default sends no signals. On collector failure, attempt to restore the
    stopped state without replacing the original exception with a cleanup error.

    For resume_stopped=True, the sequence is SIGCONT -> callback -> SIGSTOP;
    otherwise only the callback runs.
    """
    if resume_stopped:
        _resume_stopped_target(pid)
    try:
        result = callback()
    except Exception:
        if resume_stopped:
            try:
                _stop_resumed_target(pid)
            except RuntimeError:
                pass
        raise
    if resume_stopped:
        _stop_resumed_target(pid)
    return result


# --- Python target selection and launcher worker ranks ---

def _is_python_process(pid):
    """Recognize conventional interpreter names through /proc/PID/exe, not the launcher command.

    This wrapper-discovery heuristic cannot identify every renamed or embedded Python executable.
    PyStack determines whether the selected process contains a supported Python runtime.

    For example, /venv/bin/python3.12 matches, but /app/custom-runtime does not, even if the
    latter embeds Python.
    """
    executable = os.path.basename(process_tree.read_exe(pid))
    return bool(executable) and _is_generic_process_name(executable)


def _find_python_app_pid_under_wrapper(root_pid, descendants=None):
    """Return the Python application PID for an existing rank beneath nested launch wrappers.

    The launcher may report a shell or wrapper PID rather than the Python PID
    to sample. For example, rank 2 running bash -> numactl -> python should resolve
    to that Python process, not a different rank or a deeper child that inherited
    the same environment.

    Read the expected rank from the root's SLURM_PROCID, falling back to RANK.
    Direct mpi4py jobs also work without Slurm or torchrun: after MPI initializes,
    LaunchMON reads the MPI launcher's process table and gives STAT each rank's
    Python PID. Those jobs do not need this wrapper lookup or its rank environment
    variables.

    TODOs: Need to look into practical use cases where the closest Python process
      can itself be a launcher, and inherited rank variables do not prove its role.
      Current implement suffice with the use cases we have seen so far.
    """

    root_pid = int(root_pid)
    # Use the wrapper's existing rank identity; this lookup does not assign new ranks.
    root_values = process_tree.read_environ(root_pid)
    try:
        expected_rank = int(root_values.get(b"SLURM_PROCID") or
                            root_values.get(b"RANK"))
    except (TypeError, ValueError):
        return None

    expected_job = root_values.get(b"SLURM_JOB_ID") or root_values.get(b"SLURM_JOBID")
    candidates = []

    for pid in descendants if descendants is not None else process_tree.descendants(root_pid):
        if not _is_python_process(pid):
            continue
        values = process_tree.read_environ(pid)

        try:
            rank = int(values[b"RANK"])
            slurm_rank = int(values.get(b"SLURM_PROCID", rank))
        except (KeyError, TypeError, ValueError):
            continue

        job = values.get(b"SLURM_JOB_ID") or values.get(b"SLURM_JOBID")

        # Reject candidates whose rank or available job metadata disagrees with the wrapper.
        if rank != expected_rank or slurm_rank != expected_rank:
            continue
        if expected_job is not None and job != expected_job:
            continue

        # Recheck ancestry because a supplied descendant list may be stale.
        path = process_tree.process_path(pid, root_pid)
        if path and path[0] == root_pid:
            candidates.append((len(path), pid))

    if not candidates:
        return None

    # Prefer top level application over deeper children that inherited the same rank metadata.
    shallowest = min(depth for depth, _pid in candidates)
    matches = [pid for depth, pid in candidates if depth == shallowest]

    # Equally close matches are ambiguous; do not choose one by PID or discovery order.
    return matches[0] if len(matches) == 1 else None


def _torchrun_pid(root_pid, snapshot=None):
    """Find torchrun in the rank tree, including when it is the root."""
    root_pid = int(root_pid)

    entry = _command_entrypoint(process_tree.read_argv(root_pid))
    if _is_torchrun_entrypoint(entry):
        return root_pid

    for pid in process_tree.descendants_for_snapshot(root_pid, snapshot):
        entry = _command_entrypoint(process_tree.read_argv(pid))
        if _is_torchrun_entrypoint(entry):
            return pid
    return None


def _torchrun_worker_targets(root_pid, snapshot=None):
    """Return live direct children of torchrun, falling back to the launcher if none remain.

    For torchrun PID 10 with direct workers 11 and 12, return [11, 12];
    grandchildren are handled separately when child sampling is requested.
    """
    torchrun_pid = _torchrun_pid(root_pid, snapshot)

    if torchrun_pid is None:
        return None

    if snapshot is None:
        return process_tree.direct_children(torchrun_pid) or [torchrun_pid]

    return [pid for pid in snapshot.get(torchrun_pid, ())
            if os.path.exists("/proc/%s" % pid)
            and not process_tree.is_exited_process(pid)
            and not process_tree.is_container_filesystem_helper(pid)] or [torchrun_pid]


def _sampling_targets(root_pid, snapshot=None):
    """Choose targets beneath torchrun or a rank wrapper, otherwise the root and descendants.

    Wrapper discovery can omit non-Python plumbing from collection; observed
    ancestry is represented separately by _process_labels().

    For a Python root 10 with child 11 and grandchild 12, select [10, 11, 12].
    For torchrun root 10 with the same subtree, select [11, 12].
    """
    root_pid = int(root_pid)
    launcher = _torchrun_pid(root_pid, snapshot)
    if launcher is not None:
        return process_tree.descendants_for_snapshot(launcher, snapshot) or [launcher]
    descendants = process_tree.descendants_for_snapshot(root_pid, snapshot)
    if (not _is_python_process(root_pid)
            and _find_python_app_pid_under_wrapper(root_pid, descendants) is not None):
        return [pid for pid in descendants if _is_python_process(pid)]
    return [root_pid] + descendants


def _torchrun_rank(pid):
    """Return (global_rank, world_size, launcher_task_count), or None for invalid metadata.

    For RANK=3, WORLD_SIZE=4 and SLURM_NTASKS=2, return (3, 4, 2):
    worker rank 3 belongs to four workers launched through two Slurm tasks.

    Without MPI/Slurm size metadata, launcher_task_count is None; the caller must
    establish a complete worker group before using those ranks.
    """
    values = process_tree.read_environ(pid)
    try:
        rank = int(values[b"RANK"])
        world_size = int(values[b"WORLD_SIZE"])
        launcher_size = (values.get(b"OMPI_COMM_WORLD_SIZE") or values.get(b"SLURM_NTASKS")
                         or values.get(b"SLURM_NPROCS") or values.get(b"PMI_SIZE"))
        launcher_tasks = int(launcher_size) if launcher_size is not None else None
    except (KeyError, TypeError, ValueError):
        return None

    if rank < 0 or world_size <= rank or (launcher_tasks is not None and launcher_tasks < 1):
        return None

    return rank, world_size, launcher_tasks


# --- Synthetic process labels and ancestry context ---

def _sanitize_label(value):
    """Normalize names used in synthetic process labels and generic-name checks.

    For example, "Worker Pool" becomes "worker_pool" in process:thread:worker_pool.
    This removes spaces and frame separators; actual sampled frames are unchanged.
    """
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    label = label.strip("._-").lower()
    return label or "unknown"


def _is_generic_process_name(value, generic_names=()):
    """Return whether a name is too generic to use as a descriptive process label.

    For example, python3.12 is generic, train is not, and unknown is generic
    when the caller supplies _GENERIC_THREAD_NAMES.
    """
    name = _sanitize_label(os.path.basename(value))
    return name in generic_names or _PYTHON_NAME_RE.fullmatch(name) is not None


def _command_entrypoint(argv):
    """Identify the executable, Python script or module before application arguments.

    Return (kind, name), or None for inline/stdin code or unknown/incomplete options.
    For example, python -W ignore -m package.app returns ("module", "package.app").
    """
    if not argv or not argv[0]:
        return None

    executable = os.path.basename(argv[0])
    if _PYTHON_NAME_RE.fullmatch(executable) is None:
        return "executable", argv[0]

    flags_without_values = "bBdEiIOPqRsSuvx"
    remaining_args = iter(argv[1:])
    for argument in remaining_args:
        # After --, the next argument is a script even if its name starts with a dash.
        if argument == "--":
            script = next(remaining_args, None)
            if not script or script == "-":
                return None
            return "script", script

        if not argument or argument == "-":
            return None
        if not argument.startswith("-"):
            return "script", argument

        if argument == "--check-hash-based-pycs":
            mode = next(remaining_args, None)
            if mode not in ("default", "always", "never"):
                return None
            continue

        # Walk grouped flags such as -uB; options with values consume the rest of the group.
        for position, flag in enumerate(argument[1:], 1):
            if flag in "cmWX":
                # Both -Wignore and -W ignore supply a value; an empty separate value is valid.
                value = argument[position + 1:]
                if value == "":
                    value = next(remaining_args, None)
                if value is None or flag == "c":
                    return None
                if flag == "m":
                    if not value:
                        return None
                    return "module", value
                break

            if flag not in flags_without_values:
                return None
    return None


def _is_torchrun_entrypoint(entry):
    """Recognize launcher modules or the torchrun filename before label normalization."""
    if entry is None:
        return False

    kind, name = entry
    if kind == "module":
        return name in ("torch.distributed.run", "torch.distributed.launch")
    return os.path.basename(name) == "torchrun"


def _cmdline_label(pid):
    """Label the command entry point: python train.py becomes train; -c has no label."""
    entry = _command_entrypoint(process_tree.read_argv(pid))
    if entry is None:
        return None
    if _is_torchrun_entrypoint(entry):
        return "torchrun"

    kind, name = entry
    if kind == "module":
        label = name.rsplit(".", 1)[-1]
    else:
        label = os.path.splitext(os.path.basename(name))[0]

    if _is_generic_process_name(label):
        return None
    return _sanitize_label(label)


def _frame_parts(frame):
    """Extract function and source from a formatted Python frame; ignore native or unformatted frames.

    For example, "train@/app/train.py:12" gives ("train", "/app/train.py");
    a "native:poll@..." frame gives (None, None).
    """
    if frame.startswith("native:") or "@" not in frame:
        return None, None
    function, location = frame.rsplit("@", 1)
    source = location.rsplit(":", 1)[0] if ":" in location else location
    return function, source


def _stdlib_source(source):
    """Recognize conventional Python library paths while excluding site-packages and dist-packages.

    For example, /usr/lib/python3.12/threading.py matches, while
    /usr/lib/python3.12/site-packages/torch/__init__.py does not.
    """
    normalized = source.replace("\\", "/")
    return (
        "/lib/python" in normalized
        and "site-packages" not in normalized
        and "dist-packages" not in normalized
    )


def _stack_function_label(traces):
    """Choose the first non-generic application function, searching each stack from its deepest frame.

    For an application stack main -> train -> compute, choose compute;
    trailing standard-library frames do not replace that application label.
    """
    for trace in traces:
        for frame in reversed(trace.get("frames", [])):
            function, source = _frame_parts(frame)
            if function is None:
                continue
            label = _sanitize_label(function)
            if label in _GENERIC_FUNCTION_NAMES:
                continue
            if source and not _stdlib_source(source):
                return label
    return None


def _process_role(pid, root_pid, traces):
    """Choose a process label from its command, kernel name, application function or thread name.

    Try those sources in that order, using the first descriptive value found.

    Prefixes such as cmd:train and function:prepare_batch describe the evidence
    used; they do not infer application roles such as data loader or writer.
    """
    # TODO: Simplify process labels and test whether command-based launcher/worker
    # labels (e.g. torchrun and its workers) suffice. Compare readability and rank
    # merging to see whether the extra naming heuristics add meaningful value.
    cmdline_label = _cmdline_label(pid)
    if cmdline_label:
        return "cmd:%s" % cmdline_label

    comm = process_tree.read_comm(pid)
    if comm and not _is_generic_process_name(comm):
        return "comm:%s" % _sanitize_label(comm)

    function_label = _stack_function_label(traces)
    if function_label:
        return "function:%s" % function_label

    for trace in traces:
        name = trace.get("name") or ""
        if name and not _is_generic_process_name(name, _GENERIC_THREAD_NAMES):
            return "thread:%s" % _sanitize_label(name)

    if pid == root_pid:
        return "main"

    text = " ".join([comm, process_tree.read_cmdline(pid)]).lower()
    if "python" in text or "multiprocessing" in text or "spawn" in text:
        return "python-child"
    return "child"


def _synthetic_frame(function, source="stat-process-tree", line=0):
    """Encode a context label using the same function@source:line format as sampled frames.

    For example, "process:cmd:train" becomes "process:cmd:train@stat-process-tree:0".
    """
    return "%s@%s:%s" % (function, source, line)


def _process_labels(pid, root_pid, traces):
    """Build process labels along the ancestry from the rank root to the sampled PID.

    Only the sampled PID has stack evidence; its ancestors use their own process
    details. This keeps the child's function names out of its parent's label.

    For a bash parent running train.py in a child, labels can be
    ["process:cmd:bash", "process:cmd:train"], followed by the child's actual frames.
    """
    return ["process:%s" % _process_role(parent, root_pid, traces if parent == int(pid) else [])
            for parent in process_tree.process_path(pid, root_pid)]


# --- PyStack command execution and trace collection ---

def _format_pystack_error(pid, detail):
    """Explain unrecognized Python targets with PID, executable and command-line details.

    Suggest Dyninst for native binaries; preserve other collector error messages."""
    if "Could not determine python version" in detail:
        return (
            "PyStack did not recognize pid %s as a Python process (exe: %s; args: %s); use "
            "--backend dyninst for native binaries."
            % (pid, process_tree.read_exe(pid), process_tree.read_cmdline(pid) or "<unavailable>")
        )
    return detail or "pystack remote failed for pid %s" % pid


def _pystack_timeout_seconds():
    """Read the per-command timeout from STAT_PYSTACK_TIMEOUT_SECONDS.

    Use 30 seconds for missing, invalid or non-finite values, and at least one
    second for valid values. This prevents a collector from waiting indefinitely."""
    try:
        timeout = float(os.environ.get("STAT_PYSTACK_TIMEOUT_SECONDS", "30"))
    except ValueError:
        return 30.0
    return max(1.0, timeout) if math.isfinite(timeout) else 30.0


def _log_pystack_sample(pid, mode, started, outcome):
    """Report individual collector attempts only when timing diagnostics are enabled."""
    if _enabled("STAT_PYSTACK_TIMING", "0"):
        print("STAT PyStack sample: pid=%s mode=%s elapsed=%.3fs outcome=%s"
              % (pid, mode, time.monotonic() - started, outcome), file=sys.stderr, flush=True)


def _build_pystack_command(pid, include_native_threads=False, native_mode=None, exhaustive=None):
    """Build the PyStack argument list from explicit frame/thread choices and environment settings.

    Python + Native selects --native, or --native-all when threads are included;
    Python-only mode adds neither. Return an argv list rather than a shell command.
    """
    pystack = os.environ.get("STAT_PYSTACK", "pystack")
    cmd = [pystack, "--no-color", "remote"]

    if _enabled("STAT_PYSTACK_NO_BLOCK", "0"):
        cmd.append("--no-block")

    native_enabled = native_mode == "Python + Native"

    if native_enabled:
        # STAT's -w selects all threads; native mode controls only frame detail.
        if include_native_threads:
            cmd.append("--native-all")
        else:
            cmd.append("--native")

        if _enabled("STAT_PYSTACK_NATIVE_LAST", "0"):
            cmd.append("--native-last")

    if _enabled("STAT_PYSTACK_LOCALS", "0"):
        cmd.append("--locals")

    if exhaustive is None:
        exhaustive = _enabled("STAT_PYSTACK_EXHAUSTIVE", "0")

    if exhaustive:
        cmd.append("--exhaustive")

    cmd.append(str(pid))
    return cmd


def _run_pystack_remote(pid, include_native_threads=False, native_mode=None, exhaustive=None):
    """Run one PyStack command with a timeout; return stdout or raise a contextual collection error."""

    if exhaustive is None:
        exhaustive = _enabled("STAT_PYSTACK_EXHAUSTIVE", "0")

    mode = "exhaustive" if exhaustive else "normal"
    timeout = _pystack_timeout_seconds()
    started = time.monotonic()

    try:
        proc = subprocess.run(
            _build_pystack_command(pid, include_native_threads, native_mode, exhaustive),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            check=False,
            start_new_session=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        _log_pystack_sample(pid, mode, started, "timeout")
        raise RuntimeError(
            "PyStack sampling timed out after %.1fs for pid %s" % (timeout, pid)
        ) from None

    _log_pystack_sample(pid, mode, started, "ok" if proc.returncode == 0 else "exit-%s" % proc.returncode)

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(_format_pystack_error(pid, detail))

    return proc.stdout


def _sample_pystack_traces(pid, include_threads, native_mode, exhaustive=None):
    """Collect and parse one bounded attempt without changing the target state."""
    output = _run_pystack_remote(pid, include_threads, native_mode, exhaustive)
    traces = stack_parser.parse_traces(output, pid)

    if not traces:
        raise RuntimeError("pystack remote produced no parseable frames for pid %s" % pid)

    return traces


def _can_retry_with_exhaustive_search(error):
    """Return whether the error can benefit from PyStack's exhaustive runtime search.

    For example, missing Python frame structures may qualify; a permission
    denial does not. Retry policy and its opt-in setting are checked by the caller.
    Text matching is our heuristic; PyStack suggests this retry: https://bloomberg.github.io/pystack/process.html#interpreter-finalization
    """
    text = str(error).lower()

    # TODO: This is our heuristic based on errors we have seen so far. Need to look into more errors and update this.
    return (
        "could not gather enough information to extract the python frame" in text
        or "not enough information to locate the necessary data structures" in text
    )


def _trace_process(pid, root_pid, rank=None, host=None, include_threads=True, resume_stopped=False,
                   process_rank=None, native_mode=None, retry_exhaustive=False):
    """Collect one process and format its ancestry, optional thread labels and stack frames.

    Emit #process#PID once for counting; optional process_rank=(rank, world_size)
    supplies #processrank# metadata. Without threads, keep only the preferred trace.

    For PID 42 with process_rank=(1, 4), metadata starts with
    "#process#42" and "#processrank#1#4" before the context frames and stack.
    """
    def sample():
        """Collect one process, allowing an opt-in exhaustive retry for runtime-discovery failures."""
        try:
            return _sample_pystack_traces(pid, include_threads, native_mode)
        except RuntimeError as initial_error:
            if (not retry_exhaustive
                    or not _enabled("STAT_PYSTACK_EXHAUSTIVE_RETRY", "0")
                    or _enabled("STAT_PYSTACK_EXHAUSTIVE", "0")
                    or not _can_retry_with_exhaustive_search(initial_error)):
                raise
            try:
                return _sample_pystack_traces(pid, include_threads, native_mode, exhaustive=True)
            except RuntimeError as retry_error:
                raise RuntimeError(
                    "normal attempt failed: %s; exhaustive retry failed: %s"
                    % (initial_error, retry_error)
                ) from None

    traces = _run_with_optional_resume(pid, resume_stopped, sample)

    process_labels = _process_labels(pid, root_pid, traces)
    traces_to_format = traces if include_threads else traces[:1]

    # Mark the process once, independently of its per-thread traces.
    formatted = ["#process#%d" % pid]

    if process_rank is not None:
        formatted.append("#processrank#%d#%d" % process_rank)

    for trace in traces_to_format:
        formatted.extend(_synthetic_frame(label) for label in process_labels)
        if include_threads:
            name = trace.get("name") or "unknown"
            formatted.append(_synthetic_frame("thread:%s" % name))
        formatted.extend(trace["frames"])
        formatted.append("#endtrace")
    return formatted


def _trace_torchrun_workers(pid, include_threads, resume_stopped, include_children=False,
                           native_mode=None, process_snapshot=None):
    """Sample torchrun workers using their global ranks instead of launcher task ranks.

    Validate that workers have distinct ranks and agree on world size and
    launcher task count. Without an MPI/Slurm launcher count, require the complete
    worker world and report incomplete or conflicting metadata as an error.
    Otherwise return None when promotion cannot be established so the caller
    can use its ordinary process-tree path.

    For a worker with RANK=2, WORLD_SIZE=4, its sampled child processes also
    receive rank 2 metadata; their PIDs still distinguish them for process counts.
    """
    targets = _torchrun_worker_targets(pid, process_snapshot)
    if targets is None:
        return None

    ranks = [_torchrun_rank(target) for target in targets]
    valid = (
        bool(ranks)
        and all(rank is not None for rank in ranks)
        and len({rank[0] for rank in ranks}) == len(ranks)
        and len({rank[1] for rank in ranks}) == 1
        and len({rank[2] for rank in ranks}) == 1
    )
    if any(rank is not None and rank[2] is None for rank in ranks):
        # Distinct in-range ranks covering the whole world establish one complete local group.
        if not valid or len(ranks) != ranks[0][1]:
            raise RuntimeError("torchrun workers lack MPI/Slurm launcher-size metadata and do not "
                               "form a complete, consistent worker world")
    elif not valid or ranks[0][1] <= ranks[0][2]:
        return None

    formatted = []
    errors = []
    for target, rank_info in zip(targets, ranks):
        process_targets = [target]
        if include_children:
            process_targets.extend(process_tree.descendants_for_snapshot(target, process_snapshot))
        for process_target in process_targets:
            try:
                formatted.extend(
                    _trace_process(
                        process_target,
                        int(pid),
                        include_threads=bool(include_threads),
                        resume_stopped=bool(resume_stopped),
                        process_rank=rank_info[:2],
                        native_mode=native_mode,
                        retry_exhaustive=True,
                    )
                )
            except RuntimeError as exc:
                errors.append("%s: %s" % (process_target, exc))

    if not formatted:
        raise RuntimeError("; ".join(errors) or "no torchrun worker stacks")
    if errors:
        print(
            "STAT PyStack warning: omitted %d torchrun worker(s): %s"
            % (len(errors), "; ".join(error.splitlines()[0] for error in errors)),
            file=sys.stderr,
            flush=True,
        )
    return "\n".join(formatted)


# --- Sampling concurrency and per-rank results ---

def _child_collector_count(local_rank_count, target_count):
    """Limit simultaneous child collectors per rank to share the host's sampling capacity.

    Use at most 32 CPUs, divided by local rank count, and never more collectors
    than targets. Each nonempty tree can still use at least one collector.

    For example, on a host with at least 32 CPUs, four local rank trees and
    ten targets per tree allow up to eight child collectors per tree.
    """
    budget = min(32, os.cpu_count() or 1)
    return min(target_count, max(1, budget // max(1, local_rank_count)))


def _batch_collector_count(entry_count):
    """Bound rank concurrency by entry count and STAT_PYSTACK_WORKERS, defaulting to eight workers.

    For example, three entries use at most three workers with the default setting.
    """
    if entry_count <= 0:
        return 0

    raw_value = os.environ.get("STAT_PYSTACK_WORKERS", "8").strip()
    try:
        requested = int(raw_value)
    except ValueError:
        requested = 8

    if requested < 1:
        requested = 1
    return min(requested, entry_count)


def _sample_batch_entry(entry, sample_children, include_threads, resume_stopped, native_mode=None,
                        process_snapshot=None, local_rank_count=1):
    """Sample one rank, optionally including its child processes.

    entry contains the target PID, rank and host. Return the collected stack
    text, an error message and the sampling duration in seconds. On success,
    the error is empty; on failure, the stack text is empty. Catch exceptions
    here so a failed rank does not interrupt sampling of the other ranks.
    """
    pid, rank, host = entry
    started = time.monotonic()
    try:
        if sample_children:
            trace = get_trace_tree(pid, rank, host, include_threads, resume_stopped, native_mode,
                                   process_snapshot=process_snapshot, local_rank_count=local_rank_count)
        else:
            trace = get_trace(pid, resume_stopped, include_threads, native_mode, process_snapshot)
        return trace, "", time.monotonic() - started
    except Exception as exc:
        return "", "%s: %s" % (type(exc).__name__, exc), time.monotonic() - started


# --- Public sampling API: target, tree and batch ---

def get_trace(pid, resume_stopped=False, include_threads=False, native_mode=None,
              process_snapshot=None):
    """Return STAT-formatted stack traces for a process id.

    STAT_BackEnd expects one frame per line in function@source:line format, with
    #endtrace separating threads. An unrecognized executable name is still passed to PyStack
    unless wrapper discovery selects a Python descendant; it is not a reason to reject the target.
    """
    worker_traces = _trace_torchrun_workers(
        pid, include_threads, resume_stopped, native_mode=native_mode,
        process_snapshot=process_snapshot
    )
    if worker_traces is not None:
        return worker_traces

    if _torchrun_pid(pid, process_snapshot) is not None:
        return get_trace_tree(pid, include_threads=include_threads,
                              resume_stopped=resume_stopped, native_mode=native_mode,
                              process_snapshot=process_snapshot)

    if not _is_python_process(pid):
        target = _find_python_app_pid_under_wrapper(pid, process_tree.descendants_for_snapshot(pid, process_snapshot))
        if target is not None:
            if _enabled("STAT_PYSTACK_TIMING", "0"):
                print("STAT PyStack target: rank wrapper pid=%s -> python pid=%s"
                      % (pid, target), file=sys.stderr, flush=True)
            pid = target

    output = _run_with_optional_resume(
        pid,
        resume_stopped,
        lambda: _run_pystack_remote(pid, include_threads, native_mode),
    )

    traces = stack_parser.parse_output(output, pid)
    if not traces:
        raise RuntimeError("pystack remote produced no parseable frames for pid %s" % pid)

    return "\n".join(traces)


def get_trace_tree(pid, rank=-1, host="", include_threads=True, resume_stopped=False,
                   native_mode=None, process_snapshot=None, local_rank_count=1):
    """Sample a process tree with bounded concurrency and preserve target order.

    Try promoted torchrun workers first. Otherwise collect the selected subtree,
    keeping successful traces and reporting omissions through stderr and a
    synthetic warning trace. Raise when no target yields a trace.
    local_rank_count shares the child-worker budget across ranks on this host.

    For example, if a parent samples successfully but one child fails, retain
    the parent stack and report the omission rather than discarding the whole tree.
    """

    worker_traces = _trace_torchrun_workers(
        pid, include_threads, resume_stopped, include_children=True, native_mode=native_mode,
        process_snapshot=process_snapshot
    )
    if worker_traces is not None:
        return worker_traces

    all_traces = []
    errors = []
    root_pid = int(pid)
    targets = _sampling_targets(root_pid, process_snapshot)

    def sample_target(target):
        """Return this target's traces or error so one failed process does not discard the other targets."""
        try:
            return target, (
                _trace_process(
                    target,
                    int(pid),
                    int(rank),
                    host,
                    bool(include_threads),
                    bool(resume_stopped),
                    native_mode=native_mode,
                    retry_exhaustive=target != root_pid,
                )
            ), ""
        except RuntimeError as exc:
            return target, [], str(exc)

    workers = _child_collector_count(local_rank_count, len(targets))

    if workers <= 1:
        sampled = [sample_target(target) for target in targets]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            sampled = list(executor.map(sample_target, targets))

    for target, traces, error in sampled:
        if error:
            errors.append("%s: %s" % (target, error))
        else:
            all_traces.extend(traces)

    if not all_traces:
        if errors:
            raise RuntimeError("; ".join(errors))
        raise RuntimeError("pystack remote produced no parseable frames for pid %s" % pid)

    if errors:
        print("STAT PyStack warning: omitted %d process(es) after sampling failures: %s"
              % (len(errors), "; ".join(error.splitlines()[0] for error in errors)),
              file=sys.stderr, flush=True)
        all_traces.extend(_synthetic_frame(label) for label in _process_labels(root_pid, root_pid, []))
        all_traces.append(_synthetic_frame("pystack-warning:omitted-%d-processes" % len(errors)))
        all_traces.append("#endtrace")

    return "\n".join(all_traces)


def get_trace_batch(entries, sample_children=False, include_threads=True, resume_stopped=False,
                    native_mode=None):
    """Sample an ordered list of rank roots with bounded concurrency.

    Each entry is a (pid, rank, host) tuple. Results remain in entry order and
    contain (trace, error) pairs so one failed rank does not discard successful
    samples from the other ranks.
    native_mode selects Python-only or mixed Python/native frames for this batch.
    For two entries, a partial result can be [(trace_text, ""), ("", error_text)].
    The parent map is read once per batch; no process snapshot survives to the next sample.
    """

    normalized_entries = [
        (int(pid), int(rank), host or "") for pid, rank, host in entries
    ]
    worker_count = _batch_collector_count(len(normalized_entries))
    process_snapshot = process_tree.process_tree_snapshot() if normalized_entries else {}
    host_rank_counts = {}
    for _pid, _rank, host in normalized_entries:
        host_rank_counts[host] = host_rank_counts.get(host, 0) + 1
    started = time.monotonic()

    def sample(entry):
        """Apply the shared batch settings and snapshot to one rank entry."""
        return _sample_batch_entry(
            entry,
            bool(sample_children),
            bool(include_threads),
            bool(resume_stopped),
            native_mode,
            process_snapshot,
            host_rank_counts[entry[2]],
        )

    if worker_count <= 1:
        internal_results = [sample(entry) for entry in normalized_entries]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            internal_results = list(executor.map(sample, normalized_entries))

    if _enabled("STAT_PYSTACK_TIMING", "0"):
        durations = [result[2] for result in internal_results]
        if durations:
            ordered = sorted(durations)
            median = ordered[len(ordered) // 2]
            succeeded = sum(not error for _trace, error, _duration in internal_results)
            print(
                "STAT PyStack batch: ranks=%d workers=%d elapsed=%.3fs "
                "succeeded=%d failed=%d "
                "rank_min=%.3fs rank_median=%.3fs rank_max=%.3fs"
                % (
                    len(normalized_entries),
                    worker_count,
                    time.monotonic() - started,
                    succeeded,
                    len(internal_results) - succeeded,
                    ordered[0],
                    median,
                    ordered[-1],
                ),
                file=sys.stderr,
                flush=True,
            )

    return [(trace, error) for trace, error, _duration in internal_results]
