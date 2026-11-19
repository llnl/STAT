"""Read Linux process information and discover process ancestry for sampling.

Procfs readers tolerate targets exiting or becoming unreadable during discovery.
Tree helpers find children, descendants and ancestry paths, excluding exited
processes and container helpers. Batch snapshots share a parent map across ranks.
This module does not assign ranks, build labels, send signals or collect stacks.
"""

import os

# --- Process information from procfs ---

def _read_text_file(path):
    """Read a text file, returning an empty string if it disappears or cannot be read.

    Process discovery uses this for /proc files, which can disappear when a
    target exits between listing its PID and reading its details."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def read_argv(pid):
    """Read procfs arguments without losing spaces or empty arguments."""
    data = _read_text_file("/proc/%s/cmdline" % pid)
    if not data:
        return []

    argv = data.split("\0")
    # Remove only the terminator, preserving any empty arguments before it.
    if argv[-1] == "":
        argv.pop()
    return argv


def read_cmdline(pid):
    """Read procfs arguments as space-separated text for labels and diagnostics, not command execution.

    For example, "python3\\0train.py\\0" becomes "python3 train.py".
    """
    return " ".join(part for part in read_argv(pid) if part)


def read_comm(pid):
    """Read the kernel process name, or an empty string if it is unavailable."""
    return _read_text_file("/proc/%s/comm" % pid).strip()


def read_exe(pid):
    """Read /proc/PID/exe to identify the binary currently running in the target.

    Unlike command-line labels, this follows the running executable after exec.
    Return <unavailable> when the process exits or access to the link is denied.
    """
    try:
        return os.readlink("/proc/%s/exe" % pid)
    except OSError:
        return "<unavailable>"


def read_environ(pid):
    """Read the target's procfs environment for rank, world-size and job identification.

    Keep names and values as bytes to avoid assuming an encoding. Split each
    entry at its first '=' so values containing '=' remain intact. Return an
    empty mapping if the file cannot be read; callers then lack rank evidence.

    For example, "RANK=2\\0WORLD_SIZE=4\\0" produces {b"RANK": b"2", b"WORLD_SIZE": b"4"}.
    """
    try:
        with open("/proc/%s/environ" % int(pid), "rb") as environ_file:
            return dict(
                entry.split(b"=", 1)
                for entry in environ_file.read().split(b"\0")
                if b"=" in entry
            )
    except OSError:
        return {}


def is_container_filesystem_helper(pid):
    """Identify container filesystem helpers and the launch sentinel that sampling must skip."""

    # Enroot names its shell shim through argv[0]; ignore helper names in application arguments.
    command = _read_text_file("/proc/%s/cmdline" % pid).split("\0")
    name = os.path.basename(command[0])

    # TODO: this is typical enroot helper names. Add other container filesystem helpers to this list.
    helpers = {"squashfuse", "fuse-overlayfs", "fuse-shim"}

    if read_comm(pid) in helpers or name in helpers:
        return True

    # Skip the temporary container-startup shell that pauses itself until resumed.
    return name in {"sh", "dash", "bash"} and command[1:] == ["-c", "kill -STOP $$ ; exit 0", ""]


def read_ppid(pid):
    """Find the parent of the given target PID using /proc/PID/stat.

    os.getppid() only reports this adapter's own parent, so it cannot answer
    this query. Return zero if the target's record is unavailable or malformed.
    """
    stat = _read_text_file("/proc/%s/stat" % pid)

    if not stat:
        return 0

    # The process name is enclosed in parentheses and can contain spaces or ')'.
    # After its final ')', the first fields are process state and parent PID.
    end = stat.rfind(")")
    if end == -1:
        return 0

    fields = stat[end + 2 :].split()
    if len(fields) < 2:
        return 0
    try:
        return int(fields[1])
    except ValueError:
        return 0


def is_exited_process(pid):
    """Check whether procfs reports a zombie (Z) or dead (X/x) process.

    These processes cannot supply a useful stack. A stopped process is still
    eligible for sampling; an unreadable record alone does not prove it exited.
    """
    stat = _read_text_file("/proc/%s/stat" % pid)
    _comm, separator, tail = stat.rpartition(")")
    fields = tail.split()
    return bool(separator and fields and fields[0] in {"Z", "X", "x"})


# --- Process-tree discovery and snapshots ---

# - Use direct_children(pid) for immediate children, or descendants(pid) for all generations.
# - Use descendants_for_snapshot(pid, snapshot) when a batch already has a process-tree map.
# - Use process_path(pid, root_pid) for one ancestry chain used in labels and wrapper checks.
# _task_children and _scan_proc_children are discovery sources used internally by direct_children.

def _task_children(pid):
    """Find child processes created by any thread of the target process.

    Linux lists threads under /proc/PID/task/TID. Each thread's children file
    contains space-separated PIDs of its direct child processes, not thread IDs.
    Inspect every thread because a worker thread can create a subprocess too.

    For example, /proc/100/task/103/children containing "200 201" contributes
    child PIDs 200 and 201 even if the main thread's children file is empty.
    """

    children = []
    task_dir = "/proc/%s/task" % pid
    try:
        tids = os.listdir(task_dir)
    except OSError:
        return children
    for tid in tids:
        data = _read_text_file("%s/%s/children" % (task_dir, tid)).strip()
        if not data:
            continue
        for child in data.split():
            try:
                children.append(int(child))
            except ValueError:
                pass
    return children


def _scan_proc_children(pid):
    """Scan /proc for processes whose parent PID matches the requested PID.

    For example, if process 200 has parent PID 100, calling this with 100
    includes 200 in the returned list. Only direct children are returned.
    """
    children = []
    try:
        proc_entries = os.listdir("/proc")
    except OSError:
        return children
    for entry in proc_entries:
        if not entry.isdigit():
            continue
        child_pid = int(entry)
        if read_ppid(child_pid) == pid:
            children.append(child_pid)
    return children


def direct_children(pid):
    """Return eligible direct child PIDs by combining both procfs discovery methods.

    Always union per-thread children with the parent-PID scan, then remove
    duplicates, the root itself, vanished/exited processes and container helpers.
    Sort by PID so traversal has a consistent order for the same discovered tree.

    For 10 -> [11, 12] and 11 -> [13], direct_children(10) returns [11, 12];
    grandchild 13 belongs to descendants(10). PID 12 found by both methods
    appears once. This is a best-effort view: a child may exit after the check.
    """
    children = set(_task_children(pid))
    children.update(_scan_proc_children(pid))
    children.discard(pid)
    return sorted(
        child
        for child in children
        if os.path.exists("/proc/%s" % child)
        and not is_exited_process(child)
        and not is_container_filesystem_helper(child)
    )


def descendants(pid):
    """Walk eligible child processes recursively, visiting each PID at most once.

    Use direct_children() at each level, so container helpers and exited tasks
    are filtered throughout the tree. This finds grandchildren as well as children;
    unlike process_path(), it searches outward from the root rather than toward it.
    For 10 -> [11, 12] and 11 -> [13], return [11, 12, 13], excluding the root 10.
    """

    root_pid = int(pid)
    visited = {root_pid}
    descendants = []
    pending = list(direct_children(root_pid))
    index = 0

    while index < len(pending):
        child = int(pending[index])
        index += 1
        if child in visited:
            continue
        visited.add(child)
        if not os.path.exists("/proc/%s" % child):
            continue
        descendants.append(child)
        pending.extend(direct_children(child))

    return descendants


def process_path(pid, root_pid):
    """Return the live process ancestry from *root_pid* through *pid*.

    Follow parent PIDs from the target back to the root, then reverse the path
    for process labels. This returns one ancestry chain, not all descendants.
    For 10 -> [11, 12] and 11 -> [13], process_path(13, 10) returns [10, 11, 13];
    sibling 12 is not on that path. If ancestry is lost, return [13] without inventing parents.
    """

    pid = int(pid)
    root_pid = int(root_pid)
    current = pid
    reverse_path = []
    visited = set()

    while current > 0 and current not in visited:
        visited.add(current)
        reverse_path.append(current)
        if current == root_pid:
            return list(reversed(reverse_path))
        current = read_ppid(current)

    # A process may exit or be re-parented while sampling. In that case,
    # retain the sampled target without inventing an ancestry relationship.
    return [pid]


def process_tree_snapshot():
    """Build a parent-PID to child-PIDs map shared by all rank samples in one batch.

    Read /proc once to avoid repeating a full process scan for each rank.
    Rebuild the map for the next batch so newly started children can be found.

    For children 11 and 12 of PID 10, and child 13 of PID 11, the map is
    {10: [11, 12], 11: [13]}. It can be reused across ranks in this batch.
    """
    children = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return children
    for entry in entries:
        if entry.isdigit():
            parent = read_ppid(int(entry))
            if parent > 0:
                children.setdefault(parent, []).append(int(entry))
    for targets in children.values():
        targets.sort()
    return children


def descendants_for_snapshot(pid, snapshot):
    """Find descendants using the batch's parent map, or live discovery when no map is supplied.

    Recheck whether each PID exists or is an exited task or filesystem helper
    before including it, since the process tree may have changed since the scan.

    For {10: [11], 11: [12]}, starting at 10 returns [11, 12] if both are eligible;
    excluding helper 11 also prevents traversal into its descendants.
    """
    if snapshot is None:
        return descendants(pid)

    visited = {int(pid)}
    descendant_pids = []
    pending = list(snapshot.get(int(pid), ()))

    for child in pending:
        if child in visited:
            continue
        visited.add(child)
        if (not os.path.exists("/proc/%s" % child)
                or is_exited_process(child) or is_container_filesystem_helper(child)):
            continue
        descendant_pids.append(child)
        pending.extend(snapshot.get(child, ()))

    return descendant_pids
