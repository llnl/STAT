"""Parse PyStack CLI output into thread records and STAT frame lines.

This module only transforms text; process discovery, collector execution and
process labels belong to stat_py_stack. Thread priority keeps the main
application stack first while preserving frame order within each stack.
"""

import re

# --- PyStack output patterns ---

# Regex for identifying Python frame lines and extracting their source file, line number and function.
# Example from PyStack --no-color remote: (Python) File "train.py", line 12, in train.
_PYTHON_FRAME_RE = re.compile(
    r'^\s+\(Python\)\s+File "(?P<source>.*)",\s+'
    r"line (?P<line>\d+),\s+in (?P<function>.*)\s*$"
)

# Regex for identifying native C/C++frame lines; the function field may also contain a shared-object name.
_C_FRAME_RE = re.compile(
    r'^\s+\(C\)\s+File "(?P<source>.*)",\s+line (?P<line>\d+),\s+in (?P<rest>.*)\s*$'
)

# Regex for identifying thread headers that start a stack, extracting the TID and optional thread name.
_THREAD_RE = re.compile(
    r"^Traceback for thread (?P<tid>\d+)(?: \((?P<name>[^)]*)\))?.*:\s*$"
)

# Recognize synthetic filenames such as <dynamic_function_12345>.
# Their numbers can differ across ranks; normalize them so equivalent function frames merge.
_DYNAMIC_FUNCTION_SOURCE_RE = re.compile(r"<dynamic_function_[0-9]+>")


# --- Frame normalization and formatting ---

def _clean_frame_function(function):
    """Normalize missing function names and remove PyStack's trailing inline annotation.

    For example, " train (inlined) " becomes "train", and None or " " becomes "<unknown>".
    """
    function = (function or "<unknown>").strip()
    if function.endswith(" (inlined)"):
        function = function[: -len(" (inlined)")].rstrip()
    return function or "<unknown>"


def _clean_frame_source(source):
    """Strip source-path padding and give missing locations the common <unknown> label."""
    source = (source or "<unknown>").strip()
    return source or "<unknown>"


def _split_c_frame_rest(rest):
    """Separate a native function from its optional shared-object suffix.

    For example, "poll (/lib/libc.so.6)" becomes ("poll", "/lib/libc.so.6").
    """
    rest = (rest or "").strip()
    if rest.endswith(")") and " (" in rest:
        function, obj = rest.rsplit(" (", 1)
        return function.strip(), obj[:-1].strip()
    return rest, ""


def _format_frame(line):
    """Convert a Python or native frame line to function@source:line, or return None.

    Native functions gain a native: prefix. Generated Python locations are
    normalized so process-specific filenames do not split equivalent call paths.

    For example, a Python train frame at train.py line 12 becomes "train@train.py:12";
    a native poll frame becomes "native:poll@/lib/libc.so.6:0" when only its object is known.
    """
    match = _PYTHON_FRAME_RE.match(line)
    if match is not None:
        function = _clean_frame_function(match.group("function"))
        source = _clean_frame_source(match.group("source"))
        lineno = (match.group("line") or "0").strip()

        # Generated locations are process-local; retain only the function identity.
        if _DYNAMIC_FUNCTION_SOURCE_RE.fullmatch(source):
            source, lineno = "<dynamic_function>", "0"

        return "%s@%s:%s" % (function, source, lineno)

    match = _C_FRAME_RE.match(line)

    if match is not None:
        function, obj = _split_c_frame_rest(match.group("rest"))
        function = _clean_frame_function(function)
        source = _clean_frame_source(match.group("source"))
        obj = _clean_frame_source(obj) if obj else ""
        lineno = (match.group("line") or "0").strip()

        if source == "???" and obj:
            source = obj

        return "native:%s@%s:%s" % (function, source, lineno)

    return None


# --- Thread ordering and stack parsing ---

def _trace_priority(trace, pid):
    """Order thread stacks so collection without threads can select the main application stack.

    Prefer the target's main thread, then Python-named threads, then threads with
    Python frames, and finally native-only threads.

    For target PID 42, thread 42 sorts before a native-only worker thread 43,
    even when PyStack prints thread 43 first.
    """
    tid = trace.get("tid")
    name = (trace.get("name") or "").lower()
    frames = trace.get("frames") or []
    has_python = any(not frame.startswith("native:") for frame in frames)

    if pid is not None and tid == str(pid):
        return 0
    if name in {"python", "python3"}:
        return 1
    if has_python:
        return 2
    return 3


def parse_traces(output, pid=None):
    """Convert PyStack's text output into one record per thread stack.

    Each record contains the thread ID (tid), name, original index and formatted
    frames. Ignore non-frame output and empty stacks. Sort whole thread stacks
    by priority, preserving their original order for ties; frames within each
    stack stay in order. Frames without a header form an unnamed trace.

    For example, a header for thread 42 (python) produces tid="42", name="python"
    and a frames list containing that thread's formatted stack.
    """
    traces = []
    current = None

    def finish_current():
        """Retain the current thread only if it contains recognized frames."""
        if current is not None and current["frames"]:
            traces.append(current)

    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        thread_match = _THREAD_RE.match(stripped)
        if thread_match is not None:
            finish_current()
            current = {
                "tid": thread_match.group("tid"),
                "name": thread_match.group("name") or "",
                "index": len(traces),
                "frames": [],
            }
            continue

        frame = _format_frame(line)
        if frame is not None:
            if current is None:
                current = {"tid": "", "name": "", "index": len(traces), "frames": []}
            current["frames"].append(frame)

    finish_current()
    traces.sort(key=lambda trace: (_trace_priority(trace, pid), trace["index"]))
    return traces


def _format_traces(traces):
    """Convert parsed thread stacks into the line format expected by STAT_BackEnd.

    Separate stacks with #endtrace so frames from different threads are not
    interpreted as one call stack. For example, two one-frame stacks become
    [frame_a, "#endtrace", frame_b, "#endtrace"].
    """
    formatted = []
    for trace in traces:
        formatted.extend(trace["frames"])
        formatted.append("#endtrace")
    return formatted


def parse_output(output, pid=None):
    """Convert PyStack output to STAT frame lines with #endtrace separating thread stacks."""
    return _format_traces(parse_traces(output, pid))
