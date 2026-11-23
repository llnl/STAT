# Sample Options

STAT has several options for stack trace sampling ({numref}`statguiSampleMultipleDialog`).

```{figure} figs/statgui_sample_multiple_dialog.png
:name: statguiSampleMultipleDialog
:alt: The stat-gui operation toolbar.
:width: 70%
:figclass: use-figure-width

The stat-gui operation toolbar.
```

These options are defined as follows:

**Sampling Backend**
:   Choose Dyninst, PyStack, or GDB before attaching. PyStack and GDB appear when enabled in the build. For GDB, select
    the debugger executable under **Advanced → Tool Paths**. GDB supports function names with optional source lines.
    Enable **With Threads** to include additional host threads and, when using `cuda-gdb`, CUDA device traces.

**With Threads**
:   Sample helper threads in addition to the main thread.

Max Threads Per Daemon*count*
:   Allow sampling of up to *count* threads per daemon.

**With CUDA Quick**
:   When using cuda-gdb as the BE, gather less comprehensive, but faster cuda traces. Cuda frames will only show the top of the stack, not the full call path. This also defaults to display filename and line number and will not resolve the function name.

**With OpenMP**
:   Translate OpenMP stacks into logical application view (requires application built with OMPD-enabled OpenMP)

**Gather Python Traces**
:   Where applicable, gather Python script level stack traces, rather than show the Python interpreter stack traces.  This requires the Python interpreter being debugged to be built with -g and preferrably -O0.

**function only** | **module offset** | **function and pc** | **function and line**
:   Sample traces with function name only, or module name and offset, or function name with the CPU program counter, or function name with the source file and line number. When gathering the module and offset, you can later translate all of the node labels into source file and line number via the GUI (left or right click on a node).

**full list** | **count and representative**
:   Sample traces with the full task list or just the count and a single representative.  When gathering the count and representative, you can actually query an individual STAT graph node (through the left-click menu) for the full edge label, as long as the STAT session is still attached.

**Run Time Before Sample**
:   Resume the application and let it run for the specified amount of time before gathering the sample

**\# Retries/Retry Frequency** (Advanced)
:   Sometimes a process may be in a state (i.e., function prologue or epilogue) such that a complete stack trace may not be obtainable.  This option controls how many times to retry sampling and how often to wait (in microseconds) between retries to try and get a complete trace.

**\# Traces/Trace Frequency**
:   When sampling multiple traces over time, these options specify how many traces to gather per process and how long to wait between samples.

**Gather Individual Samples**
:   When sampling multiple traces over time, this option enables STAT to gather all of the intermediate 2D prefix trees in addition to the fully merged 3D prefix tree.  The traces will be displayed in individual tabs.

**Clear On Sample**
:   When sampling multiple traces over time, STAT accumulates the traces that are gathered.  This option determines whether to clear the accumulated traces when gathering additional traces.
