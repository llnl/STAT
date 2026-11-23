# stat-gui Options

-a, --attach *\[hostname:\]PID*
:   Attach to the parallel job with resource manager *\[hostname:\]PID*.

-P, --withpc
:   Sample program counter values in addition to function names.

-m, --withmoduleoffset
:   Sample module offset only.

-i, --withline
:   Sample source line number in addition to function names.

-o, --withopenmp
:   Translate OpenMP stacks to logical application view

-U, --countrep
:   Only gather edge labels with the task count and a single representative. This will improve performance at extreme (i.e., over 1 million tasks) scales.

-w, --withthreads
:   Sample stack traces from helper threads in addition to the main thread.

-y, --withpython
:   Where applicable, gather Python script level stack traces, rather than show the Python interpreter stack traces.  This requires the Python interpreter being debugged to be built with -g and preferrably -O0.

-C, --create *arg\_list*
:   Launch the application under STAT's control.  All arguments after -C are used to launch the app.  Namely, *arg\_list* is the command that you would normally use to launch your application.

-I, --serial *arg\_list*
:   Attach to a list of serial processes.  All arguments after -I are interpreted as processes.  Namely, *arg\_list* is a white-space-separated list of processes to attach to, where each process is of the form \[exe@\]\[hostname:\]PID.

-d, --debugdaemons
:   launch the daemons under the deubgger

-s, --sleep *time*
:   Sleep for *time* seconds before attaching and gathering traces.  This gives the application time to get to a hung state.

-l, --log [*FE* | *BE* | *CP*]
:   Enable debug logging of the *FE* frontend, *BE* backend, *CP* communication process, *SW* Stackwalker, *SWERR* Stackwalker on error. Multiple log options may be specified (i.e., -l FE -l BE).

-L, --logdir *log\_directory*
:   Dump logging output into *log\_directory*.  To be used with the ` --log ` option.

-M, --mrnetprintf
:   Use MRNet's printf for STAT debug logging.

-G, --gdb
:   Use (cuda-)gdb to drive the daemons. If you are using cuda-gdb and want stack traces from cuda threads, you must also explicitly specify -w.

-Q, --cudaquick
:   When using cuda-gdb as the BE, gather less comprehensive, but faster cuda traces. Cuda frames will only show the top of the stack, not the full call path. This also defaults to display filename and line number and will not resolve the function name.
