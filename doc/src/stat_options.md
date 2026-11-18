# stat-cl Options

-a, --autotopo
:   let STAT automatically create topology.

-f, --fanout *width*
:   Sets the maximum tree topology fanout to *width*.  Specify nodes to launch communications processes on with ` --nodes `.

-d, --depth *depth*
:   Sets the tree topology depth to *depth*.  Specify nodes to launch communications processes on with ` --nodes `.

-z, --daemonspernode *num*
:   Sets the number of daemons per node to *num*.

-u, --usertopology *topology*
:   Specify the number of communication nodes per layer in the tree topology, separated by dashes, with *topology*.  Specify nodes to launch communications processes on with ` --nodes `.  Example topologies: 4, 4-16, 5-20-75.

-n, --nodes *nodelist*
:   Use the specified nodes in *nodelist*.  To be used with ` --fanout `, ` --depth `, or ` --usertopology `.  Example nodes lists: host1; host1,host2; host\[1,5-7,9\].

-N, --nodesfile *filename*
:   Use the file *filename*, which should contain the list of nodes for communication processes

-A, --appnodes
:   Allow tool communication processes to be co-located on nodes running application processes.

-x, --exclusive
:   Do not use the front-end or back-end nodes for communication processes.

-p, --procs *processes*
:   Sets the maximum number of communication processes to be spawned per node to *processes*.  This should typically be set to a number less than or equal to the number of CPU cores per node.

-j, --jobid *id*
:   Append *id* to the output directory and file prefixes.  This is useful for associating STAT results with a batch job.

-r, --retries *count*
:   Attempt *count* retries per sample to try to get a complete stack trace.

-R, --retryfreq *frequency*
:   Wait *frequency* microseconds between sample retries.  To be used with the ` --retries ` option.

-P, --withpc
:   Sample program counter values in addition to function names.

-m, --withmoduleoffset
:   Sample module offset only.

-i, --withline
:   Sample source line number in addition to function names.

-o, --withopenmp
:   Translate OpenMP stacks to logical application view

-c, --comprehensive
:   Gather 5 traces: function only; module offset; function + PC; function + line; and 3D function only.

-U, --countrep
:   Only gather edge labels with the task count and a single representative. This will improve performance at extreme (i.e., over 1 million tasks) scales.

-w, --withthreads
:   Sample stack traces from helper threads in addition to the main thread.

-H, --maxdaemonthreads *count*
:   Allow sampling of up to *count* threads per daemon.

-y, --withpython
:   Where applicable, gather Python script level stack traces, rather than show the Python interpreter stack traces.  This requires the Python interpreter being debugged to be built with -g and preferrably -O0.

-t, --traces *count*
:   Gather *count* traces per process.

-T, --tracefreq *frequency*
:   Wait *frequency* milliseconds between samples.  To be used with the ` --traces ` option.

-S, --sampleindividual
:   Save all individual samples in addition to the 3D trace when using ` --traces ` option.

-C, --create *arg\_list*
:   Launch the application under STAT's control.  All arguments after -C are used to launch the app.  Namely, *arg\_list* is the command that you would normally use to launch your application.

-I, --serial *arg\_list*
:   Attach to a list of serial processes.  All arguments after -I are interpreted as processes.  Namely, *arg\_list* is a white-space-separated list of processes to attach to, where each process is of the form \[exe@\]\[hostname:\]PID.

-D, --daemon *path*
:   Specify the full path *path* to the STAT daemon executable.  Use this only if you wish to override the default.

-F, --filter *path*
:   Specify the full path *path* to the STAT filter shared object.  Use this only if you wish to override the default.

-s, --sleep *time*
:   Sleep for *time* seconds before attaching and gathering traces.  This gives the application time to get to a hung state.

-l, --log [*FE* | *BE* | *CP*]
:   Enable debug logging of the *FE* frontend, *BE* backend, *CP* communication process, *SW* Stackwalker, *SWERR* Stackwalker on error. Multiple log options may be specified (i.e., -l FE -l BE).

-L, --logdir *log\_directory*
:   Dump logging output into *log\_directory*.  To be used with the ` --log ` option.

-M, --mrnetprintf
:   Use MRNet's printf for STAT debug logging.

-X, --dysectapi *session*
:   Run the specified DySectAPI *session*.

-b, --dysectapi\_batch *secs*
:   Run the specified DySectAPI in batch mode. Session stops after *secs* seconds or detach action.

-G, --gdb
:   Use (cuda-)gdb to drive the daemons. If you are using cuda-gdb and want stack traces from cuda threads, you must also explicitly specify -w.

-Q, --cudaquick
:   When using cuda-gdb as the BE, gather less comprehensive, but faster cuda traces. Cuda frames will only show the top of the stack, not the full call path. This also defaults to display filename and line number and will not resolve the function name.
