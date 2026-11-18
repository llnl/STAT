# stat-bench Options

-a, --autotopo
:   let STAT automatically create topology.

-f, --fanout *width*
:   Sets the maximum tree topology fanout to *width*.  Specify nodes to launch communications processes on with ` --nodes `.

-d, --depth *depth*
:   Sets the tree topology depth to *depth*.  This option takes precedence over the ` --fanout ` option.  Specify nodes to launch communications processes on with ` --nodes `.

-u, --usertopology *topology*
:   Specify the number of communication nodes per layer in the tree topology, separated by dashes, with *topology*.  This option takes precedence over the ` --fanout ` and ` --depth ` options.  Specify nodes to launch communications processes on with ` --nodes `.  Example topologies: 4, 4-16, 5-20-75.

-n, --nodes *nodelist*
:   Use the specified nodes in *nodelist*.  To be used with ` --fanout `, ` --depth `, or ` --usertopology ` options.  Example nodes lists: host1; host1,host2; host\[1,5-7,9\].

-A, --appnodes
:   Allow tool communication processes to be co-located on nodes running application processes.

-p, --procs *processes*
:   Sets the maximum number of communication processes to be spawned per node to *processes*.  This should typically be set to the number of CPUs per node.

-D, --daemon *path*
:   Specify the full path *path* to the STATBenchD daemon executable.  Use this only if you wish to override the default.

-F, --filter *path*
:   Specify the full path *path* to the stat-bench filter shared object.  Use this only if you wish to override the default.

-t, --traces *count*
:   Gather *count* traces per process.

-i, --iters *count*
:   Perform *count* gathers.

-n, --numtasks *count*
:   Emulate *count* tasks per daemon.

-m, --maxdepth *depth*
:   Generate traces with a maximum depth of *depth*.

-b, --branch *width*
:   Generate traces with a max branching factor of *width*.

-e, --eqclasses *count*
:   Generate traces within *count* equivalence classes.

-U, --countrep
:   Only gather edge labels with the task count and a single representative.

-l, --log [*FE* | *BE* | *CP*]
:   Enable debug logging of the *FE* frontend, *BE* backend, or *CP* communication process. Multiple log options may be specified (i.e., -l FE -l BE).

-L, --logdir *log\_directory*
:   Dump logging output into *log\_directory*.  To be used with the ` --log ` option.

-M, --mrnetprintf
:   Use MRNet's printf for STAT debug logging.
