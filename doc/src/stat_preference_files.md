# Setting STAT Preferences and Options

## Preference Files

Several files can influence how STAT runs.  The first such file is $prefix/etc/STAT/nodes.txt, which specifies a list of hostnames, one hostname per line, on which to launch MRNet communication processes.  This file is designed to be shared by all users and should point to shared resources that all users have remote shell access to, such as login nodes.  Note that by default STAT will not test access to a node before trying to launch communication processes.  If the STAT\_CHECK\_NODE\_ACCESS environment variable is set to any value, then STAT will try to run (via remote shell) a simple test to see if the node is accessible before adding it to the MRNet tree.  Also note that nodes.txt will not be used if the -A or "Share App Nodes" option is enabled.

STAT GUI preferences can be set with an installation specific STAT.conf or user specific .STATrc file.  The installation specific file should be placed in $prefix/etc/STAT/STAT.conf, while the user specific file should be placed in $HOME/.STATrc.  Options specified in the user's .STATrc file will always take precedence over the STAT installation's .STATrc file.  Each preference file specifies one option per line of the format:

Option = Value

Here is a list of options:

Remote Host = *hostname*
:   Sets the default remote host to *hostname* to search for the job launcher process.

Remote Host Shell = *rsh|ssh*
:   Sets the default remote host shell to *rsh* or *ssh* to get a process listing on remote hosts.

Resource Manager = *Auto|Alps|Slurm*
:   Sets the default resource manager to *Alps* or *Slurm* for searching for the job launcher process, or use *Auto* to determine the resourece manager automatically.

Job Launcher = *regex*
:   Sets the default regular expression to *regex* (i.e., "mpirun|srun") for filtering the process listing for the job launcher process.

Tool Daemon Path = *path*
:   Use the STAT deamon executable installed in *path* instead of the default.

Filter Path = *path*
:   Use the STAT filter shared object installed in *path* instead of the default.

Topology Type = *automatic|depth|max fanout|custom*
:   Use the specified topology type when building the MRNet communication tree.  The *automatic* topology is typically recommended and set by default.

Topology = *topology*
:   Use *topology* for the specific topology configuration.  This should be used with the Topology Type option.  Refer to the stat-cl options to see valid Topology specifications for a given Topology Type.

Communication Nodes = *nodelist*
:   Use the nodes listed in *nodelist* for MRNet communication processes.

Check Node Access = *true|false*
:   Controls whether to check access to a node before trying to launch MRNet communication processes on it.

CP policy = *none|share app nodes|exclusive*
:   Controls where to launch communication processes.  When set to *share app nodes*, they will be launched on nodes running application processes.  On BlueGene systems, this will actually place them on the I/O nodes, and requires users to be able to access the I/O nodes via a remote shell.  When set to *exclusive*, then the communication processes will only be run on specified nodes that do not run other STAT tool processes (e.g., the STAT frontend and the back-end daemons).

Communication Processes per Node = *count*
:   Launch no more than *count* MRNet communication processes per node.

Num Traces = *count*
:   Gather *count* stack traces when sampling multiple.

Trace Frequency (ms) = *count*
:   Let the process run *count* milliseconds between multiple samples.

Num Retries = *count*
:   Attempt *count* retries to try to obtain a complete stack trace.

Retry Frequency (ms) = *count*
:   Let the process run *count* milliseconds between stack sample retries.

With Threads = *true|false*
:   Controls whether to gather stack traces from threads.

With OpenMP = *true|false*
:   Controls whether to translate OpenMP stack traces to logical application view.

Gather Python Traces = *true|false*
:   Controls whether to gather Python script level stack traces, rather than show the Python interpreter stack traces.

Sample Type = *function only|module offset|function and pc|function and line*
:   Controls the granularity of the nodes in the gathered stack traces.

Edge Type = *full list|count and representative*
:   Controls the granularity of the edges in the gathered stack traces.

DDT Path = *path*
:   Use the DDT executable installed in *path* for subset debugging.

DDT LaunchMON Prefix = *path*
:   Use the LaunchMON installation in *path* for improved DDT subset attaching, otherwise attach via hostname:PID pairs.

TotalView Path = *path*
:   Use the TotalView executable installed in *path* for subset debugging.

Additional Debugger Args = *args*
:   Add *args* to the argument list when launching TotalView or DDT.

Log Dir = *directory*
:   Write STAT debug logs to *directory*.

Log Frontend = *true|false*
:   Controls whether to enable debug logging of the STAT frontend.

Log Backend = *true|false*
:   Controls whether to enable debug logging of the STAT backend.

Log CP = *true|false*
:   Controls whether to enable debug logging of the STAT communication processes.

Log SW = *true|false*
:   Controls whether to enable debug logging of Stackwalker by the STAT backend.

Log SWERR = *true|false*
:   Controls whether to enable debug logging of Stackwalker by the STAT backend when a Stackwalker error is detected.

Use MRNet Printf = *true|false*
:   Controls whether to use MRNet's printf when writing debug logs.  This is helpful to correlate timing between STAT log messages and MRNet debug log messages, when MRNet logging is being logged (via the STAT\_MRNET\_OUTPUT\_LEVEL enviornment variable).

GDB BE = *true|false*
:   Controls whether to use (cuda-)gdb to drive the deamons.

With CUDA QUICK = *true|false*
:   When using cuda-gdb as the BE, controls whether to gather less comprehensive, but faster cuda traces. Cuda frames will only show the top of the stack, not the full call path. This also defaults to display filename and line number and will not resolve the function name.

GDB Path = *path*
:   Use the gdb executable installed in *path* for debugging when GDB BE is set to true.

```{include} stat_preferences.md
:heading-offset: 1
```

```{include} stat_environment_variables.md
:heading-offset: 1
```
