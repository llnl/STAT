# Troubleshooting Guide

## Troubleshooting

**STAT hangs when attaching to Intel MPI jobs**

When using the Intel MPI, you may need to alter your LaunchMON installation's etc/rm\_intel\_hydra.conf file and set RM\_launch\_helper=mpirun. If it is set to mpiexec.hydra, the daemons may fail to launch on remote nodes.

**stack traces are empty**

Some optimzations may make it impossible to debug, such as the GNU -fomit-frame-pointer option, which is enabled at various -O optimization levels. You can turn this off with the -fno-omit-frame-pointer flag.

**stack walks not making it to \_start**

Processes can be in portions of code from which a debugger cannot walk the stack (i.e., function prologue or epilogue).  Try the -r option to enable STAT to let the process run a bit and then retry the stack sample.

**stack walks with line number information returning ??**

Stack traces with line number information requires your code to be compiled with debug information (i.e., with the -g flag).

**/usr/lib/python2.6/site-packages/gtk-2.0/gtk/\_\_init\_\_.py :72: GtkWarning: could not open display**

Be sure to enable X-forwarding and to set your $DISPLAY environment variable.

**STATview requires gtk**

STAT requires the pygtk module to be installed.  If it is side-installed, but sure to set your $PYTHONPATH environment variable to the directory containing the pygtk module.

**ImportError: No module named STAT**

Make sure to run \`make install\` to install STAT.py in the lib/python\[version\]/site-packages directory or set your $PYTHONPATH environment variable to the directory containing STAT.py

**(ERROR): LaunchMON Engine invocation failed, exiting: No such file or directory**

Make sure the ` launchmon ` executable is in your $PATH or set the $STAT\_LMON\_LAUNCHMON\_ENGINE\_PATH engine path to the full path to the executable.

**OptionParsing (ERROR): unknown launcher: a.out**

You need to attach to your ` mpirun ` or equivalent parallel job launch process.

**OptionParsing (ERROR): the path\[/usr/local/bin/STATD\] does not exit.**

STAT looks for its components in the configured $prefix.  Be sure to run \`make install\` or set STAT\_DAEMON\_PATH to the full path to the ` STATD ` executable.

**LaunchMON prints a usage message.**

This is typically a mismatch in versions of the LaunchMON API and the LaunchMON engine.  Make sure to set your $STAT\_LMON\_LAUNCHMON\_ENGINE\_PATH enviornment variable to the full path to the appropriate ` launchmon ` executable.

**(ERROR): accepting a connection with an engine timed out**

STAT may need additional time to launch all of its daemons.  You may need to set your $LMON\_FE\_ENGINE\_TIMEOUT to a larger value, such as 600.

**UnboundLocalError: local variable 'count' referenced before assignment**

This error results from trying to open a STAT \>3.0 output with a STAT 2.X GUI.

**Hang on second attach when using CUDA GDB**

In some CUDA environments, you may need to run your application with the CUDA\_VISIBLE\_DEVICES environment variable set appropriately, otherwise cuda-gdb will hang on attach when there is already an existing cuda-gdb session on the same node. Refer to the "Simultaneous Sessions Support" section of  [http://docs.nvidia.com/cuda/cuda-gdb/index.html](<http://docs.nvidia.com/cuda/cuda-gdb/index.html>).

## Slurm Job Attachment

When attaching to a Slurm job with the Remote Connect CLI or STAT GUI, the job may still be queued, or Slurm may
show it as running while the prolog or workload is starting. SSH, launcher discovery, and STAT's own daemons can
also fail during attachment. Some messages identify the cause directly, while others only report which operation
failed.

- **Job startup.** STAT may report that Slurm is starting the job, that it cannot find a running launcher, or that a
  rank is still running `slurmstepd`. Wait until the workload starts, confirm startup in the application log, and
  try again. In the GUI, refresh the job before retrying.
- **Compute-node access.** If SSH reports `Permission denied (publickey)`, check passwordless SSH access to the
  compute node. If the job just started, retry after the prolog finishes.
- **Launcher or daemon errors.** Messages such as `Failed to launch STAT daemons` and `1 daemons reported an error`
  may need more context. If the error persists after the workload starts, save the terminal diagnostics and rerun with
  `--remote-log-dir <absolute-shared-path>` to collect remote logs.
- **Nested launchers.** If PyStack sees launcher or wrapper frames and the application runs in child processes,
  enable child-process sampling in the GUI or pass `--with-children` to the CLI. To sample a specific phase, wait
  for the application log to show that phase before attaching.

STAT attempts to detach and release its own stops after a failed PyStack attach or CLI sample. If detach also
reports a failure, check that the workload is running before trying another attachment.
