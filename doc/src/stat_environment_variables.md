# Environment Variables

Several environment variables influence STAT and its dependent packages.  Note that dependent package environment variables are prefixed with "STAT\_" to avoid conflict with other tools using that package.  The STAT process will then set the appropriate (i.e., without "STAT\_") environment variable to pass the value to the dependent package.

STAT\_PREFIX=*directory*
:   Use *directory* as the installation prefix instead of the compile-time STAT\_PREFIX macro when looking for STAT components and configuration files.

STAT\_CONNECTION\_TIMEOUT=*time*
:   Wait *time* seconds for daemons to connect to MRNet.  Upon timeout, run with the available subset.

STAT\_DAEMON\_PATH=*path*
:   Use the STAT daemon executable *path* instead of the default.  *path* must be set to the full path of the STATD executable.

STAT\_FILTER\_PATH=*path*
:   Use the STAT filter shared object *path* instead of the default.  *path* must be set to the full path of the STAT\_FilterDefinitions.so shared object file.

STAT\_FGFS\_FILTER\_PATH=*path*
:   Use the STAT FGFS filter shared object *path* instead of the default.  *path* must be set to the full path of the STAT\_FilterDefinitions.so shared object file.

STAT\_MRNET\_OUTPUT\_LEVEL=*level*
:   Enable MRNet debug logging at *level* (0-5).

STAT\_MRNET\_PORT\_BASE=*port*
:   Set the MRNet base port number to *port*.

STAT\_MRNET\_STARTUP\_TIMEOUT=*seconds*
:   Set the MRNet connection timeout to *seconds*.

STAT\_CONNECT\_TIMEOUT=*seconds*
:   Set the STAT connection timeout to *seconds*, after which STAT will try to continue with any subset of daemons that have connected.

STAT\_MRNET\_DEBUG\_LOG\_DIRECTORY=*directory*
:   Write MRNet debug log files to *directory*.

STAT\_OUTPUT\_REDIRECT\_DIR=*directory*
:   Redirect stdout and stderr to a set of hostname specific files in *directory*.

STAT\_MRN\_COMM\_PATH=*path*
:   Use the mrnet\_commnode executable *path*.  *path* must be set to the full path of the mrnet\_commnode executable.  (Deprecated along with MRNet's MRN\_COMM\_PATH)

STAT\_MRNET\_COMM\_PATH=*path*
:   Use the mrnet\_commnode executable *path*.  *path* must be set to the full path of the mrnet\_commnode executable.

STAT\_XPLAT\_RSH=*path*
:   Use the remote shell *path* for launching mrnet\_commnode processes.

STAT\_PROCS\_PER\_NODE=*count*
:   Allow up to *count* communication processes to be launched per node.

STAT\_FE\_HOSTNAME=*value*
:   Set the STAT Front End hostname to *value*.  This may be necessary for example on BlueGene systems to use the proper network interface for the I/O nodes to connect back to.

STAT\_CHECK\_NODE\_ACCESS=*value*
:   Set to any value to have STAT check user access to any specified nodes before launching communication processes.

STAT\_GROUP\_OPS=*value*
:   Set to any value to enable Stackwalker's group operations.  Group operations may help with performance when a single daemon needs to manage a large number of target processes.  This is on by default on BG/Q systems.

STAT\_LMON\_PREFIX=*path*
:   Sets the LaunchMON installation prefix to *path*.

STAT\_LMON\_LAUNCHMON\_ENGINE\_PATH=*path*
:   Use the launchmon executable *path*.  *path* must be set to the full path of the launchmon executable.

STAT\_LMON\_REMOTE\_LOGIN=*command*
:   Use the remote shell *command* for LaunchMON remote debugging.

STAT\_LMON\_DEBUG\_BES=*value*
:   Launch the backends under a debugger's control if *value* is set (must be enabled in LaunchMON configuration).

STAT\_USAGE\_LOG=*path*
:   Record usage of STAT in the file located in *path*.  *path* must be writeable by user.

STAT\_ADDR2LINE=*path*
:   Use the addr2line utility located in *path* to translate module and offset prefix trees in the stat-view GUI.

STAT\_GDB=*path*
:   Use the gdb located in *path* to drive the daemons.

STAT\_INTELGT\_MAX\_THREADS=*number*
:   Set to a positive integer to limit the number of threads used for call-stack capture. STAT will capture stack traces from only the first N threads. For example, ` STAT_INTELGT_MAX_THREADS=100 ` will capture the first 100 threads. If the value is not a positive integer, this setting is ignored and all threads are captured.

STAT\_INTELGT\_EXPR\_FILTER=*expression*
:   Set to a GDB expression to selectively filter threads for call-stack capture. Only threads where the expression evaluates to true (non-zero) will have their stack traces captured. The new line characters and system commands keywords like ` shell, call, python, source, dump, restore ` are not allowed. The expression is evaluated in the context of each thread using GDB's thread apply command. For example, ` STAT_INTELGT_EXPR_FILTER="\$_thread <= 50" ` will capture only threads with ID 50 or lower. The expression must be a valid GDB expression that returns a boolean value.

    **Security Note:** This expression is directly evaluated by GDB. For security, the expression is validated to reject dangerous GDB commands such as ` shell, call, python ` and multi-line input. Only use this feature in trusted environments where environment variables are controlled.
