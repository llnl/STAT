# Installation

STAT and its dependencies can also be built via the Spack package management tool, available at [https://github.com/spack/spack](<https://github.com/spack/spack>). Running ` spack install stat ` should build STAT and all of its dependencies, including those require for the STAT GUI. Note that three Spack variants exist. The first is ` +gui ` to enable the building of the stat-view dot file viewer and the stat-gui GUI. The second is ` +examples `, to enable building example MPI code that STAT can be tested against. This is disabled by default to avoid requiring an MPI library, but enabling it will trigger a build of MPI. The third is ` +dysect ` to enable building of the DySectAPI. The fourth is ` +fgfs ` to build with FastGlobalFileStatus and target binary broadcast support. FGFS is crucial to scalability, particularly when debugging applications that load several large dynamic libraries.

When building STAT itself, first run ` configure `.  You will need to use the ` --with-package ` options to specify the install prefix for mrnet, graphlib, launchmon, libdwarf, and dyninst.  These options will add the necessary include and library search paths to the compile options.  Refer to ` configure --help ` for exact options.  You may also wish to specify the maximum number of communication processes to launch per node with the option ` --with-procspernode=number `, generally set to the number of cores per node.

STAT creates wrapper scripts for the ` stat-cl ` and ` stat-gui ` commands.  These wrappers set appropriate paths for the launchmon and mrnet\_commnode executables, based on the ` --with-launchmon ` and ` --with-mrnet ` configure options, thus it is important to specify both of these even if they share a prefix.

STAT will try to build the GUI by default.  Building and running the STAT GUI requires SWIG, pygobject, and xdot.  If you need to modify your PYTHONPATH envirnment variable to search for side installed site-packages, you can do this by specifying ` --with-python-path=path ` during configure.  This will add the appropriate directory to the $PYTHONPATH environment variable within the stat-gui and stat-view scripts.  To disable the building of the GUI, use the ` --enable-gui=no ` configure option. Here is an example configure command:

```text
./configure --with-launchmon=/tmp/work/lee218/install \
--with-mrnet=/tmp/work/lee218/install \
--with-graphlib==/tmp/work/lee218/install \
--with-stackwalker=/tmp/work/lee218/install \
--with-libdwarf=/tmp/work/lee218/install \
--with-python=/usr/bin/python3 \
--with-python-path=/tmp/work/lee218/install/lib/python3.14/site-packages \
--prefix=/tmp/work/lee218/install
```

After running configure you just need to run:

```text
make
make install
```

Note that STAT hardcodes the paths to its daemon and filter shared object, assuming that they are in $prefix/bin and $prefix/lib respectively, thus testing should be done in the install prefix after running ` make install ` and the installation directory should not be moved.  The path to these components can, however, be overridden with the ` --daemon ` and ` --filter ` arguments.  Further, the STAT\_PREFIX environment variable can be defined to override the hardcoded paths in STAT.
STAT will also, by default, add rpaths to dependent libraries.  This behavior can be disabled by specifying ` --with-rpath=no `.  However, when doing so, you must be sure to set LD\_LIBRARY\_PATH to point to the directories containing the dependent libraries.

STAT can also be configured to use GDB as a backend instead of Dyninst. To specify the path to gdb, use the ` --with-gdb ` flag. If GPU kernal traces are desired, the path should point to a working cuda-gdb, roc-gdb, or gdb-oneapi executable. Note that STAT currently still requires Dyninst as a dependence even when using the GDB backend.

Dyninst 12 removed the local\_var.h header file that STAT uses to gather python traces. If this feature is needed, you will either need to use Dyninst version 11 or lower, or copy the local\_var.h file from [https://raw.githubusercontent.com/dyninst/dyninst/5c7e0ee327399cfae50d77f977ff6655c2ca3ae4/stackwalk/h/local\_var.h](<https://raw.githubusercontent.com/dyninst/dyninst/5c7e0ee327399cfae50d77f977ff6655c2ca3ae4/stackwalk/h/local_var.h>) to your Dyninst 12+ installation's include directory
