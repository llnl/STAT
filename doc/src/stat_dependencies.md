# Installing STAT

## Dependent Packages

STAT has several dependencies

```{eval-rst}
.. table:: STAT Dependent Packages
   :class: dependency-table
   :width: 100%
   :widths: 30 35 35

   +---------------------------------+-----------------------------------------------------+---------------------------------------+
   | Package                         | What It Does                                        | Package Web Page                      |
   +=================================+=====================================================+=======================================+
   | Graphlib version 3.0 or greater | Graph creation, merging, and export                 | https://github.com/LLNL/graphlib/     |
   +---------------------------------+-----------------------------------------------------+---------------------------------------+
   | Launchmon                       | Scalable daemon co-location                         | https://github.com/LLNL/LaunchMON     |
   +---------------------------------+-----------------------------------------------------+---------------------------------------+
   | Libdwarf                        | Debug information parsing (Required by StackWalker) | https://www.prevanders.net/dwarf.html |
   +---------------------------------+-----------------------------------------------------+---------------------------------------+
   | MRNet version 3.0 or greater    | Scalable multicast and reduction network            | https://github.com/dyninst/mrnet      |
   +---------------------------------+-----------------------------------------------------+---------------------------------------+
   | StackWalker                     | Lightweight stack trace sampling                    | https://github.com/dyninst/dyninst    |
   +---------------------------------+-----------------------------------------------------+---------------------------------------+
```

In addition, STAT requires [Python](<http://www.python.org/>) and the STAT GUI requires [PyGObject](<https://www.gtk.org/docs/language-bindings/python>), both of which are commonly preinstalled with many Linux operating systems. STAT can be built with Python 3.X. requires you to install the [xdot python package](<https://pypi.org/project/xdot/>). STAT also requires [SWIG](<http://www.swig.org/>) to generate Python wrappers for STAT's core functionality. The [Pygments](<http://pygments.org/>) Python module can optionally be installed to allow the STAT GUI to perform syntax highlighting of source code. Another GUI requirement is the [Graphviz](<http://www.graphviz.org/>) package to render the DOT format graph files.

STAT can also be optionally built with the [Fast Global File Status (FGFS)](<https://github.com/LLNL/FastGlobalFileStatus>) library.  This library helps STAT identify when a file (target binary) resides on a shared file system that may become a bottleneck if all STAT daemons try to access that file at the same time.  If so, STAT will access the file from the STAT frontend and distribute its contents to the daemons via the MRNet communication tree.  The two necessary components of FGFS can be downloaded from [https://github.com/LLNL/MountPointAttributes](<https://github.com/LLNL/MountPointAttributes>) and [https://github.com/LLNL/FastGlobalFileStatus](<https://github.com/LLNL/FastGlobalFileStatus>)

```{include} stat_installation.md
:heading-offset: 1
```
