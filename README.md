# <img src="https://github.com/LLNL/STAT/raw/develop/STATlogo.gif" alt="STAT Logo"/> STAT: the Stack Trace Analysis Tool

[![Build Status](https://travis-ci.org/LLNL/STAT.svg?branch=develop)](https://travis-ci.org/LLNL/STAT)
[![codecov](https://codecov.io/gh/LLNL/STAT/branch/develop/graph/badge.svg)](https://codecov.io/gh/LLNL/STAT)

### OVERVIEW
The Stack Trace Analysis Tool (STAT) is a highly scalable, lightweight tool that gathers and merges stack traces from all of the processes of a parallel application to form call graph prefix trees.  STAT generates two prefix trees termed 2D-trace-space and 3D-trace-space-time.  The 2D-trace-space prefix tree is a merge of a single stack trace from each task in the parallel application.  The 3D-trace-space-time prefix tree is a merge of several stack traces from each task gathered over time.  The latter provides insight into whether tasks are making progress or are in a hang state (livelock, deadlock, infiite loop, etc.).  The call graph prefix trees also identify processes equivalence classes, processes exhibitin similar behavior with respect to their call paths.  A representative task from each equivalence class can then be fed into a full-featured debugger for root cause analysis at a manageable scale.

STAT's source code also includes STATBench, a tool to emulate STAT.  STATBench enables the benchmarking of STAT on arbitrary machine architectures and applications by fully utilizing parallel resources and generating artificial stack traces.

### License
STAT is released under the Berkeley Software Distribution (BSD) license. Please see [LICENSE](https://github.com/LLNL/STAT/blob/master/LICENSE) for usage terms.

### BUILDING STAT
STAT has several dependent libraries that must be installed:
* [MRNet](https://github.com/dyninst/mrnet)
* [LaunchMON](https://github.com/LLNL/LaunchMON)
* [GraphLib](https://github.com/LLNL/graphlib)
* [Dyninst](https://github.com/dyninst/dyninst)
* [libdwarf](https://www.prevanders.net/dwarf.html)

In addition, the STAT GUI requires Python with PyGTK, both of which are commonly preinstalled with many Linux operating systems. STAT can be built with Python 2.X and PyGTK 2.X. However, starting with STAT version 4.0.1, STAT can optionally be built with Python 3.X. The use of Python 3.X also requires PyGTK version 3.X and requires you to manually install the [xdot package](https://pypi.org/project/xdot/).

Please refer to [INSTALL](/INSTALL) for detailed instructions on building STAT. The reccomended method for building STAT is to use the [Spack](https://spack.readthedocs.io) package manager with the following commands:
```
git clone https://github.com/spack/spack.git
cd spack
./bin/spack install stat
```

For STAT 4.0.1 and beyond, the spack build of STAT requires Python 3 and will require the activation of the py-xdot package:
```
./bin/spack activate py-xdot
```

For versions up to and including STAT 4.0.0, the spack build of STAT requires Python 2 and requires the activation of the py-pygtk and py-enum34 packages:
```
./bin/spack activate py-pygtk
./bin/spack activate py-enum34
```


### SUPPORTED PLATFORMS
STAT is able to run on any machine where all of the dependent libraries run.  This currently includes:
* IBM BlueGene/L
* IBM BlueGene/P
* IBM BlueGene/Q
* x86-based architectures with SLURM
* x86-based architectures with OpenMPI
* Cray XT
* Cray XE
* Cray XK

### CONTACT
Please direct any questions to [Gregory Lee](mailto:lee218@llnl.gov).
