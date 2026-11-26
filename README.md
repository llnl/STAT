# <img src="STATlogo.gif" alt="STAT Logo"/> STAT: Stack Trace Analysis Tool

[![STAT CI](https://github.com/LLNL/STAT/actions/workflows/main.yml/badge.svg?branch=develop)](https://github.com/LLNL/STAT/actions/workflows/main.yml?query=branch%3Adevelop)
[![User Guide](https://github.com/LLNL/STAT/actions/workflows/docs.yml/badge.svg?branch=develop)](https://github.com/LLNL/STAT/actions/workflows/docs.yml?query=branch%3Adevelop)

[Read the STAT User Guide](https://software.llnl.gov/STAT/)


### OVERVIEW

The Stack Trace Analysis Tool (STAT) is a highly scalable, lightweight tool that gathers and merges stack traces from all of the processes of a parallel application to form call graph prefix trees.

STAT generates two prefix trees termed 2D-trace-space and 3D-trace-space-time. The 2D-trace-space prefix tree is a merge of a single stack trace from each task in the parallel application. The 3D-trace-space-time prefix tree is a merge of several stack traces from each task gathered over time. The latter provides insight into whether tasks are making progress or are in a hang state (livelock, deadlock, infiite loop, etc.). The call graph prefix trees also identify processes equivalence classes, processes exhibitin similar behavior with respect to their call paths. A representative task from each equivalence class can then be fed into a full-featured debugger for root cause analysis at a manageable scale.

STAT's source code also includes STATBench, a tool to emulate STAT. STATBench enables the benchmarking of STAT on arbitrary machine architectures and applications by fully utilizing parallel resources and generating artificial stack traces. STAT has demonstrated scalability over 1,000,000 MPI tasks and its logarithmic scaling characteristics position it well for even larger systems.

### SUPPORTED PLATFORMS

STAT has been ported to several platforms, including Linux clusters, IBM CORAL systems (IBM Power CPUs + NVIDIA GPUs), multiple generations of Blue Gene machines, and Cray systems. Today, STAT can be installed on a wide range of Linux systems, from HPC clusters used for scientific computing to large-scale AI training clusters, including systems with NVIDIA CPUs and GPUs.

### SUPPORTED APPLICATIONS

STAT works with serial and parallel applications across programming models. It supports applications written in C, C++, Fortran, Python, or a mix of these languages. In scientific workloads, STAT can sample MPI ranks and their OpenMP or other native threads. In distributed model training, it can sample Python and C/C++ stacks from [PyTorch](https://docs.pytorch.org/docs/stable/distributed.html) with communication backends such as MPI, Gloo or the NVIDIA Collective Communications Library (NCCL).

The sampling backend determines what appears in a stack trace. By default, STAT uses [Dyninst](https://github.com/dyninst/dyninst) to collect native stack
traces, including application and library frames. With [PyStack](https://github.com/bloomberg/pystack), STAT can capture Python and native frames in the same
trace, showing which Python call led into a native library. With a configured [`cuda-gdb`](https://docs.nvidia.com/cuda/cuda-gdb/) backend, STAT can collect CUDA
device callstacks.

### INSTALLING STAT

#### Cluster Installation

The full STAT installation lives on the Linux cluster, where it collects and merges stack traces from running applications. [Spack](https://spack.io/) has been the primary way to build STAT and its dependencies, and it can reuse packages or a build cache the site already provides. A fresh Spack checkout and installation looks like below. Note that a fresh Spack installation may need to build many dependencies, which can take considerable time. If you have an existing Spack installation or a build cache, use it to speed up the installation.

```sh
git clone https://github.com/spack/spack.git
cd spack
./bin/spack install stat
```

The `develop` branch also includes a [Pixi](https://pixi.sh/)-based installer. It uses packages from conda-forge to prepare the build environment, then builds STAT from a local checkout. This is useful on clusters where Spack is difficult to set up or where building all the dependencies from source would take too long. The installer is an early addition to the `develop` branch.

```sh
bash scripts/install_stat_with_pixi.sh --install-prefix /path/to/stat-install --jobs 6
```

See the [Pixi installer usage](scripts/install_stat_with_pixi.sh) for more options.

#### Laptop/Desktop Installation (Remote Client)

The standalone client on the `develop` branch brings the GUI to a Linux or macOS laptop or desktop. It can open saved call graphs locally or connect over SSH to the cluster, where STAT collects the samples.

On macOS, Homebrew provides Python and the GTK dependencies. Pipx installs the client from `develop` and manages its environment using Homebrew's Python and GTK bindings.

```sh
brew install python pygobject3 gtk+3 graphviz pipx
pipx install --python "$(brew --prefix)/bin/python3" --system-site-packages \
  "git+https://github.com/LLNL/STAT.git@develop#subdirectory=scripts/gui"
```

For a first-time pipx setup, `pipx ensurepath` makes the installed commands available in new shells. See the [GUI Installation Guide](scripts/gui/README.md) for platform-specific setup and the [Remote Connect Guide](doc/remote-connect.md) for SSH and job attachment details.

### LICENSE

STAT is released under the Berkeley Software Distribution (BSD) license. Please see [LICENSE](https://github.com/LLNL/STAT/blob/master/LICENSE) for usage terms.

### CONTACT

Please direct any questions to [Gregory Lee](mailto:lee218@llnl.gov).
