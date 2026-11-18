(pystack)=
# PyStack Sampling

## Understanding Python application stacks

When a distributed Python application hangs or makes slow progress, a native stack may show an interpreter or
library routine without revealing the Python call that reached it. Work may also run in child processes or library
worker threads, so inspecting only the main process can leave part of the application unexplained.

STAT's [PyStack](https://github.com/bloomberg/pystack) backend collects Python call stacks and
merges matching paths across ranks. This lets you compare where ranks are executing and identify ranks that follow a
different path.

## Build and backend selection

For installation instructions, see
[scripts/install_stat_with_pixi.sh](https://github.com/llnl/STAT/blob/develop/scripts/install_stat_with_pixi.sh)
to build all dependencies and install STAT with Pixi environment.

Install PyStack first, then pass the path to its `pystack` executable, not its installation directory:

```text
./configure <other STAT options> --with-pystack=/path/to/bin/pystack
```

- PyStack support is off by default. Omitting the option or using `--without-pystack` leaves it
  disabled.
- `--with-pystack=/path/to/bin/pystack` enables support after checking that the file exists and is
  executable.
- Dyninst remains required and enabled; enabling PyStack adds another backend rather than replacing Dyninst.

Building with PyStack does not change the default sampling backend. Select it with `stat-cl -K` or
`stat-cl --pystack`. The configured executable must be available where STAT's sampling daemons
run; the configure-time executable check does not validate remote runtime availability.

## PyStack versus child-process sampling

Select `--pystack` to sample Python application processes. Add `--with-children`
to include their descendants. Child-process collection in this version requires PyStack; the child option alone
does not select a backend or enable Dyninst child sampling.

For a torchrun launcher in ML applications, STAT discovers and samples spawned rank processes without `--with-children`.
When workers expose consistent `RANK` and `WORLD_SIZE` environment variables,
STAT identifies each worker's global rank and the total worker count. Adding `--with-children`
includes each worker's descendants (e.g. data loaders) under that worker's rank.

For example, when attaching to an existing MPI launcher:

```text
stat-cl --pystack --with-children <launcher-pid>
```

Child-process sampling includes descendants such as grandchildren; it does not automatically enable thread sampling.
Add `-w` separately. Filesystem helpers and their subtrees are excluded from sampling and pause control;
confirmed exited descendants are omitted.

STAT pauses application processes and their descendants while collecting stacks. An explicit Resume continues
stopped application processes. If a process cannot be resumed, STAT reports the failure and keeps track of it
so a later Resume or Detach can retry.

## Python and native frames

`--native` requires the PyStack backend and can appear before or after `--pystack`.

| Frame detail | Without -w | With -w |
| --- | --- | --- |
| Python only (default) | Primary thread's Python stack | All Python threads' Python stacks |
| Python + C/C++ (`--native`) | Primary thread's mixed stack | All thread stacks, including native-only C/C++ workers |

```text
stat-cl --pystack --native --with-children -w <launcher-pid>
```

## Reading rank and process counts

An edge's rank list identifies the ranks contributing to that call path. When more processes than ranks contribute,
STAT appends `xN`, where `N` is the total process count on that path, not a
per-rank multiplier or a thread count. Rank selection still uses the rank IDs in brackets. Accumulated (3D) graphs
count distinct process IDs on each daemon across samples, so repeatedly sampling a process does not increase its count.

| Label | Interpretation |
| --- | --- |
| `[0-3]` | Ranks 0 through 3. |
| `[0]x3` | Rank 0, three contributing processes. |
| `[0,2]x8` | Ranks 0 and 2, eight processes in total; their per-rank distribution is not shown. |

Formatted labels may also prefix the rank count: `2:[0,2]x8` means two ranks and eight processes.
