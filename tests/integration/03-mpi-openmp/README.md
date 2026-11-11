# Dyninst: MPI main threads and OpenMP workers

This synthetic MPI/OpenMP example repeats a small workload inside a timed parallel region. We check each rank’s main-thread
path and, in the threads scenario, all application threads across the OpenMP teams.

The sampled execution state could look like this:

```text
Example: 8 MPI ranks, 4 OpenMP threads per rank

Rank main threads: pcount=8; tcount=8 with thread sampling
_start → * → main
└── run_rank_workload
    └── * → run_thread_work → process_epoch → compute_block → sampled_delay

Full OpenMP team: tcount=32
run_thread_work
└── process_epoch
    └── compute_block
        └── sampled_delay
```

## Checks

The container profile runs **4 MPI ranks with 4 OpenMP threads per rank**.

The runtime check validates the collected output and snapshots against:

### Processes

```json checks:processes
{
  "stdout": {
    "required": ["MARKER MPI\\+OPENMP FINISH"],
    "counts": [
      {"pattern": "MPI_OPENMP_DONE rank=\\d+\\b threads={num_threads}(?![0-9]) iterations=[1-9][0-9]*", "count": "num_ranks"},
      {"pattern": "MPI_OPENMP_READY rank=\\d+/{num_ranks}(?![0-9]) threads={num_threads}(?![0-9])", "count": "num_ranks"}
    ]
  },
  "dyninst": {
    "each_sample": [
      {
        "paths_any": [
          [
            "_start",
            "*",
            "main@mpi_openmp.cpp:*",
            "*run_rank_workload@mpi_openmp.cpp:*",
            "*",
            "*run_thread_work@mpi_openmp.cpp:*",
            "*process_epoch@mpi_openmp.cpp:*",
            "*compute_block@mpi_openmp.cpp:*",
            "*sampled_delay@mpi_openmp.cpp:*"
          ]
        ],
        "attribute": "pcount",
        "count": "num_ranks",
        "source": "ranks"
      }
    ],
    "forbidden": ["\\btcount=\"[0-9]+\""]
  }
}
```

### Threads

```json checks:threads
{
  "stdout": {
    "required": ["MARKER MPI\\+OPENMP FINISH"],
    "counts": [
      {"pattern": "MPI_OPENMP_DONE rank=\\d+\\b threads={num_threads}(?![0-9]) iterations=[1-9][0-9]*", "count": "num_ranks"},
      {"pattern": "MPI_OPENMP_READY rank=\\d+/{num_ranks}(?![0-9]) threads={num_threads}(?![0-9])", "count": "num_ranks"}
    ]
  },
  "dyninst": {
    "each_sample": [
      {
        "paths_any": [
          [
            "_start",
            "*",
            "main@mpi_openmp.cpp:*",
            "*run_rank_workload@mpi_openmp.cpp:*",
            "*",
            "*run_thread_work@mpi_openmp.cpp:*",
            "*process_epoch@mpi_openmp.cpp:*",
            "*compute_block@mpi_openmp.cpp:*",
            "*sampled_delay@mpi_openmp.cpp:*"
          ]
        ],
        "attribute": "pcount",
        "count": "num_ranks",
        "source": "ranks"
      },
      {
        "paths_any": [
          [
            "_start",
            "*",
            "main@mpi_openmp.cpp:*",
            "*run_rank_workload@mpi_openmp.cpp:*",
            "*",
            "*run_thread_work@mpi_openmp.cpp:*",
            "*process_epoch@mpi_openmp.cpp:*",
            "*compute_block@mpi_openmp.cpp:*",
            "*sampled_delay@mpi_openmp.cpp:*"
          ]
        ],
        "attribute": "tcount",
        "count": "num_ranks"
      },
      {
        "paths_any": [
          [
            "*run_thread_work@mpi_openmp.cpp:*",
            "*process_epoch@mpi_openmp.cpp:*",
            "*compute_block@mpi_openmp.cpp:*",
            "*sampled_delay@mpi_openmp.cpp:*"
          ]
        ],
        "attribute": "tcount",
        "count": "num_ranks * num_threads"
      }
    ]
  }
}
```

This means:

- The `threads` set requires one main-thread stack per rank and `num_ranks * num_threads` application-thread stacks.
- The `processes` set checks one main-thread stack per MPI rank, with process counts and no thread counts.
- Both sets check the expected application call chain and source lines; the `threads` set also checks the total
  application-thread count across all ranks.

[Check syntax](../TEST-CHECKS-FORMAT.md) explains the fields and wildcard matching.
