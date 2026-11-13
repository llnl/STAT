# PyStack: DataLoader children and application threads

This synthetic PyTorch workload consumes generated tensors through real DataLoader processes. An optional group of named
application threads provides a separate thread-count scenario.

We check parent `main()` stacks, loader ancestry through `run()` → `_worker_loop()`, and application-thread stacks through
`run()` → `application_thread()` when enabled.

The sampled execution state could look like this:

```text
Example: 8 parents, 2 loader workers and 2 application threads per parent

Parents: pcount=8
main

Loader children: pcount=16
run
└── _worker_loop

Application threads: tcount=16 (threads variant)
run
└── application_thread

Parent-plus-loader total: pcount=24
```

## Checks

The shared default runs **4 application ranks on one node**. Each parent has 2 loader workers (8 children); the threads
variant adds 2 application threads per parent (8 thread stacks). JSON counts follow the selected profile's rank count.

The runtime check validates the collected output and snapshots against:

### Parents and loaders

```json checks:default
{
  "dataloader_workers": true,
  "stdout": {
    "counts": [
      {"pattern": "multiprocess-done rank=\\d+/{num_ranks}(?![0-9])", "count": "num_ranks"}
    ]
  },
  "pystack": {
    "each_sample": [
      {
        "paths_any": [
          ["process:*", "process:*", "*", "run@process.py:*", "_worker_loop@worker.py:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks * num_workers",
        "process_depth": 2
      },
      {
        "paths_any": [
          ["/", "process:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks * num_processes"
      },
      {
        "paths_any": [
          ["process:*", "process:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks * num_workers",
        "process_depth": 2,
        "merged": true
      },
      {
        "paths_any": [
          ["/", "process:*", "<module>@test.py:*", "main@test.py:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks",
        "process_depth": 1
      }
    ],
    "required": ["test.py", "load_sample"],
    "across_samples": [
      {
        "paths_any": [
          ["/", "process:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks * num_processes"
      }
    ]
  }
}
```

### Extra application threads

```json checks:threads
{
  "dataloader_workers": true,
  "stdout": {
    "counts": [
      {"pattern": "multiprocess-done rank=\\d+/{num_ranks}(?![0-9])", "count": "num_ranks"},
      {"pattern": "APPLICATION_THREADS rank=\\d+ count={num_threads}(?![0-9])", "count": "num_ranks"}
    ]
  },
  "pystack": {
    "required": ["load_sample"],
    "each_sample": [
      {
        "paths_any": [
          ["process:*", "process:*", "*", "run@process.py:*", "_worker_loop@worker.py:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks * num_workers",
        "process_depth": 2
      },
      {
        "paths_any": [
          ["/", "process:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks * num_processes"
      },
      {
        "paths_any": [
          ["process:*", "thread:*", "*", "run@threading.py:*", "application_thread@test.py:*"]
        ],
        "attribute": "tcount",
        "count": "num_ranks * num_threads",
        "process_depth": 1
      },
      {
        "paths_any": [
          ["process:*", "process:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks * num_workers",
        "process_depth": 2,
        "merged": true
      },
      {
        "paths_any": [
          ["/", "process:*", "thread:*", "<module>@test.py:*", "main@test.py:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks",
        "process_depth": 1
      }
    ],
    "across_samples": [
      {
        "paths_any": [
          ["/", "process:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks * num_processes"
      }
    ]
  }
}
```

### Parent only

```json checks:parent-only
{
  "dataloader_workers": true,
  "stdout": {
    "counts": [
      {"pattern": "multiprocess-done rank=\\d+/{num_ranks}(?![0-9])", "count": "num_ranks"}
    ]
  },
  "pystack": {
    "each_sample": [
      {
        "paths_any": [
          ["<module>@test.py:*", "main@test.py:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks",
        "source": "ranks"
      }
    ],
    "forbidden": ["_worker_loop", "load_sample", "process:"]
  }
}
```

This means:

- Worker startup records establish rank, hostname, PID, parent PID and worker IDs independently of STAT’s discovery.
- Child paths must remain under their parent at process depth two, and equivalent loader branches must merge.
- The threads set adds `num_ranks * num_threads` application stacks; parent-only excludes worker/process labels while still
  checking real loader ownership from stdout.
- Parent-plus-child root totals are checked in individual snapshots and merged history.

[Check syntax](../TEST-CHECKS-FORMAT.md) explains the fields and wildcard matching.
