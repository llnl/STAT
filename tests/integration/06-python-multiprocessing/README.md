# PyStack: Python multiprocessing

This synthetic Python workload starts one loader child and one checkpoint child per parent. Their separate named functions
simulate preparation and checkpoint work without an external training dataset. It runs on host Python without a container
or PyTorch dependency; PyStack samples the parent and spawned children.

We check the parent entry plus two distinct child paths: `loader_worker()` → `prepare_batch()` and `checkpoint_worker()` →
`write_checkpoint()`.

The sampled execution state could look like this:

```text
Example: 6 parents, one child per role per parent

Parents: pcount=6
main

Child roles: pcount=6 each
run
├── loader_worker
│   └── prepare_batch
└── checkpoint_worker
    └── write_checkpoint
```

## Checks

The shared default runs **3 application ranks on one node**. Each parent has one loader and one checkpoint child, giving
3 processes in each checked child role. JSON counts follow the selected profile's rank count.

The runtime check validates the collected output and snapshots against:

```json checks:default
{
  "stdout": {
    "counts": [
      {"pattern": "multiprocess-done rank=\\d+/{num_ranks}(?![0-9])", "count": "num_ranks"},
      {"pattern": "loader-start rank=\\d+\\b", "count": "num_ranks"},
      {"pattern": "checkpoint-start rank=\\d+\\b", "count": "num_ranks"},
      {"pattern": "loader-finish rank=\\d+\\b", "count": "num_ranks"},
      {"pattern": "checkpoint-finish rank=\\d+\\b", "count": "num_ranks"}
    ]
  },
  "pystack": {
    "each_sample": [
      {
        "paths_any": [
          ["/", "process:*", "<module>@test.py:*", "main@test.py:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks",
        "process_depth": 1
      },
      {
        "paths_any": [
          ["process:*", "process:*", "*", "run@process.py:*", "loader_worker@test.py:*", "prepare_batch@test.py:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks",
        "process_depth": 2
      },
      {
        "paths_any": [
          ["process:*", "process:*", "*", "run@process.py:*", "checkpoint_worker@test.py:*", "write_checkpoint@test.py:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks",
        "process_depth": 2
      }
    ]
  }
}
```

This means:

- Each checked child role contributes one process per rank beneath its parent at process depth two.
- Loader and checkpoint START/FINISH messages are counted separately, so one role cannot conceal the other failing.
- Parent completion is printed only after the child processes have been joined successfully.

[Check syntax](../TEST-CHECKS-FORMAT.md) explains the fields and wildcard matching.
