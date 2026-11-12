# PyStack: mpi4py ranks

This small MPI application gets its rank from `MPI.COMM_WORLD`, waits in `main()` → `sampled_wait()`, then checks an
all-reduce before exiting. It requires mpi4py built for, or compatible with, the selected MPI runtime; no PyTorch is needed.

- `launch`: `stat-cl -K -C mpirun ...` starts the MPI job through LaunchMON.
- `attach`: start `mpirun ...`, wait for all ranks to initialize MPI, then attach with `stat-cl -K <launcher-pid>`.


## Checks

Both variants default to two ranks. Every sample must contain the application path for all expected ranks.

```json checks:default
{
  "stdout": {
    "forbidden": ["Traceback", "AssertionError"],
    "counts": [
      {"pattern": "MPI_READY rank=\\d+/\\d+", "count": "num_ranks"},
      {"pattern": "MPI_DONE rank=\\d+/\\d+ sum=\\d+", "count": "num_ranks"}
    ]
  },
  "pystack": {
    "each_sample": [
      {
        "paths_any": [["<module>@test.py:*", "main@test.py:*", "sampled_wait@test.py:*"]],
        "attribute": "pcount",
        "source": "ranks",
        "count": "num_ranks"
      }
    ]
  }
}
```

[Check syntax](../TEST-CHECKS-FORMAT.md) explains the fields and wildcard matching.
