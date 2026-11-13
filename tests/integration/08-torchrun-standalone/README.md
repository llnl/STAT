# PyStack: standalone torchrun

Start one local `torchrun --standalone` launcher with four CPU/Gloo workers, then attach to its PID with
`stat-cl -K -I`. No `mpirun` or `srun` starts the application. Inherited MPI/Slurm launcher-size variables are cleared
so the test checks rank discovery from standalone torchrun rather than accidentally relying on the surrounding job.

The workload waits in `main()` → `sampled_wait()`, then verifies an all-reduce and exits. It needs PyTorch but no GPU.
`TEST_RANKS_08` sets the worker count; the default is four. Under Slurm, all workers run on the batch node; its `TORCHRUN`
executable must be available there. `SRUN_OPTIONS` do not apply. With `--enroot`, the launcher and workers run inside
one local Enroot container, using the shared PID namespace for attachment.

The adapter uses workers' own rank metadata when their distinct ranks cover the complete worker world, even without
an MPI/Slurm launcher count. Incomplete or conflicting worker metadata must not silently collapse into launcher rank 0.

## Checks

Every sample must contain the connected application path for all worker ranks, even though STAT attaches to only one
launcher PID. Counting processes under launcher rank 0 is insufficient: the graph must identify worker ranks 0–3.
Completion records check that all workers continue and complete the collective after STAT detaches.
`--plain` checks workload completion without sampling. Test 08 runs in both Docker and Enroot CI selections.

```json checks:default
{
  "stdout": {
    "forbidden": ["Traceback", "AssertionError"],
    "counts": [
      {"pattern": "TORCHRUN_READY rank=\\d+/\\d+", "count": "num_ranks"},
      {"pattern": "TORCHRUN_DONE rank=\\d+/\\d+ sum=\\d+", "count": "num_ranks"}
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
