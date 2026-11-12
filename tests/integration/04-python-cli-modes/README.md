# PyStack: CLI launch and direct attach

A small Python workload stays in `main()` → `sampled_wait()` and prints a completion record after sampling.
It needs only Python. Both variants collect Python-only stacks without child or thread sampling:

- `launch`: `stat-cl -K -C <launcher> <python> ...` creates the MPI or Slurm job through LaunchMON. This checks that
  selecting PyStack preserves launch mode. `ATTACH_SETTLE_SECONDS` gives Python time to start before the first sample.
- `serial`: start one Python process, then use `stat-cl -K -I <pid>` without a job launcher. Under Slurm this process runs
  on the batch node; the configured `PYTHON` must exist there. The surrounding allocation does not set its rank count.

The container profile runs both variants directly in Docker and, with `--enroot`, inside the bundled Enroot image.
The serial variant starts Enroot locally, sharing the target PID namespace with STAT. Slurm's `SRUN_OPTIONS` apply only
to the launch variant; serial attachment does not use `srun` or its Pyxis options.

## Checks

The launch variant defaults to two ranks; serial always has one. Every snapshot must contain the connected application
path for every expected rank. Completion records establish that the workload continued after STAT detached.
The same expectations apply to both variants; `--plain` checks workload completion without invoking STAT.

```json checks:default
{
  "stdout": {
    "forbidden": ["Traceback", "OpenMPI Python rank registration failed"],
    "counts": [
      {"pattern": "PYTHON_READY rank=\\d+(?![0-9])", "count": "num_ranks"},
      {"pattern": "PYTHON_DONE rank=\\d+(?![0-9])", "count": "num_ranks"}
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
