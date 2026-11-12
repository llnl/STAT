# PyStack: torchrun workers

This test trains a small two-layer PyTorch model with DDP on synthetic data. Torchrun starts workers on each node. The
default uses CPU/Gloo, so multiple workers can run without GPUs.

We check that PyStack reaches the workers beneath torchrun and captures their application stacks. At sampling time, workers
may be in training or all-reduce.

The shared runner starts one `torchrun` launcher per node; this test's `run.sh` supplies its workload arguments.

The execution state of four application ranks could look like this:

```text
Example: 4 application ranks

Application paths: pcount=4, ranks=0–3
*torchrun
└── process:cmd:test
    └── <module> → main
        ├── distributed_train_step
        └── * → all_reduce
```

## Checks

The runtime check validates the collected output and snapshots against:

```json checks:default
{
  "stdout": {
    "forbidden": ["RendezvousConnectionError"],
    "counts": [
      {"pattern": "TORCHRUN_DONE rank=\\d+\\b steps=[1-9][0-9]*", "count": "num_ranks"}
    ]
  },
  "pystack": {
    "each_sample": [
      {
        "paths_any": [["*torchrun", "process:cmd:test", "<module>@test.py:*", "main@test.py:*"]],
        "attribute": "pcount",
        "count": "num_ranks"
      }
    ],
    "across_samples": [
      {
        "paths_any": [
          ["*torchrun", "process:cmd:test", "<module>@test.py:109", "main@test.py:86", "distributed_train_step@test.py:*"],
          ["*torchrun", "process:cmd:test", "<module>@test.py:109", "main@test.py:95", "*", "all_reduce@distributed_c10d.py:*"]
        ],
        "attribute": "pcount",
        "source": "ranks",
        "count": "num_ranks"
      }
    ]
  }
}
```

This means:

- Stdout must contain `num_ranks` completion messages of the form `TORCHRUN_DONE rank=... steps=...`, with a positive step
  count, and no `RendezvousConnectionError`.
- Every snapshot must cover all ranks through `main()`; merged samples must cover all ranks in training or all-reduce.
- `pcount` counts workers; history checks count distinct rank IDs across paths.
- Lines 109, 86 and 95 identify this example's entry, training and all-reduce callsites. Inner frames allow different
  positive source lines.

[Check syntax](../TEST-CHECKS-FORMAT.md) explains the fields and wildcard matching.
