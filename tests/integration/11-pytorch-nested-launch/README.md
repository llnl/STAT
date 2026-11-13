# PyStack: nested launcher ancestry

This synthetic DDP workload trains a small linear model on random inputs. Its launcher variants place the application
directly under Slurm, under torchrun, or beneath the nested shell chain used by generated rank scripts.

We check that STAT follows the selected launcher ancestry to the application’s `<module>` → `main()` → `training_step()` path
for every worker.

The sampled execution state could look like this:

```text
Example: 8 application workers, ranks=0–7

Nested launcher: pcount=8
bash → bash
└── test.py → main → training_step

Nested torchrun launcher: pcount=8
bash → bash
└── torchrun
    └── test.py → main → training_step
```

## Checks

The shared default runs **4 application ranks on one node**. The torchrun variants start one launcher and four workers;
the direct and nested variants start four rank processes. JSON counts follow the selected profile's worker count.

The runtime check validates the collected output and snapshots against:

### Nested

```json checks:nested
{
  "stdout": {
    "counts": [
      {"pattern": "launcher-done mode=nested rank=\\d+/{num_ranks}(?![0-9])", "count": "num_ranks"}
    ]
  },
  "pystack": {
    "each_sample": [
      {
        "paths_any": [
          [
            "/",
            "process:cmd:bash",
            "process:cmd:bash",
            "process:cmd:test",
            "<module>@test.py:*",
            "main@test.py:*",
            "training_step@test.py:*"
          ]
        ],
        "attribute": "pcount",
        "count": "num_ranks"
      }
    ]
  }
}
```

### Nested torchrun

```json checks:nested-torchrun
{
  "stdout": {
    "counts": [
      {"pattern": "launcher-done mode=nested-torchrun rank=\\d+/{num_ranks}(?![0-9])", "count": "num_ranks"}
    ]
  },
  "pystack": {
    "each_sample": [
      {
        "paths_any": [
          [
            "/",
            "process:cmd:bash",
            "process:cmd:bash",
            "process:cmd:torchrun",
            "process:cmd:test",
            "<module>@test.py:*",
            "main@test.py:*",
            "training_step@test.py:*"
          ]
        ],
        "attribute": "pcount",
        "count": "num_ranks"
      }
    ]
  }
}
```

### Direct and torchrun

```json checks:application
{
  "stdout": {
    "counts": [
      {"pattern": "launcher-done mode=[^ ]+ rank=\\d+/{num_ranks}(?![0-9])", "count": "num_ranks"}
    ]
  },
  "pystack": {
    "each_sample": [
      {
        "paths_any": [
          ["process:cmd:test", "<module>@test.py:*", "main@test.py:*", "training_step@test.py:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks"
      }
    ]
  }
}
```

This means:

- Nested checks require both bash process nodes; nested-torchrun also requires the torchrun process between the shells and
  workers.
- The direct and torchrun variants share the application block.
- Counts refer to application workers, even when one Slurm task per node launches several workers.
- Completion records must report the selected launcher mode and cover the expected number of workers.

[Check syntax](../TEST-CHECKS-FORMAT.md) explains the fields and wildcard matching.
