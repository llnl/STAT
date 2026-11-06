# Dyninst: rank-dependent recursion

This original STAT example is a synthetic MPI workload where each rank sleeps at a different recursion depth. It makes
missing frames and incorrect rank-to-stack association visible.

We check the common `main()` → `do_SendOrStall()` → `recurse()` path for every rank, and the exact recursive depth for
each rank in every sample, using the same JSON assertion.

The sampled execution state could look like this:

```text
Example: 6 MPI ranks

Common prefix: pcount=6, ranks=0–5
main
└── do_SendOrStall
    └── recurse                      rank 0 stops at depth 1
        └── recurse                  rank 1 stops at depth 2
            └── recurse              rank 2 stops at depth 3
                └── recurse          rank 3 stops at depth 4
                    └── recurse      rank 4 stops at depth 5
                        └── recurse  rank 5 stops at depth 6
```

## Checks

The container profile runs **4 MPI ranks**. JSON counts follow the actual launch configuration.

The runtime check validates the collected output and snapshots against:

```json checks:default
{
  "stdout": {},
  "dyninst": {
    "each_sample": [
      {
        "paths_any": [
          ["main@rank_test.c:*", "do_SendOrStall@rank_test.c:*", "recurse@rank_test.c:*"]
        ],
        "attribute": "pcount",
        "count": "num_ranks",
        "source": "ranks",
        "frame_count": {
          "frame": "recurse@rank_test.c:*",
          "count": "rank % 10 + 1"
        }
      }
    ]
  }
}
```

This means:

- The expected number of recursive frames is `rank % 10 + 1`, including the base frame.
- Every recursive frame needs `rank_test.c` source information and a positive line number.

[Check syntax](../TEST-CHECKS-FORMAT.md) explains the fields and wildcard matching.
