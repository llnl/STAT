# Dyninst: stalled MPI ring sender

This original STAT example is a synthetic MPI ring with rank 1 delaying its send. Other ranks can be waiting for requests or
already at the barrier when sampled.

We check rank 1 in `main()` → `do_SendOrStall()` separately from its peers in `main()` → `MPI_Waitall()` or `MPI_Barrier()`,
including their PMPI equivalents.

The sampled execution state could look like this:

```text
Example: 4 MPI ranks

Stalled sender: pcount=1, rank=1
main
└── do_SendOrStall

Accepted peer paths: combined pcount=3, ranks=0,2,3
main
├── MPI/PMPI_Waitall
└── MPI/PMPI_Barrier
```

## Checks

The container profile runs **2 MPI ranks**. JSON counts follow the actual launch configuration.

The runtime check validates the collected output and snapshots against:

```json checks:default
{
  "stdout": {
    "required": ["MPI task 1 of {num_ranks} proceeding"]
  },
  "dyninst": {
    "each_sample": [
      {
        "paths_any": [
          ["main@mpi_ringtopo.c:*", "do_SendOrStall@mpi_ringtopo.c:*"]
        ],
        "attribute": "pcount",
        "count": 1,
        "source": "ranks",
        "ranks": [1]
      },
      {
        "paths_any": [
          ["main@mpi_ringtopo.c:*", "MPI_Waitall"],
          ["main@mpi_ringtopo.c:*", "PMPI_Waitall"],
          ["main@mpi_ringtopo.c:*", "MPI_Barrier"],
          ["main@mpi_ringtopo.c:*", "PMPI_Barrier"]
        ],
        "attribute": "pcount",
        "count": "num_ranks - 1",
        "source": "ranks",
        "exclude_ranks": [1]
      }
    ]
  }
}
```

This means:

- The stalled path belongs to rank 1 only; the other assertion excludes rank 1 and expects `num_ranks - 1` processes.
- Waitall and Barrier shows ranks waiting for sender/stall rank.

[Check syntax](../TEST-CHECKS-FORMAT.md) explains the fields and wildcard matching.
