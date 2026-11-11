# JSON checks format

Each integration test keeps its expected output and sampled stacks in named JSON blocks in its README.
The shared validator reads these blocks and checks the application log and STAT graphs. Below are examples,
followed by explanations of the supported fields, call-path patterns and counts.

A runnable block starts with the fence `json checks:default`. A `default` variant in `run.sh` selects that block. Another
name, such as `json checks:threads`, defines a separate check set. Plain `json` fences are examples, not runnable checks.

## A complete example

Assume the workload runs with **eight MPI ranks and four threads per rank**. The runner passes
`--world-size 8 --num-threads 4` to the validator, so `num_ranks` means **8** and `num_threads` means **4**
in the block below.

```json checks:default
{
  "stdout": {
    "required": ["MARKER MPI\\+OPENMP FINISH"],
    "counts": [
      {"pattern": "MPI_OPENMP_DONE rank=\\d+ threads={num_threads}\\b", "count": "num_ranks"}
    ]
  },
  "dyninst": {
    "each_sample": [
      {
        "paths_any": [["main@mpi_openmp.cpp:*", "*", "*sampled_delay@mpi_openmp.cpp:*"]],
        "attribute": "pcount",
        "source": "ranks",
        "count": "num_ranks"
      }
    ]
  }
}
```

With those launch settings, this block means:

- `stdout.required`: the job log contains `MARKER MPI+OPENMP FINISH`.
- `stdout.counts` uses `count: num_ranks` to require eight DONE messages. Its `threads={num_threads}` pattern requires
  each matching message to report four threads.
- `dyninst.each_sample` applies the listed check to every snapshot. Its `paths_any` entry requires the connected
  `main → … → sampled_delay` path, allowing intermediate frames at `*`.
- That same check uses `attribute: pcount`, `source: ranks` and `count: num_ranks` to require eight distinct ranks
  on the path. With eight launched ranks and no rank subset specified, these must be rank IDs 0–7.

`stdout` describes log checks. `dyninst` and `pystack` describe stack checks for their respective backends.

## Check lists and alternatives

```json
"paths_any": [
  ["main@test.py:*", "distributed_train_step@test.py:*"],
  ["main@test.py:*", "*", "all_reduce@distributed_c10d.py:*"]
]
```

- Each inner list is one ordered call chain. This example accepts the training-step path **or** the all-reduce path.
- All alternatives contribute to one combined count; different ranks can match different paths. Neither alternative has to
  appear on its own.
- Separate objects inside `each_sample` or `across_samples` are separate required checks. To require both paths, they belong
  in separate objects.

## `each_sample` and `across_samples`

| Key | How its check list is applied |
| --- | --- |
| `each_sample` | Every listed check runs on every individual sample. |
| `across_samples` | Every listed check runs on the merged result of the collected samples. |

For `source: ranks`, an `across_samples` check requires every expected rank to match one of its listed paths in at least one
sample. The ranks can match different paths in different samples.

For one rank with accepted paths A or B:

| Sample | Captured path |
| --- | --- |
| 1 | A |
| 2 | C |

The `each_sample` check fails because C is not accepted. The `across_samples` check passes because A appeared once. For rank-
only coverage, repeating an identical check in both sections adds nothing once every sample passes.


## Frame and path syntax

| Example | Interpretation |
| --- | --- |
| `main@test.py:*` | Function `main`, source basename `test.py`, any positive line number. |
| `main@test.py:73` | The same frame at exactly line 73. |
| `*train*@*test.py:*` | Wildcards within the function and filename. |
| A standalone `"*"` | Zero or more whole frames between the adjacent named entries. |
| `process:*` or `MPI_Waitall` | Function/label matching without source information. |
| `native:*@libtorch*.so` | Native function and library basename; a positive source line is not required. |


## Process, thread and rank counts

| Field | Interpretation |
| --- | --- |
| `attribute: "pcount"` | Compare process counts on the matched paths’ terminal edges. |
| `attribute: "tcount"` | Compare thread counts on those edges. |
| `source: "ranks"` | Use distinct rank IDs as pcount instead of raw edge counts. Only valid with `pcount`. |
| `ranks: [1]` | The matched rank set must be exactly rank 1. This is an expectation, not a filter. |
| `exclude_ranks: [1]` | Remove rank 1 from the expected set. |
| `process_depth: 2` | Each matched path contains exactly two `process:*` levels. |
| `merged: true` | All equivalent matches form one selected edge. |

Without `source: ranks`, counts come from graph edges. Without `ranks` or `exclude_ranks`, every application rank is
expected. Both the total and the rank membership must match.

For example, four parents with two loader children each have eight children but only four distinct rank IDs. The child check
therefore uses edge `pcount=8`; `source: ranks` would count four.

| `count` value | Interpretation |
| --- | --- |
| `"num_ranks"` | The configured MPI rank count. |
| `"num_ranks - 1"` | One fewer than the application rank count. |
| `"num_ranks * num_threads"` | Total configured application threads. |
| `"num_ranks * num_workers"` | Total configured DataLoader children. |
| `"num_ranks * num_processes"` | Parents plus loaders: ranks × (workers + 1). |
| A positive integer | A fixed total, such as `1` for a single stalled rank. |


## Exact frame counts

```json
"frame_count": {
  "frame": "recurse@rank_test.c:*",
  "count": "rank % 10 + 1"
}
```

This optional field sits inside a path/count check, beside `paths_any`, `attribute` and `count`. It counts recursive frames
on each complete stack beneath the matched path, including frames after the matched prefix.

- `frame` names the function and its required source/line pattern.
- `count` is the exact number of those frames.


## Stdout patterns and worker ownership

| Field under `stdout` | Interpretation |
| --- | --- |
| `required: [pattern]` | Every listed pattern appears in the job log. |
| `forbidden: [pattern]` | None of the listed patterns appears. |
| `counts: [{"pattern": "...", "count": "num_ranks"}]` | Exactly `num_ranks` total matches; rank IDs are not checked. |
