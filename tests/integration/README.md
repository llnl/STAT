# STAT integration tests

Integtation tests check native application stacks in Docker and inside nested Enroot containers. Current
commits ad the following tests:

| Test | Checks |
| --- | --- |
| [01 MPI ring](01-mpi-ring/README.md) | Stalled rank 1 and waiting peers, including exact rank membership. |
| [02 MPI recursion](02-mpi-recursion/README.md) | Exact recursive frame depth and source information for each rank. |
| [03 MPI/OpenMP](03-mpi-openmp/README.md) | Main-thread paths and complete OpenMP teams, with and without thread sampling. |

Each README contains executable JSON expectations. The shared validator checks every
snapshot for rank coverage, connected call paths and expected process/thread counts.
See [check syntax](TEST-CHECKS-FORMAT.md) to change an expectation or recheck saved results.

## Run

Build the dependency image and install the checkout using the commands in
[the CI image README](../ci-image/README.md). Then run the suite from the checkout root:

```bash
docker run --rm --user 1000:1000 --cap-add SYS_PTRACE --shm-size=256m \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  -e HOME=/tmp -e ENROOT_DATA_PATH=/tmp/enroot/data \
  -e ENROOT_CACHE_PATH=/tmp/enroot/cache -e ENROOT_RUNTIME_PATH=/tmp/enroot/run \
  -v "$PWD:/workspace:ro" -v "$results:/results" stat-ci:deps \
  bash /workspace/tests/integration/run-tests.sh
```

Add `--enroot` to run the same matrix with application ranks in nested containers.
The default matrix has four scenarios: ring, recursion, OpenMP processes and OpenMP threads.
Each uses two samples and waits for successful application completion after detachment.
`--plain` runs the workloads and stdout checks without STAT. Select a subset with `01 02`
or `03:threads`. `--dry-run` lists the selected scenarios without starting programs.

The host must allow ptrace and unprivileged user namespaces, as described in the CI README.

## Results

Every invocation creates a new directory under `/results/integration`. It saves the
configuration, workload scripts, README expectations, compiler and application logs,
STAT logs and graphs, plus `summary.tsv`. Nonzero statuses are failures; missing tools
fail rather than silently skipping a CI scenario. The runner continues after individual
failures and returns nonzero if any scenario failed. Each scenario has a configured timeout.

Every available sample tree is printed with `stat-dot-tree`, whether the scenario passes or fails.
Recheck a saved scenario with:

```bash
python3 tests/integration/lib/validate_stat.py \
  --checks "$run_dir/README.md" --check-set threads \
  --output "$run_dir/validation.log" --results "$run_dir" \
  --backend dyninst --world-size 4 --num-threads 4 --sample-count 2
```

This suite covers native single-node Docker/Enroot execution. MPI-fork test 04 requires
`--with-children`, which is not available in this upstream base. Fork sampling, Slurm, direct
multi-node MPI, GPU and Python-backend workloads follow separately.
