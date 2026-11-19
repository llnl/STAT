# STAT integration tests

Integration tests check native and Python application stacks with local MPI or Slurm. The container profile runs
ranks in Docker or nested Enroot; a Slurm site profile can also select Pyxis.
Tests 01–03 form the default matrix; select the remaining tests explicitly.
The numbered order progresses from basic sampling to nested workloads:

| Test | Checks |
| --- | --- |
| [01 MPI ring](01-mpi-ring/README.md) | Stalled rank 1 and waiting peers, including exact rank membership. |
| [02 MPI recursion](02-mpi-recursion/README.md) | Exact recursive frame depth and source information for each rank. |
| [03 MPI/OpenMP](03-mpi-openmp/README.md) | Main-thread paths and complete OpenMP teams, with and without thread sampling. |
| [04 Python CLI](04-python-cli-modes/README.md) | Launch and direct attachment. |
| [05 mpi4py](05-mpi4py/README.md) | MPI Python rank discovery. |
| [06 Python multiprocessing](06-python-multiprocessing/README.md) | Parent and child process stacks. |
| [07 Native PyTorch](07-pytorch-native-compute/README.md) | Python paths into native multiplication. |
| [08 Standalone torchrun](08-torchrun-standalone/README.md) | Local worker ranks without MPI. |
| [09 Distributed PyTorch](09-torchrun-pytorch/README.md) | Training and all-reduce worker paths. |
| [10 PyTorch DataLoader](10-pytorch-multiprocessing/README.md) | Child processes and application threads. |
| [11 Nested launchers](11-pytorch-nested-launch/README.md) | Rank discovery beneath wrappers and torchrun. |

Each README contains executable JSON expectations. The shared validator checks every
snapshot for rank coverage, connected call paths and expected process/thread counts.
See [check syntax](TEST-CHECKS-FORMAT.md) to change an expectation or recheck saved results.

## Docker and Enroot

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

Add `--enroot` to run the same selected tests and variants with application ranks in nested containers.
Without `--enroot`, the application ranks run directly in the outer Docker container. The flag changes where ranks run;
it does not select or skip any tests.
The default matrix has four scenarios: ring, recursion, OpenMP processes and OpenMP threads.
Each uses two samples and waits for successful application completion after detachment.
`--plain` runs the workloads and stdout checks without STAT. Select a subset with `01 02`
or `03:threads`. `--dry-run` lists the selected scenarios without starting programs.

The host must allow ptrace and unprivileged user namespaces, as described in the CI README.

## Python CLI launch and direct attach

[Test 04](04-python-cli-modes/README.md) checks PyStack through `stat-cl -K -C` and `stat-cl -K -I` with a small Python
workload. Select `04` for both variants, or `04:launch` / `04:serial`. It needs no PyTorch or GPU.

## mpi4py launch and attachment

[Test 05](05-mpi4py/README.md) checks MPI rank discovery with a small mpi4py workload. Select `05` for both variants,
or `05:launch` / `05:attach`. With the local profile, it runs through `mpirun` directly without Slurm or torchrun.

## Native PyTorch computation

[Test 07](07-pytorch-native-compute/README.md) repeatedly multiplies CPU tensors to check native ATen frames below
Python application frames. Select `07` with the same launcher and container options as the other Python tests.

## Standalone torchrun

[Test 08](08-torchrun-standalone/README.md) starts a local CPU/Gloo worker group without MPI or Slurm launchers and
attaches to torchrun with `stat-cl -K -I`. Test 08 runs in both Docker and Enroot CI selections.

For Slurm/Pyxis, set `SRUN_OPTIONS` to the image and mount arguments in the site profile. Workload executables must exist
inside that image. `STAT_DOT_TREE` can select a call-tree utility when testing an older STAT installation.

With no test selection, both modes use 01–03. CI explicitly selects **01–11** in CPU mode
for both Docker and Enroot.
The corresponding commands inside the outer Docker container are:

```bash
# Application ranks directly in Docker.
bash /workspace/tests/integration/run-tests.sh 01 02 03 04 05 06 07 08 09 10 11

# The same application scenarios inside nested Enroot containers.
bash /workspace/tests/integration/run-tests.sh --enroot 01 02 03 04 05 06 07 08 09 10 11
```

The runner prints the execution profile, sampling mode, rank-container mode and resolved scenario list before running.
Add `--dry-run` to either command to inspect that selection without starting workloads.

The runner records each selected scenario separately and reports all failures
at the end of the matrix.
Python/PyTorch scenarios with readiness records wait for every worker before attaching STAT, with a 90-second startup limit.
