# STAT build CI

GitHub Actions runs three phases on native Linux AMD64 and ARM64 runners:

1. **Build Pixi & STAT Dependencies** builds an Ubuntu dependency image using
   `pixi.toml`, `revisions.env` and the shared dependency scripts. Separate CI steps show:

   - **1.1 Set up Pixi build environment** (`pixi` Docker target).
   - **1.2 Build Dyninst** (`dyninst` target, including libiberty and source downloads).
   - **1.3 Build Graphlib, MRNet, LaunchMON & Python tools** (`libraries` target, including PyStack).
   - **1.4 Prepare MPI & Enroot** (`runtime` target).

   Each target extends the preceding stage and reuses its layers on the same runner.

2. **Build STAT** builds the checked-out source in a temporary source copy and
   out-of-source build directory, installing into a separate results prefix.

3. **STAT Tests & Validation** checks the installed entrypoint and Python bindings,
   runs Python unit tests with a temporary working directory and virtual display, and retains
   serial and MPI smoke tests. The shared suite checks MPI/OpenMP and Python/PyTorch
   scenarios in Docker and Enroot, including rank, process, thread and native-frame counts.
   Each test prints available DOT trees even on failure. Logs and graphs are saved as CI artifacts.

CI enables `kernel.yama.ptrace_scope=0` on its disposable runner so Dyninst can attach
as the unprivileged container user. Before uploading artifacts, CI makes the results
readable by the runner, whose UID may differ from the container's UID 1000.
Local runs require a host ptrace policy that permits attachment.

PR runs explicitly check out the PR head commit. The `revisions.env` pins dependencies only.

## Run the same phases locally

From the checkout repo root, with Docker available:

```bash
# 1. Build Pixi and native dependencies.
docker build --build-arg JOBS=4 -t stat-ci:deps -f tests/ci-image/Dockerfile .
results=$(mktemp -d /tmp/stat-ci.XXXXXX)

# 2. Build this checkout, using a separate installation prefix.
docker run --rm -v "$PWD:/workspace:ro" -v "$results:/results" stat-ci:deps \
  bash /opt/stat-ci/build-stat.sh /workspace /results/stat-install

# 3. Validate and sample that installation.
chmod a+w "$results"
docker run --rm --user 1000:1000 --cap-add SYS_PTRACE --shm-size=256m \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  -e HOME=/tmp -e ENROOT_DATA_PATH=/tmp/enroot/data \
  -e ENROOT_CACHE_PATH=/tmp/enroot/cache -e ENROOT_RUNTIME_PATH=/tmp/enroot/run \
  -v "$PWD:/workspace:ro" -v "$results:/results" stat-ci:deps bash -euo pipefail -c '
    bash /opt/stat-ci/validate.sh /results/stat-install
    bash /opt/stat-ci/unit-tests.sh /workspace /results/stat-install
    timeout --kill-after=10s 120 bash /opt/stat-ci/smoke-serial.sh \
      /workspace /results/stat-install /results/serial
    timeout --kill-after=10s 150 bash /opt/stat-ci/smoke-mpi.sh \
      /workspace /results/stat-install /results/mpi plain
    timeout --kill-after=10s 150 bash /opt/stat-ci/smoke-mpi.sh \
      /workspace /results/stat-install /results/enroot enroot
    tests=(01 02 03 04 05 06 07 08 09 10 11)
    timeout --kill-after=10s 1200 bash /workspace/tests/integration/run-tests.sh "${tests[@]}"
    timeout --kill-after=10s 1200 bash /workspace/tests/integration/run-tests.sh --enroot "${tests[@]}"
  '
```

Additional configure options follow the install prefix in `build-stat.sh`; for example,
`--without-pystack` builds a separate installation for backend compatibility checks.

## Dependency cache

CI caches steps 1.1–1.4 separately for AMD64 and ARM64, including the CPU workload environment.
STAT source and smoke-test changes reuse these layers; dependency inputs, toolchain or base-image changes
rebuild affected layers. Missing or inaccessible caches trigger a normal build.
Look for `CACHED` in the build log. STAT itself is rebuilt from every checkout.

## Python workload registration

Some Python tests run through `mpirun` without using MPI themselves. The container profile enables
`STAT_TEST_MPI_REGISTER` to initialize MPI through mpi4py before these workloads start, allowing LaunchMON
to discover them with the container's OpenMPI 4 runtime.
