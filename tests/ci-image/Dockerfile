# syntax=docker/dockerfile:1.7
# Inner filesystem for MPI ranks launched through Enroot.
FROM ubuntu:24.04 AS inner
RUN apt-get update && apt-get install -y --no-install-recommends \
      openmpi-bin libopenmpi-dev python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep workload Python separate from Pixi's STAT build environment; both containers use the same CPU wheels.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv /opt/stat-workloads \
    && /opt/stat-workloads/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Register Python ranks for OpenMPI's MPIR table without changing workload entrypoints.
RUN --mount=type=cache,target=/root/.cache/pip /opt/stat-workloads/bin/pip install mpi4py
COPY tests/ci-image/stat_mpi_register.py tests/ci-image/stat_mpi_register.pth \
    /opt/stat-workloads/lib/python3.12/site-packages/

WORKDIR /workspace

# Outer image supplies the build toolchain and STAT dependencies.
FROM ubuntu:24.04 AS pixi

ARG TARGETARCH
ARG JOBS=4
ENV DEBIAN_FRONTEND=noninteractive \
    PIXI_HOME=/opt/pixi \
    PIXI_NO_PATH_UPDATE=1 \
    PIXI_CACHE_DIR=/opt/cache/pixi \
    PIP_CACHE_DIR=/opt/cache/pip \
    CARGO_HOME=/opt/cache/cargo \
    STAT_PIXI=/opt/stat-pixi \
    STAT_DEPS=/opt/stat-deps \
    PATH=/opt/pixi/bin:/opt/stat-deps/bin:/opt/stat-pixi/.pixi/envs/default/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git xz-utils bzip2 tar gzip file binutils \
    && rm -rf /var/lib/apt/lists/*

COPY tests/ci-image/pixi.toml /opt/stat-pixi/pixi.toml
# discard the package cache in the same layer so only the installed environment is exported.
RUN curl -fsSL https://pixi.sh/install.sh | sh \
    && pixi install --manifest-path /opt/stat-pixi/pixi.toml \
    && rm -rf "$PIXI_CACHE_DIR"

FROM pixi AS dyninst

COPY tests/ci-image/revisions.env tests/ci-image/install-deps.sh tests/ci-image/stat-env.sh /opt/stat-ci/
RUN --mount=type=cache,target=/opt/stat-downloads \
    JOBS="$JOBS" bash /opt/stat-ci/install-deps.sh sources
RUN JOBS="$JOBS" bash /opt/stat-ci/install-deps.sh libiberty

# Keep the expensive Dyninst layer before unrelated tools and test packages.
RUN JOBS="$JOBS" bash /opt/stat-ci/install-deps.sh dyninst

FROM dyninst AS libraries

RUN JOBS="$JOBS" bash /opt/stat-ci/install-deps.sh python-tools
RUN JOBS="$JOBS" bash /opt/stat-ci/install-deps.sh pystack
RUN JOBS="$JOBS" bash /opt/stat-ci/install-deps.sh graphlib
RUN JOBS="$JOBS" bash /opt/stat-ci/install-deps.sh mrnet
RUN JOBS="$JOBS" bash /opt/stat-ci/install-deps.sh launchmon

FROM libraries AS runtime

# Ubuntu packages support MPI workloads and the standalone GUI smoke outside Pixi.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         gcc g++ python3 graphviz openmpi-bin libopenmpi-dev openssh-server \
         python3-venv python3-gi python3-gi-cairo gir1.2-gtk-3.0 librsvg2-common xvfb xauth \
         jq squashfs-tools parallel libcap2-bin \
    && case "$TARGETARCH" in \
      amd64) sha=5955c51e88df0c5ed538d87abb5264609d4bad55fa8570745ed5d4ec66c8ab91 ;; \
      arm64) sha=267018b335815a6b32a562915de1f3fad579f7c9d6748e98c62e7369b40be421 ;; \
    esac \
    && curl -fsSL "https://github.com/NVIDIA/enroot/releases/download/v4.2.1/enroot_4.2.1-1_$TARGETARCH.deb" \
         -o /tmp/enroot.deb \
    && echo "$sha  /tmp/enroot.deb" | sha256sum -c - \
    && apt-get install -y /tmp/enroot.deb \
    && rm /tmp/enroot.deb

# prepare the existing Ubuntu UID 1000 account for key-only loopback SSH tests.
# password authentication is disabled in the test configuration; a non-locked
# password field lets sshd accept public keys without runtime account changes.
RUN test_user=$(getent passwd 1000 | cut -d: -f1) \
    && test -n "$test_user" \
    && usermod -p x "$test_user" \
    && mkdir -p /run/sshd
COPY tests/ci-image/sshd_config /opt/stat-ci/sshd_config
COPY --chmod=755 tests/ci-image/with-loopback-ssh.sh /opt/stat-ci/with-loopback-ssh.sh

# LaunchMON identifies OpenRTE through this symbol, stripped by Ubuntu packaging.
# TODO: Maybe a better way here, can be improved.
RUN objcopy --add-symbol 'orterun=.text:0,global,function' /usr/bin/orterun /tmp/orterun \
    && install -m755 /tmp/orterun /usr/bin/orterun \
    && rm /tmp/orterun

# Enroot needs its own root filesystem; bundle Ubuntu with matching MPI libraries for the nested MPI smoke.
# Tests extract this image into temporary storage, avoiding image downloads during validation.
COPY --from=inner /opt/stat-workloads /opt/stat-workloads
# mount the inner stage during compression so the unpacked rootfs is not exported.
RUN --mount=type=bind,from=inner,target=/opt/ubuntu-rootfs \
    mksquashfs /opt/ubuntu-rootfs /opt/ubuntu.sqsh \
      -comp zstd \
      -b 1M \
      -processors 2 \
      -noappend \
      -exit-on-error

# Viewer unit tests use a virtual display; keep these tools out of application rank images.
RUN apt-get update && apt-get install -y --no-install-recommends xvfb xauth \
    && rm -rf /var/lib/apt/lists/*

# Build and validation scripts consume the checkout mounted by CI; the image contains dependencies only.
COPY tests/ci-image/build-stat.sh \
    tests/ci-image/validate.sh \
    tests/ci-image/unit-tests.sh \
    tests/ci-image/smoke-serial.sh \
    tests/ci-image/smoke-mpi.sh \
    tests/ci-image/mpi-enroot.conf \
    /opt/stat-ci/

# Build scripts activate Pixi explicitly; MPI workloads use the Ubuntu runtime.
ENV PATH=/results/stat-install/bin:/opt/stat-workloads/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/results/stat-install/lib
WORKDIR /workspace
CMD ["bash"]
