# syntax=docker/dockerfile:1.7
# Inner filesystem for MPI ranks launched through Enroot.
FROM ubuntu:24.04 AS inner
RUN apt-get update && apt-get install -y --no-install-recommends openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

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
RUN curl -fsSL https://pixi.sh/install.sh | sh \
    && pixi install --manifest-path /opt/stat-pixi/pixi.toml

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

# MPI runs inside Docker and inside an Enroot image with matching MPI libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         gcc openmpi-bin libopenmpi-dev \
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

# LaunchMON identifies OpenRTE through this symbol, stripped by Ubuntu packaging.
# TODO: Maybe a better way here, can be improved.
RUN objcopy --add-symbol 'orterun=.text:0,global,function' /usr/bin/orterun /tmp/orterun \
    && install -m755 /tmp/orterun /usr/bin/orterun \
    && rm /tmp/orterun

# Enroot needs its own root filesystem; bundle Ubuntu with matching MPI libraries for the nested MPI smoke.
# Tests extract this image into temporary storage, avoiding image downloads during validation.
COPY --from=inner / /opt/ubuntu-rootfs
RUN mksquashfs /opt/ubuntu-rootfs /opt/ubuntu.sqsh \
      -comp zstd \
      -b 1M \
      -processors 2 \
      -noappend \
      -exit-on-error \
    && rm -rf /opt/ubuntu-rootfs

# Build and validation scripts consume the checkout mounted by CI; the image contains dependencies only.
COPY tests/ci-image/build-stat.sh \
    tests/ci-image/validate.sh \
    tests/ci-image/smoke-serial.sh \
    tests/ci-image/smoke-mpi.sh \
    tests/ci-image/mpi-enroot.conf \
    /opt/stat-ci/

ENV BASH_ENV=/opt/stat-ci/stat-env.sh
WORKDIR /workspace
CMD ["bash"]
