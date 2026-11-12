#!/usr/bin/env bash
set -euo pipefail

export STAT_CI_SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "${STAT_CI_SCRIPTS}/stat-env.sh"
source_dir="${1:?usage: build-stat.sh SOURCE_DIR INSTALL_PREFIX [CONFIGURE_OPTIONS ...]}"
install_prefix="${2:?usage: build-stat.sh SOURCE_DIR INSTALL_PREFIX}"
shift 2
export JOBS="${JOBS:-4}"

# Build from a temporary copy so bootstrap cannot modify the mounted checkout.
mkdir -p "$install_prefix"
source_dir="$(cd "$source_dir" && pwd)"
install_prefix="$(cd "$install_prefix" && pwd)"
work_dir="$(mktemp -d /tmp/stat-ci-build.XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/src" "$work_dir/build"
cp -a "$source_dir/." "$work_dir/src/"
source_dir="$work_dir/src"

pixi run --manifest-path "$STAT_PIXI/pixi.toml" bash -euo pipefail -c '
  source "${STAT_CI_SCRIPTS}/stat-env.sh"
  cd "$1"
  export ACLOCAL_PATH="$PIXI_PREFIX/share/aclocal:${ACLOCAL_PATH:-}"
  ./bootstrap

  # Configure against the Python and native libraries in the CI environment.
  cd "$3"
  python_include="$(python -c '\''import sysconfig; print(sysconfig.get_path("include"))'\'')"
  python_libdir="$(python -c '\''import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'\'')"
  export PYTHONINCLUDE="$python_include" PYTHONLIB="$python_libdir"
  export CPPFLAGS="-I$python_include -I$STAT_DEPS/include -I$PIXI_PREFIX/include"
  export CFLAGS="$CPPFLAGS"
  export CXXFLAGS="-DDYNINST_TBB_HAS_VERSION_H $CPPFLAGS"
  export LDFLAGS="-L$STAT_DEPS/lib -Wl,-rpath,$STAT_DEPS/lib -L$PIXI_PREFIX/lib -Wl,-rpath,$PIXI_PREFIX/lib"

  "$1/configure" \
    --prefix="$2" \
    --with-launchmon="$STAT_DEPS" \
    --with-mrnet="$STAT_DEPS" \
    --with-graphlib="$STAT_DEPS" \
    --enable-gdb \
    --with-gdb="$PIXI_PREFIX/bin/gdb" \
    --with-stackwalker="$STAT_DEPS" \
    --with-elfutils="$PIXI_PREFIX" \
    --with-python="$PIXI_PREFIX/bin/python" \
    --with-python-path="$STAT_DEPS/lib/python3.11/site-packages" \
    --with-pystack="$STAT_DEPS/bin/pystack" \
    --with-boost="$PIXI_PREFIX" \
    --with-graphviz="$PIXI_PREFIX" \
    --disable-examples "${@:4}"

  make -j "$JOBS"
  make install
' bash "$source_dir" "$install_prefix" "$work_dir/build" "$@"
