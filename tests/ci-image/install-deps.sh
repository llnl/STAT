#!/usr/bin/env bash
set -euo pipefail

export STAT_CI_SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$STAT_CI_SCRIPTS/stat-env.sh"
source "$STAT_CI_SCRIPTS/revisions.env"

# Dependency sources and build products stay outside the STAT checkout.
export STAT_SOURCES="$STAT_CI_ROOT/stat-sources"
export STAT_BUILD="$STAT_CI_ROOT/stat-build"
export STAT_DOWNLOADS="$STAT_CI_ROOT/stat-downloads"
export JOBS="${JOBS:-4}"
step="${1:?usage: install-deps.sh STEP}"
mkdir -p "$STAT_SOURCES" "$STAT_DEPS" "$STAT_BUILD" "$STAT_DOWNLOADS"

clone() {
  local url="$1" sha="$2" dir="$3"
  if [ ! -d "$STAT_SOURCES/$dir/.git" ]; then
    git clone --no-checkout "$url" "$STAT_SOURCES/$dir"
    git -C "$STAT_SOURCES/$dir" checkout --detach "$sha"
  fi
  test "$(git -C "$STAT_SOURCES/$dir" rev-parse HEAD)" = "$sha"
}

run_pixi() {
  local step="$1"
  shift
  echo "==> $step"
  pixi run --manifest-path "$STAT_PIXI/pixi.toml" bash -euo pipefail -c \
    'source "$STAT_CI_SCRIPTS/stat-env.sh"; eval "$1"' bash "$*"
}

# TODO: compare/verify the build options with Spack recipes as well.

if [ "$step" = sources ]; then
  clone https://github.com/LLNL/LaunchMON.git "$LAUNCHMON_SHA" LaunchMON
  clone https://github.com/dyninst/mrnet.git "$MRNET_SHA" mrnet
  clone https://github.com/LLNL/graphlib.git "$GRAPHLIB_SHA" graphlib
  # TODO: using fork for now, until PR is merged.
  clone https://github.com/pramodk/pystack.git "$PYSTACK_SHA" pystack
  clone https://github.com/dyninst/dyninst.git "$DYNINST_SHA" dyninst

  binutils_version=2.41
  binutils_sha=ae9a5789e23459e59606e6714723f2d3ffc31c03174191ef0d015bdf06007450

  if [ ! -s "$STAT_DOWNLOADS/binutils-$binutils_version.tar.xz" ]; then
    curl -fL --retry 3 "https://ftp.gnu.org/gnu/binutils/binutils-$binutils_version.tar.xz" \
      -o "$STAT_DOWNLOADS/binutils-$binutils_version.tar.xz"
  fi

  echo "$binutils_sha  $STAT_DOWNLOADS/binutils-$binutils_version.tar.xz" | sha256sum -c -

  mkdir -p "$STAT_SOURCES/binutils"
  tar -C "$STAT_SOURCES/binutils" --strip-components=1 \
    -xf "$STAT_DOWNLOADS/binutils-$binutils_version.tar.xz" \
    "binutils-$binutils_version/config" \
    "binutils-$binutils_version/config.guess" \
    "binutils-$binutils_version/config.sub" \
    "binutils-$binutils_version/include" \
    "binutils-$binutils_version/install-sh" \
    "binutils-$binutils_version/libiberty" \
    "binutils-$binutils_version/missing" \
    "binutils-$binutils_version/mkinstalldirs" \
    "binutils-$binutils_version/move-if-change"
fi

if [ "$step" = libiberty ]; then
  run_pixi libiberty '
    mkdir -p "$STAT_BUILD/libiberty"
    cd "$STAT_BUILD/libiberty"
    export CFLAGS="-g -O2 -fPIC"
    "$STAT_SOURCES/binutils/libiberty/configure" --prefix="$STAT_DEPS" --enable-install-libiberty
    make -j "$JOBS"
    make install
  '
fi

if [ "$step" = python-tools ]; then
  run_pixi python-tools '
    python -m pip install --no-deps --only-binary=:all: "py-spy==0.4.0"
    python -m pip install --no-deps "xdot==1.0"
    glib-compile-schemas "$PIXI_PREFIX/share/glib-2.0/schemas"
    dot -c
  '
fi

if [ "$step" = pystack ]; then
  run_pixi pystack '
    # The fork enables LTO, but conda-forge binutils cannot load GCC LTO objects
    # into nanobind-static on this toolchain.
    sed -i "/set(CMAKE_CXX_FLAGS_RELEASE.*-flto/d; /set(CMAKE_CXX_FLAGS_RELWITHDEBINFO.*-flto/d" \
      "$STAT_SOURCES/pystack/CMakeLists.txt"
    python -m pip install --prefix="$STAT_DEPS" --no-build-isolation --no-deps "$STAT_SOURCES/pystack"
  '
fi

if [ "$step" = dyninst ]; then
  run_pixi dyninst '
    libiberty_lib="$(find "$STAT_DEPS" \( -name libiberty.a -o -name libiberty.so \) -print -quit)"
    test -n "$libiberty_lib"
    cmake -S "$STAT_SOURCES/dyninst" -B "$STAT_BUILD/dyninst" \
      -DCMAKE_INSTALL_PREFIX="$STAT_DEPS" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_PREFIX_PATH="$STAT_DEPS;$PIXI_PREFIX" \
      -DBoost_ROOT_DIR="$PIXI_PREFIX" \
      -DElfUtils_ROOT_DIR="$PIXI_PREFIX" \
      -DLibIberty_ROOT_DIR="$STAT_DEPS" \
      -DLibIberty_LIBRARIES="$libiberty_lib" \
      -DTBB_ROOT_DIR="$PIXI_PREFIX" \
      -DUSE_OpenMP=ON \
      -DENABLE_STATIC_LIBS=OFF \
      -DCMAKE_INSTALL_RPATH="$STAT_DEPS/lib;$PIXI_PREFIX/lib"
    cmake --build "$STAT_BUILD/dyninst" -j "$JOBS"
    cmake --install "$STAT_BUILD/dyninst"
  '
fi

if [ "$step" = graphlib ]; then
  run_pixi graphlib '
    cmake -S "$STAT_SOURCES/graphlib" -B "$STAT_BUILD/graphlib" \
      -DCMAKE_INSTALL_PREFIX="$STAT_DEPS" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCMAKE_INSTALL_RPATH="$STAT_DEPS/lib;$PIXI_PREFIX/lib"
    cmake --build "$STAT_BUILD/graphlib" -j "$JOBS"
    cmake --install "$STAT_BUILD/graphlib"
  '
fi

if [ "$step" = mrnet ]; then
  # Bootstrap only when the source checkout needs it.
  run_pixi mrnet '
    cd "$STAT_SOURCES/mrnet"
    if test -x ./bootstrap; then
      ./bootstrap
    elif ! test -x ./configure; then
      autoreconf -ivf
    fi
    mkdir -p "$STAT_BUILD/mrnet"
    cd "$STAT_BUILD/mrnet"
    export CPPFLAGS="-I$PIXI_PREFIX/include"
    export CXXFLAGS="-I$PIXI_PREFIX/include"
    export SOFLAGS="-L$PIXI_PREFIX/lib -Wl,-rpath,$PIXI_PREFIX/lib"
    export LDFLAGS="$SOFLAGS"
    "$STAT_SOURCES/mrnet/configure" --prefix="$STAT_DEPS" --enable-shared
    make -j1
    make install
  '
fi

if [ "$step" = launchmon ]; then
  if [ -n "${STAT_LAUNCHMON_PATCH:-}" ]; then
    if git -C "$STAT_SOURCES/LaunchMON" apply --check "$STAT_LAUNCHMON_PATCH"; then
      git -C "$STAT_SOURCES/LaunchMON" apply "$STAT_LAUNCHMON_PATCH"
    else
      git -C "$STAT_SOURCES/LaunchMON" apply --reverse --check "$STAT_LAUNCHMON_PATCH"
    fi
  fi
  run_pixi launchmon '
    cd "$STAT_SOURCES/LaunchMON"
    ./bootstrap
    mkdir -p "$STAT_BUILD/launchmon"
    cd "$STAT_BUILD/launchmon"
    export ACLOCAL_PATH="$PIXI_PREFIX/share/aclocal:${ACLOCAL_PATH:-}"
    export CPPFLAGS="-I$PIXI_PREFIX/include"
    export CFLAGS="$CPPFLAGS"
    export CXXFLAGS="-std=gnu++14 $CPPFLAGS"
    export LDFLAGS="-L$PIXI_PREFIX/lib -Wl,-rpath,$PIXI_PREFIX/lib"
    "$STAT_SOURCES/LaunchMON/configure" --prefix="$STAT_DEPS" --with-test-rm=slurm
    make -j "$JOBS"
    make install
  '

  # basic sanity checks.
  test -f "$STAT_DEPS/include/walker.h"
  test -f "$STAT_DEPS/include/graphlib.h"
  test -x "$STAT_DEPS/bin/mrnet_commnode"
  test -x "$STAT_DEPS/bin/launchmon"
fi
