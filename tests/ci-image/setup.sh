#!/usr/bin/env bash
set -euo pipefail

: "${STAT_CI_ROOT:?Set STAT_CI_ROOT to a writable shared dependency directory}"
export PIXI_HOME="$STAT_CI_ROOT/pixi" PIXI_NO_PATH_UPDATE=1
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$scripts/stat-env.sh"
source_dir="${1:?usage: setup.sh SOURCE_DIR INSTALL_PREFIX}"
install_prefix="${2:?missing INSTALL_PREFIX}"

mkdir -p "$STAT_PIXI"
if ! command -v pixi >/dev/null; then
  curl -fsSL https://pixi.sh/install.sh | bash
fi
cp "$scripts/pixi.toml" "$STAT_PIXI/pixi.toml"
pixi install --manifest-path "$STAT_PIXI/pixi.toml"

# Autoconf projects such as MRNet may otherwise select host Clang with a different sysroot.
export CC="$PIXI_PREFIX/bin/cc" CXX="$PIXI_PREFIX/bin/c++"

# Use the same dependency stages as the Dockerfile, without container layers.
for step in sources libiberty dyninst python-tools pystack graphlib mrnet launchmon; do
  bash "$scripts/install-deps.sh" "$step"
done
bash "$scripts/build-stat.sh" "$source_dir" "$install_prefix"
bash "$scripts/validate.sh" "$install_prefix"
