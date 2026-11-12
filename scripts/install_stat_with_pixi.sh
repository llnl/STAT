#!/usr/bin/env bash

# Install STAT using the same Pixi environment and build scripts as CI.
# Requires Bash, curl, Git, tar, sha256sum and realpath.
#
# Clone the STAT repository and check out the desired branch, tag or commit
# containing this installer, then run it from that checkout.
#
# Usage: bash scripts/install_stat_with_pixi.sh [options]
#   --install-prefix DIR  Installation directory (default: ~/stat-install)
#   --jobs N              Build parallelism (default: JOBS or 16)
#   --reinstall LIST      Optional: reinstall only selected components.
#                         Choose stat, launchmon, dyninst, mrnet or graphlib; separate with commas.
#   -h, --help            Show this help.
#
# Initial installation:
#   git clone https://github.com/llnl/STAT.git
#   cd STAT
#   git checkout <branch-tag-or-commit>
#
#   bash scripts/install_stat_with_pixi.sh --install-prefix "$HOME/stat-install" --jobs 8
#
#
# Optional: reinstall one component, or select several:
#
#   bash scripts/install_stat_with_pixi.sh --install-prefix "$HOME/stat-install" --reinstall stat
#   bash scripts/install_stat_with_pixi.sh --install-prefix "$HOME/stat-install" --reinstall dyninst,launchmon
#
# Builds STAT from this checkout. Reinstalling a dependency deletes its cached
# source and build directories, clones the revision from tests/ci-image/revisions.env,
# and rebuilds it, then rebuilds STAT. Other dependencies remain installed.
#
# Run without --reinstall for the first installation; reinstalls require an existing prefix.
# STAT always builds from a temporary source copy. Keep the prefix at its original absolute
# path and stop applications using it before reinstalling.


set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
shared="$repo/tests/ci-image"
prefix="${STAT_INSTALL_LOCATION:-$HOME/stat-install}"
jobs="${JOBS:-16}"
reinstall=

fail() {
  echo "install-stat: $*" >&2
  exit 2
}

while (( $# )); do
  case "$1" in
    --install-prefix|--jobs|--reinstall)
      (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] ||
        fail "$1 requires a value"

      case "$1" in
        --install-prefix)
          prefix="$2"
          ;;
        --jobs)
          jobs="$2"
          ;;
        --reinstall)
          reinstall="$2"
          ;;
      esac

      shift 2
      ;;

    -h|--help)
      # Use the header comments as help so usage stays in one place.
      sed -n '2,/^set -/s/^# \{0,1\}//p' "$0"
      exit 0
      ;;

    *)
      fail "unknown option: $1"
      ;;
  esac
done

[[ "$jobs" =~ ^[1-9][0-9]*$ ]] ||
  fail "--jobs must be a positive integer"

if [[ -n "$reinstall" ]]; then
  [[ "$reinstall" != ,* && "$reinstall" != *, && "$reinstall" != *,,* ]] ||
    fail "empty component in --reinstall"

  IFS=, read -r -a components <<< "$reinstall"

  for component in "${components[@]}"; do
    case "$component" in
      stat|launchmon|dyninst|mrnet|graphlib)
        ;;
      *)
        fail "unknown reinstall component: $component"
        ;;
    esac
  done
fi

prefix=$(realpath -m -- "$prefix")

# The CI helper copies the checkout: neither path may contain the other.
case "$prefix/" in
  "$repo/"*)
    fail "install outside the source checkout"
    ;;
esac

case "$repo/" in
  "$prefix/"*)
    fail "source checkout must be outside the install prefix"
    ;;
esac

if [[ -n "$reinstall" ]]; then
  [[ -x "$prefix/bin/stat-cl" && -f "$prefix/stat-pixi/pixi.toml" ]] ||
    fail "incomplete installation: $prefix"
fi

mkdir -p "$prefix"

export STAT_CI_ROOT="$prefix" JOBS="$jobs"
source "$shared/stat-env.sh"

# Use the compiler toolchain from the Pixi build environment.
export CC="$PIXI_PREFIX/bin/cc" CXX="$PIXI_PREFIX/bin/c++"

if [[ -z "$reinstall" ]]; then
  bash "$shared/setup.sh" "$repo" "$prefix"
else
  # Fixed order also avoids rebuilding duplicates in the list.
  for component in dyninst graphlib mrnet launchmon; do
    [[ ",$reinstall," = *",$component,"* ]] || continue

    # The LaunchMON checkout uses different capitalization from its build directory.
    source_component="$component"
    [[ "$component" != launchmon ]] || source_component=LaunchMON

    # Start fresh for this dependency while retaining all installed prerequisites.
    rm -rf -- \
      "$prefix/stat-sources/$source_component" \
      "$prefix/stat-build/$component"

    bash "$shared/install-deps.sh" sources "$source_component"
    bash "$shared/install-deps.sh" "$component"
  done

  # Rebuild STAT once against the selected dependency updates.
  bash "$shared/build-stat.sh" "$repo" "$prefix"
  bash "$shared/validate.sh" "$prefix"
fi

printf '\nSTAT installation complete: %s/bin/stat-cl\n' "$prefix"
