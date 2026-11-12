#!/usr/bin/env bash
# Run the checkout's Python unit tests against the installed STAT modules.
set -euo pipefail

# --- Test environment ---

scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$scripts/stat-env.sh"

source_dir="${1:?usage: unit-tests.sh SOURCE_DIR INSTALL_PREFIX}"
prefix="${2:?missing INSTALL_PREFIX}"

# Prefer the tested installation while retaining dependency modules from stat-env.sh.
export PYTHONPATH="$prefix/lib/python3.11/site-packages:$PYTHONPATH"
export GSETTINGS_SCHEMA_DIR="$PIXI_PREFIX/share/glib-2.0/schemas"

# --- Isolated test execution ---

# Some existing tests write relative logs; keep the mounted checkout read-only.
work_dir=$(mktemp -d /tmp/stat-unit-tests.XXXXXX)
trap 'rm -rf "$work_dir"' EXIT
cd "$work_dir"

# Viewer tests require a display even when CI has no graphical desktop.
xvfb-run -a "$PIXI_PREFIX/bin/python" -m unittest discover \
  -s "$source_dir/tests/unit" \
  -t "$source_dir"
