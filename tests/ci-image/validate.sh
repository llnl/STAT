#!/usr/bin/env bash

# validate the installed STAT installation with sanity checks.

set -euo pipefail

prefix="${1:?usage: validate.sh INSTALL_PREFIX}"
python="${STAT_CI_ROOT:-/opt}/stat-pixi/.pixi/envs/default/bin/python"

"$prefix/bin/stat-cl" --help >/dev/null 2>&1
# Check bindings with STAT's build interpreter without activating Pixi for workloads.
PYTHONPATH="$prefix/lib/python3.11/site-packages" "$python" -c 'import STAT; import _STAT'

echo "PASS: STAT entrypoint and Python binding"
