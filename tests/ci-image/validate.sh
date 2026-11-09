#!/usr/bin/env bash

# validate the installed STAT installation with sanity checks.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/stat-env.sh"

prefix="${1:?usage: validate.sh INSTALL_PREFIX}"
export PYTHONPATH="$prefix/lib/python3.11/site-packages:$PYTHONPATH"

"$prefix/bin/stat-cl" --help >/dev/null 2>&1
python -c 'import STAT; import _STAT'

echo "PASS: STAT entrypoint and Python binding"
