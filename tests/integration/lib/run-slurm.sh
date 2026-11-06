#!/usr/bin/env bash
set -euo pipefail
# Slurm copies this script to its spool; the runner path must be passed explicitly.
runner="$1"
shift
exec bash "$runner" "$@"
