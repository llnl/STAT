#!/usr/bin/env bash
set -euo pipefail

source_dir="${1:?usage: smoke-serial.sh SOURCE_DIR INSTALL_PREFIX RESULTS_DIR}"
prefix="${2:?missing INSTALL_PREFIX}"
results="${3:?missing RESULTS_DIR}"

# Use a fresh directory so previous graphs cannot satisfy these checks.
mkdir "$results"
cc -g -O0 "$source_dir/examples/src/serial.c" -o "$results/serial"
cd "$results"

./serial &
app_pid=$!
trap 'kill "$app_pid" 2>/dev/null || true; wait "$app_pid" 2>/dev/null || true' EXIT
sleep 1

# Take one snapshot and check the example's two function names.
timeout --kill-after=10s 60 "$prefix/bin/stat-cl" -i -S -t 1 -I "$app_pid" \
  2>&1 | tee stat-cl.log

# check if function names are present in the snapshot.
grep -q 'myfun' stat_results/*/*.2D.dot
grep -q 'mysleep' stat_results/*/*.2D.dot
echo "PASS: serial snapshot contains myfun and mysleep"

# STAT writes numbered samples followed by the final merged graph; print the last one.
dot_files=(stat_results/*/*.2D.dot)
"$prefix/bin/stat-dot-tree" --charset ascii "${dot_files[-1]}"
