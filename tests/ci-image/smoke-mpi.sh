#!/usr/bin/env bash
set -euo pipefail

scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$scripts/stat-env.sh"

source_dir="${1:?usage: smoke-mpi.sh SOURCE_DIR INSTALL_PREFIX RESULTS_DIR [plain|enroot]}"
prefix="${2:?missing INSTALL_PREFIX}"
results="${3:?missing RESULTS_DIR}"
mode="${4:-plain}"

mkdir "$results"
OMPI_CC=/usr/bin/gcc /usr/bin/mpicc -g -O0 "$source_dir/examples/src/mpi_ringtopo.c" -o "$results/ring"
cd "$results"

launcher=(/usr/bin/orterun --oversubscribe --bind-to none -n 2)
case "$mode" in
  plain) ;;
  enroot)
    enroot create -n ubuntu /opt/ubuntu.sqsh
    launcher+=(enroot start
               --conf "$scripts/mpi-enroot.conf"
               --mount /results:/results
               --mount /tmp:/tmp
               ubuntu)
    ;;
  *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac

# Attach through the launcher; LaunchMON discovers the rank PIDs automatically.
"${launcher[@]}" "$PWD/ring" 30 >application.log 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT
sleep 5
kill -0 "$pid"
timeout --kill-after=10s 60 "$prefix/bin/stat-cl" -i -S -t 1 "$pid" \
  2>&1 | tee stat-cl.log

# Check the application frames and both ranks in the collected graph.
grep -q 'function="do_SendOrStall"' stat_results/*/*.2D.dot
grep -q 'function="main"' stat_results/*/*.2D.dot
grep -Fq 'bv="[0-1]"' stat_results/*/*.2D.dot

# Let both ranks finish normally after STAT detaches.
wait "$pid"
trap - EXIT
echo "PASS: $mode MPI snapshot contains application frames and both ranks"

# STAT writes numbered samples followed by the final merged graph; print the last one.
dot_files=(stat_results/*/*.2D.dot)
"$prefix/bin/stat-dot-tree" --charset ascii "${dot_files[-1]}"
