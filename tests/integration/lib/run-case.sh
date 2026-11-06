#!/usr/bin/env bash
set -euo pipefail

config="$1"
TEST_DIR="$2"
variant="$3"
RUN_DIR="$4"
mode="$5"
rank_container="$6"
source "$config"
source "$RUN_DIR/run.sh"
id="${TEST_DIR##*/}"
rank_setting="TEST_RANKS_${id%%-*}"
ranks="${!rank_setting}"

cd "$RUN_DIR"
pid=
finish() {
  status=$?
  trap - EXIT
  if [ -n "$pid" ]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi

  # Keep sampled trees visible even when attachment or validation fails.
  if [ "$mode" = stat ]; then
    shopt -s nullglob
    graphs=(stat_results/*/*.2D.dot)
    if [ "${#graphs[@]}" = 0 ]; then
      echo "No sample graphs were produced."
    fi
    for graph in "${graphs[@]}"; do
      echo "Call tree: $graph"
      "$STAT_PREFIX/bin/stat-dot-tree" --charset ascii "$graph" ||
        echo "Could not render $graph" >&2
    done
  fi
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM

# Preserve compiler diagnostics and the exact workload used by this run.
build_workload > "$RUN_DIR/compiler.log" 2>&1
launcher=("${MPI_LAUNCHER[@]}" -n "$ranks")
if [ "$rank_container" = enroot ]; then
  launcher+=(enroot start --conf "$TEST_DIR/../../ci-image/mpi-enroot.conf"
             --mount /results:/results --mount /tmp:/tmp "$STAT_TEST_ENROOT_NAME")
fi
printf 'Launch: '
printf '%q ' "${launcher[@]}" "${APPLICATION[@]}"
printf '\n'
"${launcher[@]}" "${APPLICATION[@]}" > application.log 2>&1 &
pid=$!

# Attach to the MPI launcher; LaunchMON discovers the ranks automatically.
if [ "$mode" = stat ]; then
  sleep "$ATTACH_SETTLE_SECONDS"
  kill -0 "$pid"
  interval_ms=$(LC_ALL=C awk -v seconds="$SAMPLE_INTERVAL" 'BEGIN { printf "%.0f", seconds * 1000 }')
  stat=("$STAT_PREFIX/bin/stat-cl" -i -S -t "$SAMPLE_COUNT" -T "$interval_ms")
  [ "$variant" != threads ] || stat+=(-w)
  "${stat[@]}" "$pid" 2>&1 | tee stat-cl.log
fi
wait "$pid"
pid=
cat application.log

# Check completion and every sample against the saved README expectations.
cat application.log > validation.log
[ "$mode" != stat ] || cat stat-cl.log >> validation.log
checks=(--checks "$RUN_DIR/README.md" --check-set "$variant"
        --output "$RUN_DIR/validation.log" --world-size "$ranks" --num-threads "$OMP_NUM_THREADS")
if [ "$mode" = stat ]; then
  checks+=(--results "$RUN_DIR" --backend dyninst --sample-count "$SAMPLE_COUNT")
fi
python3 "$TEST_DIR/../lib/validate_stat.py" "${checks[@]}"
