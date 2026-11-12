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

BACKEND=${BACKEND:-dyninst}
WITH_CHILDREN=${WITH_CHILDREN:-0}
STAT_INVOCATION=attach

case "$variant" in
  threads) APP_THREADS=2 ;;
  parent-only) WITH_CHILDREN=0 ;;
esac

id="${TEST_DIR##*/}"
id="${id%%-*}"
rank_setting="TEST_RANKS_$id"
ranks="${!rank_setting}"
workers=$ranks
launcher_ranks=$ranks

# Launch one torchrun parent per node; count its application workers separately for sampling checks.
if [ "$id" = 09 ] || { [ "$id" = 11 ] && [[ "$variant" == *torchrun ]]; }; then
  launcher_ranks=$NODES
  workers=$((NODES * WORKERS_PER_NODE))
fi

if [ "$EXECUTION" = slurm ]; then
  MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)}
else
  MASTER_ADDR=${MASTER_ADDR:-${HOSTS[0]%%:*}}
fi

export MASTER_ADDR MASTER_PORT WORKERS_PER_NODE NODES TORCHRUN PYTHON CPU_ONLY

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

    # History checks use the merged graph; include it when explaining a failure.
    if [ "$status" != 0 ]; then
      graphs+=(stat_results/*/*.3D.dot)
    fi

    if [ "${#graphs[@]}" = 0 ]; then
      echo "No sample graphs were produced."
    fi

    for graph in "${graphs[@]}"; do
      echo "Call tree: $graph"
      "${STAT_DOT_TREE:-$STAT_PREFIX/bin/stat-dot-tree}" --charset ascii "$graph" ||
        echo "Could not render $graph" >&2
    done
  fi

  exit "$status"
}

trap finish EXIT
trap 'exit 143' TERM

# Preserve compiler diagnostics and the exact workload used by this run.
build_workload > "$RUN_DIR/compiler.log" 2>&1

if [ "$STAT_INVOCATION" = serial ]; then
  # Direct attachment samples a local process, without MPI or Slurm rank discovery.
  # A local launcher may own several workers even though STAT attaches to one PID.
  workers=${SERIAL_WORKERS:-1}
  launcher=()
elif [ "$EXECUTION" = slurm ]; then
  launcher=(srun --mpi="$MPI_MODE" --nodes="$NODES" --ntasks="$launcher_ranks"
            --cpus-per-task="$OMP_NUM_THREADS" --kill-on-bad-exit=1 "${SRUN_OPTIONS[@]}")
else
  launcher=("${MPI_LAUNCHER[@]}" -n "$launcher_ranks")

  if [ "$EXECUTION" = local ]; then
    hosts=$(IFS=,; echo "${HOSTS[*]}")
    launcher+=(--host "$hosts")
  fi
fi

if [ "$rank_container" = enroot ]; then
  launcher+=(enroot start --conf "$TEST_DIR/../../ci-image/mpi-enroot.conf"
             --mount /results:/results --mount /tmp:/tmp --mount /workspace:/workspace
             "$STAT_TEST_ENROOT_NAME")
fi

interval_ms=$(LC_ALL=C awk -v seconds="$SAMPLE_INTERVAL" 'BEGIN { printf "%.0f", seconds * 1000 }')
stat=("$STAT_PREFIX/bin/stat-cl" -i -S -t "$SAMPLE_COUNT" -T "$interval_ms")
[ "$BACKEND" != pystack ] || stat+=(-K)
[ "$WITH_CHILDREN" != 1 ] || stat+=(--with-children)
[ "${WITH_NATIVE:-0}" != 1 ] || stat+=(--native)
[ "$variant" != threads ] || stat+=(-w)

if [ "$STAT_INVOCATION" = launch ] && [ "$mode" = stat ]; then
  # STAT owns job creation; inherited output keeps tee open until the workload finishes.
  "${stat[@]}" -s "$ATTACH_SETTLE_SECONDS" -C "${launcher[@]}" "${APPLICATION[@]}" \
    2>&1 | tee stat-cl.log application.log
else
  printf 'Launch: '
  printf '%q ' "${launcher[@]}" "${APPLICATION[@]}"
  printf '\n'

  "${launcher[@]}" "${APPLICATION[@]}" > application.log 2>&1 &
  pid=$!

  # Attach to the launcher, or to the local Python PID in serial mode.
  if [ "$mode" = stat ]; then
    # Python launchers can be discoverable before their application workers exist.
    if [ -n "${READY_PATTERN:-}" ]; then
      deadline=$((SECONDS + 90))
      until [ "$(grep -oE "$READY_PATTERN" application.log | wc -l)" -ge "$workers" ]; do
        kill -0 "$pid" || { cat application.log; exit 1; }

        if [ "$SECONDS" -ge "$deadline" ]; then
          echo "Timed out waiting for $workers workload readiness records: $READY_PATTERN" >&2
          exit 1
        fi

        sleep 1
      done
    fi

    sleep "$ATTACH_SETTLE_SECONDS"
    kill -0 "$pid"

    [ "$STAT_INVOCATION" != serial ] || stat+=(-I)
    "${stat[@]}" "$pid" 2>&1 | tee stat-cl.log
  fi

  wait "$pid"
  pid=
fi

cat application.log

# Check completion and every sample against the saved README expectations.
cat application.log > validation.log
if [ "$mode" = stat ] && [ "$STAT_INVOCATION" != launch ]; then
  cat stat-cl.log >> validation.log
fi

check_set="${CHECK_SET:-$variant}"
[ "$variant" != srun ] || check_set=default
[ "$variant" != torchrun ] || check_set=default
[ "$variant" != direct ] || check_set=application
[ "$id:$variant" != 11:torchrun ] || check_set=application
[ "$variant" != nested-torchrun ] || check_set=nested-torchrun

expected_threads=$OMP_NUM_THREADS
if [ "$id" = 10 ]; then
  expected_threads=$((APP_THREADS > 0 ? APP_THREADS : 1))
fi

checks=(--checks "$RUN_DIR/README.md" --check-set "$check_set"
        --output "$RUN_DIR/validation.log" --world-size "$workers"
        --num-threads "$expected_threads" --num-workers "$LOADER_WORKERS")
if [ "$mode" = stat ]; then
  checks+=(--results "$RUN_DIR" --backend "$BACKEND" --sample-count "$SAMPLE_COUNT")
fi

python3 "$TEST_DIR/../lib/validate_stat.py" "${checks[@]}"
