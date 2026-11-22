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
  gdb|gdb-threads) BACKEND=gdb ;;
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
flux_job=

# STAT attaches to this helper PID once Flux has published the MPI rank table.
start_flux_debug_launcher() {
  local deadline=$((SECONDS + 30))
  local log=flux-debug.log

  flux job attach --read-only --debug --show-exec --label-io "$flux_job" \
    </dev/null >/dev/null 2>"$log" &
  pid=$!

  # The helper reports readiness as "<timestamp>s: exec.shell.start".
  while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do
    if grep -qsE '^[^ ]+s: exec[.]shell[.]start([[:space:]]|$)' "$log"; then
      return 0
    fi
    sleep 0.1
  done

  echo "Flux debug launcher did not become ready for job $flux_job" >&2
  cat "$log" >&2
  return 1
}

finish() {
  status=$?
  trap - EXIT

  if [ -n "$pid" ]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi

  if [ -n "$flux_job" ]; then
    flux cancel "$flux_job" 2>/dev/null || true
  fi

  # Keep sampled trees visible even when attachment or validation fails.
  if [ "$mode" != plain ]; then
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
elif [ "$EXECUTION" = flux ]; then
  launcher=(flux submit -N "$NODES" -n "$launcher_ranks" -t "${CASE_TIMEOUT}s" --output=application.log)
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
stat=("$STAT_PREFIX/bin/stat-cl" -S -t "$SAMPLE_COUNT" -T "$interval_ms")

[ "${WITH_LINES:-1}" != 1 ] || stat+=(-i)
[ "$BACKEND" != pystack ] || stat+=(-K)
[ "$BACKEND" != gdb ] || stat+=(-G)
[ "$WITH_CHILDREN" != 1 ] || stat+=(--with-children)
[ "${WITH_NATIVE:-0}" != 1 ] || stat+=(--native)
case "$variant" in
  threads|gdb-threads) stat+=(-w) ;;
esac

if [ "$STAT_INVOCATION" = launch ] && [ "$mode" = stat ]; then
  # STAT owns job creation; inherited output keeps tee open until the workload finishes.
  "${stat[@]}" -s "$ATTACH_SETTLE_SECONDS" -C "${launcher[@]}" "${APPLICATION[@]}" \
    2>&1 | tee stat-cl.log application.log
else
  printf 'Launch: '
  printf '%q ' "${launcher[@]}" "${APPLICATION[@]}"
  printf '\n'

  if [ "$EXECUTION" = flux ]; then
    flux_job=$("${launcher[@]}" "${APPLICATION[@]}")
    echo "Flux application job: $flux_job"
  else
    "${launcher[@]}" "${APPLICATION[@]}" > application.log 2>&1 &
    pid=$!
  fi

  # sample the launched application unless this is a workload-only run.
  if [ "$mode" != plain ]; then
    if [ "$EXECUTION" = flux ]; then
      if [ "$mode" = stat ]; then
        start_flux_debug_launcher
      elif ! flux job wait-event -t 30 -p guest.exec.eventlog "$flux_job" shell.start; then
        flux job eventlog "$flux_job" >&2 || true
        exit 1
      fi
    fi

    # Python launchers can be discoverable before their application workers exist.
    if [ -n "${READY_PATTERN:-}" ]; then
      deadline=$((SECONDS + 90))
      until [ "$(grep -oE "$READY_PATTERN" application.log | wc -l)" -ge "$workers" ]; do
        [ -z "$pid" ] || kill -0 "$pid" || { cat application.log; exit 1; }

        if [ "$SECONDS" -ge "$deadline" ]; then
          echo "Timed out waiting for $workers workload readiness records: $READY_PATTERN" >&2
          exit 1
        fi

        sleep 1
      done
    fi

    sleep "$ATTACH_SETTLE_SECONDS"
    [ -z "$pid" ] || kill -0 "$pid"

    if [ "$mode" = remote ]; then
      remote=(python3 "$TEST_DIR/../lib/sample-remote.py"
              --prefix "$STAT_PREFIX" --samples "$SAMPLE_COUNT" --interval "$SAMPLE_INTERVAL"
              --results "$RUN_DIR/stat_results/remote" --backend "$BACKEND")
      if [ "$EXECUTION" = flux ]; then
        remote+=(--flux-job "$flux_job")
      else
        remote+=(--pid "$pid")
      fi
      [ "$STAT_INVOCATION" != serial ] || remote+=(--serial)
      [ "${WITH_LINES:-1}" != 0 ] || remote+=(--function-only)
      [ "$WITH_CHILDREN" != 1 ] || remote+=(--with-children)
      [ "${WITH_NATIVE:-0}" != 1 ] || remote+=(--native)
      [ "$variant" != threads ] || remote+=(--threads)
      "${remote[@]}" 2>&1 | tee stat-cl.log
    else
      [ "$STAT_INVOCATION" != serial ] || stat+=(-I)
      "${stat[@]}" "$pid" 2>&1 | tee stat-cl.log
    fi
  fi

  if [ "$EXECUTION" = flux ]; then
    flux job status "$flux_job"
    flux_job=
  fi

  if [ -n "$pid" ]; then
    wait "$pid"
    pid=
  fi
fi

cat application.log

# protocol acknowledgements are checked by the remote helper, not CLI log strings.
if [ "$mode" = remote ]; then
  common=(--checks "$RUN_DIR/README.md" --check-set "${CHECK_SET:-$variant}"
          --world-size "$workers" --num-threads "$OMP_NUM_THREADS")
  python3 "$TEST_DIR/../lib/validate_stat.py" "${common[@]}" --output "$RUN_DIR/application.log"
  python3 "$TEST_DIR/../lib/validate_stat.py" "${common[@]}" --results "$RUN_DIR" \
    --backend "$BACKEND" --sample-count "$SAMPLE_COUNT"
  exit
fi

# Check completion and every sample against the saved README expectations.
cat application.log > validation.log
if [ "$mode" = stat ] && [ "$STAT_INVOCATION" != launch ]; then
  cat stat-cl.log >> validation.log
fi

check_set="${CHECK_SET:-$variant}"
case "$id:$variant" in
  03:gdb) check_set=processes ;;
  03:gdb-threads) check_set=threads ;;
  *:gdb) check_set=default ;;
esac
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
