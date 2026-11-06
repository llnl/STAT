#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config="$root/config-container.sh"
mode=stat
rank_container=plain
dry_run=0
selected=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --config) config="${2:?--config needs a file}"; shift ;;
    --plain) mode=plain ;;
    --stat) mode=stat ;;
    --enroot) rank_container=enroot ;;
    --dry-run) dry_run=1 ;;
    -h|--help)
      echo "Usage: $0 [--config FILE] [--plain|--stat] [--enroot] [--dry-run] [01 02 03:threads ...]"
      exit 0 ;;
    --*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) selected+=("$1") ;;
  esac
  shift
done
config=$(realpath "$config")
source "$config"
[ "${#selected[@]}" -gt 0 ] || selected=(01 02 03)

# Resolve the complete matrix before starting workloads or creating results.
cases=()
for selection in "${selected[@]}"; do
  id="${selection%%:*}"
  case "$id" in
    01) test_dir="$root/01-mpi-ring" ;;
    02) test_dir="$root/02-mpi-recursion" ;;
    03) test_dir="$root/03-mpi-openmp" ;;
    *) echo "Unknown test: $id" >&2; exit 2 ;;
  esac
  source "$test_dir/run.sh"
  variants=("${VARIANTS[@]}")
  if [[ "$selection" == *:* ]]; then
    variant="${selection#*:}"
    if [[ " ${VARIANTS[*]} " != *" $variant "* || -z "$variant" ]]; then
      echo "Unknown variant: $selection" >&2
      exit 2
    fi
    variants=("$variant")
  fi
  for variant in "${variants[@]}"; do
    cases+=("$test_dir:$variant")
  done
done
if [ "$dry_run" = 1 ]; then
  printf '%s\n' "${cases[@]}"
  exit 0
fi

mkdir -p "$RESULTS_ROOT"
results=$(mktemp -d "$RESULTS_ROOT/$rank_container-$mode.XXXXXX")
cp "$config" "$results/config.sh"
printf 'scenario\texit_status\n' > "$results/summary.tsv"
echo "Results: $results"

# Extract once per matrix. The named container belongs only to this invocation.
export STAT_TEST_ENROOT_NAME="stat-test-$(basename "$results")"
if [ "$rank_container" = enroot ]; then
  enroot create -n "$STAT_TEST_ENROOT_NAME" "$ENROOT_IMAGE"
  trap 'enroot remove -f "$STAT_TEST_ENROOT_NAME"' EXIT
fi

failed=0
for entry in "${cases[@]}"; do
  test_dir="${entry%:*}"
  variant="${entry##*:}"
  scenario="${test_dir##*/}-$variant"
  run_dir="$results/$scenario"
  mkdir "$run_dir"
  cp "$test_dir/README.md" "$run_dir/README.md"
  cp "$test_dir/run.sh" "$run_dir/run.sh"
  status=0
  timeout --kill-after=10s "$CASE_TIMEOUT" bash "$root/lib/run-case.sh" \
    "$results/config.sh" "$test_dir" "$variant" "$run_dir" "$mode" "$rank_container" \
    > "$run_dir/runner.log" 2>&1 || status=$?
  cat "$run_dir/runner.log"
  printf '%s\t%s\n' "$scenario" "$status" >> "$results/summary.tsv"
  [ "$status" = 0 ] || failed=1
done
cat "$results/summary.tsv"
exit "$failed"
