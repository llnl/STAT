#!/usr/bin/env bash
set -euo pipefail
TEST_DIR=$1
variant=$2
bash -c '
  set -euo pipefail
  test_dir=$1
  variant=$2
  if [ "$variant" = nested-torchrun ]; then
    command=("$TORCHRUN" --nnodes="$NODES" --nproc-per-node="$WORKERS_PER_NODE"
      --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT"
      --rdzv-id="stat-${SLURM_JOB_ID:-local}")
  else
    command=("$PYTHON" -u)
  fi
  command+=("$test_dir/test.py" --launcher "$variant" --iterations 120 --step-sleep 0.15)
  [ "$CPU_ONLY" = 0 ] || command+=(--cpu)
  "${command[@]}"
  exit 0
' nested-rank "$TEST_DIR" "$variant"
exit 0
