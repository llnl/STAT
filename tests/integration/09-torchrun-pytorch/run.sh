# Workload settings for the shared integration runner.
VARIANTS=(default)
BACKEND=pystack
WITH_CHILDREN=1
READY_PATTERN=TORCHRUN_READY

build_workload() {
  APPLICATION=("$TORCHRUN" --nnodes="$NODES" --nproc-per-node="$WORKERS_PER_NODE"
    --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT"
    --rdzv-id="stat-${SLURM_JOB_ID:-local}" "$TEST_DIR/test.py"
    --duration "$DURATION" --batch-size 64 --width 1024)
  [ "$CPU_ONLY" = 0 ] || APPLICATION+=(--cpu)
}
