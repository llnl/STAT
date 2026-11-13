# Workload settings for the shared integration runner.
VARIANTS=(nested nested-torchrun direct torchrun)
BACKEND=pystack
WITH_CHILDREN=1
READY_PATTERN=launcher-ready

build_workload() {
  case "$variant" in
    nested|nested-torchrun)
      APPLICATION=(bash "$TEST_DIR/nested-rank.sh" "$TEST_DIR" "$variant") ;;
    direct|torchrun)
      if [ "$variant" = torchrun ]; then
        APPLICATION=("$TORCHRUN" --nnodes="$NODES" --nproc-per-node="$WORKERS_PER_NODE"
          --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT"
          --rdzv-id="stat-${SLURM_JOB_ID:-local}" "$TEST_DIR/test.py" --launcher "$variant")
      else
        APPLICATION=("$PYTHON" -u "$TEST_DIR/test.py" --launcher "$variant")
      fi
      APPLICATION+=(--iterations 120 --step-sleep 0.15)
      [ "$CPU_ONLY" = 0 ] || APPLICATION+=(--cpu) ;;
  esac
}
