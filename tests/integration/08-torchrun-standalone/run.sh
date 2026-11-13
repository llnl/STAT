# Isolate worker-rank discovery from MPI and Slurm launcher metadata.
VARIANTS=(default)
BACKEND=pystack
CHECK_SET=default
READY_PATTERN=TORCHRUN_READY

build_workload() {
  STAT_INVOCATION=serial
  SERIAL_WORKERS=$ranks
  # These inherited values must not supply the standalone launcher's rank count.
  unset OMPI_COMM_WORLD_SIZE SLURM_NTASKS SLURM_NPROCS PMI_SIZE
  unset RANK WORLD_SIZE LOCAL_RANK SLURM_PROCID
  unset STAT_TEST_MPI_REGISTER STAT_TEST_MPI_REGISTERED
  APPLICATION=("$TORCHRUN" --standalone --nnodes=1 --nproc-per-node="$SERIAL_WORKERS"
    "$TEST_DIR/test.py" --duration "$DURATION")
}
