# Workload settings are sourced by the shared runner.
VARIANTS=(default)
OPTIONAL_VARIANTS=(gdb)

build_workload() {
  "$MPI_CC" "${NATIVE_FLAGS[@]}" "$TEST_DIR/mpi_ringtopo.c" -o "$RUN_DIR/mpi_ringtopo"
  APPLICATION=("$RUN_DIR/mpi_ringtopo" "$DURATION")
}
