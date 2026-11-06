# Both variants run the same team; only STAT thread sampling changes.
VARIANTS=(processes threads)

build_workload() {
  "$MPI_CXX" "${OPENMP_FLAGS[@]}" "$TEST_DIR/mpi_openmp.cpp" -o "$RUN_DIR/mpi_openmp"
  APPLICATION=(/usr/bin/env "OMP_NUM_THREADS=$OMP_NUM_THREADS" "OMP_DYNAMIC=$OMP_DYNAMIC"
               "OMP_PROC_BIND=$OMP_PROC_BIND" "$RUN_DIR/mpi_openmp" --duration "$DURATION")
}
