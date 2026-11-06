# Each rank sleeps at its own recursion depth.
VARIANTS=(default)

build_workload() {
  "$MPI_CC" "${NATIVE_FLAGS[@]}" "$TEST_DIR/rank_test.c" -o "$RUN_DIR/rank_test"
  APPLICATION=("$RUN_DIR/rank_test" "$DURATION")
}
