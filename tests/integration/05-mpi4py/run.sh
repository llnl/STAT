# Exercise MPI process-table discovery through both CLI entry paths.
VARIANTS=(launch attach)
BACKEND=pystack
CHECK_SET=default
READY_PATTERN=MPI_READY

build_workload() {
  STAT_INVOCATION="$variant"
  # The workload initializes MPI itself; do not let CI's Python startup hook register it.
  unset STAT_TEST_MPI_REGISTER STAT_TEST_MPI_REGISTERED
  APPLICATION=("$PYTHON" -u "$TEST_DIR/test.py" --duration "$DURATION")
}
