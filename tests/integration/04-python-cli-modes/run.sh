# One workload isolates backend selection from the CLI's launch and attachment paths.
VARIANTS=(launch serial)
BACKEND=pystack
CHECK_SET=default
READY_PATTERN=PYTHON_READY

build_workload() {
  STAT_INVOCATION="$variant"
  APPLICATION=("$PYTHON" -u "$TEST_DIR/test.py" --duration "$DURATION")
  [ "$variant" != serial ] || APPLICATION+=(--serial)
}
