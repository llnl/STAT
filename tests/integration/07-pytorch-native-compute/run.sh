# Keep native coverage inside one compute loop, without training or loader phases.
VARIANTS=(default)
BACKEND=pystack
CHECK_SET=default
WITH_NATIVE=1
READY_PATTERN=NATIVE_READY

build_workload() {
  APPLICATION=("$PYTHON" -u "$TEST_DIR/test.py" --duration "$DURATION")
}
