# Workload settings for the shared integration runner.
VARIANTS=(default)
BACKEND=pystack
WITH_CHILDREN=1

build_workload() {
  APPLICATION=("$PYTHON" -u "$TEST_DIR/test.py" --duration "$DURATION")
}
