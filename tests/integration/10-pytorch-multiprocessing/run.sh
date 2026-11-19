# Workload settings for the shared integration runner.
VARIANTS=(default threads parent-only)
BACKEND=pystack
WITH_CHILDREN=1
READY_PATTERN=APPLICATION_THREADS

build_workload() {
  APPLICATION=("$PYTHON" -u "$TEST_DIR/test.py" --duration "$DURATION"
    --workers "$LOADER_WORKERS" --threads "$APP_THREADS")
}
