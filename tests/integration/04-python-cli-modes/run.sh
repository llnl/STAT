# One workload isolates backend selection from the CLI's launch and attachment paths.
VARIANTS=(launch serial)
if [ "$EXECUTION" = slurm ]; then
  OPTIONAL_VARIANTS+=(slurm-wrapper)
fi
BACKEND=pystack
WITH_LINES=0
CHECK_SET=default
READY_PATTERN=PYTHON_READY

build_workload() {
  STAT_INVOCATION="$variant"
  APPLICATION=("$PYTHON" -u "$TEST_DIR/test.py" --duration "$DURATION")
  if [ "$variant" = serial ]; then
    APPLICATION+=(--serial)
  elif [ "$variant" = slurm-wrapper ]; then
    STAT_INVOCATION=attach
    MPI_MODE=none
    WITH_NATIVE=1
    # Bash stays registered with Slurm while PyStack finds its Python child by SLURM_PROCID alone.
    APPLICATION=(bash -c 'unset RANK; "$@" & wait "$!"' bash "${APPLICATION[@]}")
  fi
}
