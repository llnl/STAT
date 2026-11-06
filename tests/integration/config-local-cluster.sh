# Copy and adjust for hosts with shared source, installation and result paths.
source "$STAT_TEST_ROOT/lib/defaults.sh"
# Slots across hosts must cover the largest selected MPI rank count.
HOSTS=(localhost:4)

# Initialize MPI and STAT's runtime environment before invoking the runner.
# If mpirun is a shell wrapper, select its real executable (often orterun) here.
MPI_LAUNCHER=(mpirun --bind-to none --map-by node)
