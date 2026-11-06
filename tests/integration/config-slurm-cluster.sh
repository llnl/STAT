# Copy and adjust for the selected Slurm cluster and MPI installation.
source "$STAT_TEST_ROOT/lib/defaults.sh"
EXECUTION=slurm
NODES=2
MPI_MODE=pmix
JOB_TIME=00:10:00
# Site options, for example: (--partition=debug --nodelist=node01,node02).
SBATCH_OPTIONS=()
