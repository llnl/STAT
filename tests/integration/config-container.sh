# Ubuntu MPI and compilers match the MPI libraries in the nested Enroot image.
source "$STAT_TEST_ROOT/lib/defaults.sh"
EXECUTION=container
STAT_PREFIX=/results/stat-install
RESULTS_ROOT=/results/integration
MPI_LAUNCHER=(/usr/bin/orterun --oversubscribe --bind-to none)
MPI_CC=/usr/bin/mpicc
MPI_CXX=/usr/bin/mpicxx
export OMPI_CC=/usr/bin/gcc OMPI_CXX=/usr/bin/g++
ENROOT_IMAGE=/opt/ubuntu.sqsh
