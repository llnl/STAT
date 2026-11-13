# Ubuntu MPI and compilers match the MPI libraries in the nested Enroot image.
source "$STAT_TEST_ROOT/lib/defaults.sh"
EXECUTION=container
STAT_PREFIX=/results/stat-install
RESULTS_ROOT=/results/integration
MPI_LAUNCHER=(/usr/bin/orterun --oversubscribe --bind-to none)
MPI_CC=/usr/bin/mpicc
MPI_CXX=/usr/bin/mpicxx
export OMPI_CC=/usr/bin/gcc
export OMPI_CXX=/usr/bin/g++

PYTHON=/opt/stat-workloads/bin/python
TORCHRUN=/opt/stat-workloads/bin/torchrun

# OpenMPI needs Python ranks to register before LaunchMON can discover them.
export STAT_TEST_MPI_REGISTER=1

CASE_TIMEOUT=180

ENROOT_IMAGE=/opt/ubuntu.sqsh
