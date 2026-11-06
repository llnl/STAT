# Toolchain and sampling settings for native MPI inside the CI Docker image.

STAT_PREFIX=/results/stat-install
RESULTS_ROOT=/results/integration

# CI runners may have fewer CPU slots than ranks; avoid binding OpenMP teams to one core.
MPI_LAUNCHER=(/usr/bin/orterun --oversubscribe --bind-to none)

# Use Ubuntu's MPI wrappers and compilers, matching the MPI libraries inside Enroot.
MPI_CC=/usr/bin/mpicc
MPI_CXX=/usr/bin/mpicxx
export OMPI_CC=/usr/bin/gcc OMPI_CXX=/usr/bin/g++

NATIVE_FLAGS=(-O0 -g -fno-omit-frame-pointer)
OPENMP_FLAGS=(-O1 -g -fopenmp -fno-omit-frame-pointer)

export OMP_NUM_THREADS=4 OMP_DYNAMIC=false OMP_PROC_BIND=false

# Number of ranks for each test.
TEST_RANKS_01=2
TEST_RANKS_02=4
TEST_RANKS_03=4

SAMPLE_COUNT=2

# Times are in seconds. The workload must stay alive through attachment and sampling.
SAMPLE_INTERVAL=0.1
ATTACH_SETTLE_SECONDS=5
DURATION=20
CASE_TIMEOUT=120

ENROOT_IMAGE=/opt/ubuntu.sqsh
