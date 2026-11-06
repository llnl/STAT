# Shared workload and sampling settings; profiles override environment-specific values.

EXECUTION=local
STAT_PREFIX=${STAT_PREFIX:-/usr/local}
RESULTS_ROOT=${STAT_TEST_RESULTS:-$PWD/tests/integration/results}

MPI_LAUNCHER=(mpirun --bind-to none --map-by node)
HOSTS=(localhost:4)

MPI_CC=mpicc
MPI_CXX=mpicxx
NATIVE_FLAGS=(-O0 -g -fno-omit-frame-pointer)
OPENMP_FLAGS=(-O1 -g -fopenmp -fno-omit-frame-pointer)

export OMP_NUM_THREADS=4 OMP_DYNAMIC=false OMP_PROC_BIND=false

TEST_RANKS_01=2
TEST_RANKS_02=4
TEST_RANKS_03=4

SAMPLE_COUNT=2

# Seconds; workloads must remain alive through attachment and sampling.
SAMPLE_INTERVAL=0.1
ATTACH_SETTLE_SECONDS=5
DURATION=20
CASE_TIMEOUT=120
