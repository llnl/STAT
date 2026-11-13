# Shared workload and sampling settings; profiles override environment-specific values.

EXECUTION=local
NODES=1
STAT_PREFIX=${STAT_PREFIX:-/usr/local}
RESULTS_ROOT=${STAT_TEST_RESULTS:-$PWD/tests/integration/results}

MPI_LAUNCHER=(mpirun --bind-to none --map-by node)
SRUN_OPTIONS=()
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

# Optional GPU/Python workloads; sites can override these in their profile.
PYTHON=${PYTHON:-python3}
TORCHRUN=${TORCHRUN:-torchrun}
LOADER_WORKERS=${LOADER_WORKERS:-2}
APP_THREADS=${APP_THREADS:-0}
WORKERS_PER_NODE=${WORKERS_PER_NODE:-4}
CPU_ONLY=${CPU_ONLY:-1}
MASTER_ADDR=${MASTER_ADDR:-}
MASTER_PORT=${MASTER_PORT:-29500}
TEST_RANKS_04=2
TEST_RANKS_05=2
TEST_RANKS_06=3
TEST_RANKS_07=2
TEST_RANKS_08=4
TEST_RANKS_09=2
TEST_RANKS_10=4
TEST_RANKS_11=4
