// Synthetic MPI/OpenMP workload for STAT rank-main-thread and full application-team coverage.

#include <mpi.h>
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

struct ThreadResult {
    long long iterations;
    unsigned long long checksum;
};

// The checked worker chain is run_thread_work → process_epoch → compute_block → sampled_delay.
__attribute__((noinline))
static unsigned long long sampled_delay(unsigned long long value) {
    // This leaf pause keeps a real application source/line frame available for sampling.
    usleep(20000);
    return value ^ 0x9e3779b97f4a7c15ULL;
}

__attribute__((noinline))
static unsigned long long compute_block(unsigned long long value) {
    const auto next = sampled_delay(value + 17);
    return next * 33 + 7;
}

__attribute__((noinline))
static unsigned long long process_epoch(unsigned long long value) {
    const auto next = compute_block(value ^ (value >> 13));
    return next + (value & 0xff);
}

__attribute__((noinline))
static ThreadResult run_thread_work(int rank, int thread, double duration) {
    using Clock = std::chrono::steady_clock;
    const auto deadline = Clock::now() + std::chrono::duration<double>(duration);
    unsigned long long checksum = static_cast<unsigned long long>(rank * 1000 + thread + 1);
    long long iterations = 0;
    do {
        checksum = process_epoch(checksum);
        ++iterations;
    } while (Clock::now() < deadline);
    return {iterations, checksum};
}

__attribute__((noinline))
static ThreadResult run_rank_workload(int rank, int size, double duration) {
    long long iterations = 0;
    unsigned long long checksum = 0;

    // Every rank creates the configured team; the threads assertion expects ranks × threads.
#pragma omp parallel reduction(+ : iterations, checksum)
    {
        // One READY record per rank reports the actual team size for stdout validation.
#pragma omp single
        {
            std::printf("MPI_OPENMP_READY rank=%d/%d threads=%d\n", rank, size,
                        omp_get_num_threads());
            std::fflush(stdout);
        }

        // Worker and main threads share this application chain; runtime entry frames can differ.
        const auto result = run_thread_work(rank, omp_get_thread_num(), duration);
        iterations += result.iterations;
        checksum += result.checksum;
    }
    return {iterations, checksum};
}

// The main-thread path starts at _start → * → main → run_rank_workload.
int main(int argc, char** argv) {
    double duration = 60.0;
    if (argc == 3 && std::strcmp(argv[1], "--duration") == 0) {
        duration = std::strtod(argv[2], nullptr);
    }

    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "OpenMP runtime error: MPI thread support=%d\n", provided);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    int rank = 0;
    int size = 0;

    // Distinct MPI rank IDs establish process coverage independently of the larger thread total.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
        std::printf("MARKER MPI+OPENMP START\n");
        std::fflush(stdout);
    }

    // Process-only sampling still runs the same OpenMP team but must omit numeric thread counts.
    const auto result = run_rank_workload(rank, size, duration);

    // Each rank must finish with its configured thread count and a positive iteration count.
    std::printf("MPI_OPENMP_DONE rank=%d threads=%d iterations=%lld checksum=%llu\n",
                rank, omp_get_max_threads(), result.iterations, result.checksum);
    std::fflush(stdout);
    MPI_Finalize();

    // Rank zero supplies the required finish marker.
    if (rank == 0) std::printf("MARKER MPI+OPENMP FINISH\n");
    return 0;
}
