"""Opt-in OpenMPI registration for Python integration workloads."""
import os

if (os.environ.get("STAT_TEST_MPI_REGISTER") == "1"
        and "OMPI_COMM_WORLD_RANK" in os.environ
        and "STAT_TEST_MPI_REGISTERED" not in os.environ):
    # OpenMPI 4 creates its MPIR table after registration. Children must not register again.
    os.environ["STAT_TEST_MPI_REGISTERED"] = "1"
    try:
        from mpi4py import MPI
    except Exception as error:
        import sys
        print(f"OpenMPI Python rank registration failed: {error}", file=sys.stderr, flush=True)
        os._exit(1)
