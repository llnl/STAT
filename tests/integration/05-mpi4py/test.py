#!/usr/bin/env python3
"""Initialize MPI and keep a Python application frame available for STAT sampling."""

import argparse
import time

from mpi4py import MPI


def sampled_wait(duration):
    """Keep all ranks in a recognizable Python frame until sampling has finished."""
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, required=True)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    comm.Barrier()
    print(f"MPI_READY rank={rank}/{size}", flush=True)
    sampled_wait(args.duration)
    total = comm.allreduce(rank + 1, op=MPI.SUM)
    assert total == size * (size + 1) // 2, total
    print(f"MPI_DONE rank={rank}/{size} sum={total}", flush=True)


if __name__ == "__main__":
    main()
