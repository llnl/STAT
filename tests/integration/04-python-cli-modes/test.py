#!/usr/bin/env python3
"""Keep a recognizable Python stack alive through CLI launch or direct attachment."""

import argparse
import os
import time


def sampled_wait(duration):
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--serial", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")

    # A local serial target is independent of the surrounding Slurm allocation.
    rank = 0 if args.serial else int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("SLURM_PROCID", "0")))
    print(f"PYTHON_READY rank={rank}", flush=True)
    sampled_wait(args.duration)
    print(f"PYTHON_DONE rank={rank}", flush=True)


if __name__ == "__main__":
    main()
