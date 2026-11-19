#!/usr/bin/env python3
"""Keep standalone torchrun workers in a Python frame and verify CPU collective completion."""

import argparse
from datetime import timedelta
import time

import torch
import torch.distributed as dist


def sampled_wait(duration):
    """Keep each worker in a recognizable application frame while STAT samples it."""
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, required=True)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")

    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        dist.barrier()
        print(f"TORCHRUN_READY rank={rank}/{size}", flush=True)
        sampled_wait(args.duration)
        total = torch.tensor(rank + 1)
        dist.all_reduce(total)
        assert total.item() == size * (size + 1) // 2, total.item()
        print(f"TORCHRUN_DONE rank={rank}/{size} sum={total.item()}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
