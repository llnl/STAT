#!/usr/bin/env python3
"""synthetic parent/loader/checkpoint workload for STAT child-process discovery."""

import argparse
import math
import multiprocessing as mp
import os
import socket
import time

def prepare_batch(duration):
    # this frame terminates the checked loader_worker → prepare_batch child path.
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        time.sleep(0.05)
        sum(math.sin(index) for index in range(4096))


def loader_worker(rank, duration):
    # the README counts one loader start and finish record for every application rank.
    print(f"loader-start rank={rank} pid={os.getpid()}", flush=True)
    prepare_batch(duration)
    print(f"loader-finish rank={rank} pid={os.getpid()}", flush=True)


def write_checkpoint(duration):
    # the separate checkpoint path ends here; no actual checkpoint file is needed for this synthetic test.
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        time.sleep(0.1)


def checkpoint_worker(rank, duration):
    # checkpoint start/finish records are counted independently of loader records.
    print(f"checkpoint-start rank={rank} pid={os.getpid()}", flush=True)
    write_checkpoint(duration)
    print(f"checkpoint-finish rank={rank} pid={os.getpid()}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30)
    args = parser.parse_args()

    # every expected rank must have a parent, loader and checkpoint application path.
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    context = mp.get_context("spawn")

    # each child role contributes one process per rank at process depth two.
    children = [
        context.Process(target=loader_worker, args=(rank, args.duration)),
        context.Process(target=checkpoint_worker, args=(rank, args.duration)),
    ]

    print(
        f"multiprocess-start rank={rank}/{world_size} host={socket.gethostname()}",
        flush=True,
    )
    for child in children:
        child.start()

    # successful parent completion is allowed only after both children have exited successfully.
    failed = []
    for child in children:
        child.join(timeout=args.duration + 5)
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
            if child.is_alive():
                child.kill()
                child.join(timeout=5)

        if child.exitcode != 0:
            failed.append(f"{child.name}: exit={child.exitcode}")
    if failed:
        raise RuntimeError("Child process failed: " + "; ".join(failed))

    # the stdout check expects one completion record per parent rank.
    print(f"multiprocess-done rank={rank}/{world_size}", flush=True)


if __name__ == "__main__":
    # the parent main entry and each child's run@process.py ancestry are checked separately.
    main()
