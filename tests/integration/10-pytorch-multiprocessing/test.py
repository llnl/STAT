#!/usr/bin/env python3
"""synthetic PyTorch loader workload for STAT parent/child and application-thread checks."""

import argparse
import json
import os
import socket
import threading
import time

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticDataset(Dataset):
    def __len__(self):
        return 100000

    def __getitem__(self, index):
        # load_sample must remain visible somewhere in the loader stacks.
        return load_sample(index)


def load_sample(index):
    time.sleep(0.05)
    return torch.full((128,), float(index % 32))


def consume_batch(batch):
    return torch.sin(batch).sum().item()


def worker_ready(worker_id):
    """Report real DataLoader ownership independently of STAT's process discovery."""

    # rank/host/pid/ppid/worker_id independently establish each real worker's ownership.
    record = "DATALOADER_WORKER " + json.dumps({
        "rank": int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("SLURM_PROCID", "0"))),
        "host": socket.gethostname(), "pid": os.getpid(),
        "ppid": os.getppid(), "worker_id": worker_id,
    }) + "\n"

    # one pipe write keeps simultaneous worker records from sharing a line.
    os.write(1, record.encode())


def application_thread(stop):
    """Keep a recognizable application thread alive throughout sampling."""

    # the threads variant counts run@threading.py → application_thread once per configured thread.
    while not stop.wait(0.05):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    # these application rank IDs must all appear on the checked parent and child paths.
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))

    # real DataLoader processes supply run@process.py → _worker_loop frames below each parent.
    loader = DataLoader(
        SyntheticDataset(),
        batch_size=16,
        num_workers=args.workers,
        worker_init_fn=worker_ready,
        persistent_workers=args.workers > 0,
    )

    print(
        f"multiprocess-start rank={rank}/{world_size} host={socket.gethostname()} "
        f"workers={args.workers}",
        flush=True,
    )

    # declared worker totals and parent identity are compared with the worker startup records.
    print("DATALOADER_PARENT " + json.dumps({
        "rank": rank, "host": socket.gethostname(), "pid": os.getpid(),
        "workers": args.workers,
    }), flush=True)

    # fork DataLoader workers before starting additional application threads.
    # the graph checks child process depth, equivalent-branch merging and the parent-plus-child root total.
    batches = iter(loader)
    stop = threading.Event()
    threads = [threading.Thread(target=application_thread, args=(stop,),
                                name="application-worker") for _ in range(args.threads)]
    for thread in threads:
        thread.start()

    # stdout must report the configured application-thread count for every rank.
    print(f"APPLICATION_THREADS rank={rank} count={len(threads)}", flush=True)

    deadline = time.monotonic() + args.duration
    steps = 0
    try:
        while time.monotonic() < deadline:
            for batch in batches:
                consume_batch(batch)
                steps += 1
                if time.monotonic() >= deadline:
                    break
            if time.monotonic() < deadline:
                batches = iter(loader)
    finally:
        stop.set()
        for thread in threads:
            thread.join()

    # stdout requires one parent completion record per rank, including parent-only sampling.
    print(f"multiprocess-done rank={rank}/{world_size} steps={steps}", flush=True)


if __name__ == "__main__":
    # the parent path keeps this module/main entry; parent-only checks exclude worker/process labels.
    main()
