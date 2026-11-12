#!/usr/bin/env python3
"""synthetic DDP example for STAT sampling, using random data and a small model.

torchrun workers stay active so STAT can capture training or all-reduce stacks.
the README also rejects torchrun's RendezvousConnectionError in the job log.
"""

from __future__ import annotations

import argparse
import datetime
import os
import socket
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def distributed_train_step(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    # the training path requires this named frame, test.py source and a positive line number.
    inputs = torch.randn(batch_size, width, device=device)
    targets = torch.randint(0, width, (batch_size,), device=device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)

    # the DDP backward pass synchronizes gradients, providing a distributed training stack.
    loss.backward()
    optimizer.step()
    return loss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--cpu", action="store_true",
        help="Use CPU/Gloo to test multiple workers without multiple GPUs",
    )
    args = parser.parse_args()

    # the PyStack graph keeps torchrun → process:cmd:test context; the check counts these workers and their expected RANK IDs.
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    if not args.cpu:
        torch.cuda.set_device(local_rank)
    device = torch.device("cpu") if args.cpu else torch.device("cuda", local_rank)

    dist.init_process_group(
        "gloo" if args.cpu else "nccl",
        timeout=datetime.timedelta(minutes=10),
    )

    model = torch.nn.Sequential(
        torch.nn.Linear(args.width, args.width),
        torch.nn.GELU(),
        torch.nn.Linear(args.width, args.width),
    ).to(device)
    ddp = DistributedDataParallel(model, device_ids=None if args.cpu else [local_rank])
    optimizer = torch.optim.AdamW(ddp.parameters())

    print(
        f"TORCHRUN_READY rank={rank}/{dist.get_world_size()} "
        f"host={socket.gethostname()} pid={os.getpid()}",
        flush=True,
    )
    deadline = time.monotonic() + args.duration
    stop = torch.zeros(1, dtype=torch.int32, device=device)
    step = 0

    # every snapshot must cover all workers across the two accepted paths below.
    while True:
        # the README matches main's exact callsite line here, then any positive line in distributed_train_step.
        loss = distributed_train_step(ddp, optimizer, args.batch_size, args.width, device)

        if step % 100 == 0:
            print(f"TORCHRUN_STEP rank={rank} step={step} loss={loss.item():.6f}", flush=True)
        step += 1

        stop.fill_(int(time.monotonic() >= deadline))

        # this callsite → * → all_reduce@distributed_c10d.py:* is the alternate path; MAX keeps ranks stopping together.
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            break

    dist.barrier()
    dist.destroy_process_group()

    # the stdout check expects num_ranks TORCHRUN_DONE records, each with steps > 0.
    print(f"TORCHRUN_DONE rank={rank} steps={step}", flush=True)
    return 0


if __name__ == "__main__":
    # both accepted paths include this exact <module> → main entrypoint and its source line.
    raise SystemExit(main())
