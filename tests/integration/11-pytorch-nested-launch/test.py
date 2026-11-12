#!/usr/bin/env python3
"""synthetic DDP workload for STAT discovery through direct, torchrun and nested-shell launchers."""

import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def forward_pass(model, inputs):
    return model(inputs)


def compute_loss(outputs):
    return outputs.square().mean()


def backward_pass(loss):
    loss.backward()


def optimizer_step(optimizer):
    optimizer.step()


def training_step(model, optimizer, inputs, step_sleep):
    # this source-labelled frame is the application endpoint for every launcher variant.
    optimizer.zero_grad(set_to_none=True)
    outputs = forward_pass(model, inputs)
    loss = compute_loss(outputs)
    backward_pass(loss)
    optimizer_step(optimizer)
    time.sleep(step_sleep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--step-sleep", type=float, default=0.15)
    parser.add_argument("--cpu", action="store_true", help="Use CPU/Gloo for multi-rank tests on one-GPU nodes")
    args = parser.parse_args()

    # worker rank IDs, rather than the smaller number of torchrun launchers, define expected coverage.
    rank = int(os.environ.get("RANK") or os.environ.get("OMPI_COMM_WORLD_RANK")
               or os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("WORLD_SIZE") or os.environ.get("OMPI_COMM_WORLD_SIZE")
                     or os.environ.get("SLURM_NTASKS", "1"))
    if not args.cpu:
        torch.cuda.set_device(0)
    device = torch.device('cpu' if args.cpu else 'cuda')
    dist.init_process_group(
        "gloo" if args.cpu else "nccl",
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        rank=rank,
        world_size=world_size,
    )

    model = DistributedDataParallel(torch.nn.Linear(1024, 1024, bias=False).to(device),
                                    device_ids=None if args.cpu else [0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    inputs = torch.randn(16, 1024, device=device)
    print(f"launcher-ready mode={args.launcher} rank={rank}/{world_size}", flush=True)
    for iteration in range(args.iterations):
        # the graph must reach main → training_step beneath the selected process-launcher chain.
        training_step(model, optimizer, inputs, args.step_sleep)
        if rank == 0 and (iteration + 1) % 100 == 0:
            print(f"launcher-progress mode={args.launcher} iteration={iteration + 1}/{args.iterations}", flush=True)

    dist.destroy_process_group()

    # the selected README block requires the launcher mode and one completion record per worker rank.
    print(f"launcher-done mode={args.launcher} rank={rank}/{world_size}", flush=True)


if __name__ == "__main__":
    # nested cases require bash → bash → [torchrun →] process:cmd:test before this module/main entry.
    torch.set_num_threads(int(os.environ.get("STAT_THREADS_PER_RANK", "1")))
    main()
