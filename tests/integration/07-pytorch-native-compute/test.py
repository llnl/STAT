"""Keep CPU elementwise multiplication active while STAT samples native PyTorch frames."""

import argparse
import os
import time

import torch


def native_compute(rank, duration):
    """Allocate and warm up before readiness, then repeat multiplication until the deadline."""
    x = torch.ones(4096, 4096)
    y = torch.full_like(x, 1.0001)
    result = torch.empty_like(x)
    torch.mul(x, y, out=result)

    operations = 0
    start = time.monotonic()
    print(f'NATIVE_READY rank={rank} kernel=mul', flush=True)
    while time.monotonic() - start < duration:
        torch.mul(x, y, out=result)
        operations += 1
    print(f'NATIVE_DONE rank={rank} kernel=mul operations={operations} '
          f'elapsed={time.monotonic() - start:.2f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=30, help='compute window in seconds')
    args = parser.parse_args()
    if not 0 < args.duration <= 120:
        parser.error('--duration must be greater than zero and at most 120 seconds')

    rank = next((os.environ[name] for name in
                 ('RANK', 'OMPI_COMM_WORLD_RANK', 'SLURM_PROCID') if name in os.environ), '0')
    torch.set_num_threads(1)
    with torch.inference_mode():
        native_compute(rank, args.duration)


if __name__ == '__main__':
    main()
