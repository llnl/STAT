# Native PyTorch computation

Every snapshot must contain each rank in `main -> native_compute`. The merged
history must also contain the native `at::_ops::mul_out::call` frame on that
application path for at least one rank. Tensor allocation happens before readiness; the
remaining window repeatedly executes CPU elementwise multiplication.

Select `07` with the shared runner in Docker, Enroot, local MPI or Slurm. The workload
uses `test.py` and needs CPU PyTorch; no GPU is required.
The shared CI selection runs it in both Docker and Enroot.
A successful run must also complete positive work on every rank after detachment.

```json checks:default
{
  "stdout": {
    "counts": [{"pattern": "NATIVE_DONE rank=\\d+ kernel=mul operations=[1-9]\\d*", "count": "num_ranks"}]
  },
  "pystack": {
    "each_sample": [{
      "paths_any": [["main@test.py:*", "native_compute@test.py:*"]],
      "attribute": "pcount", "count": "num_ranks", "source": "ranks"
    }],
    "across_samples": [{
      "paths_any": [
        ["main@test.py:*", "native_compute@test.py:*", "*", "native:at::_ops::mul_out::call*@libtorch_cpu.so"]
      ],
      "attribute": "pcount", "min_count": 1, "source": "ranks"
    }]
  }
}
```
