# nsight_compute_tutorial

One GEMM (`C = A @ B`, fp16 in / fp32 accumulate / fp16 out) written four ways —
TileLang, Triton, hand-written CUDA with WMMA, and cuBLAS — so they can be
compared against each other and profiled under Nsight Compute.

## Setup

```bash
uv sync
```

That creates `.venv/` with torch, Triton, TileLang, **and a pip-installed CUDA
toolkit** (nvcc, headers, static libs). No system CUDA install is needed, and
nothing in this repo hard-codes a toolkit path: `examples/cuda_env.py`
discovers everything at run time from whichever environment is active, and
detects the GPU's `sm_XX` from `nvidia-smi`.

The toolkit wheels are pinned to the 13.0.x line to match the CUDA runtime
headers torch 2.13 ships. See the comments in `pyproject.toml` — a mismatched
nvcc makes every TileLang kernel fail to compile.

## Benchmark

```bash
source .venv/bin/activate
cd examples

python bench.py          # default sweep: 1024, 2048, 4096
python bench.py 8192     # any sizes you like
```

All four implementations are timed by the same function on the same tensors,
and each is checked against an fp32 reference. The CUDA numbers come from the
kernel in `gemm_cuda.cu` itself: `bench.py` extracts the source above the naive
reference and binds it to torch, so the benchmarked code is literally the same
kernel the standalone binary runs.

## Standalone CUDA build

`gemm_cuda.cu` also builds on its own — it is self-verifying (against a naive
kernel) and self-benchmarking:

```bash
cd examples
make run                 # or: make && ./gemm_cuda [M N K]
```

The Makefile asks `cuda_env.py` for nvcc's location and the GPU arch, and falls
back to this repo's `.venv` if you have not activated it. Override either:

```bash
make ARCH=sm_90
make CUDA_HOME=/usr/local/cuda-13.0
```

## Layout

| path | |
|---|---|
| `examples/bench.py` | runs and times all four implementations |
| `examples/cuda_env.py` | CUDA toolkit / arch discovery; also a CLI for the Makefile |
| `examples/gemm_tilelang.py` | TileLang kernel |
| `examples/gemm_triton.py` | Triton kernel, same tiling |
| `examples/gemm_cuda.cu` | hand-written WMMA kernel + standalone main() |
| `examples/Makefile` | standalone build |

`examples/build/` (generated linker shims) and `examples/gemm_cuda` are build
output and are gitignored; `make clean` removes them.
