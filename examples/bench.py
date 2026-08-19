"""Benchmark TileLang / Triton / hand-written CUDA / cuBLAS on one GEMM.

All four are timed by the same function on the same tensors, so the numbers
are directly comparable. The CUDA kernel source is extracted verbatim from
gemm_cuda.cu (everything above the naive reference) and bound to torch, so
the benchmarked code is literally the same kernel the standalone binary runs.
"""

import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so this runs from any working directory

import cuda_env

cuda_env.activate()  # CUDA_HOME + nvcc on $PATH, discovered, never hard-coded

import torch


def bench(fn, warmup=25, iters=200):
    """Median of `iters` timed launches, after warmup. Returns ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def build_cuda_ext():
    from torch.utils.cpp_extension import load_inline

    src = (HERE / "gemm_cuda.cu").read_text()
    kernel_only = src.split("// Naive reference")[0]  # kernel + #defines, no main()
    binding = """
#include <torch/extension.h>
torch::Tensor gemm(torch::Tensor A, torch::Tensor B) {
  int M = A.size(0), K = A.size(1), N = B.size(1);
  auto C = torch::empty({M, N}, A.options());
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(NTHREADS);
  gemm_kernel<<<grid, block>>>(
      reinterpret_cast<const half*>(A.data_ptr()),
      reinterpret_cast<const half*>(B.data_ptr()),
      reinterpret_cast<half*>(C.data_ptr()), M, N, K);
  return C;
}
"""
    return load_inline(
        name="gemm_cuda_ext",
        cpp_sources=["#include <torch/extension.h>\ntorch::Tensor gemm(torch::Tensor, torch::Tensor);"],
        cuda_sources=[kernel_only + binding],
        functions=["gemm"],
        extra_cuda_cflags=["-O3", f"-arch={cuda_env.gpu_arch()}", "-std=c++17"],
        # pip CUDA ships only libcudart.so.13; ld wants an unversioned libcudart.so
        extra_ldflags=[f"-L{cuda_env.link_shim_dir()}"],
        verbose=False,
    )


def main():
    import sys

    sizes = [int(x) for x in sys.argv[1:]] or [1024, 2048, 4096]

    import gemm_tilelang
    import gemm_triton

    ext = build_cuda_ext()

    print(f"\n{torch.cuda.get_device_name(0)}   fp16 in / fp32 accum / fp16 out")
    for S in sizes:
        m = n = k = S
        torch.manual_seed(0)
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        ref = (a.float() @ b.float()).half()

        # TileLang bakes M/N/K in as compile-time constants -> one kernel per shape.
        tl_kernel = gemm_tilelang.build(m, n, k)

        impls = {
            "TileLang": lambda: tl_kernel(a, b),
            "Triton": lambda: gemm_triton.matmul(a, b),
            "CUDA (WMMA, hand-written)": lambda: ext.gemm(a, b),
            "cuBLAS (torch a @ b)": lambda: a @ b,
        }

        print(f"\n--- GEMM {m}x{n}x{k} ---")
        print(f"{'implementation':<28} {'ms':>9} {'TFLOP/s':>9} {'max abs err':>12}  ok")
        print("-" * 72)
        flops = 2.0 * m * n * k
        for name, fn in impls.items():
            out = fn()
            err = (out.float() - ref.float()).abs().max().item()
            ms = bench(fn)
            tol = 0.5 * (k / 1024) ** 0.5
            print(f"{name:<28} {ms:>9.4f} {flops / (ms * 1e-3) / 1e12:>9.1f} {err:>12.4f}  "
                  f"{'yes' if err < tol else 'NO'}")
    print()


if __name__ == "__main__":
    main()
