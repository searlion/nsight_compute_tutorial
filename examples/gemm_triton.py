"""Triton GEMM -- functional equivalent of gemm_tilelang.py.

Same tiling (128x128x32), same fp16 in / fp32 accumulate / fp16 out,
same 3-stage pipeline, same 128 threads (num_warps=4).
"""

import torch
import triton
import triton.language as tl

BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, NUM_WARPS = 128, 128, 32, 3, 4


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # TileLang's T.Kernel(ceildiv(N,bn), ceildiv(M,bm)) -> (bx, by).
    # Triton has no implicit grid object, so we read the ids explicitly.
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Unlike T.copy, every access is a raw pointer we compute ourselves.
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Equivalent of T.alloc_fragment(..., accum_dtype) + T.clear.
    # Note: no shared-memory declaration anywhere -- Triton decides that itself.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # num_stages is a decorator/launch hint in Triton, not a loop construct.
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)          # <- the T.gemm equivalent
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def matmul(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "incompatible shapes"
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_stages=NUM_STAGES, num_warps=NUM_WARPS,
    )
    return c
