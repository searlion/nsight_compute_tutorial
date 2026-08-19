// Hand-written CUDA GEMM: C = A @ B, fp16 in / fp32 accumulate / fp16 out.
// Functional equivalent of gemm_tilelang.py, same 128x128x32 block tiling,
// tensor cores via WMMA, double-buffered shared memory.
//
// Build (the Makefile discovers nvcc and the GPU arch, nothing to edit):
//   make
// Run:
//   ./gemm_cuda [M N K]

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define BM 128            // block tile M
#define BN 128            // block tile N
#define BK 32             // block tile K
#define WM 2              // warps along M
#define WN 4              // warps along N
#define NWARPS (WM * WN)  // 8 warps
#define NTHREADS (NWARPS * 32)

#define WTILE_M (BM / WM)  // 64 rows of C per warp
#define WTILE_N (BN / WN)  // 32 cols of C per warp
#define FM (WTILE_M / 16)  // 4 wmma frags along M
#define FN (WTILE_N / 16)  // 2 wmma frags along N

#define APAD 8
#define BPAD 8
#define AS_LD (BK + APAD)  // 40
#define BS_LD (BN + BPAD)  // 136

#define CUDA_CHECK(x)                                                                    \
  do {                                                                                   \
    cudaError_t e = (x);                                                                 \
    if (e != cudaSuccess) {                                                              \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);      \
      exit(1);                                                                           \
    }                                                                                    \
  } while (0)

// ---------------------------------------------------------------------------
// The kernel. Everything TileLang inferred -- thread->data mapping, the async
// pipeline, the fragment layouts, the swizzle -- is spelled out by hand here.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(NTHREADS) void gemm_kernel(
    const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
    int M, int N, int K) {
  // Two shared buffers per operand: one being computed on, one being filled.
  __shared__ half As[2][BM * AS_LD];
  __shared__ half Bs[2][BK * BS_LD];

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int warp_m = warp / WN;  // 0..1
  const int warp_n = warp % WN;  // 0..3

  const int block_m = blockIdx.y * BM;
  const int block_n = blockIdx.x * BN;

  // --- global -> shared index math, done once ------------------------------
  // A tile is 128x32 halves = 512 float4; 256 threads -> 2 float4 each.
  const int a_row = tid / 4;        // 0..63, +64 on second iteration
  const int a_col8 = (tid % 4) * 8; // byte-vector column offset (8 halves)
  // B tile is 32x128 halves = 512 float4; 2 float4 each.
  const int b_row = tid / 16;        // 0..15, +16 on second iteration
  const int b_col8 = (tid % 16) * 8;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  const int ktiles = (K + BK - 1) / BK;

  // Manual prologue: fill stage 0. TileLang's T.Pipelined generates this.
  {
    const int k0 = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      int r = a_row + i * 64;
      const half* src = A + (size_t)(block_m + r) * K + k0 * BK + a_col8;
      *(float4*)(&As[0][r * AS_LD + a_col8]) = *(const float4*)src;
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      int r = b_row + i * 16;
      const half* src = B + (size_t)(k0 * BK + r) * N + block_n + b_col8;
      *(float4*)(&Bs[0][r * BS_LD + b_col8]) = *(const float4*)src;
    }
  }
  __syncthreads();

  for (int kt = 0; kt < ktiles; ++kt) {
    const int cur = kt & 1;
    const int nxt = cur ^ 1;

    // Prefetch the next K-tile into the other buffer while we compute on this one.
    if (kt + 1 < ktiles) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        int r = a_row + i * 64;
        const half* src = A + (size_t)(block_m + r) * K + (kt + 1) * BK + a_col8;
        *(float4*)(&As[nxt][r * AS_LD + a_col8]) = *(const float4*)src;
      }
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        int r = b_row + i * 16;
        const half* src = B + (size_t)((kt + 1) * BK + r) * N + block_n + b_col8;
        *(float4*)(&Bs[nxt][r * BS_LD + b_col8]) = *(const float4*)src;
      }
    }

    // --- the actual math: 2 wmma k-steps per 32-wide K tile ---------------
#pragma unroll
    for (int kf = 0; kf < BK / 16; ++kf) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af[FM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf[FN];
#pragma unroll
      for (int i = 0; i < FM; ++i) {
        int r = warp_m * WTILE_M + i * 16;
        wmma::load_matrix_sync(af[i], &As[cur][r * AS_LD + kf * 16], AS_LD);
      }
#pragma unroll
      for (int j = 0; j < FN; ++j) {
        int c = warp_n * WTILE_N + j * 16;
        wmma::load_matrix_sync(bf[j], &Bs[cur][(kf * 16) * BS_LD + c], BS_LD);
      }
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
    }
    __syncthreads();
  }

  // --- epilogue: fp32 accumulators -> fp16 global -------------------------
  // Reuse the (now dead) shared tiles as a float staging area, 256 floats/warp.
  float* stage = reinterpret_cast<float*>(&As[0][0]) + warp * 256;
  const int lane = tid % 32;
#pragma unroll
  for (int i = 0; i < FM; ++i) {
#pragma unroll
    for (int j = 0; j < FN; ++j) {
      wmma::store_matrix_sync(stage, acc[i][j], 16, wmma::mem_row_major);
      __syncwarp();
      int base_r = block_m + warp_m * WTILE_M + i * 16;
      int base_c = block_n + warp_n * WTILE_N + j * 16;
      // 32 lanes write 256 elements: 8 each.
      for (int e = lane; e < 256; e += 32) {
        int r = e / 16, c = e % 16;
        C[(size_t)(base_r + r) * N + (base_c + c)] = __float2half(stage[e]);
      }
      __syncwarp();
    }
  }
}

// Naive reference for correctness (fp32 accumulate).
__global__ void gemm_naive(const half* A, const half* B, half* C, int M, int N, int K) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= M || c >= N) return;
  float s = 0.f;
  for (int k = 0; k < K; ++k) s += __half2float(A[(size_t)r * K + k]) * __half2float(B[(size_t)k * N + c]);
  C[(size_t)r * N + c] = __float2half(s);
}

int main(int argc, char** argv) {
  int M = 1024, N = 1024, K = 1024;
  if (argc == 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
  printf("GEMM %dx%dx%d  (block %dx%dx%d, %d threads)\n", M, N, K, BM, BN, BK, NTHREADS);

  size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;
  half *hA = (half*)malloc(szA * 2), *hB = (half*)malloc(szB * 2);
  half *hC = (half*)malloc(szC * 2), *hR = (half*)malloc(szC * 2);
  srand(0);
  for (size_t i = 0; i < szA; ++i) hA[i] = __float2half((rand() / (float)RAND_MAX) * 2.f - 1.f);
  for (size_t i = 0; i < szB; ++i) hB[i] = __float2half((rand() / (float)RAND_MAX) * 2.f - 1.f);

  half *dA, *dB, *dC, *dR;
  CUDA_CHECK(cudaMalloc(&dA, szA * 2)); CUDA_CHECK(cudaMalloc(&dB, szB * 2));
  CUDA_CHECK(cudaMalloc(&dC, szC * 2)); CUDA_CHECK(cudaMalloc(&dR, szC * 2));
  CUDA_CHECK(cudaMemcpy(dA, hA, szA * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, szB * 2, cudaMemcpyHostToDevice));

  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(NTHREADS);
  gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  dim3 g2((N + 15) / 16, (M + 15) / 16), b2(16, 16);
  gemm_naive<<<g2, b2>>>(dA, dB, dR, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(hC, dC, szC * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hR, dR, szC * 2, cudaMemcpyDeviceToHost));
  double maxerr = 0;
  for (size_t i = 0; i < szC; ++i) {
    double d = fabs((double)__half2float(hC[i]) - (double)__half2float(hR[i]));
    if (d > maxerr) maxerr = d;
  }
  printf("max abs diff vs naive: %.5f  -> %s\n", maxerr, maxerr < 0.5 ? "OK" : "MISMATCH");

  // benchmark
  const int warm = 20, iters = 200;
  for (int i = 0; i < warm; ++i) gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (int i = 0; i < iters; ++i) gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
  cudaEventRecord(t1); CUDA_CHECK(cudaEventSynchronize(t1));
  float ms = 0; cudaEventElapsedTime(&ms, t0, t1); ms /= iters;
  printf("latency: %.4f ms  ->  %.1f TFLOP/s\n", ms, 2.0 * M * N * K / (ms * 1e-3) / 1e12);
  return 0;
}
