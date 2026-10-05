// =====================================================================
// sgemm_naive.cu — Kernel 0：Naive 基线（AR001）
// ---------------------------------------------------------------
// 设计（刻意"教科书式"，禁止任何优化 —— 它是 1x 加速比基准，详设 §5）：
//   * 2D 线程块 16x16，每线程计算一个 C[row][col]。
//   * 刻意将 row 映射到 threadIdx.x（教科书常见写法，未考虑合并访存）：
//       - warp 内相邻线程访问 A/B/C 时按行方向跨步（stride = K 或 N），
//         每次访存拆成 32 个独立事务 → DRAM 流量爆炸；
//       - 预期 ncu 证据：sectors/request >> 4、long scoreboard stall 主导。
//   * 无 shared memory、无复用、K 维串行累加。
// 本机实测（4096^3, Quadro RTX 5000, sm_75, 2026-10-04）：155.22 GFLOPS
//   （= 理论 FP32 峰值 11.15 TF 的 1.4%；计时证据与推导见
//    results/bottleneck_analysis.md Kernel 0 节）。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>

namespace {

__global__ void sgemm_naive_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    // 刻意的不合并映射：x -> 行，y -> 列
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
            acc += A[row * (long long)K + kk] * B[(long long)kk * N + col];
        }
        C[row * (long long)N + col] = acc;
    }
}

}  // namespace

void sgemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    sgemm_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[sgemm_naive] launch failed: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}
