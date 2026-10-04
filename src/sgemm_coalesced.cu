// =====================================================================
// sgemm_coalesced.cu — Kernel 1：访存合并（AR002）
// ---------------------------------------------------------------
// 相对 Kernel 0 的唯一变化：交换线程映射，使 warp 内 32 个线程
//   * 连续读 B（相邻线程相邻列 → 128B 对齐合并事务）
//   * 连续写 C（同理）
//   * A 的访问退化为同地址广播（warp 内 32 线程读同一 A[row][kk]）
// 仍然【无分块、无复用】：A/B 被重复读取 O(N)/O(M) 次，保持 memory-bound，
// 为 AR003 提供"重复读取流量 ≈ 理论最小值数十倍"的证据（E4 实验）。
// smem 用量：0。预期性能：~740.44 GFLOPS（6.52x vs naive）。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>

namespace {

__global__ void sgemm_coalesced_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K) {
    // 合并映射：x -> 列（跨 N），y -> 行（跨 M）
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int kk = 0; kk < K; ++kk) {
            // warp 内：B 访问连续（合并），A 访问同地址（硬件广播）
            acc += A[row * (long long)K + kk] * B[(long long)kk * N + col];
        }
        C[row * (long long)N + col] = acc;   // 合并写回
    }
}

}  // namespace

void sgemm_coalesced(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);   // 与 naive 相同的块形状：保证 A/B 对比只有映射变量
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    sgemm_coalesced_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[sgemm_coalesced] launch failed: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}
