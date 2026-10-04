// =====================================================================
// sgemm_smem_1d.cu — Kernel 2：Shared Memory 32x32 分块 + 1D TM=8（AR003）
// ---------------------------------------------------------------
// Tile / 线程分配：
//   * Block Tile：BM = BN = 32（一个 block 负责 C 的 32x32 子块）
//   * BK：K 向扫描步长，模板参数消融 {8,16,32}（CLI --bk，默认 16）
//   * 线程块：128 线程 = (32, 4)：tx ∈ [0,32) → 列，ty ∈ [0,4) → 行组
//   * 1D Thread Tiling：每线程 TM = 8，负责一列上的 8 个连续行输出
//     C[by*32 + ty*8 + i][bx*32 + tx]，i ∈ [0,8)
//     —— 计算密度 x8，同步/线程开销摊薄 x8
// smem 布局（含 padding 理由）：
//   * As[32][BK + 4] ：A tile（行 x k），pad=4 削弱写入冲突；
//       计算侧读取 As[ty*8+i][kk] 在 warp 内为同地址广播（无冲突）。
//   * Bs[BK][32 + 4] ：B tile（k x 列），pad=4；
//       计算侧读取 Bs[kk][tx] 为 32 连续 bank（无冲突）。
//   * 加载：线程按线性下标跨步搬运（含边界补零），全局侧按行段合并。
// 预期性能：~1.43 TFLOPS（1.94x vs coalesced）。
// 预期新瓶颈（AR004 输入）：barrier/short scoreboard 上升、load:FMA 比高。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>

namespace sgemm {
int g_smem1d_bk = 16;
}

namespace {

template <int BK>
__global__ void sgemm_smem_1d_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int TM = 8;                 // 每线程 8 个输出（1D thread tiling）
    constexpr int PAD = 4;                // smem padding（消写冲突，ncu 复核）

    __shared__ float As[BM][BK + PAD];    // A tile: 32 行 x BK 个 k
    __shared__ float Bs[BK][BN + PAD];    // B tile: BK 个 k x 32 列

    const int bx = blockIdx.x;            // 列方向 tile
    const int by = blockIdx.y;            // 行方向 tile
    const int tx = threadIdx.x;           // [0,32) -> 列
    const int ty = threadIdx.y;           // [0,4)  -> 行组（每组 8 行）
    const int tid = ty * blockDim.x + tx; // [0,128)

    float acc[TM];
#pragma unroll
    for (int i = 0; i < TM; ++i) acc[i] = 0.0f;

    const int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        const int k0 = t * BK;

        // ---- 协作加载 A tile（边界补零）----
        for (int idx = tid; idx < BM * BK; idx += 128) {
            const int r = idx / BK;
            const int c = idx % BK;
            const int gr = by * BM + r;
            const int gk = k0 + c;
            As[r][c] = (gr < M && gk < K) ? A[(long long)gr * K + gk] : 0.0f;
        }
        // ---- 协作加载 B tile（边界补零）----
        for (int idx = tid; idx < BK * BN; idx += 128) {
            const int r = idx / BN;      // k 方向
            const int c = idx % BN;      // n 方向
            const int gk = k0 + r;
            const int gc = bx * BN + c;
            Bs[r][c] = (gk < K && gc < N) ? B[(long long)gk * N + gc] : 0.0f;
        }
        __syncthreads();

        // ---- 从 smem 累加：每 k 步 1 次 B 读 + 8 次 FMA ----
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const float b = Bs[kk][tx];                    // warp 内 32 连续 bank
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                acc[i] += As[ty * TM + i][kk] * b;         // warp 内同地址广播
            }
        }
        __syncthreads();
    }

    // ---- 写回：warp 内 32 线程写同一行的 32 个连续列（合并）----
    const int col = bx * BN + tx;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = by * BM + ty * TM + i;
        if (row < M && col < N) {
            C[(long long)row * N + col] = acc[i];
        }
    }
}

}  // namespace

void sgemm_smem_1d_ext(const float* A, const float* B, float* C,
                       int M, int N, int K, int bk) {
    dim3 block(32, 4);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    cudaError_t err;
    switch (bk) {   // BK 消融：8/16/32（AR003 T003）
        case 8:  sgemm_smem_1d_kernel<8> <<<grid, block>>>(A, B, C, M, N, K); break;
        case 32: sgemm_smem_1d_kernel<32><<<grid, block>>>(A, B, C, M, N, K); break;
        default: sgemm_smem_1d_kernel<16><<<grid, block>>>(A, B, C, M, N, K); break;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[sgemm_smem_1d] launch failed (bk=%d): %s\n",
                     bk, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

void sgemm_smem_1d(const float* A, const float* B, float* C, int M, int N, int K) {
    sgemm_smem_1d_ext(A, B, C, M, N, K, sgemm::g_smem1d_bk);
}
