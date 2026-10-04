// =====================================================================
// sgemm_2d_tile.cu — Kernel 3：二维寄存器分块 128x128x8（AR004）
// ---------------------------------------------------------------
// Block Tile：BM = BN = 128，BK = 8；线程块 16x16 = 256 线程。
// Thread Tile：TM = TN = 8 —— 每线程 64 个 fp32 累加器 c[8][8]，
//   负责子块 C[by*128 + ty*8 + i][bx*128 + tx*8 + j]，i,j ∈ [0,8)。
// 内层（每 k 一步）：读 smem 片段 a[8]、b[8]，做 64 次外积 FMA
//   c[i][j] += a[i] * b[j] —— 最大化寄存器复用与 ILP（全展开）。
// smem 布局与 padding 理由：
//   * As[128][8 + 5] ：PAD_A=5 —— a 片段读 As[ty*8+i][kk]，warp 内 ty∈{0,1}
//     两行相差 8，行距 13 floats → 8*13=104 ≡ 8 (mod 32) → 无 bank conflict；
//     若 PAD_A=4（行距 12）：8*12=96 ≡ 0 (mod 32) → 2-way conflict。
//     加载写入侧：行距 13 与 32 互素 → 16 行映射 16 个不同 bank → 无冲突。
//   * Bs[8][128 + 4] ：b 片段读 Bs[kk][tx*8+j]，warp 内 16 个 tx 相距 8
//     → 4-way bank conflict【已知且记录】：pad 无法消除（stride-8 固有），
//     由 AR005 的 swizzle 治理；此处保留作为 ncu 证据链（E7/E9）。
// 资源：64 acc + 16 片段 + 寻址 ≈ 100+ 寄存器；__launch_bounds__(256, lb)
//   显式控制 ptxas 分配，lb ∈ {1,2} 消融（CLI --lb）；spill=0 是硬门。
// 预期性能：~3.51 TFLOPS（2.46x vs smem_1d）。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>

namespace sgemm {
int g_tile2d_min_blocks = 1;
}

namespace {

constexpr int BM = 128, BN = 128, BK = 8;   // block tile
constexpr int TM = 8,   TN = 8;             // thread tile
constexpr int PAD_A = 5;                    // 见文件头注释
constexpr int PAD_B = 4;                    // 装载写入侧消冲突；读侧冲突为已知项

template <int MIN_BLOCKS>
__global__ void __launch_bounds__(256, MIN_BLOCKS)
sgemm_2d_tile_kernel(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int M, int N, int K) {
    __shared__ float As[BM][BK + PAD_A];    // [128][13]
    __shared__ float Bs[BK][BN + PAD_B];    // [8][132]

    const int bx = blockIdx.x;              // 列方向 tile（步长 128）
    const int by = blockIdx.y;              // 行方向 tile
    const int tx = threadIdx.x;             // [0,16)
    const int ty = threadIdx.y;             // [0,16)
    const int tid = ty * 16 + tx;

    float c[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) c[i][j] = 0.0f;

    // 加载任务划分：
    //   A tile 128x8 / 256 线程 → 每线程 1 行段 x 4 个 k（两个 4float 段/两线程一行）
    //   B tile 8x128 / 256 线程 → 每线程 1 个 k 行 x 4 列
    const int load_a_row = tid >> 1;            // [0,128) tile 内行
    const int load_a_k   = (tid & 1) * 4;       // 0 或 4
    const int load_b_k   = tid >> 5;            // [0,8)  tile 内 k
    const int load_b_col = (tid & 31) * 4;      // [0,128) tile 内列段

    const int a_glb_row = by * BM + load_a_row;
    const int b_glb_col = bx * BN + load_b_col;

    const int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        const int k0 = t * BK;

        // ---- 加载 A tile（含边界补零；warp 覆盖 16 行 x 8 k → 行段 32B）----
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int gr = a_glb_row;
            const int gk = k0 + load_a_k + i;
            As[load_a_row][load_a_k + i] =
                (gr < M && gk < K) ? A[(long long)gr * K + gk] : 0.0f;
        }
        // ---- 加载 B tile（含边界补零；warp 覆盖 128 连续列 → 合并）----
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int gk = k0 + load_b_k;
            const int gc = b_glb_col + i;
            Bs[load_b_k][load_b_col + i] =
                (gk < K && gc < N) ? B[(long long)gk * N + gc] : 0.0f;
        }
        __syncthreads();

        // ---- 外积主循环：每 k 一步，8+8 次 smem 读 → 64 FMA ----
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[ty * TM + i][kk];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[kk][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    c[i][j] = a[i] * b[j] + c[i][j];
        }
        __syncthreads();
    }

    // ---- 写回（标量；AR005 升级为 2x float4）----
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = by * BM + ty * TM + i;
        if (row < M) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = bx * BN + tx * TN + j;
                if (col < N) C[(long long)row * N + col] = c[i][j];
            }
        }
    }
}

}  // namespace

void sgemm_2d_tile_ext(const float* A, const float* B, float* C,
                       int M, int N, int K, int lb) {
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    cudaError_t err;
    if (lb >= 2) {   // minBlocks=2：约束 regs<=128（2 block/SM），消融用
        sgemm_2d_tile_kernel<2><<<grid, block>>>(A, B, C, M, N, K);
    } else {
        sgemm_2d_tile_kernel<1><<<grid, block>>>(A, B, C, M, N, K);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[sgemm_2d_tile] launch failed (lb=%d): %s\n",
                     lb, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

void sgemm_2d_tile(const float* A, const float* B, float* C, int M, int N, int K) {
    sgemm_2d_tile_ext(A, B, C, M, N, K, sgemm::g_tile2d_min_blocks);
}
