// =====================================================================
// sgemm_vec4.cu — Kernel 4：float4 向量化（AR005）
// ---------------------------------------------------------------
// 在 2D 寄存器分块（128x128x8, 8x8）骨架上做三处 16B 向量化：
//   (1) global→smem 加载用 float4（指令数/事务数 /4）
//   (2) A tile 以转置布局 [BK][BM] 写入 smem：全局侧仍沿 K 维 float4 读取，
//       写入为 4 次标量散射；配合 PAD_A=4：
//         - 转置写：bank = ((kq*4+kk)*132 + row) mod 32 = (kq*4+kk)*4 + row
//           → warp 内 16 行 x 2 kq 全部错开 → 0 冲突
//         - 片段读：a-frag = 2x float4（As[kk][ty*8..ty*8+7]），warp 内仅
//           2 个 16B 段（ty=0/1），banks 0..3 / 8..11 → 0 冲突
//   (3) B tile 保持 [BK][BN]，但用 XOR swizzle（CUTLASS 风格，16B 单位）：
//         physical_unit = logical_unit ^ (kk & 7)
//       消除 b-frag 的 stride-8 4-way 冲突（AR004 记录的已知瓶颈）：
//       读侧 half-warp 16 线程 → 16 个不同 16B 单位；写侧 warp 32 单位全错开。
//   回写：每线程每行 8 个连续输出 = 2x float4，warp 级合并。
// 主路径条件（16B 对齐）：N % 4 == 0 且 K % 4 == 0 且指针 16B 对齐；
//   不满足 → 回退 sgemm_2d_tile（标量、任意尺寸），--verbose 打印路径。
// M 任意（行守卫）；边界 quad 由"N%4==0 ⇒ quad 全有或全无"化简。
// 预期性能：~5.84 TFLOPS（1.66x vs tile2d）；bank conflict ≈ 0。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>
#include <cstdint>

namespace {

constexpr int BM = 128, BN = 128, BK = 8;
constexpr int TM = 8,   TN = 8;
constexpr int PAD_A = 4;                       // As 行距 132 floats（16B 对齐保持）

union Frag8 {                                  // 片段寄存器：可按 float4 装载、按 float 使用
    float4 v4[2];
    float  f[8];
};

__device__ __forceinline__ float4 load_zero_guard(const float* base, long long idx,
                                                  bool valid) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (valid) v = *reinterpret_cast<const float4*>(base + idx);
    return v;
}

__global__ __launch_bounds__(256)
sgemm_vec4_kernel(const float* __restrict__ A,
                  const float* __restrict__ B,
                  float* __restrict__ C,
                  int M, int N, int K) {
    // smem：A 转置 [BK][BM+PAD]；B swizzle [BK][BN]（32 个 16B 单位/行）
    __shared__ __align__(16) float As[BK][BM + PAD_A];
    __shared__ __align__(16) float Bs[BK][BN];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;   // block(16,16)
    const int tid = ty * 16 + tx;

    float c[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) c[i][j] = 0.0f;

    // 加载划分（每线程恰好 1 个 float4 / tile / 矩阵）：
    //   A: row = tid/2, kquad = tid%2  （128 行 x 2 个 4-k 段）
    //   B: krow = tid/32, unit = tid%32（8 行 x 32 个 16B 单位）
    const int ld_a_row  = tid >> 1;
    const int ld_a_kq   = tid & 1;
    const int ld_b_krow = tid >> 5;
    const int ld_b_unit = tid & 31;

    const int a_glb_row = by * BM + ld_a_row;
    const int b_glb_col = bx * BN + ld_b_unit * 4;

    const int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; ++t) {
        const int k0 = t * BK;

        // ---- (1)(2) A：沿 K 维 float4 读，转置散射写 smem ----
        {
            const bool valid = (a_glb_row < M) && (k0 + ld_a_kq * 4 < K);
            const float4 av = load_zero_guard(
                A, (long long)a_glb_row * K + k0 + ld_a_kq * 4, valid);
            const float* af = reinterpret_cast<const float*>(&av);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                As[ld_a_kq * 4 + kk][ld_a_row] = af[kk];
            }
        }
        // ---- (1)(3) B：沿 N 维 float4 读（warp 128 列合并），swizzle 写 ----
        {
            const bool valid = (k0 + ld_b_krow < K) && (b_glb_col < N);
            const float4 bv = load_zero_guard(
                B, (long long)(k0 + ld_b_krow) * N + b_glb_col, valid);
            // 物理 16B 单位 = 逻辑单位 ^ (krow & 7)
            reinterpret_cast<float4*>(&Bs[ld_b_krow][0])[ld_b_unit ^ (ld_b_krow & 7)] = bv;
        }
        __syncthreads();

        // ---- 外积主循环：每 k 一步 2+2 次 LDS.128 → 64 FMA ----
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            Frag8 a, b;
            const float4* as_row = reinterpret_cast<const float4*>(&As[kk][0]);
            a.v4[0] = as_row[ty * 2];
            a.v4[1] = as_row[ty * 2 + 1];

            const float4* bs_row = reinterpret_cast<const float4*>(&Bs[kk][0]);
            const int sw = kk & 7;
            b.v4[0] = bs_row[(tx * 2) ^ sw];
            b.v4[1] = bs_row[(tx * 2 + 1) ^ sw];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    c[i][j] = a.f[i] * b.f[j] + c[i][j];
        }
        __syncthreads();
    }

    // ---- 回写：每行 2x float4（N%4==0 ⇒ quad 全有或全无）----
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = by * BM + ty * TM + i;
        if (row < M) {
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int col = bx * BN + tx * TN + q * 4;
                if (col < N) {
                    float4 v;
                    v.x = c[i][q * 4 + 0];
                    v.y = c[i][q * 4 + 1];
                    v.z = c[i][q * 4 + 2];
                    v.w = c[i][q * 4 + 3];
                    *reinterpret_cast<float4*>(&C[(long long)row * N + col]) = v;
                }
            }
        }
    }
}

}  // namespace

void sgemm_vec4(const float* A, const float* B, float* C, int M, int N, int K) {
    // 主路径条件：N、K 为 4 的倍数且三个指针 16B 对齐（cudaMalloc 天然满足）
    const bool aligned =
        (N % 4 == 0) && (K % 4 == 0) &&
        (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0);

    if (!aligned) {
        if (sgemm::g_verbose) {
            std::printf("[vec4] size %dx%dx%d -> scalar fallback (tile2d)\n", M, N, K);
        }
        sgemm_2d_tile(A, B, C, M, N, K);   // 谓词化回退路径（任意尺寸正确）
        return;
    }

    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_vec4_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[sgemm_vec4] launch failed: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}
