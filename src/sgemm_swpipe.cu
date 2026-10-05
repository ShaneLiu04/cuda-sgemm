// =====================================================================
// sgemm_swpipe.cu — Kernel 6：单缓冲软件流水（AR007 扩展变体）
// ---------------------------------------------------------------
// 动机：sm_75（Quadro RTX 5000）无 cp.async 异步拷贝硬件，AR006 两个
// cp.async 变体在本机均为负收益（0.88x / 0.98x vs vec4）。本版改用
// 纯软件流水：把「每 tile 的全局 LDG」提前一轮发射到寄存器，
// store→smem 与 compute 交替执行，LDG 延迟被 64 FMA/kstep 的计算覆盖。
//
// 布局与 vec4 完全一致（bank 冲突结论直接继承，见 sgemm_vec4.cu 头注释）：
//   - tile：BM=BN=128, BK=8；每线程 TM=TN=8（64 累加器），block 16x16=256 线程
//   - A 转置 As[BK][BM+PAD_A=4]（全局沿 K float4 读、4 次标量散射写；PAD
//     使 warp 内转置写/片段读均 0 冲突）
//   - B 保持 Bs[BK][BN]，16B 单位 XOR swizzle（unit ^ (krow & 7)），
//     消除 b-frag stride-8 4-way 冲突
//   - 搬运划分：A: row=tid/2, kquad=tid%2；B: krow=tid/32, unit=tid%32
//
// 流水（单缓冲，2 阶段 / 缓冲数=1）：
//   预取 tile0 → [ store(t)→smem | S1 同步 | 预取(t+1)→reg |
//                  compute(t) 从 smem | S2 同步 ] 循环 → 回写 C
//   屏障契约：S1 = tile t 全块就绪才计算；S2 = 全块读完 As/Bs 后
//   下一轮才可覆写（单缓冲写读隔离的正确性核心）。
//   寄存器预算：64 acc + 8（a_reg/b_reg 持久）+ 瞬态片段；
//   __launch_bounds__(256, 2) = 128 regs 封顶（AR008 实测固化，见 kernel 体内
//   注释；spill=0 为硬门）。
//
// 主路径条件（16B 对齐）：N%4==0 且 K%4==0 且三指针 16B 对齐；
//   不满足 → 回退 sgemm_2d_tile（任意尺寸正确），--verbose 打印。
// 正确性与 vec4 同累加顺序（主路径逐位一致）。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>
#include <cstdint>

namespace {

constexpr int BM = 128, BN = 128, BK = 8;
constexpr int TM = 8,   TN = 8;
constexpr int PAD_A = 4;                       // As 行距 132 floats（16B 对齐保持）

__device__ __forceinline__ float4 load_zero_guard(const float* base, long long idx,
                                                  bool valid) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (valid) v = *reinterpret_cast<const float4*>(base + idx);
    return v;
}

// tile 主体（AR008 参数化：z=blockIdx.z 切片，k-tile 区间 [z*tps, min((z+1)*tps, num_tiles))，
// 输出 Out = Out_base + z*M*N）。单波路径（grid.z=1, tps=num_tiles）与原语义逐位等价：
// t0=0、t1=num_tiles、Out=C。空片（t0>=t1）自然写全零——swsk 的 sk>tiles 边界语义。
//
// 资源契约（AR008 实测固化）：固定 __launch_bounds__(256, 2) = 128 regs 封顶 /
// 2-block 占用包络。参数化新增 tps/num_tiles/z 寻址后，无约束（minBlocks=1）实例
// 自然分配 130 regs → 1 block/SM，1024³ 实测 4314 vs 4617 GF（-7.0%，wave 敏感
// 尺寸靠双 block 驻留重叠 store/compute）；AR007 基线（127 regs）本属 2-block
// 级，故 128-reg 封顶即历史默认行为的等价物，LB 消融旋钮对 swpipe/swsk 语义
// 收窄为 no-op 并移除（--lb 仅剩 tile2d/ws，详设 §4.6 同步）。
__global__ __launch_bounds__(256, 2)
void sgemm_swpipe_kernel(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ Out_base,
                         int M, int N, int K,
                         int tps, int num_tiles) {
    // smem：A 转置 [BK][BM+PAD]；B swizzle [BK][BN]（与 vec4 相同布局）
    __shared__ __align__(16) float As[BK][BM + PAD_A];
    __shared__ __align__(16) float Bs[BK][BN];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int z  = blockIdx.z;                  // split-K 切片号（单波路径恒 0）
    const int t0 = z * tps;
    const int t1 = min(t0 + tps, num_tiles);
    float* __restrict__ Out = Out_base + (long long)z * M * N;

    const int tx = threadIdx.x, ty = threadIdx.y;   // block(16,16)
    const int tid = ty * 16 + tx;

    float c[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) c[i][j] = 0.0f;

    // 搬运划分（与 vec4 完全一致，每线程每 tile 各 1 个 float4）
    const int ld_a_row  = tid >> 1;
    const int ld_a_kq   = tid & 1;
    const int ld_b_krow = tid >> 5;
    const int ld_b_unit = tid & 31;

    const int a_glb_row = by * BM + ld_a_row;
    const int b_glb_col = bx * BN + ld_b_unit * 4;

    // ---- 流水预取：本片首 tile 的 A/B float4 先入寄存器（越界零守卫）----
    const int k_first = t0 * BK;
    float4 a_reg = load_zero_guard(
        A, (long long)a_glb_row * K + k_first + ld_a_kq * 4,
        (t0 < t1) && (a_glb_row < M) && (k_first + ld_a_kq * 4 < K));
    float4 b_reg = load_zero_guard(
        B, (long long)(k_first + ld_b_krow) * N + b_glb_col,
        (t0 < t1) && (k_first + ld_b_krow < K) && (b_glb_col < N));

    for (int t = t0; t < t1; ++t) {
        // ---- ① 寄存器 → smem（A 转置散射；B swizzle 直拷）----
        {
            const float* af = reinterpret_cast<const float*>(&a_reg);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                As[ld_a_kq * 4 + kk][ld_a_row] = af[kk];
            }
        }
        {
            // 物理 16B 单位 = 逻辑单位 ^ (krow & 7)
            reinterpret_cast<float4*>(&Bs[ld_b_krow][0])[ld_b_unit ^ (ld_b_krow & 7)] = b_reg;
        }
        __syncthreads();   // S1：tile t 全块就绪

        // ---- ② 预取 tile t+1 → 寄存器（LDG 提前发射，延迟由 ③ 覆盖）----
        if (t + 1 < t1) {
            const int k0 = (t + 1) * BK;
            a_reg = load_zero_guard(
                A, (long long)a_glb_row * K + k0 + ld_a_kq * 4,
                (a_glb_row < M) && (k0 + ld_a_kq * 4 < K));
            b_reg = load_zero_guard(
                B, (long long)(k0 + ld_b_krow) * N + b_glb_col,
                (k0 + ld_b_krow < K) && (b_glb_col < N));
        }

        // ---- ③ 计算主循环（与 vec4 相同：每 k 一步 2+2 LDS.128 → 64 FMA）----
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float4 a0, a1, b0, b1;
            const float4* as_row = reinterpret_cast<const float4*>(&As[kk][0]);
            a0 = as_row[ty * 2];
            a1 = as_row[ty * 2 + 1];
            const float4* bs_row = reinterpret_cast<const float4*>(&Bs[kk][0]);
            const int sw = kk & 7;
            b0 = bs_row[(tx * 2) ^ sw];
            b1 = bs_row[(tx * 2 + 1) ^ sw];
            const float af[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bf[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    c[i][j] = af[i] * bf[j] + c[i][j];
        }
        __syncthreads();   // S2：全块读完 As/Bs，下一轮才可覆写
    }

    // ---- 回写：每行 2x float4（N%4==0 ⇒ quad 全有或全无），warp 合并 ----
    // （split-K 模式写部分积 Out = P + z*M*N；空片 z 自然写全零）
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
                    *reinterpret_cast<float4*>(&Out[(long long)row * N + col]) = v;
                }
            }
        }
    }
}

}  // namespace

namespace sgemm { namespace detail {

// swpipe tile 主体的参数化复用入口（AR008 swsk 共享；单波路径由本文件 wrapper 使用）
// grid = (grid_n, grid_m, grid_z)；Out_base 需为可容纳 grid_z 个 M*N 切片的基址
// （单波：Out_base=C, grid_z=1；split-K：Out_base=P, grid_z=sk）。
// 资源：单一实例 __launch_bounds__(256,2) = 128 regs / 2-block 包络（见 kernel 注释）。
void swpipe_tile_grid(const float* A, const float* B, float* Out_base,
                      int M, int N, int K, int tps, int num_tiles,
                      int grid_n, int grid_m, int grid_z) {
    const dim3 grid(grid_n, grid_m, grid_z);
    const dim3 block(16, 16, 1);
    sgemm_swpipe_kernel<<<grid, block>>>(A, B, Out_base, M, N, K, tps, num_tiles);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[swpipe_tile_grid] launch failed: %s\n",
                     cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

}}  // namespace sgemm::detail

void sgemm_swpipe(const float* A, const float* B, float* C, int M, int N, int K) {
    // 主路径条件：与 vec4 相同（N、K 为 4 的倍数且三指针 16B 对齐）
    const bool aligned =
        (N % 4 == 0) && (K % 4 == 0) &&
        (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0);

    if (!aligned) {
        if (sgemm::g_verbose) {
            std::printf("[swpipe] size %dx%dx%d -> scalar fallback (tile2d)\n", M, N, K);
        }
        sgemm_2d_tile(A, B, C, M, N, K);   // 谓词化回退路径（任意尺寸正确）
        return;
    }

    const int num_tiles = (K + BK - 1) / BK;
    const int grid_n = (N + BN - 1) / BN;
    const int grid_m = (M + BM - 1) / BM;
    // 单波路径：z=1、tps=num_tiles —— 与参数化前语义逐位等价
    sgemm::detail::swpipe_tile_grid(A, B, C, M, N, K, num_tiles, num_tiles,
                                    grid_n, grid_m, 1);
}
