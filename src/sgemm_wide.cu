// =====================================================================
// sgemm_wide.cu — Kernel 8：宽块占用率攻坚（AR009）
// ---------------------------------------------------------------
// 动机（AR008 四门 v2 遗留根因）：swpipe 系 256 线程 × TM=TN=8（64 acc）
// → 128 regs → 2 block/SM = 50% 占用封顶；512³(sk4) 仅 33% 占用 → 36%
// of peak。AR008 ws 消融已证明"50% 占用内重排指令流"无益——唯一出路
// 是抬占用率本身。本版把每线程累加器 64 → 32（TM=4×TN=8），块加宽到
// 512 线程（16 warp）：
//
//   tile：BM=BN=128, BK=8；block = 512 线程；每线程 TM=4 × TN=8（32 acc）
//   寄存器预算（design §4.2.3）：32 acc + 4 预取 + 12 片段 + ~14 寻址
//   ≈ 62 ≤ __launch_bounds__(512, 2) 上限 64 → 2 block/SM
//   = 1024 线程/SM（sm_75 上限恰 1024）= 32 warp = **100% 占用**
//   （regs 2×512×64 = 65536 恰满 64K/SM；smem 2×8320B = 16640B）
//   LB=1 为 spill 保底实例（≤128 regs，1 block/SM = 50%，单块 warp
//   翻倍仍改善中尺寸聚合占用：1024³ 67% vs swpipe 33%）。
//
// 布局/流水结论三承 swpipe（AR006-008 证实的正确结构，见
// sgemm_swpipe.cu 头注释）：
//   - smem：A 转置 As[BK][BM+PAD_A=4]（PAD 消转置写/片段读冲突）；
//     B 保持 Bs[BK][BN]，16B 单位 XOR swizzle（unit ^ (krow & 7)）
//   - bank 冲突论证（与 vec4/swpipe 逐位同型）：计算网格 ty=tid>>4,
//     tx=tid&15——A 读 as_row[ty]（warp 内 16 线程广播 2 地址，0 冲突）；
//     B 读 (tx*2)^sw / (tx*2+1)^sw（tx∈0..15 与 vec4 完全一致）
//   - 搬运划分（每 tile A+B 共 512 quads = 512 线程 × 恰 1 quad/线程）：
//       tid <  256 : A loader（row=tid>>1, kq=tid&1，同 swpipe）
//       tid >= 256 : B loader（krow=(tid-256)>>5, unit=(tid-256)&31，同 swpipe）
//     预取寄存器因单角色从 8 降到 4（统一 ld_reg 槽位，角色分支下
//     活跃区间不重叠）
//   - 流水：单缓冲双 __syncthreads（S1=tile 就绪 / S2=读完方可覆写），
//     预取混合流（LDG 提前一轮入寄存器，延迟被 32 FFMA/kstep 覆盖）
//
// 吞吐自检（design §4.2.4，每 SM 每 kstep 满驻留 2 block）：
//   LDS 2×512×3×16B = 48KB → 384 cy @128B/cy；FFMA 2×512×32 = 32768
//   → 512 cy @64/cy——计算仍主导（LSU 利用率 ~75%，比 swpipe 的 50%
//   紧张；用 +50% smem 流量换 +100% 占用）。
//
// 主体参数化（AR009 wsk 复用）：z=blockIdx.z 切片，k-tile 区间
// [z*tps, min((z+1)*tps, num_tiles))，输出 Out = Out_base + z*M*N。
// 单波路径（grid.z=1, tps=num_tiles）与单 kernel 语义逐位等价；
// 空片（t0>=t1）自然写全零——wsk 的 sk>tiles 边界语义。
//
// 主路径条件（16B 对齐）：N%4==0 且 K%4==0 且三指针 16B 对齐；
// 不满足 → 回退 sgemm_2d_tile（任意尺寸正确），--verbose 打印。
// 正确性与 swpipe 同累加序（k 升序 FMA 链）——主路径逐位一致
//（T002 数值序专项验证）。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>
#include <cstdint>

namespace sgemm {
int g_wide_min_blocks = 2;  // --wlb（默认 2 = 100% 占用目标；1 = 保底消融）
}

namespace {

constexpr int BM = 128, BN = 128, BK = 8;
constexpr int TM = 4,   TN = 8;                  // 每线程 32 acc（宽块核心取舍）
constexpr int PAD_A = 4;                         // As 行距 132 floats（16B 对齐保持）
constexpr int THREADS = 512;                     // 16 warp（sm_75 恰容纳 2 block/SM）

__device__ __forceinline__ float4 load_zero_guard(const float* base, long long idx,
                                                  bool valid) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (valid) v = *reinterpret_cast<const float4*>(base + idx);
    return v;
}

// tile 主体（AR009 参数化，镜像 swpipe 契约）：z 切片 + Out_base 偏移。
// 资源契约：LB=2 实例 ≤64 regs / 0 spill = 2 block/SM = 100% 占用
//（T002 ptxas 审计判据）；LB=1 实例 ≤128 regs / 0 spill 保底。
template <int LB>
__global__ __launch_bounds__(THREADS, LB)
void sgemm_wide_kernel(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ Out_base,
                       int M, int N, int K,
                       int tps, int num_tiles) {
    // smem：A 转置 [BK][BM+PAD]；B swizzle [BK][BN]（与 swpipe 相同布局）
    __shared__ __align__(16) float As[BK][BM + PAD_A];
    __shared__ __align__(16) float Bs[BK][BN];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int z  = blockIdx.z;                  // split-K 切片号（单波路径恒 0）
    const int t0 = z * tps;
    const int t1 = min(t0 + tps, num_tiles);

    const int tid = threadIdx.x;                // block(512)：一维扁平
    // 计算网格（全 512 线程）：行 32×TM4=128，列 16×TN8=128
    const int ty = tid >> 4;                    // 0..31
    const int tx = tid & 15;                    // 0..15

    float c[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) c[i][j] = 0.0f;

    // 搬运角色（静态划分，每 tile 每线程恰 1 quad；与 swpipe 同型）。
    // 基址/守卫预计算（ptxas 64-reg 封顶下的寄存器瘦身：per-tile 64 位
    // 乘加压成一次加法，Out 指针延迟到回写期才物化）
    const bool is_a_loader = (tid < 256);
    const int ld_a_row  = tid >> 1;             // 0..127（A loader）
    const int ld_a_kq   = tid & 1;              // 0..1
    const int ld_b_krow = (tid - 256) >> 5;     // 0..7（B loader）
    const int ld_b_unit = (tid - 256) & 31;     // 0..31

    const long long a_base = (long long)(by * BM + ld_a_row) * K + ld_a_kq * 4;
    const long long b_base = (long long)ld_b_krow * N + (bx * BN + ld_b_unit * 4);
    const bool a_row_ok = (by * BM + ld_a_row) < M;
    const bool b_col_ok = (bx * BN + ld_b_unit * 4) < N;
    const int  k_a_max  = K - ld_a_kq * 4;      // k0 < k_a_max ⇔ 四连 k 全有效
    const int  k_b_max  = K - ld_b_krow;        // k0 < k_b_max ⇔ 该行有效

    // ---- 流水预取：本片首 tile 的 1×float4 先入寄存器（越界零守卫）----
    //（统一 ld_reg 槽位：A/B loader 活跃区间不重叠，ptxas 复用同一组寄存器）
    const int k_first = t0 * BK;
    float4 ld_reg;
    if (is_a_loader) {
        ld_reg = load_zero_guard(
            A, a_base + k_first,
            (t0 < t1) && a_row_ok && (k_first < k_a_max));
    } else {
        ld_reg = load_zero_guard(
            B, b_base + (long long)k_first * N,
            (t0 < t1) && b_col_ok && (k_first < k_b_max));
    }

    for (int t = t0; t < t1; ++t) {
        // ---- ① 寄存器 → smem（A 转置散射 / B swizzle 直拷，角色分工）----
        if (is_a_loader) {
            const float* af = reinterpret_cast<const float*>(&ld_reg);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                As[ld_a_kq * 4 + kk][ld_a_row] = af[kk];
            }
        } else {
            // 物理 16B 单位 = 逻辑单位 ^ (krow & 7)
            reinterpret_cast<float4*>(&Bs[ld_b_krow][0])[ld_b_unit ^ (ld_b_krow & 7)] = ld_reg;
        }
        __syncthreads();   // S1：tile t 全块就绪

        // ---- ② 预取 tile t+1 → 寄存器（LDG 提前发射，延迟由 ③ 覆盖）----
        if (t + 1 < t1) {
            const int k0 = (t + 1) * BK;
            if (is_a_loader) {
                ld_reg = load_zero_guard(A, a_base + k0,
                                         a_row_ok && (k0 < k_a_max));
            } else {
                ld_reg = load_zero_guard(B, b_base + (long long)k0 * N,
                                         b_col_ok && (k0 < k_b_max));
            }
        }

        // ---- ③ 计算主循环：每 k 一步 1+2 LDS.128 → 32 FFMA ----
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float4 a0, b0, b1;
            const float4* as_row = reinterpret_cast<const float4*>(&As[kk][0]);
            a0 = as_row[ty];                     // As[kk][ty*4 .. +3]（16 线程广播）
            const float4* bs_row = reinterpret_cast<const float4*>(&Bs[kk][0]);
            const int sw = kk & 7;
            b0 = bs_row[(tx * 2) ^ sw];
            b1 = bs_row[(tx * 2 + 1) ^ sw];
            const float af[TM] = {a0.x, a0.y, a0.z, a0.w};
            const float bf[TN] = {b0.x, b0.y, b0.z, b0.w,
                                  b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    c[i][j] = af[i] * bf[j] + c[i][j];
        }
        __syncthreads();   // S2：全块读完 As/Bs，下一轮才可覆写
    }

    // ---- 回写：每行 2x float4（N%4==0 ⇒ quad 全有或全无），warp 合并 ----
    //（split-K 模式写部分积 Out = P + z*M*N；空片 z 自然写全零）
    float* __restrict__ Out = Out_base + (long long)z * M * N;
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

// wide tile 主体的参数化复用入口（AR009 wsk 共享；单波路径由本文件
// wrapper 使用）。grid = (grid_n, grid_m, grid_z)；Out_base 需为可容纳
// grid_z 个 M*N 切片的 16B 对齐基址（单波：Out_base=C, grid_z=1；
// split-K：Out_base=P, grid_z=sk）。LB 实例由 g_wide_min_blocks 选择。
void wide_tile_grid(const float* A, const float* B, float* Out_base,
                    int M, int N, int K, int tps, int num_tiles,
                    int grid_n, int grid_m, int grid_z) {
    const dim3 grid(grid_n, grid_m, grid_z);
    const dim3 block(THREADS, 1, 1);
    if (g_wide_min_blocks >= 2) {
        sgemm_wide_kernel<2><<<grid, block>>>(A, B, Out_base, M, N, K, tps, num_tiles);
    } else {
        sgemm_wide_kernel<1><<<grid, block>>>(A, B, Out_base, M, N, K, tps, num_tiles);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[wide_tile_grid] launch failed: %s\n",
                     cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

}}  // namespace sgemm::detail

void sgemm_wide(const float* A, const float* B, float* C, int M, int N, int K) {
    // 主路径条件：与 swpipe 相同（N、K 为 4 的倍数且三指针 16B 对齐）
    const bool aligned =
        (N % 4 == 0) && (K % 4 == 0) &&
        (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0);

    if (!aligned) {
        if (sgemm::g_verbose) {
            std::printf("[wide] size %dx%dx%d -> scalar fallback (tile2d)\n", M, N, K);
        }
        sgemm_2d_tile(A, B, C, M, N, K);   // 谓词化回退路径（任意尺寸正确）
        return;
    }

    const int num_tiles = (K + BK - 1) / BK;
    const int grid_n = (N + BN - 1) / BN;
    const int grid_m = (M + BM - 1) / BM;
    // 单波路径：z=1、tps=num_tiles —— 与单 kernel 语义逐位等价
    sgemm::detail::wide_tile_grid(A, B, C, M, N, K, num_tiles, num_tiles,
                                  grid_n, grid_m, 1);
}
