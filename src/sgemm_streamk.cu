// =====================================================================
// sgemm_streamk.cu — Kernel 10：Stream-K 统一调度（AR011 T002）
// ---------------------------------------------------------------------
// 动机（AR010 终态 + results/2026107.md）：两项结构性浪费统一消除——
//   ① 波量化：deep/dsk 的 grid 块数未必 48（SM 数）整倍数（2048³ 128 块
//      = 2.67 波，尾波 -11%）；
//   ② 归约第二 kernel：dsk direct 归约 41μs@1024³ + 启动 ~3μs + P 自片
//      DRAM 往返；1024³ 门缺口折算仅 1.05μs——归约侧结构性节省即翻门。
//
// 块映射（design §4.1 问题 1 方案 B，§4.2.1 数学）：
//   迭代空间单元 u = (C-tile c, k-tile kt)，tile-major 线性化 u = c·nt + kt，
//   总量 TOT = tiles·nt；[0, TOT) 连续切成 B = 48·W 块（波填充率恒 100%），
//   每块 U = ceil(TOT/B) 单元。C-tile c 由连续块号闭区间覆盖：
//     b_lo(c) = (c·nt)/U，  b_hi(c) = (min((c+1)·nt, TOT) - 1)/U
//   （块号升序 ≡ k 升序，cover(c) = b_hi-b_lo+1）；块 b 对 tile c 的
//   k-tile 区间 [t0, t1)：t0 = b·U - c·nt（截 0），t1 = min((b+1)·U-c·nt, nt)。
//   例 1024³ W=2：tiles=32, nt=128, U=43 → tile0 切点 {43,86} 与 dsk sk3
//   的 tps=43 逐点重合（bitwise 锚链构造基础，tests 专项）。
//
// 票据归并（design §4.2.2 + T002 修订，per-tile 票据 = AR008 全局单块否决案
// 的正确变体：无全局串行化）：
//   每 tile c、每覆盖块：compute → store P（tile-local）→ 每写者线程
//   __threadfence()（release：fence 只 flush 调用线程自身写，P 为全线程
//   分摊写故逐线程 fence）→ __syncthreads() 块收敛 → tid==0 单线程
//   atomicAdd(tick[c])（每块 +1）经共享标志广播；old==cover-1 的赢家
//   （最后完成者）全线程执行归并（块级统一分支无 warp 发散）。
//   归并 = F2 全 P 归并（T002 修订默认）：赢家将 c[i][j] 清零复用为累加器，
//   按 s=b_lo..b_hi 升序 __ldcs 读全部 P 切片（含 own）——链
//   0+P[b_lo]+…+P[b_hi] 与 dsk 全 P 归约链**恒逐位一致**，且与 winner 身份
//   无关（确定性天然成立；F1 寄存器代入在 winner 居中且 cover≥3 时错结合序，
//   降级为 cover==2 备选，见 design §4.2.2 修订注）。
//   票据自清洁：赢家归并写 C 后 tid==0 写 tick[c]=0（跨 launch 由流序保证；
//   首配及 realloc 由 ensure() cudaMemset 兜底——launch 中途异常即进程退出，不回收）。
//   cover(c)==1 快路径：唯一覆盖块跳过 P/票据，直写 C（deep 同型，零 P 流量）。
//
// 主体自包含（design §4.1 问题 3 方案 B）：deep 计算主体逐拷贝——
//   tile BM=256×BN=128×BK=8，block 16×16=256 线程，每线程 TM=16×TN=8=128
//   acc；A 转置 As[BK][BM+PAD4]（loader 逐行、计算期 warp 广播 0 冲突）；
//   B Bs[BK][BN] 16B XOR swizzle（unit^(krow&7)，swpipe 已证 0 冲突域）；
//   寄存器预取流水（2×A quad + 1×B quad）；LDS:FFMA = 6:128 = 1:21.3；
//   模板 <DBUF> 双实例：1 = 双缓冲单同步（As[2]+Bs[2]=24832B，默认），
//   0 = 单缓冲双同步（swpipe 同构）。**c 边界屏障（DBUF=1 专属）**：块跨
//   tile 时上一 c 末 compute(t1-1) 与下一 c 首 store(t0) 可能同 buf 奇偶
//   （nt 奇数时），c 循环顶补 __syncthreads() 隔离（DBUF=0 自带 S2 免）。
//   寄存器预算：deep 同基线（243 regs dbuf1）+ 票据/归并寻址 ~10；
//   __launch_bounds__(256,1) = 255 封顶，0 spill 硬门（build.log 审计）。
//
// P workspace：per-block 槽位索引（T002 修订）slice(b,c) = b·SLOTS +
// (c - c_lo(b))，SLOTS = ceil(U/nt)+1；容量 (B·SLOTS)·BM·BN floats
// （1024³ W=2 = 96×2×128KB = 24MB）；tile-local 布局 [row_l·BN+col_l]，
// 越界 (row≥M/col≥N) 槽位不写不读入 C（谓词同 deep epilogue；未写槽的
// 陈旧垃圾只进不落盘的 c[i][j] 死槽，无害）。grow-only RAII（dsk 模式）。
//
// 数值序：与 dsk 全 P 归约链（v1/direct 同括号序）逐位一致（切点重合时，
// 见 tests 专项）；cover==1 快路径与 deep 单波逐位一致。--dbuf 不改变数值。
//
// 主路径条件（16B 对齐）：N%4==0 且 K%4==0 且三指针 16B 对齐；不满足 →
// 回退 sgemm_2d_tile。TOT < 48（不满一波）→ 旁路 sgemm_deep 单波路径
// （与 deep 逐位一致，tests bitwise 锚链素材）。
//
// 同步契约（与 sgemm_deep.cu 的防漂移军规）：deep 主体布局/流水/swizzle
// 变更必须同步本文件逐拷贝主体（AGENTS §2 自包含 + design §5.1）。
//
// AR012 T004 FR2：SWZ 模板维（实例 ×2）——swz=1 时 c→(m,n) 解码处套用
// detail::swizzle_tile 分组列序重排（c 先线性分解 (n0,m0)=(c%gn, c/gn)，
// 再重排得物理 (m',n')；组宽 G = 运行时参数 swz_g = g_swzg）。**切点/
// cover/票据/P 槽位索引全部保持线性 c 空间不变**（slice(b,c)/tick[c]/b_lo/
// b_hi 原样），仅物理 tile 坐标重排 → 每 tile 的 K 链与归并参与集不变
// → swz on/off 输出逐位一致（bitwise 锚）。寄存器警示：本 kernel 255 regs
// 顶格（launch_bounds(256,1) 包络），remap 临时量短命于 c 循环头，ptxas
// 0 spill 硬门审计（SWZ=0 实例编译期剔除全部 remap 代码，现役 codegen
// 不变——AR011 T004 手法）。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>

namespace sgemm {
int g_streamk_waves = 0;   // --waves：0 = auto（T005 校准 = 恒 W=1，
                           // streamk_ar011.csv 实测 W>1 单调负）
int g_l2_persist = 0;      // --persist：1 = 计时区内 accessPolicyWindow 钉 C
                           // + 计时区后强制复位（先于 cuBLAS 锚定，协议纪律）
double g_l2_hit = 0.8;     // --hit：hitRatio（0.5..1.0）
int g_skred = 0;           // --skred：AR012 FR3，cover 归约路径（0 = 融合票据
                           // 归并现状；1 = cover-only 独立归约 kernel，T005
                           // Green 接入；链≡F2 → bitwise）
int g_launch_skred = -1;   // 接线探针（T001）：wrapper 每次 launch 快照 g_skred
}

namespace {

constexpr int BM = 256, BN = 128, BK = 8;
constexpr int TM = 16,  TN = 8;
constexpr int PAD_A = 4;                        // As 行距 260 floats（16B 对齐保持）
constexpr int TILE_FLOATS = BM * BN;            // P 槽位容量（128KB/tile 槽）

__device__ __forceinline__ float4 load_zero_guard(const float* base, long long idx,
                                                  bool valid) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (valid) v = *reinterpret_cast<const float4*>(base + idx);
    return v;
}

// ---- Stream-K 主体（deep 计算主体逐拷贝 + tile 循环 + 三路径 epilogue）----
// grid = (B, 1, 1) 一维；block (16, 16)。空块（b·U >= TOT）立即退出。
template <int DBUF, int SWZ = 0>
__global__ __launch_bounds__(256, 1)
void sgemm_streamk_kernel(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           float* __restrict__ P,        // [B*SLOTS][BM*BN] 槽位
                           unsigned int* __restrict__ tick, // [tiles] 自清洁票据
                           int M, int N, int K,
                           int U, int nt, int tiles, int grid_n, int SLOTS,
                           int swz_g) {
    __shared__ __align__(16) float As[DBUF ? 2 : 1][BK][BM + PAD_A];
    __shared__ __align__(16) float Bs[DBUF ? 2 : 1][BK][BN];
    __shared__ int s_winner;                       // 票据赢家标志（tid==0 写，全块读）

    const int b = blockIdx.x;
    const int TOT = tiles * nt;
    if (b * U >= TOT) return;                     // 空块（--waves 手动放大时出现）

    const int u_lo = b * U;
    const int u_hi = min((b + 1) * U, TOT);
    const int c_lo = u_lo / nt;
    const int c_hi = (u_hi - 1) / nt;

    const int tx = threadIdx.x, ty = threadIdx.y;   // block(16,16)
    const int tid = ty * 16 + tx;

    // ---- 搬运划分（deep 同型：A 逐行 2 quad；B krow/unit swizzle loader）----
    const int ld_a_row  = tid;
    const int ld_b_krow = tid >> 5;
    const int ld_b_unit = tid & 31;

    for (int c = c_lo; c <= c_hi; ++c) {
        // c 边界屏障（DBUF=1 专属，见文件头注释：nt 奇数时 buf 奇偶碰撞；
        // DBUF=0 每 tile 自带 S2 屏障，天然安全）
        if (DBUF && c > c_lo) __syncthreads();

        int bx = c % grid_n;                        // 本 tile 的全局 tile 坐标
        int by = c / grid_n;                        // （线性 c 空间 → (n0, m0)）
        if (SWZ) {   // AR012 T004：分组列序重排（切点/票据/P 索引保持线性
            //           c 空间不变，仅物理坐标变 → 每 tile 链不变 → bitwise）
            const int grid_m = tiles / grid_n;
            int m2, n2;
            sgemm::detail::swizzle_tile(bx, by, grid_n, grid_m, 1, swz_g,
                                        &m2, &n2);
            by = m2;
            bx = n2;
        }
        const int t0 = max(u_lo - c * nt, 0);
        const int t1 = min(u_hi - c * nt, nt);     // t0 < t1 由 c∈[c_lo,c_hi] 保证

        const int a_glb_row = by * BM + ld_a_row;
        const int b_glb_col = bx * BN + ld_b_unit * 4;

        // ---- 本 c 段的覆盖块闭区间（归并参与集；块号升序 ≡ k 升序）----
        const int b_lo = (c * nt) / U;
        const int b_hi = (min((c + 1) * nt, TOT) - 1) / U;
        const bool solo = (b_lo == b_hi);          // cover==1 快路径判定

        // ---- acc 清零（128 独立 FMA 链基座；赢家归并段亦复用此模式）----
        float cacc[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) cacc[i][j] = 0.0f;

        // ---- 流水预取：本段首 tile（deep 主体逐拷贝自此至 compute 循环）----
        const int k_first = t0 * BK;
        float4 a_reg0 = load_zero_guard(
            A, (long long)a_glb_row * K + k_first,
            (a_glb_row < M) && (k_first + 0 < K));
        float4 a_reg1 = load_zero_guard(
            A, (long long)a_glb_row * K + k_first + 4,
            (a_glb_row < M) && (k_first + 4 < K));
        float4 b_reg = load_zero_guard(
            B, (long long)(k_first + ld_b_krow) * N + b_glb_col,
            (k_first + ld_b_krow < K) && (b_glb_col < N));

        for (int t = t0; t < t1; ++t) {
            // ① 寄存器 → smem（本 tile 目标缓冲）
            const int buf = DBUF ? (t & 1) : 0;
            {
                const float af0[4] = {a_reg0.x, a_reg0.y, a_reg0.z, a_reg0.w};
                const float af1[4] = {a_reg1.x, a_reg1.y, a_reg1.z, a_reg1.w};
#pragma unroll
                for (int kk = 0; kk < 4; ++kk)
                    As[buf][kk][ld_a_row] = af0[kk];
#pragma unroll
                for (int kk = 0; kk < 4; ++kk)
                    As[buf][4 + kk][ld_a_row] = af1[kk];
            }
            {
                reinterpret_cast<float4*>(&Bs[buf][ld_b_krow][0])
                    [ld_b_unit ^ (ld_b_krow & 7)] = b_reg;
            }
            __syncthreads();   // S：tile t 全块就绪

            // ② 预取 tile t+1（段尾不预取）
            if (t + 1 < t1) {
                const int k0 = (t + 1) * BK;
                a_reg0 = load_zero_guard(
                    A, (long long)a_glb_row * K + k0,
                    (a_glb_row < M) && (k0 + 0 < K));
                a_reg1 = load_zero_guard(
                    A, (long long)a_glb_row * K + k0 + 4,
                    (a_glb_row < M) && (k0 + 4 < K));
                b_reg = load_zero_guard(
                    B, (long long)(k0 + ld_b_krow) * N + b_glb_col,
                    (k0 + ld_b_krow < K) && (b_glb_col < N));
            }

            // ③ 计算主循环：4×LDS.128(A 广播) + 2×LDS.128(B swizzle)
            //    + 128 FFMA（LDS:FFMA = 1:21.3，deep 逐拷贝）
#pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                float4 a0, a1, a2, a3, b0, b1;
                const float4* as_row = reinterpret_cast<const float4*>(&As[buf][kk][0]);
                a0 = as_row[ty * 4];
                a1 = as_row[ty * 4 + 1];
                a2 = as_row[ty * 4 + 2];
                a3 = as_row[ty * 4 + 3];
                const float4* bs_row = reinterpret_cast<const float4*>(&Bs[buf][kk][0]);
                const int sw = kk & 7;
                b0 = bs_row[(tx * 2) ^ sw];
                b1 = bs_row[(tx * 2 + 1) ^ sw];
                const float af[16] = {a0.x, a0.y, a0.z, a0.w,
                                      a1.x, a1.y, a1.z, a1.w,
                                      a2.x, a2.y, a2.z, a2.w,
                                      a3.x, a3.y, a3.z, a3.w};
                const float bf[8] = {b0.x, b0.y, b0.z, b0.w,
                                     b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j)
                        cacc[i][j] = af[i] * bf[j] + cacc[i][j];
            }
            if (!DBUF) __syncthreads();   // DBUF=0：S2 全块读完方可覆写
        }

        // ---- epilogue 三路径 ----
        if (solo) {
            // 路径 A（cover==1 快路径）：直写 C（deep 全局 epilogue 同型，
            // 零 P 流量、零原子；与 deep 单波逐位一致）
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int row = by * BM + ty * TM + i;
                if (row < M) {
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const int col = bx * BN + tx * TN + q * 4;
                        if (col < N) {
                            float4 v;
                            v.x = cacc[i][q * 4 + 0];
                            v.y = cacc[i][q * 4 + 1];
                            v.z = cacc[i][q * 4 + 2];
                            v.w = cacc[i][q * 4 + 3];
                            *reinterpret_cast<float4*>(&C[(long long)row * N + col]) = v;
                        }
                    }
                }
            }
            continue;                              // 下一 c
        }

        // 路径 B/C 公共：store P（tile-local；越界行列不写——赢家对越界位
        // 的读值只进不落盘的死槽 cacc，无害，见文件头 workspace 注释）
        float* psl = P + ((size_t)b * SLOTS + (c - c_lo)) * TILE_FLOATS;
        {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int row = by * BM + ty * TM + i;
                if (row < M) {
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const int col = bx * BN + tx * TN + q * 4;
                        if (col < N) {
                            float4 v;
                            v.x = cacc[i][q * 4 + 0];
                            v.y = cacc[i][q * 4 + 1];
                            v.z = cacc[i][q * 4 + 2];
                            v.w = cacc[i][q * 4 + 3];
                            *reinterpret_cast<float4*>(
                                &psl[(ty * TM + i) * BN + tx * TN + q * 4]) = v;
                        }
                    }
                }
            }
        }

        // release：P store 全 device 可见先于票据递增（CUDA threadFenceReduction
        // 样板模式的单线程票据变体）。__threadfence() 只 flush 调用线程自身的
        // 写，故每个写过 P 的线程各 fence 一次；块收敛后 tid==0 单线程递增票据
        // （每块 +1 而非每线程 +1）并经共享标志广播赢家（Green 调试 2026-10-07：
        // 修复全线程 atomicAdd 致计数 ×256、赢家提前误触发、归并读到未写槽）
        __threadfence();
        __syncthreads();
        if (tid == 0)
            s_winner = (atomicAdd(&tick[c], 1u) ==
                        (unsigned int)(b_hi - b_lo)) ? 1 : 0;
        __syncthreads();

        if (!s_winner) {
            continue;                              // 路径 B：非赢家，下一 c
        }

        // 路径 C（赢家 = 最后完成者）：F2 全 P 归并（T002 修订默认）——
        // cacc 清零复用为累加器，s 升序读全部切片（含 own，own 的 P 已在
        // 路径 B/C 公共段写入）：链 0+P[b_lo]+…+P[b_hi] 与 dsk 全 P 归约
        // 链恒逐位一致，且与 winner 身份无关（确定性天然成立）
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) cacc[i][j] = 0.0f;
        for (int s = b_lo; s <= b_hi; ++s) {
            const float* pbs = P + ((size_t)s * SLOTS + (c - (s * U) / nt))
                               * TILE_FLOATS;
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int row_l = ty * TM + i;
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    // __ldcs：P 一次性流读（dsk v3 归约先例）；越界行列的
                    // 槽位读值只进死槽 cacc（不落盘），in-bounds 安全
                    const float4 v = __ldcs(reinterpret_cast<const float4*>(
                        &pbs[row_l * BN + tx * TN + q * 4]));
                    cacc[i][q * 4 + 0] += v.x;
                    cacc[i][q * 4 + 1] += v.y;
                    cacc[i][q * 4 + 2] += v.z;
                    cacc[i][q * 4 + 3] += v.w;
                }
            }
        }
        // 归并结果写 C（deep epilogue 同谓词；默认 store——AR010 T007
        // __stwt 写穿反压教训）+ 票据自清洁（跨 launch 由流序保证）
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int row = by * BM + ty * TM + i;
            if (row < M) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const int col = bx * BN + tx * TN + q * 4;
                    if (col < N) {
                        float4 v;
                        v.x = cacc[i][q * 4 + 0];
                        v.y = cacc[i][q * 4 + 1];
                        v.z = cacc[i][q * 4 + 2];
                        v.w = cacc[i][q * 4 + 3];
                        *reinterpret_cast<float4*>(&C[(long long)row * N + col]) = v;
                    }
                }
            }
        }
        if (tid == 0) tick[c] = 0u;                // 自清洁：赢家单线程写
    }
}

// ---- grow-only workspace（RAII；dsk 模式逐拷贝；程序退出统一释放）------
struct Workspace {
    float*         p    = nullptr;
    unsigned int*  tick = nullptr;
    size_t         cap_p    = 0;      // P 容量（floats）
    size_t         cap_tick = 0;      // tick 容量（uint32）
    ~Workspace() {
        if (p) cudaFree(p);
        if (tick) cudaFree(tick);
    }
    void ensure(size_t p_floats, size_t ticks) {
        if (cap_p < p_floats) {
            if (p) cudaFree(p);
            cudaError_t err = cudaMalloc(&p, p_floats * sizeof(float));
            if (err != cudaSuccess) {
                p = nullptr; cap_p = 0;
                std::fprintf(stderr,
                             "[streamk] P workspace alloc failed (%zu floats): %s\n",
                             p_floats, cudaGetErrorString(err));
                std::exit(EXIT_FAILURE);
            }
            cap_p = p_floats;        // P 无需清零：仅真实槽被写后被读
        }
        if (cap_tick < ticks) {
            if (tick) cudaFree(tick);
            cudaError_t err = cudaMalloc(&tick, ticks * sizeof(unsigned int));
            if (err != cudaSuccess) {
                tick = nullptr; cap_tick = 0;
                std::fprintf(stderr,
                             "[streamk] tick workspace alloc failed (%zu): %s\n",
                             ticks, cudaGetErrorString(err));
                std::exit(EXIT_FAILURE);
            }
            // tick 必须零起始（票据计数语义）；此后靠赢家自清洁跨迭代保零
            cudaError_t e2 = cudaMemset(tick, 0, ticks * sizeof(unsigned int));
            if (e2 != cudaSuccess) {
                std::fprintf(stderr, "[streamk] tick memset failed: %s\n",
                             cudaGetErrorString(e2));
                std::exit(EXIT_FAILURE);
            }
            cap_tick = ticks;
        }
    }
};
Workspace g_ws;   // 静态实例：首次调用分配，之后零开销

}  // namespace

void sgemm_streamk(const float* A, const float* B, float* C,
                   int M, int N, int K) {
    // 主路径条件（与 deep 同条件）：N/K 4 倍数 + 三指针 16B 对齐
    const bool aligned =
        (N % 4 == 0) && (K % 4 == 0) &&
        (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0);
    if (!aligned) {
        if (sgemm::g_verbose) {
            std::printf("[streamk] size %dx%dx%d -> scalar fallback (tile2d)\n",
                        M, N, K);
        }
        sgemm_2d_tile(A, B, C, M, N, K);
        return;
    }

    const int nt    = (K + BK - 1) / BK;
    const int tiles = ((N + BN - 1) / BN) * ((M + BM - 1) / BM);
    const int grid_n = (N + BN - 1) / BN;
    const long long TOT_ll = (long long)tiles * nt;
    if (TOT_ll < 48) {
        // 不满一波（如 256³ TOT=16）：旁路 deep 单波路径——与 deep 逐位
        // 一致（tests bitwise 锚链素材），dispatch 层由 swsk 守擂小尺寸
        if (sgemm::g_verbose) {
            std::printf("[streamk] TOT=%lld < 48 -> bypass to deep single-wave\n",
                        TOT_ll);
        }
        sgemm_deep(A, B, C, M, N, K);
        return;
    }

    // W 选择（design §4.2.6）：--waves 0 = auto。T005 sweep 实测校准（AR011，
    // streamk_ar011.csv，3 轮中位）：W=1 全尺寸最优、W>1 单调负（P 税 +
    // 块/SM 占用损失），auto 公式坍缩为恒 W=1；显式 1..8 直接采用（消融/
    // 实验用；B>TOT 时多余块空退出，波填破坏但正确）
    int W;
    if (sgemm::g_streamk_waves > 0) {
        W = sgemm::g_streamk_waves;
    } else {
        W = 1;   // T005 校准（原 floor(TOT/768) 公式方向性错误，负结果归档）
    }
    const int B_blocks = 48 * W;
    const int U = (int)((TOT_ll + B_blocks - 1) / B_blocks);
    const int SLOTS = (U + nt - 1) / nt + 1;   // per-block 槽位（design T002 修订）

    g_ws.ensure((size_t)B_blocks * SLOTS * TILE_FLOATS, (size_t)tiles);

    const dim3 grid(B_blocks);
    const dim3 block(16, 16, 1);
    sgemm::g_launch_skred = sgemm::g_skred;   // T001 接线探针（独立 kernel 由 T005 接入）
    // AR012 T004：SWZ 实例维（c-decode 重排；组宽经参数；SWZ=0 现役 codegen
    // 不变）+ swz 接线探针快照（与 deep/dsk 咽喉点同语义）
    const int swz   = (sgemm::g_swz != 0) ? 1 : 0;
    const int swz_g = sgemm::g_swzg;
    sgemm::g_launch_swz = sgemm::g_swz;
    if (sgemm::g_deep_dbuf) {
        if (swz)
            sgemm_streamk_kernel<1, 1><<<grid, block>>>(
                A, B, C, g_ws.p, g_ws.tick, M, N, K, U, nt, tiles, grid_n,
                SLOTS, swz_g);
        else
            sgemm_streamk_kernel<1, 0><<<grid, block>>>(
                A, B, C, g_ws.p, g_ws.tick, M, N, K, U, nt, tiles, grid_n,
                SLOTS, swz_g);
    } else {
        if (swz)
            sgemm_streamk_kernel<0, 1><<<grid, block>>>(
                A, B, C, g_ws.p, g_ws.tick, M, N, K, U, nt, tiles, grid_n,
                SLOTS, swz_g);
        else
            sgemm_streamk_kernel<0, 0><<<grid, block>>>(
                A, B, C, g_ws.p, g_ws.tick, M, N, K, U, nt, tiles, grid_n,
                SLOTS, swz_g);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[streamk] launch failed: %s\n",
                     cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
    if (sgemm::g_verbose) {
        std::printf("[streamk] %dx%dx%d: W=%d B=%d U=%d SLOTS=%d tiles=%d "
                    "nt=%d TOT=%lld P=%.1fMB\n",
                    M, N, K, W, B_blocks, U, SLOTS, tiles, nt, TOT_ll,
                    (double)B_blocks * SLOTS * TILE_FLOATS * 4 / (1024 * 1024));
    }
}
