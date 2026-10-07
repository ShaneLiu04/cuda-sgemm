// =====================================================================
// sgemm_deep.cu — Kernel 9：128-acc 深寄存器分块（AR010 攻坚 LDS.128 带宽墙）
// ---------------------------------------------------------------------
// 动机（AR009 三重证据链收尾）：sm_75 FP32 的真墙不是占用率（wide LB1==LB2、
// 半填充>满填充、2048³ 50% 占用反 +2.2%），而是 LDS.128 带宽/发射压力——
// acc-per-LDS.128 经验律：wide 8 acc/LDS → 31% peak、swpipe 16 → 46-54%、
// cuBLAS（深分块 ~21+）→ 84%。本版把每线程累加器 64 → 128（TM16×TN8），
// acc-per-LDS 抬到 21.3，以 128 条独立 FMA 链的 ILP 替代 TLP（占用率
// 50% → 25%），直接检验经验律并对齐 cuBLAS 设计点。
//
// 几何（TM>TN 的必然性，design §4.1 问题 1 方案 B 弃选依据）：
//   - tile：BM=256, BN=128, BK=8；每线程 TM=16 × TN=8（128 acc），
//     block 16x16 = 256 线程（行 16×TM16=256，列 16×TN8=128）
//   - TN=16 会使 B 片段成 stride-4 quad 访问（16 连续列/线程），任意 16B
//     XOR swizzle 均留 mod-8 双残留类 → 8-way bank conflict 不可救；
//     TM=16 时 B 片段保持 stride-2 quad（swpipe 已实证的 swizzle 消解域）
//   - LDS:FFMA = (4 A + 2 B) LDS.128 : 128 FFMA = 1:21.3（swpipe 1:16）
//
// smem 布局（第五次继承 vec4/swpipe 结论，design §4.2.1 逐项论证）：
//   - A 转置 As[BK][BM+PAD_A=4]：loader 逐行负责（row=tid，每线程 2 个
//     k-quad = 行内 32B 连续 LDG），STS 8 标量散射 As[kk][tid]——连续 tid
//     → 连续 bank（行距 260 ≡ 4 mod 32），0 冲突；计算期 A 片段
//     As4[kk][ty*4..ty*4+3] 为 warp 级广播（16 lanes 同地址），0 冲突
//   - B 保持 Bs[BK][BN]，16B 单位 XOR swizzle（unit ^ (krow&7)）——loader
//     划分 krow=tid>>5 / unit=tid&31 与 swpipe 逐位同型，计算期
//     (tx*2)^sw / (tx*2+1)^sw 与 vec4/swpipe 完全同型（AR006 ncu 锚定
//     0 冲突），0 冲突
//
// 流水（模板 <int DBUF> 两实例，--dbuf 消融）：
//   DBUF=0（默认，swpipe 同构）：单缓冲双同步——
//     预取 tile0 → [ store(t)→smem | S1 | 预取(t+1)→reg | compute(t) | S2 ]
//   DBUF=1（延迟对冲路径）：双缓冲单同步（As[2]+Bs[2]=24832B ≤ 48KB 静态
//     上限）——[ store(t)→buf[t&1] | S | 预取(t+1) | compute(t)←buf[t&1] ]；
//     正确性：store(t+1) 写 buf[(t+1)&1]，其上一读者 compute(t-1) 与之间
//     隔着 iter t 的 barrier（全块 compute(t-1) 必已完结）；收益：下一 tile
//     的 STS 与本 tile 的 LDS/FFMA 重叠，同步 2→1/ tile
//   寄存器预算：128 acc + 12 预取（2 A quad + 1 B quad）+ 24 瞬态片段 +
//   ~20 寻址 ≈ 184；__launch_bounds__(256, 1) = 255 regs 封顶，0 spill 硬门
//   （256×255=65280 ≤ 64K → 1 block/SM = 8 warp = 25% 占用 = 设计点）
//
// 数值序：与 swpipe/vec4 主路径逐位一致（同 k 升序 FMA 链——TM/TN 重排
// 不改变逐元素 c[i][j] 的 k 递增累加序；--dbuf 亦不改变数值）。
//
// 主路径条件（16B 对齐）：N%4==0 且 K%4==0 且三指针 16B 对齐；
//   不满足 → 回退 sgemm_2d_tile（任意尺寸正确），--verbose 打印。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>
#include <cstdint>

namespace sgemm {
int g_deep_dbuf = 1;   // --dbuf：1 = 双缓冲单同步（默认，AR010 T004 实测全尺寸
                       // +3~14% 数据裁定），0 = 单缓冲双同步
int g_deep_bpf = 0;    // --bpf：AR011 FR3a，B(kk+1) 寄存器预取（仅 DBUF=1 族）
int g_deep_phase = 0;  // --phase：AR011 FR3b，kk 轮转错相（仅 DBUF=1 族）
int g_swz = 0;         // --swz：AR012 FR2，L2 块序 swizzle（0 = 线性光栅现状；
                       // 1 = 分组列序 remap，T003 Green 接入 device 侧）
int g_swzg = 8;        // --swzg：组宽 G（n-tiles/组；4/8/16）
int g_launch_swz = -1; // 接线探针（T001）：wrapper 每次 launch 快照 g_swz
}

namespace {

constexpr int BM = 256, BN = 128, BK = 8;
constexpr int TM = 16,  TN = 8;
constexpr int PAD_A = 4;                        // As 行距 260 floats（1040B，
                                                // 16B 对齐保持：260*4 % 16 == 0）

__device__ __forceinline__ float4 load_zero_guard(const float* base, long long idx,
                                                  bool valid) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (valid) v = *reinterpret_cast<const float4*>(base + idx);
    return v;
}

// tile 主体（参数化：z=blockIdx.z 切片，k-tile 区间 [z*tps, min((z+1)*tps,
// num_tiles))，输出 Out = Out_base + z*M*N。单波路径（grid.z=1, tps=num_tiles）
// 与原语义逐位等价；空片（t0>=t1）自然写全零——dsk 的 sk>tiles 边界语义。
// AR010 T007 last-slice-direct：LAST_DIRECT=1 实例中 z==gridDim.z-1 的切片
// 直写 Out_last（=C，默认 store——归约将读 C），其余片照写 P；LAST_DIRECT=0
//（单波与旧消融路径）编译期剔除全部直写代码——与旧实现指令序列逐字一致
//（运行时 last_direct 参数是 ptxas 重排诱因，实测 deep@1024³ -4.9%：寄存器
// 247/241→243 重分配连带流水重排；模板化后非直写路径恢复旧代码生成）。
// （__stwt 写穿已实测回退：main +16μs 反压损失，无归约收益。）
//
// AR011 T004 延迟覆盖消融（design §4.2.5，模板 <BPF, PHASE> 默认 0 = 现役
// 路径 codegen 不变——回归门：旧 4 实例 bitwise 全绿 + 性能 ±1% 噪声带）：
//   - BPF=1（FR3a）：compute 循环每 kk 步开头先发射 B(kk+1) 的 2×LDS.128
//     入第二组寄存器（+8 regs 刀口），本步 128 FFMA 覆盖其延迟；kk=BK-1 不
//     预取；B(kk) 自 kk=1 起由上步预取寄存器供给（kk=0 首步直取）。
//     **不改变加法链序** → BPF on/off 必须 bitwise 一致（tests 专项）。
//   - PHASE=1（FR3b）：warp w（w=tid>>5，0..7；warp 内统一无 lane 发散）
//     按逻辑步 s 处理 kk'=(s+w)&7——屏障后全 warp 同 kk 齐射 LDS 的同相
//     停顿被错相打散；A 广播与 B swizzle 均以实际 kk' 寻址（swizzle 消解
//     域与 kk 取值无关，任意固定 sw 均 0 冲突）。**改变逐元素 k 加法序**
//     （每线程 c[i][j] 的 k 递增序被轮转）→ rel≤1e-4 双参考 + 确定性
//     双跑门（数值口径分级，design §4.2.5）。
//   实例矩阵受控：BPF/PHASE 仅 DBUF=1 族 + LAST_DIRECT∈{0,1}，组合
//   (bpf,phase)∈{(1,0),(0,1),(1,1)}（{on,off}² 因果分解需要 both-on，
//   T004 修订）；DBUF=0 仅现役 <0,ld,0,0>。
template <int DBUF, int LAST_DIRECT, int BPF = 0, int PHASE = 0>
__global__ __launch_bounds__(256, 1)
void sgemm_deep_kernel(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ Out_base,
                       float* __restrict__ Out_last,
                       int M, int N, int K,
                       int tps, int num_tiles) {
    // smem：A 转置 [DBUF][BK][BM+PAD]；B swizzle [DBUF][BK][BN]
    __shared__ __align__(16) float As[DBUF ? 2 : 1][BK][BM + PAD_A];
    __shared__ __align__(16) float Bs[DBUF ? 2 : 1][BK][BN];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int z  = blockIdx.z;                  // split-K 切片号（单波恒 0）
    const int t0 = z * tps;
    const int t1 = min(t0 + tps, num_tiles);
    // 注意：Out 指针下沉到回写段计算（AR010 T007 实测：头部计算+双指针跨
    // 主循环活跃会扰动 ptxas 寄存器分配 241→243，main -3.4%；回写前才需要）

    const int tx = threadIdx.x, ty = threadIdx.y;   // block(16,16)
    const int tid = ty * 16 + tx;

    float c[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) c[i][j] = 0.0f;

    // ---- 搬运划分（每 tile A 512 quads + B 256 quads = 768 = 256 线程 × 3）----
    // A：row = tid（0..255 逐行负责），每线程该行 2 个 k-quad（行内 32B 连续）
    // B：krow = tid>>5（0..7），unit = tid&31（0..31）——与 swpipe 逐位同型
    const int ld_a_row  = tid;
    const int ld_b_krow = tid >> 5;
    const int ld_b_unit = tid & 31;

    const int a_glb_row = by * BM + ld_a_row;
    const int b_glb_col = bx * BN + ld_b_unit * 4;

    // ---- 流水预取：本片首 tile 的 2×A quad + 1×B quad 先入寄存器 ----
    const int k_first = t0 * BK;
    float4 a_reg0 = load_zero_guard(
        A, (long long)a_glb_row * K + k_first,
        (t0 < t1) && (a_glb_row < M) && (k_first + 0 < K));
    float4 a_reg1 = load_zero_guard(
        A, (long long)a_glb_row * K + k_first + 4,
        (t0 < t1) && (a_glb_row < M) && (k_first + 4 < K));
    float4 b_reg = load_zero_guard(
        B, (long long)(k_first + ld_b_krow) * N + b_glb_col,
        (t0 < t1) && (k_first + ld_b_krow < K) && (b_glb_col < N));

    for (int t = t0; t < t1; ++t) {
        // ---- ① 寄存器 → smem（本 tile 目标缓冲）----
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
            // 物理 16B 单位 = 逻辑单位 ^ (krow & 7)——swpipe 同型
            reinterpret_cast<float4*>(&Bs[buf][ld_b_krow][0])
                [ld_b_unit ^ (ld_b_krow & 7)] = b_reg;
        }
        __syncthreads();   // S：tile t 全块就绪（DBUF=1 下同时保证 compute(t-2)
                           //    对 buf 的读取已完结——design §2 时序论证）

        // ---- ② 预取 tile t+1 → 寄存器（LDG 提前发射，延迟由 ③ 覆盖）----
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

        // ---- ③ 计算主循环：每 kk 一步 4×LDS.128(A 广播) + 2×LDS.128(B
        //      swizzle) + 128 FFMA —— LDS:FFMA = 1:21.3 ----
        //      BPF=1：步首先发射 B(kk+1) LDS（延迟由本步 FFMA 覆盖）；
        //      PHASE=1：warp w 的 kk 序轮转 (s+w)&7（见文件头 AR011 注释）
        const int wid = tid >> 5;                 // warp 号 0..7（PHASE 用）
        float4 b0_pf, b1_pf;                      // BPF 预取寄存器组（+8 regs）
#pragma unroll
        for (int s = 0; s < BK; ++s) {
            const int kk = PHASE ? ((s + wid) & 7) : s;
            // BPF=1：预取发射——目标 = 下一逻辑步 s+1 将消费的 kk（PHASE=1
            // 时为 (s+1+wid)&7，kk=7 步的下一消费值回绕到 0；末步 s=BK-1
            // 无后继不预取）
            float4 b0n, b1n;
            if (BPF && s < BK - 1) {
                const int kkn = PHASE ? ((s + 1 + wid) & 7) : (kk + 1);
                const float4* bsn =
                    reinterpret_cast<const float4*>(&Bs[buf][kkn][0]);
                const int swn = kkn & 7;
                b0n = bsn[(tx * 2) ^ swn];
                b1n = bsn[(tx * 2 + 1) ^ swn];
            }
            float4 a0, a1, a2, a3, b0, b1;
            const float4* as_row = reinterpret_cast<const float4*>(&As[buf][kk][0]);
            a0 = as_row[ty * 4];
            a1 = as_row[ty * 4 + 1];
            a2 = as_row[ty * 4 + 2];
            a3 = as_row[ty * 4 + 3];
            if (BPF && s > 0) {
                b0 = b0_pf;                       // B(kk) 已于上步预取入寄存器
                b1 = b1_pf;
            } else {
                const float4* bs_row =
                    reinterpret_cast<const float4*>(&Bs[buf][kk][0]);
                const int sw = kk & 7;
                b0 = bs_row[(tx * 2) ^ sw];
                b1 = bs_row[(tx * 2 + 1) ^ sw];
            }
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
                    c[i][j] = af[i] * bf[j] + c[i][j];
            if (BPF && s < BK - 1) {
                b0_pf = b0n;                      // 寄存器组交换（unroll 后
                b1_pf = b1n;                      // 由编译器重命名消除）
            }
        }
        if (!DBUF) __syncthreads();   // DBUF=0：S2 全块读完 As/Bs 方可覆写
        // DBUF=1：无 S2——buf 周期 2，覆写权由下一轮 S 的 barrier 语义保证
    }

    // ---- 回写：每行 2×float4（N%4==0 ⇒ quad 全有或全无），16 行 × 2 quads ----
    // （split-K 模式写部分积 Out = P + z*M*N；空片 z 自然写全零）
    float* __restrict__ Out;
    if (LAST_DIRECT && (z == gridDim.z - 1)) {
        Out = Out_last;                         // 末片直写 C
    } else {
        Out = Out_base + (long long)z * M * N;  // P 片（或单波 C）
    }
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
                    // 统一默认 store（AR010 T007 实测：__stwt 写穿使 main +16μs
                    // ——写穿流量与 A/B 读 miss 反压；写回吸收 L2 延迟排出更优）
                    *reinterpret_cast<float4*>(&Out[(long long)row * N + col]) = v;
                }
            }
        }
    }
}

}  // namespace

namespace sgemm { namespace detail {

// deep tile 主体的参数化复用入口（AR010 dsk 共享；单波路径由本文件 wrapper
// 使用）。grid = (grid_n, grid_m, grid_z)；Out_base 需为可容纳 grid_z 个
// M*N 切片的 16B 对齐基址（单波：Out_base=C, grid_z=1；split-K：P, grid_z=sk）。
// AR010 T007：Out_last = 末片输出目标；传 Out_last==Out_base 关闭直写
// （last_direct=0，与原实现行为逐字一致）；dsk 主路径传 C（末片直写 C，
// P 片 __stwt 写穿）。
// 资源：双实例均 __launch_bounds__(256,1) = ≤255 regs / 1-block（25% 占用）
// 包络；DBUF 实例由 g_deep_dbuf 选择（--dbuf）。
void deep_tile_grid(const float* A, const float* B, float* Out_base,
                    float* Out_last,
                    int M, int N, int K, int tps, int num_tiles,
                    int grid_n, int grid_m, int grid_z) {
    const dim3 grid(grid_n, grid_m, grid_z);
    const dim3 block(16, 16, 1);
    sgemm::g_launch_swz = sgemm::g_swz;   // T001 接线探针（device remap 由 T003 接入）
    const int last_direct = (Out_last != Out_base) ? 1 : 0;
    // AR011 T004：BPF/PHASE 消融实例矩阵（受控，design §4.2.5）——仅 DBUF=1
    // 族 × (bpf,phase)∈{00,10,01,11}（both-on 支撑 {on,off}² 因果分解）；
    // DBUF=0 仅现役 <0,ld,0,0>（main.cu 已对 --dbuf 0 + --bpf/--phase 打
    // [note]，此层防御性忽略）
    if (sgemm::g_deep_dbuf) {
        const int b = (sgemm::g_deep_bpf != 0) ? 1 : 0;
        const int p = (sgemm::g_deep_phase != 0) ? 1 : 0;
#define LAUNCH_DEEP(BPF_, PH_)                                            \
        do {                                                              \
            if (last_direct)                                              \
                sgemm_deep_kernel<1, 1, BPF_, PH_><<<grid, block>>>(      \
                    A, B, Out_base, Out_last, M, N, K, tps, num_tiles);   \
            else                                                          \
                sgemm_deep_kernel<1, 0, BPF_, PH_><<<grid, block>>>(      \
                    A, B, Out_base, Out_last, M, N, K, tps, num_tiles);   \
        } while (0)
        if (b && p)      LAUNCH_DEEP(1, 1);
        else if (b)      LAUNCH_DEEP(1, 0);
        else if (p)      LAUNCH_DEEP(0, 1);
        else             LAUNCH_DEEP(0, 0);
#undef LAUNCH_DEEP
    } else {
        if (last_direct)
            sgemm_deep_kernel<0, 1><<<grid, block>>>(A, B, Out_base, Out_last,
                                                     M, N, K, tps, num_tiles);
        else
            sgemm_deep_kernel<0, 0><<<grid, block>>>(A, B, Out_base, Out_last,
                                                     M, N, K, tps, num_tiles);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[deep_tile_grid] launch failed: %s\n",
                     cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

}}  // namespace sgemm::detail

void sgemm_deep(const float* A, const float* B, float* C, int M, int N, int K) {
    // 主路径条件：与 swpipe 相同（N、K 为 4 的倍数且三指针 16B 对齐）
    const bool aligned =
        (N % 4 == 0) && (K % 4 == 0) &&
        (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0);

    if (!aligned) {
        if (sgemm::g_verbose) {
            std::printf("[deep] size %dx%dx%d -> scalar fallback (tile2d)\n", M, N, K);
        }
        sgemm_2d_tile(A, B, C, M, N, K);   // 谓词化回退路径（任意尺寸正确）
        return;
    }

    const int num_tiles = (K + BK - 1) / BK;
    const int grid_n = (N + BN - 1) / BN;
    const int grid_m = (M + BM - 1) / BM;
    // 单波路径：z=1、tps=num_tiles —— 与参数化前语义逐位等价
    // （Out_last==Out_base==C ⇒ last_direct=0，全默认 store，行为不变）
    sgemm::detail::deep_tile_grid(A, B, C, C, M, N, K, num_tiles, num_tiles,
                                  grid_n, grid_m, 1);
}
