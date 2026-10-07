// =====================================================================
// sgemm_kernels.h — 六版 SGEMM kernel 的统一接口与注册表
// 契约（冻结，详设 §4.1）：
//   行主序 A[M x K] * B[K x N] = C[M x N]，严格 FP32，支持任意 M/N/K > 0。
// 每版 kernel 必须自包含（一个 .cu 文件），并遵守 AGENTS.md 注释契约。
// =====================================================================
#pragma once

#include <string>

using SgemmFn = void (*)(const float* A, const float* B, float* C,
                         int M, int N, int K);

// ---- 六版 kernel（冻结签名）------------------------------------------
void sgemm_naive    (const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_coalesced(const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_smem_1d  (const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_2d_tile  (const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_vec4     (const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_cpasync  (const float* A, const float* B, float* C, int M, int N, int K);

// ---- 消融/扩展变体（GPU 环境实验用，不改变冻结签名）-------------------
// smem1d 的 BK 消融：bk ∈ {8,16,32}（默认 32，AR007 实测固化）
void sgemm_smem_1d_ext(const float* A, const float* B, float* C,
                       int M, int N, int K, int bk);
// tile2d 的 __launch_bounds__ minBlocks 消融：lb ∈ {1,2}
void sgemm_2d_tile_ext(const float* A, const float* B, float* C,
                       int M, int N, int K, int lb);
// cp.async 方案乙（A: float4+转置+寄存器预取；B: cp.async+swizzle）
void sgemm_cpasync_v2(const float* A, const float* B, float* C, int M, int N, int K);
// K6 软件流水（AR007 扩展变体）：A/B float4 寄存器预取 + 单缓冲 + vec4 布局，
// 无 cp.async 依赖（sm_75 可用）；__launch_bounds__ minBlocks 消融 lb ∈ {1,2}
void sgemm_swpipe(const float* A, const float* B, float* C, int M, int N, int K);
// K6 变体（AR008）：split-K + 确定性固定序归约；内部 grow-only workspace（RAII，
// 冻结签名下的必然取舍，见 AR008 design §4.2.1）；--sk 片数旋钮（默认 4，sk=1
// 旁路直走 swpipe 单波路径）
void sgemm_swpipe_sk(const float* A, const float* B, float* C, int M, int N, int K);
// Kernel 7（AR008）：warp 专属化 producer/consumer 软件流水——2 producer warp
// （LDG→reg→STS 灌多级 smem 环）+ 8 consumer warp（纯 LDS+FFMA），PTX named
// barriers（bar.sync/bar.arrive id,count，sm_75 合法）握手，无 cp.async 依赖
void sgemm_ws(const float* A, const float* B, float* C, int M, int N, int K);
// 自动选核（AR008 T006）：几何 dispatch（blocks=ceil(M/128)*ceil(N/128) 带判
// + K 钳制），表值由 T004/T005 实测回填（≤4→swsk12 / ≤64→swsk4 / 其余→swpipe）
void sgemm_auto(const float* A, const float* B, float* C, int M, int N, int K);
// Kernel 8（AR009）：宽块占用率攻坚——512 线程（16 warp）× 128×128 tile，
// 每线程 TM=4×TN=8（32 acc），寄存器预算 ≤64（__launch_bounds__(512,2)）
// → 2 block/SM = 1024 线程 = 100% 占用（破除 swpipe 系 128 regs→50% 墙）。
// 搬运双角色：tid<256 载 A / tid>=256 载 B（每线程恰 1×float4）；布局/流水/
// swizzle 结论三承 swpipe（A 转置+PAD4、B XOR swizzle、单缓冲双同步、
// 寄存器预取混合流）。LB 消融 --wlb（g_wide_min_blocks，默认 2）。
void sgemm_wide(const float* A, const float* B, float* C, int M, int N, int K);
// Kernel 8 变体（AR009）：wide 的 split-K——主体经 detail::wide_tile_grid
// 参数化复用（z 切片 + Out_base），归约复用 detail::swsk_reduce（与 swsk
// 共享单一确定性数值路径）；--sk 旋钮对 swsk/wsk 双生效。
void sgemm_wsk(const float* A, const float* B, float* C, int M, int N, int K);
// Kernel 9（AR010）：128-acc 深寄存器分块——tile 256×128×BK8，block 256 线程
// （16×16），每线程 TM=16×TN=8 = 128 累加器，LDS:FFMA = 6 LDS.128 : 128 FFMA
// = 1:21.3（acc-per-LDS 16→21.3，直击 AR009 实证的 LDS.128 带宽墙）；TM>TN
// 几何使 B 片段保持 stride-2 quad（swpipe swizzle 消冲突域逐位继承）、A 片段
// warp 级广播；占用 1 block/SM = 8 warp = 25%，以 128 条独立 FMA 链的 ILP
// 替代 TLP（与 AR009 占用率证伪构成完整消融矩阵）。__launch_bounds__(256,1)
// 下 ptxas 实测 247 regs（dbuf0）/241 regs（dbuf1），0 spill 硬门（256×247=63,232
// 恰满 64K 寄存器堆）。--dbuf 消融（g_deep_dbuf，默认 1，AR010 T004 实测
// dbuf1 全尺寸 +3~14% 固化）：0 = 单缓冲双同步（swpipe 同构，12416B）；
// 1 = 双缓冲单同步（As[2]+Bs[2]=24832B，同步 2→1/tile，25% 占用下的延迟对冲路径）。
void sgemm_deep(const float* A, const float* B, float* C, int M, int N, int K);
// Kernel 9 变体（AR010）：deep 的 split-K——主体经 detail::deep_tile_grid
// 参数化复用（z 切片 + Out_base），归约复用 detail::swsk_reduce；波几何：
// 1024³ grid 4×8=32 blocks × sk3 = 96 = 2 精确波（1 block/SM × 48 slots）；
// --sk 旋钮对 swsk/wsk/dsk 三生效。
void sgemm_dsk(const float* A, const float* B, float* C, int M, int N, int K);
// Kernel 10（AR011）：Stream-K 统一调度——迭代空间 (C-tile, k-tile) tile-major
// 线性化后连续切分为 48·W 块（波填充率恒 100%，消灭波量化），变长 k 区间；
// per-tile 票据归并（atomic + __threadfence，赢家按块号升序=z 升序归并部分积，
// 链序与 dsk direct 归约逐位对齐），归约第二 kernel/启动开销/P 自片 DRAM 往返
// 一并消除；cover==1 的 tile 走直写快路径（deep 同型）。--waves 波数旋钮
// （g_streamk_waves，默认 0 = auto）。计算主体自包含（deep 主体逐拷贝，
// design §4.1 问题 3 裁定：防 ptxas 重排 + 自包含军规）。
void sgemm_streamk(const float* A, const float* B, float* C, int M, int N, int K);

// ---- cuBLAS FP32 基线（严格 CUBLAS_DEFAULT_MATH，详设 §4.2）-----------
void sgemm_cublas(const float* A, const float* B, float* C, int M, int N, int K);

// ---- 内部复用入口（跨编译单元；AR008 swsk 共享 swpipe tile 主体）-------
// 语义：grid=(grid_n, grid_m, grid_z)，切片 z 处理 k-tile 区间
// [z*tps, min((z+1)*tps, num_tiles))，输出写 Out_base + z*M*N。
// Out_base 需为可容纳 grid_z 个 M*N 切片的 16B 对齐基址
// （单波：Out_base=C, grid_z=1, tps=num_tiles；split-K：Out_base=P, grid_z=sk）。
// 资源：单一实例 __launch_bounds__(256,2) = 128 regs / 2-block 包络（AR008
// 实测固化：无约束实例 130 regs → 1 block/SM，1024³ -7%；详见 .cu 注释）。
namespace sgemm { namespace detail {
void swpipe_tile_grid(const float* A, const float* B, float* Out_base,
                      int M, int N, int K, int tps, int num_tiles,
                      int grid_n, int grid_m, int grid_z);
// AR009：wide tile 主体参数化入口（wsk 共享；契约同 swpipe_tile_grid，
// block=512 线程，LB 实例由 g_wide_min_blocks 选择）
void wide_tile_grid(const float* A, const float* B, float* Out_base,
                    int M, int N, int K, int tps, int num_tiles,
                    int grid_n, int grid_m, int grid_z);
// AR010：deep tile 主体参数化入口（dsk 共享；契约同 swpipe/wide_tile_grid，
// block=256 线程（16×16），DBUF 实例由 g_deep_dbuf 选择）。
// T007 last-slice-direct：Out_last = 末片输出目标；传 Out_last==Out_base
// 关闭直写（行为与旧签名逐字一致）；dsk 主路径传 C。
void deep_tile_grid(const float* A, const float* B, float* Out_base,
                    float* Out_last,
                    int M, int N, int K, int tps, int num_tiles,
                    int grid_n, int grid_m, int grid_z);
// AR009：确定性归约提升（自 swpipe_sk.cu launch_reduce；swsk/wsk 共享，
// C[i] = Σ_z P[z*MN+i] 固定 z 序，逐位可复现）
void swsk_reduce(const float* P, float* C, long long mn, int sk);
// AR010 T007：last-slice-direct 归约（dsk 主路径）——P 含 sk-1 个切片，
// C[i] = ((0+P_0)+…+P_{sk-2})+C[i]，链序与 swsk_reduce v1 逐位一致
void swsk_reduce_direct(const float* P, float* C, long long mn, int sk);
}}

// ---- 消融旋钮（由 main CLI 注入；默认值与详设 §4.6 一致）----------------
namespace sgemm {
extern int g_smem1d_bk;          // 默认 32（AR007 实测固化：bk32 比 bk16 +8.3%）
extern int g_tile2d_min_blocks;  // 默认 1
extern int g_swsk_slices;        // --sk，默认 4（AR008 swsk split-K 片数，1..16）
extern int g_ws_stages;          // ws 环深度，默认 3（消融 2/3）
extern int g_ws_min_blocks;      // ws __launch_bounds__ 消融，默认 1（1/2）
extern int g_ws_prod_warps;      // ws producer warp 数，默认 2（消融 1/2）
extern int g_wide_min_blocks;    // wide/wsk __launch_bounds__ 消融，默认 2（1/2；
                                 //   LB=2 = 64 regs 100% 占用目标，LB=1 = spill 保底）
extern int g_deep_dbuf;          // deep/dsk smem 双缓冲消融，默认 1（0/1；
                                 //   1 = 双缓冲单同步（AR010 T004 实测全尺寸
                                 //   +3~14% 数据裁定），0 = 单缓冲双同步）
extern int g_reduce_ilp2;        // swsk/wsk/dsk 归约实现消融，默认 0（0/1/3；
                                   //   0 = v1 1 f4/线程，1 = v2 2 f4/线程（dsk -4%
                                   //   负结果），3 = v3 4 f4/线程 + __ldcs/__stcs
                                   //   流式（AR010 T007 攻坚，逐位等价））
extern int g_streamk_waves;      // --waves，Stream-K 波数 W，默认 0 = auto 公式
                                   //   clamp(floor(TOT/(48·16)),1,8)（AR011 design
                                   //   §4.2.6；T005 sweep 校准后 auto v4 表接管）
extern int g_l2_persist;         // --persist，L2 persistence 钉 C 开关，默认 0；
                                   //   1 = 计时区内 accessPolicyWindow 钉 C +
                                   //   计时区后强制复位（先于 cuBLAS 锚定，协议纪律）
extern double g_l2_hit;          // --hit，accessPolicyWindow hitRatio，默认 0.8
                                    //   （0.5..1.0；C > persisting 上限时按比例钉入）
extern int g_deep_bpf;           // --bpf，deep/dsk B 片段 kk+1 寄存器预取消融，
                                    //   默认 0（AR011 FR3a；链序不变 → bitwise 门；
                                    //   仅 DBUF=1 族实例化）
extern int g_deep_phase;         // --phase，deep/dsk kk 轮转错相消融，默认 0
                                    //   （AR011 FR3b；warp w 序 kk'=(s+w)&7；改变
                                    //   k 加法序 → rel≤1e-4 + 确定性双跑门；
                                    //   仅 DBUF=1 族实例化）
extern bool g_verbose;           // 打印回退路径细节（测试断言路径覆盖用）
}

// ---- 注册表 ----------------------------------------------------------
namespace sgemm {
inline constexpr int K_NAIVE = 0, K_COALESCED = 1, K_SMEM1D = 2, K_TILE2D = 3,
                      K_VEC4 = 4, K_CPASYNC = 5, K_CPASYNC2 = 6, K_SWPIPE = 7,
                      K_SWSK = 8, K_WS = 9, K_CUBLAS = 10, K_AUTO = 11,
                      K_WIDE = 12, K_WSK = 13, K_DEEP = 14, K_DSK = 15,
                      K_STREAMK = 16,
                      KERNEL_COUNT = 17;

inline const char* kernel_name(int id) {
    static const char* names[KERNEL_COUNT] = {
        "naive", "coalesced", "smem1d", "tile2d", "vec4", "cpasync", "cpasync2",
        "swpipe", "swsk", "ws", "cublas", "auto", "wide", "wsk", "deep", "dsk",
        "streamk"};
    return (id >= 0 && id < KERNEL_COUNT) ? names[id] : "unknown";
}

// 按名称查 id；未找到返回 -1
inline int kernel_id(const std::string& name) {
    for (int i = 0; i < KERNEL_COUNT; ++i)
        if (name == kernel_name(i)) return i;
    return -1;
}

inline SgemmFn kernel_fn(int id) {
    // AR011 T002 Green：streamk 接入（T001 Red 期间槽位为 nullptr）
    static const SgemmFn fns[KERNEL_COUNT] = {
        sgemm_naive, sgemm_coalesced, sgemm_smem_1d, sgemm_2d_tile,
        sgemm_vec4,  sgemm_cpasync,   sgemm_cpasync_v2,
        sgemm_swpipe, sgemm_swpipe_sk, sgemm_ws, sgemm_cublas, sgemm_auto,
        sgemm_wide, sgemm_wsk, sgemm_deep, sgemm_dsk, sgemm_streamk};
    return (id >= 0 && id < KERNEL_COUNT) ? fns[id] : nullptr;
}
}  // namespace sgemm
