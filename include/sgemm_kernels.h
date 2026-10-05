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
}}

// ---- 消融旋钮（由 main CLI 注入；默认值与详设 §4.6 一致）----------------
namespace sgemm {
extern int g_smem1d_bk;          // 默认 32（AR007 实测固化：bk32 比 bk16 +8.3%）
extern int g_tile2d_min_blocks;  // 默认 1
extern int g_swsk_slices;        // --sk，默认 4（AR008 swsk split-K 片数，1..16）
extern int g_ws_stages;          // ws 环深度，默认 3（消融 2/3）
extern int g_ws_min_blocks;      // ws __launch_bounds__ 消融，默认 1（1/2）
extern int g_ws_prod_warps;      // ws producer warp 数，默认 2（消融 1/2）
extern bool g_verbose;           // 打印回退路径细节（测试断言路径覆盖用）
}

// ---- 注册表 ----------------------------------------------------------
namespace sgemm {
inline constexpr int K_NAIVE = 0, K_COALESCED = 1, K_SMEM1D = 2, K_TILE2D = 3,
                      K_VEC4 = 4, K_CPASYNC = 5, K_CPASYNC2 = 6, K_SWPIPE = 7,
                      K_SWSK = 8, K_WS = 9, K_CUBLAS = 10, K_AUTO = 11,
                      KERNEL_COUNT = 12;

inline const char* kernel_name(int id) {
    static const char* names[KERNEL_COUNT] = {
        "naive", "coalesced", "smem1d", "tile2d", "vec4", "cpasync", "cpasync2",
        "swpipe", "swsk", "ws", "cublas", "auto"};
    return (id >= 0 && id < KERNEL_COUNT) ? names[id] : "unknown";
}

// 按名称查 id；未找到返回 -1
inline int kernel_id(const std::string& name) {
    for (int i = 0; i < KERNEL_COUNT; ++i)
        if (name == kernel_name(i)) return i;
    return -1;
}

inline SgemmFn kernel_fn(int id) {
    static const SgemmFn fns[KERNEL_COUNT] = {
        sgemm_naive, sgemm_coalesced, sgemm_smem_1d, sgemm_2d_tile,
        sgemm_vec4,  sgemm_cpasync,   sgemm_cpasync_v2,
        sgemm_swpipe, sgemm_swpipe_sk, sgemm_ws, sgemm_cublas, sgemm_auto};
    return (id >= 0 && id < KERNEL_COUNT) ? fns[id] : nullptr;
}
}  // namespace sgemm
