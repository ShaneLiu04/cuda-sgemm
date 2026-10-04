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
// smem1d 的 BK 消融：bk ∈ {8,16,32}
void sgemm_smem_1d_ext(const float* A, const float* B, float* C,
                       int M, int N, int K, int bk);
// tile2d 的 __launch_bounds__ minBlocks 消融：lb ∈ {1,2}
void sgemm_2d_tile_ext(const float* A, const float* B, float* C,
                       int M, int N, int K, int lb);
// cp.async 方案乙（A: float4+转置+寄存器预取；B: cp.async+swizzle）
void sgemm_cpasync_v2(const float* A, const float* B, float* C, int M, int N, int K);

// ---- cuBLAS FP32 基线（严格 CUBLAS_DEFAULT_MATH，详设 §4.2）-----------
void sgemm_cublas(const float* A, const float* B, float* C, int M, int N, int K);

// ---- 消融旋钮（由 main CLI 注入；默认值与 srs 一致）-------------------
namespace sgemm {
extern int g_smem1d_bk;          // 默认 16
extern int g_tile2d_min_blocks;  // 默认 1
extern bool g_verbose;           // 打印回退路径细节（测试断言路径覆盖用）
}

// ---- 注册表 ----------------------------------------------------------
namespace sgemm {
inline constexpr int K_NAIVE = 0, K_COALESCED = 1, K_SMEM1D = 2, K_TILE2D = 3,
                      K_VEC4 = 4, K_CPASYNC = 5, K_CPASYNC2 = 6, K_CUBLAS = 7,
                      KERNEL_COUNT = 8;

inline const char* kernel_name(int id) {
    static const char* names[KERNEL_COUNT] = {
        "naive", "coalesced", "smem1d", "tile2d", "vec4", "cpasync", "cpasync2", "cublas"};
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
        sgemm_vec4,  sgemm_cpasync,   sgemm_cpasync_v2, sgemm_cublas};
    return (id >= 0 && id < KERNEL_COUNT) ? fns[id] : nullptr;
}
}  // namespace sgemm
