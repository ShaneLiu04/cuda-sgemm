// =====================================================================
// sgemm_wide_sk.cu — Kernel 8 变体 wsk：wide 的 split-K（AR009 T003）
// ---------------------------------------------------------------------
// 动机（AR009 srs FR2/FR3）：中/小尺寸 wave 饥饿区（512³ 16 blocks /
// 1024³ 64 blocks < 48 SM 波容量），wide 的 512 线程宽块使每 block warp
// 数翻倍（16 vs swsk 的 8）——同 blocks 数下并行度 ×2。G1/G2 的打穿
// 候选：wsk@512³(sk4) 聚合占用 67%（swsk4 现状 33%）、1024³(sk3) 精确
// 2 波（192 blocks = 2×96）、256³(sk12) 50%（swsk12 现状 25%，P=3MB
// 恰入 L2 4MB）。
//
// 主体复用：wide tile 主体经参数化共享（detail::wide_tile_grid，见
// sgemm_wide.cu）——512 线程双角色流水、A 转置 + B swizzle、单缓冲双
// 同步、LB 实例选择（--wlb 对 wide/wsk 双生效）全部继承，本文件不重复
// 实现。
//
// 归约复用：detail::swsk_reduce（自 sgemm_swpipe_sk.cu 提升，AR009
// 跨单元共享）——确定性固定 z 序，与 swsk 数值路径单一来源，逐位一致。
//
// workspace：冻结签名下的既定取舍（AR008 design §4.2.1 同构）——静态
// RAII grow-only（只增不减，warmup 期完成分配，计时区零分配）。本文件
// 自备实例（与 swsk 的 workspace 相互独立，不跨单元共享内存管理）。
//
// 边界语义：sk=1 旁路直走 sgemm_wide 单波路径（零归约开销）；sk >
// num_tiles 时空片（t0>=t1）自然写全零；对齐不满足回退 sgemm_2d_tile。
// --sk 旋钮对 swsk/wsk 双生效（main.cu note 提示）。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr int BM = 128, BN = 128, BK = 8;   // 与 wide tile 主体一致

// ---- grow-only workspace（RAII；程序退出统一释放）--------------------
struct Workspace {
    float*  p   = nullptr;
    size_t  cap = 0;        // 容量（floats）
    ~Workspace() {
        if (p) cudaFree(p);
    }
    void ensure(size_t floats) {
        if (cap >= floats) return;
        if (p) cudaFree(p);
        cudaError_t err = cudaMalloc(&p, floats * sizeof(float));
        if (err != cudaSuccess) {
            p = nullptr; cap = 0;
            std::fprintf(stderr, "[wsk] workspace alloc failed (%zu floats): %s\n",
                         floats, cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
        cap = floats;
    }
};
Workspace g_wsk_ws;  // 静态实例：首次调用分配，之后零开销

}  // namespace

void sgemm_wsk(const float* A, const float* B, float* C,
               int M, int N, int K) {
    const int sk = sgemm::g_swsk_slices;

    // 对齐不满足 → 谓词化回退（与 wide 同条件；split-K 主路径依赖
    // float4 写 P 与归约 float4 读 P/C）
    const bool aligned =
        (N % 4 == 0) && (K % 4 == 0) &&
        (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0);
    if (!aligned) {
        if (sgemm::g_verbose) {
            std::printf("[wsk] size %dx%dx%d -> scalar fallback (tile2d)\n", M, N, K);
        }
        sgemm_2d_tile(A, B, C, M, N, K);
        return;
    }

    // sk=1 旁路：直走 wide 单波路径（零 workspace / 零归约开销）
    if (sk < 2) {
        if (sgemm::g_verbose) {
            std::printf("[wsk] sk=%d -> bypass to wide single-wave\n", sk);
        }
        sgemm_wide(A, B, C, M, N, K);
        return;
    }

    const int num_tiles = (K + BK - 1) / BK;
    const int grid_n = (N + BN - 1) / BN;
    const int grid_m = (M + BM - 1) / BM;
    const int tps = (num_tiles + sk - 1) / sk;   // 每片 k-tile 数（ceil）

    // workspace：sk 个 M*N 切片（grow-only；空片由主体写零，全量有效）
    g_wsk_ws.ensure((size_t)sk * M * N);

    // ① split-K 主体：grid=(grid_n, grid_m, sk)，Out=P（wide tile 主体）
    sgemm::detail::wide_tile_grid(A, B, g_wsk_ws.p, M, N, K,
                                  tps, num_tiles, grid_n, grid_m, sk);
    // ② 确定性归约：P → C（固定 z 序；与 swsk 共享单一数值路径）
    sgemm::detail::swsk_reduce(g_wsk_ws.p, C, (long long)M * N, sk);
}
