// =====================================================================
// sgemm_deep_sk.cu — Kernel 9 变体 dsk：deep 的 split-K + 确定性固定序归约
// ---------------------------------------------------------------------
// 动机（AR010）：deep 128×256 tile 单波 grid 偏小——512³ 仅 8 blocks /
// 48 SM（17% 占用面），1024³ 32 blocks（1.33 波尾空转）。split-K 沿 K 维
// 切 z 片：grid=(n_t, m_t, sk)。波几何（1 block/SM × 48 slots/波）：
//   512³  8 blocks × sk6  = 48  = 1 精确波
//   1024³ 32 blocks × sk3  = 96  = 2 精确波
//   256³  2 blocks × sk{8,16} = 16/32（K 切片过浅，预期弱——消融实证）
//
// 主体复用：deep tile 主体经参数化共享（detail::deep_tile_grid，
// 见 sgemm_deep.cu）——TM16×TN8=128 acc、A 广播 + B swizzle 布局、
// 寄存器预取流水、DBUF 实例选择全部继承，本文件不重复实现。
//
// 归约（确定性）：默认（--rv2=0）走 T007 last-slice-direct——主体末片直写
// C、P 片统一默认 store（__stwt 写穿已实测回退：main +16μs 反压损失、
// 无归约收益，见 sgemm_deep.cu 头注与 AR010 T007 消融），归约
// C = P[0..sk-2]+C（链序与全 P 归约 v1 逐位一致，P 流量 12→8MB@1024³sk3，
// 归约 47→43μs 实测，流量地板 355 GB/s）；--rv2≠0 回退旧全 P 路径
// （detail::swsk_reduce v1/v2/v3，swsk/wsk 单一数值路径共享）。
//
// workspace：冻结签名下的既定取舍（AR008 design §4.2.1 先例）——静态
// RAII grow-only。P 容量极值：1024³ sk3 = 12MB。
//
// 边界语义：sk > num_tiles 时空片（t0>=t1）自然写全零；sk=1 旁路直走
// sgemm_deep 单波路径；对齐不满足（N%4/K%4/16B）回退 sgemm_2d_tile。
//
// --verbose 分段计时（AR010 新增，仅 verbose 模式——正常计时零开销）：
// 打印 main kernel / reduce 两段 μs，用于 G2@256³ 的 split-K 开销定界。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>

namespace {
constexpr int BM = 256, BN = 128, BK = 8;

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
            std::fprintf(stderr, "[dsk] workspace alloc failed (%zu floats): %s\n",
                         floats, cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
        cap = floats;
    }
};
Workspace g_ws;  // 静态实例：首次调用分配，之后零开销

}  // namespace

void sgemm_dsk(const float* A, const float* B, float* C,
               int M, int N, int K) {
    const int sk = sgemm::g_swsk_slices;

    // 对齐不满足 → 谓词化回退（与 deep 同条件；split-K 主路径依赖
    // float4 写 P 与归约 float4 读 P/C）
    const bool aligned =
        (N % 4 == 0) && (K % 4 == 0) &&
        (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0);
    if (!aligned) {
        if (sgemm::g_verbose) {
            std::printf("[dsk] size %dx%dx%d -> scalar fallback (tile2d)\n",
                        M, N, K);
        }
        sgemm_2d_tile(A, B, C, M, N, K);
        return;
    }

    // sk=1 旁路：直走 deep 单波路径（零 workspace / 零归约开销）
    if (sk < 2) {
        if (sgemm::g_verbose) {
            std::printf("[dsk] sk=%d -> bypass to deep single-wave\n", sk);
        }
        sgemm_deep(A, B, C, M, N, K);
        return;
    }

    const int num_tiles = (K + BK - 1) / BK;
    const int grid_n = (N + BN - 1) / BN;
    const int grid_m = (M + BM - 1) / BM;
    const int tps = (num_tiles + sk - 1) / sk;   // 每片 k-tile 数（ceil）

    // workspace：sk-1 个 M*N 切片已够（末片直写 C；按 sk 全量保留余量，
    // legacy 消融路径仍需 sk 个）
    g_ws.ensure((size_t)sk * M * N);

    // --verbose 分段计时（正常路径零开销：事件仅 verbose 模式创建）
    cudaEvent_t ev0 = nullptr, ev1 = nullptr, ev2 = nullptr;
    if (sgemm::g_verbose) {
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        cudaEventCreate(&ev2);
        cudaEventRecord(ev0);
    }

    // AR010 T007 last-slice-direct（默认 --rv2=0）：末片直写 C（P 片统一
    // 默认 store——__stwt 实测回退），归约读 sk-1 片 P + 热 C——P 流量
    // 12→8MB@1024³sk3（实测归约 47→43μs，地板 355 GB/s）。
    // --rv2≠0 = 旧全 P 消融路径（v1/v2/v3 归约，行为与 AR008-T006 逐字一致）
    const bool direct = (sgemm::g_reduce_ilp2 == 0);

    // ① split-K 主体：grid=(grid_n, grid_m, sk)，Out=P（direct：末片 Out_last=C）
    sgemm::detail::deep_tile_grid(A, B, g_ws.p, direct ? C : g_ws.p,
                                  M, N, K, tps, num_tiles, grid_n, grid_m, sk);
    // ② 确定性归约：direct ⇒ P[0..sk-2]+C；legacy ⇒ 全 P（v1/v2/v3）
    if (sgemm::g_verbose) cudaEventRecord(ev1);
    if (direct) {
        sgemm::detail::swsk_reduce_direct(g_ws.p, C, (long long)M * N, sk);
    } else {
        sgemm::detail::swsk_reduce(g_ws.p, C, (long long)M * N, sk);
    }
    if (sgemm::g_verbose) {
        cudaEventRecord(ev2);
        cudaDeviceSynchronize();
        float t_main = 0.f, t_red = 0.f;
        cudaEventElapsedTime(&t_main, ev0, ev1);
        cudaEventElapsedTime(&t_red, ev1, ev2);
        std::printf("[dsk] %dx%dx%d sk=%d: main %.1f us + reduce %.1f us "
                    "(reduce share %.1f%%)\n",
                    M, N, K, sk, t_main * 1e3, t_red * 1e3,
                    100.0 * t_red / (t_main + t_red));
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        cudaEventDestroy(ev2);
    }
}
