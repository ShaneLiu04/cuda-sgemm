// =====================================================================
// sgemm_auto.cu — AR008 T006 自动选核（几何 dispatch，T004/T005 实测回填）
// ---------------------------------------------------------------------
// dispatch 表（blocks = ceil(M/128) * ceil(N/128)，即 128x128 tile 划分的
// 2D grid 块数；48 SMs / 2 block 驻留 = 96 槽，Quadro RTX 5000 实测标定）：
//   blocks <= 4   → swsk sk=12  （256^3 类：4 blocks=0.083 波饥饿区；
//                                  x12 → 48 blocks 恰满单波，实测 1508.7 GF
//                                  全场最优，超 smem1d +7.9% / cuBLAS +27%）
//   blocks <= 64  → swsk sk=4   （512^3/1024^3 类：实测最优片数 4；
//                                  512^3 3542 GF / 1024^3 ~5200 GF）
//   blocks >  64  → swpipe 单波 （2048^3+=多波饱和区，split-K 的 P 写出
//                                  开销纯反噬：2048^3 sk1=7143 > sk16=5441）
// K 钳制：sk = min(表值, ceil(K/8))（k-tile 数不足时不超切；sk<2 语义等价
//   swpipe 单波旁路）。
// 冻结签名下的内部旋钮注入：临时改写 g_swsk_slices 调用 swsk 后还原
//   （save/restore，不污染同进程后续 kernel 的旋钮状态）。
// 边界：对齐/N%4/K%4 由 swsk/swpipe 内部回退 tile2d 兜底（本层不管）。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>
#include <cstdint>

namespace {

// dispatch 表（T004/T005 实测；blocks 为 128x128 tile 2D grid 块数）
int pick_sk(long long blocks, int tiles) {
    int sk;
    if (blocks <= 4) {
        sk = 12;    // 256^3 类单波填满（4 blocks x 12 = 48 SMs）
    } else if (blocks <= 64) {
        sk = 4;     // 512^3 / 1024^3 类实测最优
    } else {
        sk = 1;     // 多波饱和区：单波 swpipe
    }
    if (sk > tiles) sk = tiles > 0 ? tiles : 1;   // K 钳制（不超切 k-tile）
    if (sk < 1) sk = 1;
    return sk;
}

}  // namespace

void sgemm_auto(const float* A, const float* B, float* C,
                int M, int N, int K) {
    const long long grid_m = (M + 127) / 128;
    const long long grid_n = (N + 127) / 128;
    const long long blocks = grid_m * grid_n;
    const int tiles = (K + 7) / 8;
    const int sk = pick_sk(blocks, tiles);

    if (sgemm::g_verbose) {
        std::printf("[auto] %dx%dx%d blocks=%lld tiles=%d -> %s (sk=%d)\n",
                    M, N, K, blocks, tiles,
                    sk >= 2 ? "swsk" : "swpipe", sk);
    }

    if (sk < 2) {
        sgemm_swpipe(A, B, C, M, N, K);   // 单波路径（含对齐回退）
        return;
    }
    const int saved = sgemm::g_swsk_slices;   // save/restore：不污染全局旋钮
    sgemm::g_swsk_slices = sk;
    sgemm_swpipe_sk(A, B, C, M, N, K);
    sgemm::g_swsk_slices = saved;
}
