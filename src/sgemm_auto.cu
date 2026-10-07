// =====================================================================
// sgemm_auto.cu — AR011 T006 auto v4 自动选核（几何 dispatch）
// ---------------------------------------------------------------------
// dispatch 表 v4（AR010 v3 基础上吸收 AR011 实测裁决；blocks =
// ceil(M/128)*ceil(N/128) 即 128x128 tile 2D grid 块数；48 SMs；deep 族
// 1 blk/SM = 48 槽/波，256x128 tile 网格 = blocks/2 记作 t）：
//   blocks <= 4   → swsk sk=6   （256³ 重度饥饿区，AR010 细扫守擂；
//                                 AR011 T005 复测 1553.5 再证）
//   blocks <= 16  → swsk sk=3   （512³ 单带：4273 >> streamk_w1 1777
//                                 （U=11 P 税爆炸，T005）> dsk_sk3 3121）
//   blocks <= 64  → dsk 几何 sk（sk = clamp(ceil(96/t), 2, 16)：
//                                 dsk 网格 = t·sk 命中 48·W 整波。
//                                 t=32（1024³/1000x1016/256x4096/4096x256）
//                                 → sk=3 与 v3 实测带一致（6279/6157）；
//                                 t 更小（矩形窄带）自动升 sk 补波，
//                                 首测回填见 auto_ar011.csv）
//   blocks >  64  → t = blocks/2 deep tiles：
//                   t%48==0（整波）或 t>192（尾波占比 <1/3 波，税小）
//                     → deep dbuf1（4096³ t=512 实测 deep 8994 >
//                     streamk_w1 8919，T005）
//                   其余（1<t/48<=4 且带零头尾波）→ streamk W=1
//                     （2048³ t=128：48 块恒 1 波 + solo 快路径，8721 =
//                     +2.1% over deep（波量化税实测）= 85.4% cuBLAS，
//                     T005；1024x2048 t=64 首测回填）
// K 钳制：sk = min(表值, ceil(K/8))（k-tile 不足不超切；sk<2 语义等价单波）。
// 冻结签名下的内部旋钮注入：临时改写 g_swsk_slices / g_deep_dbuf /
// g_streamk_waves 调用目标 kernel 后还原（save/restore，不污染同进程
// 后续 kernel 的旋钮状态）。
// 边界：对齐/N%4/K%4 由各 kernel 内部回退路径兜底（本层不管）。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>
#include <cstdint>

namespace {

// dispatch 表 v4（AR011 T003/T004/T005 实测裁决回填；证据链
// streamk_ar011.csv / lat_cover_ar011.csv + bottleneck 闭环 #6/#7）
int pick_sk(long long blocks) {
    if (blocks <= 4) return 6;    // 256³ 重度饥饿区（swsk）
    if (blocks <= 16) return 3;   // 512³ 单带（swsk）
    if (blocks <= 64) return 0;   // 1024³ 带（dsk，几何 sk 公式接管）
    return 1;                     // 2048³+ 多波区（deep / streamk W=1 细分）
}

}  // namespace

void sgemm_auto(const float* A, const float* B, float* C,
                int M, int N, int K) {
    const long long grid_m = (M + 127) / 128;
    const long long grid_n = (N + 127) / 128;
    const long long blocks = grid_m * grid_n;
    // deep 256x128 tile 数（deep grid；精确式，blocks/2 在 grid_n 奇数时失准）
    const long long t = ((M + 255) / 256) * grid_n;
    const int tiles = (K + 7) / 8;
    const int zone_sk = pick_sk(blocks);
    // 几何 sk 公式（FR5）：dsk 网格 = t·sk 命中 48·W 整波（W=2 起点，
    // 1024³ t=32 → sk=3 与 v3 实测带一致）；clamp(2,16) + K 钳制
    int sk = zone_sk;
    if (zone_sk == 0) {
        long long want = (48 * 2 + t - 1) / t;   // ceil(96/t)
        if (want < 2) want = 2;
        if (want > 16) want = 16;
        sk = (int)want;
    }
    if (sk > tiles) sk = tiles > 0 ? tiles : 1;

    // 区带 -> 目标 kernel：0=swsk(sk>=2)，1=dsk(sk>=2)，2=deep(单波)，
    // 3=streamk W=1（多波区尾波消除）
    int target;
    if (blocks <= 16) {
        target = 0;
    } else if (blocks <= 64) {
        target = (sk >= 2) ? 1 : 2;
    } else {
        const bool exact_wave = (t % 48) == 0;      // deep 整波（零尾波税）
        const bool big_grid = t > 192;              // 尾波占比小（4096³ 实测 deep 胜）
        target = (exact_wave || big_grid) ? 2 : 3;
    }
    const char* name = target == 0 ? "swsk"
                      : target == 1 ? "dsk"
                      : target == 2 ? "deep" : "streamk";

    if (sgemm::g_verbose) {
        std::printf("[auto] %dx%dx%d blocks=%lld t=%lld tiles=%d -> %s "
                    "(sk=%d, dbuf=%d, W=%d)\n",
                    M, N, K, blocks, t, tiles, name, sk,
                    target == 0 ? 0 : 1, target == 3 ? 1 : 0);
    }

    if (target == 0) {
        const int saved_sk = sgemm::g_swsk_slices;
        const int saved_db = sgemm::g_deep_dbuf;   // dsk/deep 不经此路径，防御性还原
        sgemm::g_swsk_slices = sk;
        sgemm::g_deep_dbuf = saved_db;
        sgemm_swpipe_sk(A, B, C, M, N, K);
        sgemm::g_swsk_slices = saved_sk;
        return;
    }
    if (target == 1) {
        const int saved_sk = sgemm::g_swsk_slices;
        const int saved_db = sgemm::g_deep_dbuf;
        sgemm::g_swsk_slices = sk;
        sgemm::g_deep_dbuf = 1;                     // AR010 实测 dbuf1 全尺寸 +3~14%
        sgemm_dsk(A, B, C, M, N, K);
        sgemm::g_swsk_slices = saved_sk;
        sgemm::g_deep_dbuf = saved_db;
        return;
    }
    if (target == 3) {
        const int saved_w = sgemm::g_streamk_waves;
        const int saved_db = sgemm::g_deep_dbuf;
        sgemm::g_streamk_waves = 1;                 // T005 校准：恒 W=1
        sgemm::g_deep_dbuf = 1;
        sgemm_streamk(A, B, C, M, N, K);
        sgemm::g_streamk_waves = saved_w;
        sgemm::g_deep_dbuf = saved_db;
        return;
    }
    const int saved_db = sgemm::g_deep_dbuf;
    sgemm::g_deep_dbuf = 1;                         // deep 单波同样固化 dbuf1
    sgemm_deep(A, B, C, M, N, K);
    sgemm::g_deep_dbuf = saved_db;
}
