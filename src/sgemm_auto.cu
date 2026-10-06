// =====================================================================
// sgemm_auto.cu — AR010 T006 auto v3 自动选核（几何 dispatch，T004/T005
//                  同会话 1620 MHz 稳态实测回填；v2 表为 AR009 数据）
// ---------------------------------------------------------------------
// dispatch 表 v3（blocks = ceil(M/128)*ceil(N/128) 即 128x128 tile 2D grid
// 块数；48 SMs；deep 族 1 blk/SM = 48 槽/波，256x128 tile 网格 = blocks/2）：
//   blocks <= 4   → swsk sk=6   （256³ 重度饥饿区：T005 sk6=1560-1570 >
//                                  sk5 1477 / sk7 1533（细扫）> dsk_sk12 1346 /
//                                  smem1d 1394；均衡切片假说证伪，分割并行度
//                                  与归约税的权衡峰在 sk6）
//   blocks <= 16  → swsk sk=3   （512³ 单带：swsk_sk3 4295 守擂 > dsk_sk6 3882
//                                  （deep tile 2x 大 → 同 blocks 需 2x sk →
//                                  P 税 6MB + K 切片浅，T004 实测裁定））
//   blocks <= 64  → dsk sk=3 dbuf1（1024³/1000x1016 带：6296/6079 >
//                                  swsk_sk3 5405/5215（+16.5%/+16.6%）；
//                                  dsk 网格 = blocks/2 x sk3 = 48..96 = 1-2 精确波，
//                                  波量子化兑现；dbuf1 单同步 +10.5%）
//   blocks >  64  → deep dbuf1  （2048³+ 多波饱和区：8569/8113 > swpipe
//                                  7103/6630（+20.8%/+22.3%）；split-K P 写出
//                                  与归约开销在无饥饿时纯反噬，取单波 deep）
// K 钳制：sk = min(表值, ceil(K/8))（k-tile 不足不超切；sk<2 语义等价单波）。
// 冻结签名下的内部旋钮注入：临时改写 g_swsk_slices / g_deep_dbuf 调用目标
// kernel 后还原（save/restore，不污染同进程后续 kernel 的旋钮状态）。
// 边界：对齐/N%4/K%4 由各 kernel 内部回退路径兜底（本层不管）。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>
#include <cstdint>

namespace {

// dispatch 表 v3（AR010 T004/T005 同会话 1620 稳态实测回填；依据注记见上，
// 完整证据链 deep_ar010.csv / g2_ar010.csv + tasks.md T004/T005）
int pick_sk(long long blocks) {
    if (blocks <= 4) return 6;    // 256³ 重度饥饿区（swsk）
    if (blocks <= 16) return 3;   // 512³ 单带（swsk）
    if (blocks <= 64) return 3;   // 1024³ 带（dsk，deep 网格 48-96 精确波）
    return 1;                     // 2048³+ 多波饱和区（deep 单波）
}

}  // namespace

void sgemm_auto(const float* A, const float* B, float* C,
                int M, int N, int K) {
    const long long grid_m = (M + 127) / 128;
    const long long grid_n = (N + 127) / 128;
    const long long blocks = grid_m * grid_n;
    const int tiles = (K + 7) / 8;
    const int zone_sk = pick_sk(blocks);
    const int sk = zone_sk > tiles ? (tiles > 0 ? tiles : 1) : zone_sk;

    // 区带 -> 目标 kernel：0=swsk(sk>=2)，1=dsk(sk>=2)，2=deep(单波)
    const int target = (blocks <= 16) ? 0 : (sk >= 2 ? 1 : 2);
    const char* name = target == 0 ? "swsk" : (target == 1 ? "dsk" : "deep");

    if (sgemm::g_verbose) {
        std::printf("[auto] %dx%dx%d blocks=%lld tiles=%d -> %s (sk=%d, dbuf=%d)\n",
                    M, N, K, blocks, tiles, name, sk,
                    target == 0 ? 0 : 1);
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
        sgemm::g_deep_dbuf = 1;                     // T004 实测 dbuf1 全尺寸 +3~14%
        sgemm_dsk(A, B, C, M, N, K);
        sgemm::g_swsk_slices = saved_sk;
        sgemm::g_deep_dbuf = saved_db;
        return;
    }
    const int saved_db = sgemm::g_deep_dbuf;
    sgemm::g_deep_dbuf = 1;                         // deep 单波同样固化 dbuf1
    sgemm_deep(A, B, C, M, N, K);
    sgemm::g_deep_dbuf = saved_db;
}
