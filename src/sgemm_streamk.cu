// =====================================================================
// sgemm_streamk.cu — Kernel 10：Stream-K 统一调度（AR011 T002 实现中）
// ---------------------------------------------------------------------
// 动机（AR010 终态 + 2026107 分析）：两项结构性浪费可被统一消除——
//   ① 波量化：deep/dsk 的 grid 块数未必是 48（SM 数）整倍数（2048³ 128 块
//      = 2.67 波，尾波 -11%）；
//   ② 归约第二 kernel：dsk direct 归约 41μs@1024³ + 启动 ~3μs + P 自片
//      DRAM 往返；1024³ 门缺口折算仅 1.05μs——任何归约侧结构性节省即翻门。
//
// 方案（AR011 design §4.1/§4.2，三问三案裁定）：
//   - 块映射：迭代空间 (C-tile c, k-tile kt) tile-major 线性化 u = c·nt + kt，
//     [0, TOT) 连续切成 B = 48·W 块（波填充率恒 100%），每块 U = ceil(TOT/B)
//     单元；C-tile c 由连续块号区间 [b_lo(c), b_hi(c)] 覆盖（块号升序 ≡ k 升序）。
//   - 归并：per-tile 票据（atomic 票据 + __threadfence release-acquire）；
//     最后完成块按块号升序归并部分积，own 位置以寄存器 acc 代入——
//     z 升序括号链与 dsk direct 归约逐位对齐（design §4.2.2 论证）；
//     票据自清洁（赢家归零，跨 launch 由流序保证）。
//   - cover(c)==1 快路径：唯一覆盖块跳过 P/票据，直写 C（deep 同型）。
//   - 主体自包含：deep 计算主体（TM16×TN8=128 acc、A 转置+PAD4、B XOR
//     swizzle、寄存器预取流水、DBUF 双缓冲）逐拷贝（design §4.1 问题 3：
//     不共享函数——防 ptxas 重排病灶 + AGENTS §2 自包含军规）。
//
// 状态：T001 Red——本文件当前仅含旋钮定义（main.cu 绑定链接所需），
// kernel 主体与注册接入在 T002 Green 落地（注册槽位现为 nullptr）。
// =====================================================================
#include "sgemm_kernels.h"

namespace sgemm {
int g_streamk_waves = 0;   // --waves：0 = auto 公式 clamp(floor(TOT/(48·16)),1,8)
                           // （design §4.2.6；T005 sweep 校准后 auto v4 表接管）
int g_l2_persist = 0;      // --persist：1 = 计时区内 accessPolicyWindow 钉 C
                           // + 计时区后强制复位（先于 cuBLAS 锚定，协议纪律）
double g_l2_hit = 0.8;     // --hit：hitRatio（0.5..1.0）
}
