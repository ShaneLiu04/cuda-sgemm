# AR009 ST 验收报告（wide-occupancy）— 2026-10-06

> 集中回报（auto 模式：开发各阶段不询问，ST 统一汇报）。全部数字实测可复现，
> 原始 CSV 行级 gpu_state 逐行可核。

## 1. 验收范围

srs FR1（512 线程宽块 wide/wsk 攻坚占用率墙）+ FR2（auto v2 重标定）+ FR3（三门 v2 复判）。
任务 T001-T008 全数 passing（specs/changes/AR009-wide-occupancy/tasks.md）。

## 2. ST 正确性门（全绿）

| 检查 | 结果 |
|------|------|
| 回归套件（12 kernel × 尺寸/边界/回退） | **128/128 ALL PASS** |
| 数值同源专项（bitwise memcmp 临时工具） | wide==swpipe 逐位 YES（1024³/1000×1016×1024/256³）；wsk==swsk 逐位 YES（1024³sk4、256³sk12 max_abs 双方一致） |
| 确定性 | wsk 双独立进程 max_abs 全位一致；swsk 归约固定 z 序 |
| memcheck | wide/wsk 主路径+回退 0 errors |
| racecheck | wide 0 hazards（单缓冲双同步契约） |
| ptxas 审计（build.log） | wide LB=1 = 79 regs / LB=2 = **64 regs/0 spill**（2×512×64=65536 恰满 64K → 100% 占用判据达成）；wsk 复用 wide 实例 |

## 3. 五门 v3 判定（compare_ar009_paired.md，3 PASS / 2 FAIL）

| 门 | 判定 | 关键数据 |
|----|------|---------|
| G1@512³ ≥75% cuBLAS | **PASS（75.03%）** | auto v2 4275.5 / cuBLAS 5698.8（swsk sk3 调点） |
| G1@1024³ | **FAIL（64.20%）** | 5377.3/8377.1；缺口 906 GF 架构性（LDS 墙） |
| G2@256³ ≥1618.2 | **FAIL（95.4%）** | swsk_sk6 1544.3；超同会话 cuBLAS **1.240×** |
| G3@4096³ ≥7.0TF | **PASS（102.6%）** | swpipe 3 轮中位 7185.3（批 A 双峰如实披露） |
| G4' auto v2 ≥4/6 +2% | **PASS** | 512³ +21.5% / 1024³ +13.3% / 1000×1016 +13.3% / 256³ +2.2% |
| G5' dispatch 保真 <2pp | **PASS** | 最大 0.33pp |

用户指令为 G1/G2/G3 三门全过——**实测裁定：G3 首过（AR008 FAIL→PASS），G1@512³ 刀锋过，
G1@1024³/G2 未达**。按数据真实性军规如实汇报，不放宽判定。

## 4. 核心科学结论（负结果 + 新规律）

1. **占用率假说证伪（三重独立证据）**：①wide LB1(50%)≈LB2(100%) 全尺寸同速；
   ②半填充 sk3(1 blk/SM) 反超满填充 sk6（wsk +18%/swsk +25%@512³）；
   ③LB 效应 ±0-4% 符号翻转，2048³ 50% 占用反而 +2.2%。
2. **真墙 = LDS.128 带宽**：wide 4LDS/64FFMA=1:16 vs swpipe 平衡点 3:32；
   wide/wsk best-vs-best 全尺寸 0.57-0.75× 于 swsk 最优。swpipe 处 sm_75 FP32 Pareto 点。
3. **稳态测量纪律**：1620 MHz 持续态为诚实基准（8/8 探针 = 5405.03 丝毫不差）；
   boost 行为瞬态彩票；4096³ 批间双峰（同 declared 时钟 ±13%）披露为方法学边界案例。

## 5. 图表交付（fig16-21，make_figures.py 可复现）

| 图 | 内容 | 数据 |
|----|------|------|
| fig16 | wide 结构 + 资源包络 + LSU 墙早期信号 | smoke_wide + build.log |
| fig17 | pre-wave 扫描（半填充>满填充、精确波、wsk 落败） | ablation_ar009.csv |
| fig18 | LB 消融三探针裁决 | ablation_wlb_ar009.csv |
| fig19 | auto v2 + G4' + 稳态纪律 | auto_ar009 + boost_lottery |
| fig20 | 五门 v3 判定 | paired_ar009.csv |
| fig21 | 全 kernel 阶梯 v3（wide/wsk 入榜） | paired_ar009.csv |

## 6. 事故与披露记录（军规执行）

- performance.csv 误删恢复（T002，63 行备份还原 + 54 行重跑，7 行散跑记损）
- sgemm_bench 旧二进制空指针 AV（T003，重建后全通，无数据污染）
- T003 冒烟 2.07× 钟态混杂虚高（T004 同会话修正为 0.83×，冒烟行保留为证）
- auto v2 首轮 dispatch 误判（boost 态行污染，attempt A 36 行归档 dispatchA.csv）
- 256³ 冷启动伪影（空闲降频首轮 0.44×，热身 spin 后消失，T007 协议纳入）
- 4096³ 批间双峰（attempt A 6.5T vs 正式 7.3T，同 declared 时钟——G3 附态报告）

## 7. 结论

AR009 以**负结果 + 可复现证据链**的方式关闭了占用率路线：wide/wsk 保留为教学阶梯
Kernel 8（结构完整、数值同源、sanitizer 全清、100% 占用达成），其证伪价值与
auto v2（+21.5%@512³）、G3 首过（7185.3 GF）、稳态测量纪律共同构成本 AR 交付。
建议归档：specs/changes/AR009-wide-occupancy → archive（含 5 份 CSV + 6 图 + 判定文书）。
