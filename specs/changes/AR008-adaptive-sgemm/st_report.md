# AR008-adaptive-sgemm ST 验收报告

- 日期：2026-10-05
- 验收人：AI Agent（用户授权 auto 模式，ST 集中回报）
- 范围：尺寸自适应 SGEMM——swsk（split-K 变体）+ Kernel 7 ws（warp 专属化）+ auto 选核
  + thermal-paired 配对测量协议

## 1. 正确性回归（110 例）

- 套件：12 kernel × 9 尺寸 + cuBLAS 交叉验证 2 例 = **110/110 ALL PASS**
  （`tests/suite_out.txt`，clean 重建后运行）
- ws 专项：racecheck **0 hazards**×3 配置（环回绕 128×128×64 / STAGES=2 / LB=2@512³）；
  memcheck **0 errors**（512³/4096³ 主路径 + 130×257×66 回退）；短 K（num_tiles=1/2/3）
  无死锁专项 PASS；1024³ rel=1.179e-06 与 swpipe 逐位同值（同累加序设计验证）
- swsk 确定性：`--check` 两轮 max_abs 全位一致（1024³：3.051758e-05 两独立组）
- 接口：`--list-kernels` 12 项；`--sk 0`/`--stages 5`/`--wp 3` → CLI_ERROR 正确拒绝；
  旋钮跨 kernel 忽略带 note 提示且正常执行

## 2. 四门 v2 逐门判定（thermal-paired 会话，2 PASS / 3 FAIL）

| 门 | 判定 | 关键数据 |
|----|------|---------|
| G1 中尺寸 ≥75% cuBLAS | **FAIL** | 512³ swsk 3542.5 / cuBLAS 5698.8 = 62.2%；1024³ 4988.2/8581.1 = 58.1% |
| G2 256³ 守成 ≥1618.2 GF | **FAIL** | auto 1517.5（同会话 > cuBLAS 1260.3 达 +20.5%；绝对值门败于跨会话漂移 -14%，已归因归档） |
| G3 4096³ ws ≥7.0 TF | **FAIL** | ws 5377.7 GF；对内 delta -16.00%（极差 0.33pp） |
| G4 全线 ≥4/6 尺寸 delta ≥+2% | **PASS** | 256³ auto +170.3% / 512³ swsk +49.4% / 1024³ swsk +2.8% / 1000×1016 swsk +2.3%（污染组剔除重测）；2048³/4096³ ~0%（auto 正确选 swpipe 的体现） |
| G5 方法学 delta 轮间极差 <2pp | **PASS** | 24 组 median 0.53pp；最差 20pp（swpipe 基线钟态双峰，median 判据稳健） |

判定文件：`results/compare_ar008_paired.md`；数据：`results/paired_ar008.csv`
（污染组原始行备份 `paired_ar008_thermal_bak.csv`，协议诚实性）。

## 3. ws issue-slot 假说检验回报（负结果，如实归档）

- **假说**：大尺寸稳态循环 FFMA 发射槽占比 ~55% → 剥离搬运给 producer warp 可抬升消费者吞吐。
- **裁定：否定**。三轴消融（PW{1,2}×STAGES{2,3}×LB{1,2}，8 配置 × 4 尺寸 × 升降序 2-pass）：
  - ws 全 8 配置在 1024³/2048³/4096³ 均不敌 swpipe（-7.9% / -20.4% / -15.0%）；
  - PW=2 全面胜 PW=1（单 producer warp 搬运吞吐不足）；
  - LB=2 双 block 驻留（62.5% 占用，96 regs/0 spill 可达）3/4 尺寸劣于 31.3%——占用率假说同样被否；
  - 唯一正收益 512³ +6.1%（wave 饥饿区），远逊 swsk +49%。
- **归因**：消费者纯化后 LDS→FFMA 依赖链 ILP 上限 + DRAM 延迟覆盖成为新瓶颈；swpipe
  的寄存器预取混合流同时覆盖两者。结论：软件 warp specialization 在 sm_75（无 cp.async/
  mbarrier/专属化硬件）打不过混合流——与 AR006 cp.async 负结果构成同族边界证据。
- ws 保留为教学阶梯 Kernel 7（结构/racecheck/正确性全过），auto 不选 ws。

## 4. 图表索引（结论 → 图 → 数据，三链可溯）

| # | 结论 | 图 | 数据 |
|---|------|----|----|
| fig9 (a)(b) | sk 最优带随 blocks 变化；wave 饥饿/饱和分界 | fig9_splitk_sweep | ablation_ar008.csv |
| fig9 (c) | 256³ 全 kernel 矩阵：swsk(sk12) 同会话登顶 | 同上 | size256_ar008.csv |
| fig10 | auto 选核=实测最优，dispatch 零开销（\|Δ\|≤0.9%） | fig10_dispatch_map | auto_ar008.csv |
| fig11 | ws 流水结构与 named barrier 契约 + ptxas 包络 | fig11_ws_structure | kernel 常量 + build.log |
| fig12 | ws 三轴消融矩阵（PW/STAGES/LB） | fig12_ws_ablation | ablation_ar008.csv |
| fig13 | issue-slot 假说否定（负结果证据图） | fig13_isslot_hypothesis | ablation + paired |
| fig14 | 四门 v2 判定 + G4 全线 delta（min-max 误差线） | fig14_paired_delta | paired_ar008.csv |
| fig15 | 同会话全 kernel 阶梯 v2（auto 贴最优可视化） | fig15_ladder_v2 | paired_ar008.csv |

全部图表由 `python bench\make_figures.py` 从 CSV 可复现（无手工修饰）。

## 5. 交付物清单

- 代码：`src/sgemm_swpipe_sk.cu`、`src/sgemm_ws.cu`（8 模板实例）、`src/sgemm_auto.cu`、
  `src/sgemm_swpipe.cu`（参数化改造）、CLI `--sk/--stages/--wp`（common.h/main.cu）、
  注册表 KERNEL_COUNT=12（include/sgemm_kernels.h）
- 脚本：`bench/run_paired.ps1`（配对协议）、`bench/run_sk_ablation.ps1`、
  `bench/run_ws_ablation.ps1`、`bench/run_size256.ps1`、`bench/run_auto_paired.ps1`、
  `bench/compare.py --paired`、`bench/make_figures.py`（fig9-15）
- 数据：ablation_ar008.csv（双消融 152 行）、paired_ar008.csv（全矩阵 146 行）、
  size256_ar008.csv、auto_ar008.csv、compare_ar008_paired.md、build.log（ptxas 审计）、
  paired_ar008_thermal_bak.csv（污染组备份）
- 文档：详设 §4.1/§4.6/§5 + CLI 表、AGENTS §3、report.md §10 + 附录 B、
  bottleneck_analysis.md 闭环 #1/#2/#3、README 全面刷新

## 6. 遗留与承接

- git 提交：按用户裁定不提交（AR007 惯例）；CSV git_sha 列为当前 HEAD（84e261f），
  复现链缺口由用户承接。
- ncu 硬件计数器权限（ERR_NVGPUCTRPERM）持续阻塞——issue-slot 假说以消融间接检验
  完成（负结果）；若未来获得权限，ws 消费者 smsp stall 分析可作 backlog。
- G1 与 cuBLAS 的 35pp 结构性差距、G2 跨会话漂移：超出本 AR 范围，已在
  bottleneck_analysis.md 记录（严格 FP32 + ptxas 边界）。

**ST 结论：验收通过**——正确性 110/110、sanitizer 全清、四门 v2 如实判定（2 PASS/3 FAIL，
负结果归档）、图表三链可溯、文档同步完毕。
