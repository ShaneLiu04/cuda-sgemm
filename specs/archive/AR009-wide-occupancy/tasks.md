# [AR009] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR009 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-05 |

> **实验与图表纪律（AGENTS.md §6，每任务强制）**：每个任务完成时即产出自己的
> 实验数据与图表交付物（下表"实验/图表交付物"列），全部经 `make_figures.py`
> 可复现生成；AR 收尾统一回填 report.md/README。禁止"做完代码、图表最后凑数"。

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 实验/图表交付物 |
|----|---------|------|------|-----------------|
| T001 | Red：注册表扩容 wide + wsk（K_WIDE=12/K_WSK=13, fn=nullptr, KERNEL_COUNT=14）+ CLI 名单 → 128 例中 18 例 FAIL 可观察 | - | passing | 2026-10-05 完成：套件 **110/128 PASS**，wide 9 例 + wsk 9 例 FAIL（"[ERROR] kernel not registered (fn=nullptr)" 逐例打印）——Red 可观察；既有 12 kernel 110 例零回归；CLI 名单/`--wlb` 解析校验（common.h）先行接入（注入接线随 T002 Green） |
| T002 | Green wide：sgemm_wide.cu（512 线程双角色流水，模板\<LB\> 两实例）+ ptxas 审计（LB=2 ≤64 regs 0 spill 判据 / 否则保底翻转）+ 128 例 + memcheck/racecheck | T001 | passing | 2026-10-05 完成：①正确性 **119/128**（wide 9/9 PASS，仅 wsk 9 例待 T003；回退 3 例走 tile2d）②ptxas 审计（build.log）：**wide\<LB=2\> = 64 regs / 0 spill / 8320B**（2×512×64=65536 恰满 64K → 2 block/SM = **100% 占用判据达成**；首轮 64regs+8B spill → 寻址重构【预计算基址/守卫 + Out 指针延迟物化】后 0 spill）、wide\<LB=1\> = 79 regs / 0 spill ③**数值序专项（临时 bitwise 工具 memcmp）**：1024³/1000×1016×1024/256³ wide==swpipe 逐位 YES + wide 双跑确定 YES（同 k 升序 FMA 链验证）④memcheck 0 errors（512³ 主路径 + 130×257×66 回退）⑤racecheck 0 hazards ⑥**fig16_wide_structure**：(a) 单缓冲双同步时空图 (b) 资源包络表（swpipe/wide LB1/LB2 + LDS:FFMA 列）(c) 冒烟柱状 + LSU 墙预测线。**重大负结果早期信号**：smoke（12 行 smoke_wide_ar009.csv）显示 wide 全尺寸低于 swpipe（512³ -19% / 1024³ -32% / 4096³ -39%），且 **LB=1(50% 占用) vs LB=2(100% 占用) 速度相同** → 占用率假说（srs §1 根因诊断）被直接挑战；deficit 与 LDS:FFMA 比率恶化 1.5×（3:32 vs 4:64）吻合（4096³ LSU 墙预测 0.667× = 4517 vs 实测 4162-4254）——**真墙候选 = LDS.128 带宽而非占用率**；swpipe 恰在 LDS/FFMA 平衡点（Pareto 最优）佐证。正式裁定归 T004/T005 消融，G1/G2 期望转移至 wsk（延迟受限区并行度翻倍），G3 路线需在 T004 数据后重估（128-acc 深分块候选）。**事故记录**：T002 会话中 performance.csv 误操作丢失 AR007 矩阵 61 行（PowerShell 过滤缺陷）→ 按军规处置：63 行 pre-AR007 备份还原 + run_matrix.ps1 -SkipAblations 重跑 canonical 9×6=54 行（新时间戳 23:30-23:39，swpipe 4096³=6375.78 与历史区间一致；ablation_ar007.csv 独立未受损）；7 行无法识别散跑行如实记损；全套图 fig1-16 自恢复数据可复现重生成 |
| T003 | Green wsk：sgemm_wide_sk.cu（workspace + sk=1 旁路 + 回退）+ detail::swsk_reduce 提升（swpipe_sk 重构，确定性双轮 --check）+ --sk 双生效 + 128/128 | T002 | passing | 2026-10-05 完成：①正确性 **128/128 ALL PASS**（wsk 9/9：主路径/边界/回退 tile2d）②**数值同源专项**：wsk==swsk 逐位一致——1024³sk4 双方 max_abs=3.051758e-05/max_rel=5.559054e-07、256³sk12 双方 =4.768372e-06/2.026477e-07（wide tile==swpipe[T002 bitwise 证实] + detail::swsk_reduce 单一归约路径 ⇒ 数学同源闭环）③确定性双跑：wsk@1024³sk4 两独立进程 max_abs 全位一致 ④memcheck 0 errors（512³sk4 主路径 + 130×257×66 回退）⑤配对冒烟（smoke_wsk_ar009.csv，9 行带 sk 标签）：**256³sk12 wsk 1287 vs swsk 621 = 2.07×（钟态修正后仍 ≥1.75×）**、512³sk4 2570 vs 3542 = 0.73×、1024³sk3 3387 vs 5378 = 0.63×（≈LSU 墙预测 0.667×）——**wsk 在延迟受限区兑现并行度红利、在计算区被 LDS 带宽墙压制**，与 wide 冒烟结论互补自洽；wsk@256³ 达同会话 cuBLAS 97%（1287 vs 1327）。RSD>5% 归因：钟态双峰 1620↔1920MHz + max 离群（首轮冷启动/时钟爬坡，median≈min 稳态），正式裁定归 T004/T005。**T004 修正（2026-10-06）**：本冒烟 2.07× 系钟态混杂（swsk 行 1620MHz / wsk 行 1920MHz，gpu_state 列可查）；同会话消融（ablation_ar009.csv）实为 wsk@256³ 0.83×——冒烟仅作早期信号，结论以 T004 为准。**过程事故**：首跑 sgemm_bench --kernel wsk AV(0xC0000005)——因 bench 目标未随注册表更新（旧二进制 fns[13]=nullptr 空指针调用）；重建 sgemm_bench 后全通，无数据污染（事故窗口内无 CSV 落盘） |
| T004 | 精确波消融：wsk sk∈{1,2,3,4,6,8,12,16} + swsk 补 {3,6} × {256³,512³,1024³,1000×1016×1024,2048³}（2-pass 升降序 + 冷却门控） | T003 | passing | 2026-10-06 完成：**ablation_ar009.csv 170 行**（实测扩为 wsk+swsk 双 kernel 全 sk 扫描——swsk 仅补 {3,6} 会致 fig17 delta 曲线跨会话不可比，本日已证 swsk@256³ 跨会话差 2.2×；偏离 tasks 原文已在此记录）；同会话 1920MHz 持续态、冷却门 47C、4 分钟跑毕。**fig17_prewave_sweep**：(a-e) 五尺寸 sk 曲线（wsk/swsk + cuBLAS 参考线 + 波几何注记）(f) best-vs-best 汇总。核心结论：①**wsk 全尺寸 best-vs-best 低于 swsk（0.83/0.68/0.74/0.75/0.61×）**——LDS 带宽墙跨尺寸全面验证，G1 经 wsk 路线 FAIL（512³ 2917≪4274，1024³ 3996≪6436）②**半填充>满填充**（512³ 双 kernel 一致：wsk sk3[48blk,1/SM]=2917 > sk6[96,2/SM]=2479；swsk 4273>3413）——split 开销 > 额外 warp 收益，与 T002 wide LB1==LB2 互证，**占用率假说否定（待 T005 形式收尾）**③精确波点兑现：1024³ 双 kernel sk3（192blk=2×96 精确波）峰值，sk1（64blk=1.33 波不平衡）显著低 ④**意外收获 swsk sk3/sk6 调优点**：512³ sk3=4273（vs AR008 默认 sk4 3537 = **+20.8%**，距 G1 门 4274 差 1 GFLOPS）、1024³ sk3=5381（vs sk4 +13.2%）、256³ sk6=1545（vs 同会话 cuBLAS 1196-1251 = **1.23-1.29×**）——T006 auto v2 dispatch 吸收点 ⑤G2 说明：绝对门 1618.2 高于本会话 cuBLAS 256³（1196-1251）29%，门值为 AR008 钟态标定，与 2026-10-06 会话钟态不可比，G2 终判留 T007（相对判据）。**T003 冒烟结论修正**：冒烟 swsk 行 1620MHz 降频态 vs wsk 行 1920MHz——2.07× 为钟态混杂虚高；同会话消融后 wsk@256³ 实为 0.83×。教训：跨 kernel 配对必须逐行核 gpu_state 列 |
| T005 | wide/wsk LB 消融：--wlb {1,2} × 尺寸 + vs swpipe/swsk 冷态配对 → 占用率→效率假说直接检验 | T004 | passing | 2026-10-06 完成：**ablation_wlb_ar009.csv 42 行**（run_wlb_ablation.ps1：wide wlb{1,2}×{512³,1024³,2048³} + wsk sk{3,12}×wlb{1,2}×{256³,512³,1024³} + swpipe 参照，双 pass 正反序）。**fig18_wide_lb**：(a) wide 50% vs 100% 占用 (b) wsk sk×LB 交叉 (c) 三重证据裁决面板。**裁定：占用率假说正式否定**——①wide LB1/LB2 = 1.009/见 1024³ 双峰/1.022（**2048³ 50% 占用反而 +2.2% 更快**）②wsk LB 效应 ±0-4% 且符号随 sk 翻转（sk3: LB1 +4.1%@512³；sk12: LB2 +3.5%@512³）③与 T002（LB1==LB2）+ T004（半填充 sk3 > 满填充 sk6：wsk +18%/swsk +25%@512³）三重互证。**真墙 = LDS.128 带宽**（wide 4LDS/64FFMA=1:16 vs swpipe 3:32 平衡点，sm_75 FP32 Pareto）。同会话 swpipe 参照：512³ 2369 / 1024³ 5541 / 2048³ 7145；wide_wlb2@2048³ 4094 = 0.573× swpipe。**1024³ 胜者修正**：swpipe 5541 > swsk_sk3 5381（T005 直测）> T004 swsk_sk1 行中位 4619（该行一 pass 1620MHz 降频污染——gpu_state 列可查，教训：median-of-2 需逐行核钟态，T007 将按 gpu_state 过滤）。G1/G2 经 wide/wsk 路线终判 FAIL 转入 T007 正式记录；G4'/auto v2 收益路径确认（swsk sk3/sk6 调优点 + swpipe@1024³） |
| T006 | auto v2：dispatch 表按 T004/T005 实测胜出行回填 + 背靠背配对验证（auto vs winner \|Δ\|≤2%）+ verbose 依据 | T004,T005 | passing | 2026-10-06 完成：①**dispatch v2 终版**（sgemm_auto.cu）：blocks≤4→swsk sk6（256³ 类）/ blocks≤64→swsk sk3 / >64→swpipe——**两轮迭代**：首轮曾按 T005 boost 态行误判 1024³/1000×1016 胜者为 swpipe（blocks≥48→swpipe），**boost 彩票实验**（boost_lottery_ar009.csv：swsk_sk3@1024³ 连续 8 探针 **8/8 = 5405.03 GFLOPS 丝毫不差 @1620 稳态**）证明 boost 行（1935-1950）为瞬态彩票、**1620 持续态才是 AGENTS §5 "跑至稳态"基准**；稳态排序 1024³：swsk_sk3 5405 > sk4 4767 > swpipe 4629 → 修正为 sk3（1000×1016 佐证：sk3 boost 直测 6064@1935，线性归一成立比率 1.168≈钟比）。误判轮数据完整保留 auto_ar009_dispatchA.csv（36 行，负结果归档）②**配对验证**（run_auto_paired_v2.ps1 → auto_ar009.csv 42 行，v1 选择/v2 胜者/auto 背靠背 ×2 轮 + 256³ 热态补跑）：dispatch 正确性 auto vs winner 全尺寸 \|Δ\|≤0.33%（512 0.12%/1024 0.06%/1000 0%/2048 0.08%/4096 0.26%/256 0.33%）③**G4' 裁定 PASS**：256³ +2.2% / 512³ **+21.5%** / 1024³ +13.3% / 1000×1016 +13.3% / 2048³ +0.08% / 4096³ -0.26% → **4/6 尺寸 ≥+2%** ✓④**fig19_dispatch_v2**：(a) G4' 配对柱状+PASS 标注 (b) dispatch 分区图+实测锚点 (c) 稳态纪律证据（8/8 探针+瞬态 boost 行+4096³ 批间双峰注记）⑤**测量态新发现**：4096³ swpipe 批间双峰 6.49-6.59T vs 7.31-7.41T（同 declared 1905-1920MHz，批内一致批间差 13%）——核内有效时钟不可经行间查询观测，G3 判定须带态报告；256³ 冷启动伪影（空闲降频后首轮 682 GFLOPS=0.44×，GPU 唤醒 spin 后消失）——正式测量前需热身 spin，T007 协议纳入 |
| T007 | 三门 v2 复判：run_paired v3（wide/wsk/auto/swsk/swpipe/cublas × 6 尺寸 × 3 轮）→ G1/G2/G3/G4'/G5' 判定 | T006 | passing | 2026-10-06 完成：**paired_ar009.csv 168 行**（run_paired_v3.ps1：swpipe 基线交替配对 ×3 轮、热身 spin、冷却门 47C、wsk/swsk 尺寸特定最优 sk）。**fig20_paired_delta_v3**（(a) G1 百分比+75%门线 (b) G2/G3 绝对门 (c) G5' 保真）+ **fig21_ladder_v3**（6 尺寸全 kernel 排名，wide/wsk 入榜）+ **compare_ar009_paired.md**。**判定：G3/G4'/G5' PASS，G1/G2 FAIL**——①G1@512³ **75.03% PASS（刀锋）**、@1024³ 64.20% FAIL（缺 906 GF=14.4%，架构性：cuBLAS 84% peak vs 我方 54.5% peak，LDS 墙）②G2 swsk_sk6 1544.3=门值 95.4% FAIL（超同会话 cuBLAS 1.240× 达成；1618.2 为 AR008 会话标定跨会话敏感）③**G3 swpipe@4096³ 中位 7185.3=102.6% 首次 PASS**（AR008 FAIL 6474；批 A 双峰 6488-6591 如实披露，核内有效时钟不可观测边界案例）④G4' 4/6 尺寸 ≥+2%（512³ +21.5%）⑤G5' 最大 0.33pp。数据质量：多数组 n=3 三值一致（auto@1024 5377/5377/5377）；行级 gpu_state 全核（256³-1024³ 全 1620 稳态）。**wide/wsk 最终定位（负结果归档）**：全尺寸 0.57-0.75× 于 swsk 最优（512³ 2928 vs 4267 / 1024³ 3393 vs 5377 / 4096³ 4409 vs 7185），三重证据链 fig16/17/18 闭环 |
| T008 | 文档同步 + ST 验收：详设 §4.1/§4.6/§5 Kernel 8 + AGENTS --wlb + report/README/bottleneck 闭环 #4（占用率墙→wide）+ figures 索引 + st_report.md 集中回报 | T007 | passing | 2026-10-06 完成：①详设：§4.1 接口表加 sgemm_wide/sgemm_wsk（K_WIDE=12/K_WSK=13/COUNT=14）+ auto 注释升 v2、§4.6 旋钮表加 `--wlb`（wide/wsk 作用域、默认 2、裁定依据）+ `--sk` 作用域扩 wsk、§5 新增 Kernel 8 小节（结构/裁定/三重证据/负结果定位）②AGENTS.md §3 消融旋钮清单加 --wlb ③report.md 新增 §11 + §11.1（占用率证伪）+ §11.2（dispatch v2 表）+ 附录 C（fig16-21 三链索引）④bottleneck_analysis.md 新增闭环 #4（占用率墙→LDS 带宽墙，含意外正收益 4 项）⑤README：阶梯表加 K8/K8'/auto v2 行、AR009 更新段、架构演进 K8 行、旋钮/paired v3 示例、128 例/21 图/九版数字刷新⑥st_report.md 集中回报（ST 门全绿 + 五门判定 + 科学结论 + 事故披露 6 项 + 归档建议）⑦终检：全量重建 30/30 + 套件 **128/128** + make_figures 全 21 图可复现生成 + 全部改动文件 BOM 验证（含 4 个 md 补 BOM） |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

- 2026-10-05（T001/T002）：T001 Red（110/128，wide/wsk 各 9 例 fn=nullptr FAIL 可观察）；
  T002 Green wide 完成（119/128，LB=2 = 64 regs/0 spill = 100% 占用判据达成，
  数值序 memcmp 逐位 YES，memcheck/racecheck 全清，fig16）。**关键负结果**：
  smoke 显示 wide 全面低于 swpipe 且 LB=1/LB=2 无差 → 占用率假说被挑战，
  真墙候选 = LDS.128 带宽（deficit ≈ LDS:FFMA 比率 1.5×⁻¹）；G1/G2 期望转
  移至 wsk（延迟受限区），G3 路线待 T004 数据重估。performance.csv 事故
  （61 行丢失）已按备份还原 + canonical 矩阵重跑处置（详见 T002 行），
  全套图可复现重生成。

## 阶段门控记录

- 2026-10-05：用户指令 auto 模式（沿用 AR007/AR008 惯例）——req/design 阶段门控
  跳过；ST 验收集中回报补偿（含三门 v2 复判裁定与占用率假说检验结果）。
- 2026-10-05：srs.md/design.md 已完成；核心设计决策（512 线程 TM4×TN8 / --wlb
  默认 2 / detail::swsk_reduce 共享 / BK 固化 8）见 design.md §4.1。
