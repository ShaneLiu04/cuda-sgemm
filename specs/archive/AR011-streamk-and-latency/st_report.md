# AR011 ST 验收报告（streamk-and-latency）— 2026-10-07

> 集中回报（auto 模式：开发各阶段不询问，ST 统一汇报）。全部数字实测可复现，
> 原始 CSV 行级 gpu_state 逐行可核。

## 1. 验收范围

srs FR1（Kernel 10 streamk 统一调度）+ FR2（L2 persistence，T003 裁定 N/A 收口）+
FR3（deep 延迟覆盖消融）+ FR4（auto v4）+ FR5（五门 v5 复判）+ FR6（补充尺寸首测）。
任务 T001-T009 全数 passing（specs/changes/AR011-streamk-and-latency/tasks.md）。

## 2. ST 正确性门（全绿）

| 检查 | 结果 |
|------|------|
| 回归套件（17 kernel × 尺寸/边界/回退） | **163/163 ALL PASS**（T009 终检全量重建后复跑确认） |
| AR011 bitwise 锚（memcmp exact） | **7/7**：streamk 旁路==deep / W=1 cover==deep / W=2==dsk(sk2,direct) / W=2 tile0==dsk(sk3,direct) / streamk auto W=5 双跑确定性 / deep BPF 链同序 / deep PHASE 双跑确定性 + PHASE rel≤1e-4（5.096e-07） |
| 确定性 | 票据累加单线程 atomicAdd 变体（跨 launch 流序自清洁）+ 归约固定序 |
| memcheck | streamk W=1 @1024³/2048³ **0/0 errors**（T008）；全版本历史 memcheck 干净 |
| racecheck | streamk 票据/P 访问 0 hazards（T002 sanitize 四连 0） |
| ptxas 审计（build.log 全量重建） | streamk 255/255 regs 0 spill；deep 10 模板实例全 0 spill（现役 247/243）；**ws stages=2 消融实例 4×8B spill 显式豁免记录**（sgemm_ws.cu 头注：非默认消融路径、消融可比性优先，AR011 T009 清偿） |

## 3. 五门 v5 判定（compare_ar011_paired.md，paired_ar011.csv 207 行）

| 门 | 判定 | 关键数据 |
|----|------|---------|
| G1@1024³ ≥75% cuBLAS | **MISS（刀锋）** | auto_v4（dsk 路径）6279.5 / 8383.4 = **74.90%**（逐轮 74.85-74.95；差 0.10pp ≈ 0.4μs） |
| G6@2048³ ≥85%（新门） | **MISS（刀锋）** | streamk W=1 8773.3-8784 / 末锚 10341.9 = **84.85%**（差 0.15pp ≈ 3.2μs；r1 冷锚 9278 排除——误用即伪 PASS +9.8pp） |
| G1@512³ 回归 ≥75.14% | **PASS（边缘）** | 75.01%（-0.13pp 会话噪声，落于 AR010 自身轮值域 [4270.1, 4291.9]；r1 75.20 达线） |
| G2@256³ 钟态匹配 | **PASS** | %peak 15.58（+10.3% like-for-like）；1860 投影 1781.2 ≥ 1618.2；反超 cuBLAS 122.2% |
| G3@4096³ ≥7.0TF | **PASS** | deep 9058.1（%peak 78.0，AR010 68.1 → +9.9pp） |
| G4''' auto v4 ≥4/6 +2% | **MISS** | 仅 2048³ +2.36%（1/6，尾波区带兑现）；无 < -2% 回退 |
| G5''' 保真 ≤2pp | **PASS** | canonical 17/18 ≤0.77pp（256³ r1 一次性瞬态披露，AR010 同源已知）；轮间极差 median 0.72pp；补充尺寸以 T006 逐行冷却对为准 |

G1@1024³ 责任链闭合：L2 钉 C 能力墙 N/A（T003）→ BPF/PHASE 双因子负（T004）→
streamk@1024³ = dsk 的 87%（T005）→ E-A 归因 main ~42μs + reduce ~43μs 结构性（T007）。
两个比例门 0.10-0.15pp 刀锋未翻，**按军规如实 MISS、门线不放宽**。

## 4. 核心科学结论

1. **"多波精确填充"假说否定**：W=2 整波（2048³ 96 blocks = 2 精确波）8355 GF <
   deep 2.67 波 8542 GF——Stream-K 的赢面在 **W=1 tile 聚合**（48 blocks 恒 1 波 +
   solo 快路径免票据免 P 回合）；W>1 全尺寸单调负（P 缓冲税 + 块/SM 占用损失）。
2. **尾波量化税兑现**：deep 82.50% → streamk W=1 84.85% = **+2.35pp**（理论尾波
   ~-11% 部分回收，余量 = split 结构税：P 流量 + 融合归约 1 block/SM MLP 减半）。
3. **G1@1024³ 缺口结构性**：E-A 四分类归因（同相停顿/步首依赖排除——BPF/PHASE 双负；
   尾 tile/波量化确认；barrier 歪斜未决需 ncu）；缺口 84.8μs = main ~42 + reduce ~43μs。
4. **L2 persistence 能力墙**：TU104 sm_75 无 cudaAccessPolicyWindow persistence（T003
   实证 N/A）——FR2 以负结果收口，旋钮保留复现。
5. **K=4096 时钟制度纪律 + 锚敏感性纪律（协议级）**：冷启动 1620 vs in-seq 热
   1920-1935 MHz（dsk 差 -13%）——同轮同制度才可比、保真须逐行冷却；2048³ cublas
   r1 冷锚 9278 vs 稳态末锚 10337-10342——比例门以稳态锚为分母、方向从严。

## 5. 图表交付（fig29/31-34，make_figures.py 可复现）

| 图 | 内容 | 数据 |
|----|------|------|
| fig29 | streamk 结构（工作包/票据/solo/旁路/融合归约） | 设计文档 + smoke |
| fig31 | FR3 消融双负（BPF/PHASE 假说否定） | lat_cover_ar011.csv（24 行） |
| fig32 | W sweep 三联（W=1 最优/多波否定/1024³ 差距） | streamk_ar011.csv（165 行） |
| fig33 | auto v4 dispatch + G4'''/G5''' + 补充首测 | auto_ar011.csv（24 行） |
| fig34 | 五门 v5 终判（比例门刀锋/锚敏感性/三代演进） | paired_ar011.csv（207 行） |

## 6. 事故与披露记录（军规执行）

- T002 票据 bug（AR010 遗留：单线程 atomicAdd 依赖假设错误）——Red 阶段套件 146/155 抓出，修复 160/160
- T004 数据溯源违例自纠：CSV git=92bd546（陈旧 configure 烙印）→ commit 后 reconfigure 重跑，全行携带正确 sha
- T005/T006 补充行剥离未备份事故 → .bak 备份纪律补齐重测
- T006 run-1 "4096×256 盲区" 为 auto 末位行降频伪影（gpu_state 1740/1620 留证）**已撤回**，双制度重测收口
- T008 v5 补充尺寸 in-seq warm ramp 伪影（auto@pos3 vs pick@pos2 斜坡污染 -4~-6pp）→ 协议结论：K=4096 保真须逐行冷却（environment.md §9）
- PS 5.1 教训累积：`&&` 不可用（env.cmd 包裹）、`-File` 无 BOM UTF-8 中文注释 GBK 错配破坏语法（.ps1 必须 BOM + Parser 验证）、`>` 重定向 = UTF-16LE
- report.md §12 转义污染（`\t`→TAB/`\r` 吞字 9 处）T009 发现并修复；paper/interview 扫描无同源污染

## 7. 结论

AR011 以 **Kernel 10 streamk（尾波税 +2.35pp 兑现、W=1 tile 聚合最优、多波假说否定）+
auto v4（决策表 10/10、保真全绿）+ 五门 v5（4 PASS / 3 MISS，两比例门 0.10-0.15pp 刀锋
诚实归档）+ 测量方法学（时钟制度纪律/锚敏感性纪律）+ 交付文档（report §13 / paper §5.2b/
interview 故事 8b/11b）**完成交付。十七版严格 FP32 阶梯收官：4096³ 9058 GF（%peak 78.0，
57.7× vs naive）、2048³ 8773 GF（cuBLAS 84.85%，55.9×）、256³ 反超 cuBLAS 122.2%、
163/163 + 7 bitwise 锚 + 豁免外全 0 spill + sanitizer 全清。
建议归档：specs/changes/AR011-streamk-and-latency → archive（含 4 份 AR011 CSV + 5 图 +
判定文书 + E-A 归因 + 需求源 2026107.md）。
