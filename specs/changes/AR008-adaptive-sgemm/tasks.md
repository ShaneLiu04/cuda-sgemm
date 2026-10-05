# [AR008] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR008 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-05 |

> **实验与图表纪律（AGENTS.md §6，每任务强制）**：每个任务完成时即产出自己的
> 实验数据与图表交付物（下表"实验/图表交付物"列），全部经 `make_figures.py`
> 可复现生成；AR 收尾统一回填 report.md/README。禁止"做完代码、图表最后凑数"。

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 实验/图表交付物 |
|----|---------|------|------|-----------------|
| T001 | thermal-paired 协议：run_paired.ps1（交替配对+冷却门控）+ compare.py --paired（对内 delta） | - | passing | 2026-10-05 完成：自证实验（swpipe vs vec4/cublas @512³+1024³ ×3 轮）——512³ vec4 delta 极差 0.13pp、cublas 0.44pp；跨会话比值一致（2.42× vs AR007 2.44×）而绝对值漂 ~12% → 共模相消验证成立；1024³ vec4 一轮 -3.29% 离群（极差 2.31pp，median 判据稳健）；产出 paired_ar008_selfcheck.csv + compare_ar008_selfcheck.md |
| T002 | Red：注册表扩容 swsk + ws（fn=nullptr）→ 101 例中 18 例 FAIL 可观察 | - | passing | 2026-10-05 完成：套件 83/101 PASS，swsk 9 例 + ws 9 例 FAIL（"[ERROR] kernel not registered (fn=nullptr)" 逐例打印）——Red 可观察 |
| T003 | Green swsk：split-K 主 kernel（swpipe 参数化复用）+ 确定性归约 + workspace RAII + CLI --sk + 101/101 | T002 | passing | 2026-10-05 完成：①正确性 92/101（仅 ws 9 例待 T007；swsk 9/9 PASS，rel 分布 2.2e-07~2.7e-06，与单 kernel 版同量级——split-K 累加序变化的预期幅度）②ptxas 审计：swpipe tile 单实例 **128 regs / 8320B smem / 0 spill**（= AR007 LB=2 基线逐位同级）+ reduce 61 regs / 0 spill，归档 build.log ③确定性证据：1024³ --check 两次 max_abs=3.051758e-05 / max_rel=5.559054e-07 全位一致（两独立组）④性能证据（同会话 45°C 冷态对照）：512³ swpipe 2373.7 vs swsk(sk4) **3542.5（+49.2%）**、1024³ 4619.3 vs 4745.5（+2.7%）、4096³ 6481.8 vs 6290.6（-2.9%）——wave 饥饿假说首次实测证实，大尺寸负收益为 auto dispatch 几何分界提供依据。**LB 旋钮收窄决策**（实测驱动）：参数化后无约束实例自然分配 130 regs → 1 block/SM，1024³ 实测 4314 vs 4617 GF（-7.0%）；AR007 基线 127 regs 本属 2-block 级 → tile 主体固化 `__launch_bounds__(256,2)`=128 regs 封顶，--lb 对 swpipe/swsk 移除（仅剩 tile2d，ws 用自有旋钮），详设 §4.6 + AGENTS §3 已同步 |
| T004 | swsk 消融：sk ∈ {1,2,4,8,12,16} × {256³,512³,1024³,1000×1016×1024,2048³} | T003 | passing | 2026-10-05 完成（bench/run_sk_ablation.ps1：双向遍历 2-pass median 消线性热漂移 + 冷却门控 + 行名标记 swsk_skN）：**fig9_splitk_sweep (a)(b)** + ablation_ar008.csv（70 行）。结论（2-pass median）：256³ 最优 **sk12=1511 GF**（4 blocks→48=恰满单波；sk16=64 blocks 尾波反噬→1217）、512³ 最优 **sk4=3542**（16→64 blocks）、1024³ 最优 **sk4≈5215**（sk2/sk4 在时钟浮动 1665↔1950MHz 下混叠，median 判据）、1000×1016×1024 最优 sk4≈5050、2048³ 最优 **sk1=7143**（256 blocks=5.3 波已饱和，sk↑ 单调递减，P 写出纯反噬）——wave 饥饿区/饱和区分界实证，auto dispatch 几何依据 |
| T005 | 256³ 守成专项：swsk@256³ vs smem1d(bk32)=1618.2；无净收益如实归档 | T004 | passing | 2026-10-05 完成（bench/run_size256.ps1 → size256_ar008.csv 同会话全 kernel 矩阵）：**结论反转——swsk(sk12)=1508.7 今会话 256³ 全场最优**，超 smem1d(1398.1) +7.9%、超 cuBLAS(1184.8) +27.3%；**fig9_splitk_sweep (c)**（10 kernel 柱状 + blocks/波数标注：128-tile 系仅 4 blocks=0.083 波→swpipe 560，smem1d 64 blocks=1.33 波→1398，swsk sk12 48 blocks=1.0 波→1509）。绝对值仍低于 AR007 守成基准 1618.2（跨会话漂移 -14%，今会话 cublas 亦 -7.5%），G2 门最终裁定以 T009 paired 会话为准；dispatch 表 256³-class 改选 swsk sk12 |
| T006 | --kernel auto：dispatch 表（T004 实测回填）+ CLI + 选中=实测最优验证 | T004 | passing | 2026-10-05 完成：sgemm_auto.cu（blocks=ceil(M/128)·ceil(N/128) 带判：≤4→swsk12 / ≤64→swsk4 / 其余→swpipe；K 钳制 sk≤ceil(K/8)；g_swsk_slices save/restore 防旋钮污染）；注册表 K_AUTO=11（KERNEL_COUNT=12，套件 110 例 auto 9/9 PASS → 101/110 仅 ws 残留）；CLI --kernel auto 接线。验证：①verbose dispatch 日志 6/6 尺寸选核正确 ②**run_auto_paired.ps1 背靠背配对**（winner↔auto ×2 轮，同钟态）：对内 delta 全部 \|Δ\|≤0.9%（256³ -0.3% / 512³ ~0 / 1024³ -0.1% / 1000×1016 ~0 / 2048³ -0.4% / 4096³ +0.4%，容差 ±2%）——dispatch 开销不可测。**fig10_dispatch_map**：(a) (M,N) 空间三带着色地图 + 6 门尺寸标注 (b) 配对验证柱状图（auto vs winner + delta 标签）。数据 auto_ar008.csv。附注：4096³ 今会话配对 swpipe=6542-6600 GF 与 AR007 冷态 6584.8 一致；单跑 auto 时钟爬坡（1620→1950）致 median 偏低（2048³ 6079 vs 配对 7160）已由配对协议消除——WDDM 动态时钟证据链再添一例 |
| T007 | Green ws（Kernel 7）：producer/consumer + named barriers + 3 级环 + 回退 + 101/101 + racecheck | T002 | passing | 2026-10-05 完成：sgemm_ws.cu（320 线程=2P+8C warp；模板 \<LB,STAGES\> 4 实例；full[s]/empty[s] named barriers id 1..6 count=320——64 arrive+256 sync 补齐释放；producer LDG→reg→STS 灌环（A 转置+B swizzle 布局第三次继承），consumer 纯 LDS.128+64FFMA/kstep + 独占回写；首轮 t<STAGES 免等 empty，短 K 无死锁）。**①正确性 110/110 ALL PASS**（ws 9/9：1024³ rel=1.179e-06 与 swpipe 逐位同值=同累加序验证；回退 3 例走 tile2d）；**②racecheck 0 hazards×3 配置**（环回绕 128×128×64 默认/STAGES=2/LB=2@512³）；③memcheck 0 errors（512³/4096³ 主路径+130×257×66 回退）；④短 K 专项（num_tiles=1/2/3）无死锁全 PASS；⑤ptxas 审计（build.log）：ws\<1,3\> 默认 **128 regs/0 spill/24960B**、ws\<1,2\> 129/0/16640B、**ws\<2,3\> 96 regs/0 spill/24960B（2 block/SM 达标：61440 regs+49920B smem）**、ws\<2,2\> 96 regs+**8B spill（记录取舍，T008 裁定）**；CLI `--stages {2,3}` 新增（common.h 校验+main 注入+AGENTS §3+详设 §4.6/CLI 表同步）。**fig11_ws_structure**：(a) 协议时空 Gantt（屏障事件+run-ahead 括注，结构常量模拟） (b) ptxas 资源包络表。性能冒烟（早期信号，T008 正式裁定）：512³ ws(LB=2) **2492.9 vs swpipe 2365.3（+5.4%）**、1024³ LB=2 3912.6 vs 4617.7（-15.3%）、4096³ LB=1 5590.6 vs 6669.2（-16.2%）——issue-slot 假说大尺寸方向初步阴性，正合设计预期"负结果如实归档"路径 |
| T008 | ws 消融：warp 配比（2+8/1+8）、环深度（2/3）、launch_bounds（1/2）+ 冷态配对 vs swpipe | T007 | passing | 2026-10-05 完成：kernel 增设 PW 模板轴（`--wp {1,2}`，8 实例全编译；ptxas 审计入 build.log：LB=1 系 125-130 regs/0 spill，LB=2 系 96 regs，STAGES=2×LB=2 恒 8B spill 其余 0 spill；110/110 回归绿）。bench/run_ws_ablation.ps1（8 配置×4 尺寸×升降序 2-pass + 冷却门控 + swpipe/cublas 参考行 + 行名标记 ws_pw{P}_st{S}_lb{L}）→ ablation_ar008.csv。**结论（2-pass median）**：①**PW=2 全面胜 PW=1**（512³ 2517 vs 2310，4096³ 5500 vs 5104——单 producer warp 搬运吞吐喂不动 8 consumer warp）；②**LB=2 占用率假说被否**（62.5% 占用配置 3/4 尺寸劣于 31.3%，唯 1024³ pw2_st2_lb2=4753 为 ws 峰值仍 -7.9% vs swpipe）；③STAGES 2/3 差异小且混合（环深 3 的延迟覆盖非瓶颈）；④**issue-slot 假说（大尺寸）否定**：ws 全 8 配置在 1024³/2048³/4096³ 均不敌 swpipe（-7.9%/-20.4%/-15.0%），G3 门（4096³≥7.0TF）FAIL——"pre-Ampere 软件流水边界"负结果归档（设计 §6.5 预案兑现）；512³ 最佳 ws +6.1% vs swpipe 但远逊 swsk（+49%），auto dispatch 表无需改动（大尺寸仍选 swpipe）。**fig12_ws_ablation**（8 配置×4 尺寸 GF 矩阵 + swpipe/cublas 参考线 + ptxas 资源包络表）+ **fig13_isslot_hypothesis**（假说判定 delta 图 + 绝对阶梯图，正/负结果均上图） |
| T009 | AR008 全矩阵（paired 协议）：auto/swsk/ws/swpipe/cublas × 6 尺寸 → 四门 v2 判定 | T001,T006,T008 | passing | 2026-10-05 完成：run_paired.ps1（baseline swpipe；挑战者 swsk/ws/auto/cublas；6 尺寸 × 3 轮背靠背 + 冷却门控）→ paired_ar008.csv + compare_ar008_paired.md。**污染组补救**：1000×1016 组首测冷却超时（48C）被标 thermal-contaminated → 污染行备份至 paired_ar008_thermal_bak.csv 后剔除、冷却重测干净组（swsk +2.34% HIT 维持）。**四门 v2 裁定（2 PASS / 3 FAIL）**：G1 **FAIL**（512³ swsk 3542.5/5698.8=62.2%、1024³ 4988.2/8581.1=58.1% < 75%）；G2 **FAIL**（256³ auto 1517.5 < 1618.2 跨会话守成基准；但同会话 > cuBLAS 1260.3 达 +20.5%——绝对值 FAIL 归因跨会话漂移 -14% 已归档）；G3 **FAIL**（4096³ ws 5377.7 GF / delta -16.00% 极差 0.33pp——issue-slot 假说否定，负结果如实归档）；G4 **PASS**（4/6 命中：256³ auto +170.3%、512³ swsk +49.4%、1024³ swsk +2.8%、1000×1016 swsk +2.3%；miss：2048³ +0.0%、4096³ -0.2%——大尺寸本就选 swpipe，delta~0 是 auto 正确性的体现）；G5 **PASS**（全部 24 组 delta 轮间极差 median 0.53pp；最差 20pp 出现在 1024³/1000×1016 的 swpipe 基线钟态双峰 1620↔1935MHz，median 判据稳健，已归因）。**fig14_paired_delta**（三挑战者×6 尺寸 delta 矩阵 min-max 误差线 + 四门判定面板）+ **fig15_ladder_v2**（同会话 5 kernel×6 尺寸阶梯 + 自研最优虚线=auto 贴线可视化） |
| T010 | 文档同步：详设 §4.1/§4.3/§4.6/§5 Kernel 7 + AGENTS --sk + report/README/figures 总刷新 | T009 | passing | 2026-10-05 完成：①详设——§4.1 swsk/ws/auto 声明齐、§4.6 新增 --stages/--wp 行 + CLI 表更新 + --lb 行（T003 已同步）、§5 Kernel 7 节补三轴消融与 T008/T09 负结果裁定；②AGENTS §3 旋钮表新增 --stages/--wp；③report.md 追加 §10（AR008 全章：动机/交付表/四门 v2 判定/§10.1 ws 负结果专节/§10.2 auto dispatch 表）+ **附录 B 图表索引**（8 结论→图→CSV 三链可溯）；④bottleneck_analysis.md 追加闭环 #1 wave 饥饿→swsk（证实）、#2 issue-slot→ws（否定，负结果归档）、#3 热漂移→paired 协议（证实，G5 0.53pp）；⑤README——badges 110/110、kernel 表新增 swsk/ws/auto 行（负结果行诚实标注 ⚠）、架构演进 K6'/K7/auto 三行、快速开始更新（--sk/--stages/--wp 旋钮 + run_paired + compare --paired 命令）、0 spill 注记（ws 消融实例 8B spill 取舍说明）。**图表核对单**：AR008 新图 6 张（fig9-15 共 7 文件，fig9 为 AR008 扩展三面板）全部由 make_figures.py 从 CSV 可复现、全部入 report 附录 B + README 引用（fig15 阶梯/fig14 判定经 §10 与 compare 链接）；旧图 fig1-8 未动（数据源 performance.csv 主表未增 AR008 行——swsk/ws/auto 性能证据在分流 CSV，fig15_ladder_v2 为同会话替代，报告已注明） |
| T011 | ST 验收：101 例回归 + 四门 v2 逐门判定 + ws 假说检验回报 + 图表索引集中回报 | T010 | passing | 2026-10-05 完成：**st_report.md**（specs/changes/AR008-adaptive-sgemm/）。①clean 重建后 **110/110 ALL PASS**（12 kernel × 9 尺寸 + cuBLAS 交叉）；②接口测试：--list-kernels 12 项、--sk 0/--stages 5/--wp 3 → CLI_ERROR、旋钮跨 kernel note 提示正常执行；③四门 v2：G1 FAIL（62.2%/58.1%）、G2 FAIL（1517.5，同会话 +20.5% vs cuBLAS）、G3 FAIL（ws 5377.7/-16.00% 假说否定）、G4 PASS（4/6）、G5 PASS（0.53pp）；④ws 假说检验负结果专节回报；⑤图表索引节（8 图三链可溯）；⑥交付物清单 + 遗留承接（git 不提交沿用 AR007 裁定、ncu 权限 backlog）。**ST 结论：验收通过** |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

- 2026-10-05（T003）：swsk Green 完成。swpipe tile 主体参数化（z 切片 + Out_base 偏移，
  单波路径与原语义逐位等价）；swpipe_sk.cu（grow-only RAII workspace + 固定 z 序归约
  kernel + sk=1 旁路 + 对齐回退）；CLI --sk（1..16 越界 CLI_ERROR）全链接线。
  套件 92/101（仅 ws 待 T007）。**LB 收窄**：--lb 移出 swpipe/swsk（实测依据见 T003 行），
  详设 §4.6 / AGENTS §3 已同步；build.log 已归档新 ptxas 审计。性能证据：512³ +49.2%
  （3542.5 GF，wave 饥饿论证实）、4096³ -2.9%（auto 分界依据）。遗留：fig_splitk_sweep
  数据在 T004 扫描中产出（本任务性能点为单点对照，不单独成图）。
- 2026-10-05（T006/T007）：auto 选核验证收尾（背靠背配对 |Δ|≤0.9%，fig10）；
  ws Green 完成（110/110 ALL PASS，racecheck 0 hazards×3 配置，memcheck 全清，
  ptxas 4 实例审计入 build.log，fig11 结构时空图）。CLI 新增 --stages {2,3}
  （AGENTS §3/详设 §4.6/CLI 表三处同步）。ws 性能早期信号：512³(LB=2) +5.4%、
  大尺寸 -15~-21%——假说检验留待 T008 消融与 T009 冷态配对正式裁定。
- 2026-10-05（T008-T011，收尾会话）：ws 增设 PW 模板轴（--wp，8 实例，110/110 回归绿）；
  三轴消融（fig12）裁定 PW=2 胜、LB=2 占用假说否、ws 大尺寸全败 swpipe（fig13 负结果
  归档）；T009 全矩阵配对（144 组 + 污染组剔除重测，fig14/15）四门 v2 = 2 PASS/3 FAIL
  （G1/G2/G3 FAIL 如实归档，G4/G5 PASS）；T010 文档总刷新（report §10+附录 B、
  bottleneck 闭环 ×3、README、详设、AGENTS）；T011 ST 验收通过（st_report.md）。
  AR008 全部 11 任务 passing。

## 阶段门控记录

- 2026-10-05：用户指令 auto 模式（"使用auto的harnessX这个skill进行深度的优化"）——req/design 阶段门控跳过，
  沿用 AR007 惯例；ST 验收集中回报补偿（含 ws issue-slot 假说检验结果与四门 v2 裁定）。
- 2026-10-05：用户追加指令（设计阶段）：**实验与图表纪律写进每一个任务**——每任务完成即交付
  实验/图表（上表新增"实验/图表交付物"列），并同步立入 AGENTS.md §6（长期军规）。
