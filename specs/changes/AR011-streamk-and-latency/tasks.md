# [AR011] 任务清单

| 字段 | 内容 |
|------|------|
| AR 编号 | AR011 |
| 对应 srs.md | ./srs.md |
| 对应 design.md | ./design.md |
| 创建日期 | 2026-10-07 |

> **实验与图表纪律（AGENTS.md §6，每个 AR 强制）**：每个任务在完成时就明确自己的
> 实验数据与图表交付物（下表"实验/图表交付物"列），全部可由 `make_figures.py`
> 复现生成；AR 收尾统一回填 report.md/README/paper/interview，禁止"做完代码、
> 图表最后凑数"。

## 任务列表

| ID | 任务简述 | 依赖 | 状态 | 实验/图表交付物 |
|----|---------|------|------|-----------------|
| T001 | Red：注册 K_STREAMK=16（fn=nullptr 红观察）+ `--persist/--hit/--waves` 旋钮注册与 CLI 校验 + 票据 workspace RAII 骨架声明 + **P0 权限申请发起**（admin/TCC，条件任务 T007 的外部前提，申请动作无代码依赖） | - | passing | 2026-10-07 完成（auto 模式）：**注册表 16→17**（K_STREAMK=16, "streamk"，fns 槽位 nullptr Red 态）；`--waves(0..8)/--persist(0|1)/--hit(0.5..1.0)` 三旋钮全链落地（CliOptions/usage/解析/校验/main 绑定 + [note] 语义与 --sk/--dbuf 先例同型）；`src/sgemm_streamk.cu` 骨架（旋钮定义 + 头部注释契约全量 + T002 状态声明，注册槽保持 nullptr）；CMakeLists 双目标入列。**验证**：①`--list-kernels` 17 项 ✓（git=92bd546 头部）；②CLI 越界三连 `--waves 9/--persist 2/--hit 1.1` → CLI_ERROR **exit=1** ✓，`--hit 0.4` → exit=1 ✓，advisory [note]（waves/persist/hit 各场景）→ exit=0 ✓（cmd /v:on 延迟展开验证）；③**Red 观察：sgemm_test 146/155 PASS**——基线 146 全绿（零回归）+ streamk 9 项 `[ERROR] kernel 'streamk' not registered (fn=nullptr)` FAIL ✓（tests/test_correctness.cu:88 守卫，Red 状态可观察非崩溃，AR010 T001 模式复现）；**P0 申请记录落 environment.md §7**（待用户 admin 操作；条件语义与授权后回填路径已声明）。坑：PS 直跑 exe DLL 缺失（需 tools\env.cmd 包裹）、增量构建首跑 cl.exe 不在 PATH（同因）、cmd `%ERRORLEVEL%` 需 `/v:on !！` 延迟展开 |
| T002 | Green：Stream-K 主体（sgemm_streamk.cu，**自包含**——deep 计算主体逐拷贝（design §4.1 问题 3 裁定，ptxas 稳定性 + 自包含军规），非函数级共享）——tile-major 连续切分块映射（4.2.1 数学）+ cover==1 快路径 + per-tile 票据归并（atomic + `__threadfence` + 固定 z 序寄存器代入，4.2.2）+ ptxas 硬门（0 spill 逐实例审计，F1 寄存器代入→F2 全 P 归并回退阶梯）+ 146/146 + 分级数值门（bitwise 锚链/rel≤1e-4 + 确定性双跑）+ memcheck/racecheck 专项 | T001 | pending | 待办：ptxas 逐实例记录（build_t002.log）；146/146 + 数值专项结果；racecheck（票据/fence 路径）0 hazards 证据；冒烟（1024³/2048³ vs dsk direct 同会话初判 + 波填充率 = 100% 核验）；fig：stream-K 结构图（块映射/票据/归并三联） |
| T003 | FR2 L2 persistence 钉 C：`--persist {0,1}` + `--hit {0.6,0.8,1.0}` 旋钮落 common.h + dsk 集成（cudaDeviceSetLimit + accessPolicyWindow + 计时后 reset）+ main/reduce 分段净效应配对（sgemm_deep_sk.cu:99-136 verbose 基建复用） | T001 | pending | 待办：同会话 on/off 配对 × hitRatio 三档 × {1024³,1000×1016}（SGEMM_CSV 分流）；归约段净节省 μs + main 段退化 % 双指标表；G1@1024³ 复判数据（预期翻门）；fig：persistence 消融（分段条形 + 净效应瀑布） |
| T004 | FR3 延迟覆盖消融：FR3a B 片段 kk 预取（`template <int BPF>` 双实例，+8 regs 刀口 251/255，**ptxas 逐实例审计硬门**）+ FR3b kk 轮转错相（`template <int PHASE>`，零寄存器）；FR3a bitwise 门（链序不变）/ FR3b rel≤1e-4 双参考 + 确定性双跑（数值口径分级声明同步详设） | T001 | pending | 待办：v4 配对协议 @1024³/2048³ × {on,off}² 双实验（4 组）+ ptxas 寄存器/实测性能双记录；胜者判定（+≥2% 进 auto v4 候选池，0/- 归档负结果）；fig：延迟覆盖消融对比（含 LSU/FFMA 利用率标注，P0 解锁时补 ncu 证据列） |
| T005 | Stream-K 性能 sweep：waves W∈{1..6}（tile-major 切分已由 design §4.1 裁定，仅扫 W）× 全六尺寸 + 1024³/2048³ 主战场深扫（对照 dsk direct/deep/cuBLAS 同会话）+ auto 公式校准 | T002,T004 | pending | 待办：streamk_ar011.csv（SGEMM_CSV 分流）；1024³ 总时间分解（main+归并融合 vs 297+41μs 基线）+ 2048³ 波量化消除验证（128 blocks/2.67 波 → 恒 48 整倍数）；票据赢家尾块不均衡实测披露；W 最优表（auto v4 输入）；fig：Stream-K vs dsk/deep 全尺寸阶梯 + 1024³ 时间预算瀑布 |
| T006 | auto v4 + 几何 sk 公式：dispatch 表吸收胜者（streamk/persist/FR3 胜者，按 T003/T004/T005 实测裁决）+ 矩形尺寸 sk = clamp(ceil(48·W/blocks), 2, 16) 泛化 + 补充尺寸（256×4096、4096×256、1024×2048）全量首测 + A-B-A-B 保真验证 | T005 | pending | 待办：auto_ar011.csv（A-B-A-B 双 rep + 47°C 冷却门 + GPU 时钟 spin）；auto vs winner 保真表（10/12 ≤0.6pp 口径 + 补充尺寸首测列）；fig：v4 dispatch 全景图（含补充尺寸入表标注） |
| T007 | FR1 条件归因收口（P0）：admin/TCC 可得 → E-A 主核 stall 四分类分解（同相停顿/barrier 歪斜/步首依赖/尾 tile，至少排除或确认两个）+ E-B cuBLAS@1024³ launch config 反测 + E-C 锁频五门复判；不可得 → 环境声明落盘 + E-A 替代协议（分段计时 + T004 消融间接归因）+ %peak 不变量口径延续。**不阻塞 T002-T006，T008 门判前必须收口** | T001 | pending | 待办：profile/streamk_ar011/ ncu 原始数据（.ncu-rep + csv）或替代协议数据；environment.md 更新（时钟策略/权限状态）；E-B cuBLAS 反测表（blocks/waves/占用推断）；fig（若解锁）：stall 分解饼图 + 锁频 vs WDDM 口径对照 |
| T008 | 门 v5 终判：run_paired v5（v4 协议继承：cublas 首末锚定 + 47°C 冷却 + GPU 时钟 spin + tools\env.cmd 包裹 + 追加前删旧文件）× deep/dsk/streamk/auto v4/胜者变体/swsk/swpipe/cublas 全 kernel × 六尺寸 + 补充尺寸 → G1@1024³ 翻门 / G6@2048³ 新门 / G1@512³、G2、G3 回归 / G4''' 增量 / G5''' 保真 / 正确性与资源硬门全量复判 | T006,T007 | pending | 待办：paired_ar011.csv + compare_ar011_paired.md（终判表，AR010 体例）；六门判定逐门证据行；锁频口径并行列（若 E-C 完成）；fig：门判终态总览（AR009→AR010→AR011 三代对比） |
| T009 | 工程卫生 + 交付收尾：sgemm_deep_sk.cu:15-17,109 过时 `__stwt`/"45→~33μs" 注释修正（注释契约违规清偿）+ ws/wsk 4×8B spill 根治或显式豁免记录 + 详设同步（Kernel 10 条目/旋钮表/CLI/AR 表/术语）+ AGENTS 旋钮段 + report.md §13 + bottleneck 闭环 #6 + paper/interview 附录回填 + st_report.md 集中回报 + 2026107.md 归档入 AR011 目录 + 全量自检（重建 N/N 无警告 + 146/146 + 全图复现 ≤72KB + BOM 完整）+ gitee 推送 | T008 | pending | 待办：五件套文档回填记录；重建 35+/35+ 与全图复现日志；st_report.md；推送 commit hash |

## 状态说明

- `pending`：未开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，自测通过
- `failed`：开发失败，待修复

## 门控记录

> 每任务每会话可自由追加，记录关键决策。

## 阶段门关键记录

- 2026-10-07：用户指示 auto 模式（沿用 AR007-AR010 先例）对 results/2026107.md
 （AR010 收官优化分析报告）做深度分析与优化 → 开启 AR011。req/design 阶段
 门控跳过，ST 验收集中回报；需求源 = 2026107 报告（P0-P3 路线图 + §5 AR011
 框架草案 FR1-FR5 正式化扩展为 FR1-FR6，新增工程卫生 FR6 与 G6@2048³ 新门）。
- 2026-10-07：srs.md 关键口径预声明——①P0 全部设计为条件任务（admin/TCC 结果
 不阻塞主线，环境缺失不得虚构数据）；②数值口径分级（FR3a/链序不变路径 bitwise
 锚链延续；FR3b/Stream-K 若链序对齐失败则 rel≤1e-4 + 确定性双跑）；③AR008
 last-block 否决案的翻案声明（全局单块串行化 ≠ per-tile 票据，srs §2 FR4）。
