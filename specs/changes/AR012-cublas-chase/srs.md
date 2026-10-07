# AR012 srs — cuBLAS 追赶：E-B 零权限反汇编 + L2 块序 swizzle + 1024³ 结构混合

| 字段 | 内容 |
|------|------|
| AR 编号 | AR012 |
| AR 主题 | cublas-chase（主循环发射效率与 L2 足迹攻坚） |
| 关联 SR | 无独立 SR（承接 AR011 st_report §4 终态 + 本会话"全尺寸反超 cuBLAS"推断分析） |
| 日期 | 2026-10-07 |
| 状态 | Approved（auto 模式，用户指令"按照推断深度优化"） |

## 1. 背景与问题（AR011 终态 + 归因收敛）

AR011 十七版阶梯终态（canonical 同会话锚）：

| 尺寸 | 我方最优 | vs cuBLAS | 差距 |
|------|---------|-----------|------|
| 256³ | swsk 1551.2 | **+22.2% 反超** | — |
| 512³ | swsk 4273 | 75.01% | 边缘 PASS |
| 1024³ | dsk 6279.5 | 74.90% | miss 0.10pp = 1.05μs（G1 刀锋） |
| 2048³ | streamk_w1 8773 | 84.85% | miss 0.15pp = 3.2μs（G6 刀锋） |
| 4096³ | deep 9058 | 88.18%（%peak 78.0） | 主循环发射效率 |

**核心判断**：cuBLAS@2048³ ≈ 10337 GF ≈ 88% %peak，我方 ≈ 75% %peak。尾波/调度侧
我方已到极限（尾波税 +2.35pp 已兑现、W=1 tile 聚合已确认最优）；剩余 13pp 在**主循环
FFMA 发射效率与存储通路**。九条负结果（L2 persistence 能力墙 / BPF / PHASE / both /
多波 Stream-K / ws warp 专属化 / 占用率 / streamk@1024³ 翻门失败 / …）已把盲打空间
打光——E-A 归因四分类仍有 1 项未决，无 ncu 计数器无法定向。

**三条未试杠杆**（本会话分析锁定，构成 FR1-FR3）：

1. **E-B 静态反汇编（零权限）**：cuBLAS 88% %peak 的配方就在 `libcublas64_12.dll`
   的 cubin 里。当年 E-B 被误捆进 P0 条件任务（AR011 T007）从未执行——反汇编本身
   不需要任何 GPU 权限。
2. **L2 块序 swizzle**：deep 为 2D 线性光栅（`bx=n` fastest 行扫）、streamk 为 1D
   线性——**17 版阶梯从未做过块序重排**。2048³ A+B = 32MB vs L2 4MB，活跃波足迹
   决定命中；cuBLAS/CUTLASS 均用分组光栅化（threadblock swizzle）。
3. **1024³ 2-blocks/SM 混合**：streamk_w1 输 dsk 的根因 = 48 blocks 1 块/SM
   （MLP 减半）+ split 税；把 cover 工作切到 96 blocks（2 块/SM）保 tile 聚合
   + 恢复 MLP——缺口仅 1.05μs，是最便宜的 G1 翻门尝试。

## 2. 功能需求（FR）

### FR1 E-B cuBLAS 静态反汇编归因（P1，零权限，独立主线）

**描述**：对 cuBLAS@{1024³, 2048³, 4096³} 实际调用的 SGEMM kernel 做静态反汇编，
提取其结构性配方，归因 13pp 发射效率差距的构成。

**触发条件**：AR012 启动即执行（与 FR2/FR3 并行，无依赖）。

**期望行为**：
1. `tools/cublas_disasm.py`：nsys profile（trace=cuda，免 admin）跑 cuBLAS bench
   → 提取 kernel 名（cublas 命名通常内嵌 tile 配置）；
2. `cuobjdump -xelf` 从 `libcublas64_12.dll` 提取目标 cubin → `--dump-sass` 反汇编；
3. 特征提取：寄存器用量、smem 布局与 cp.async（LDGSTS）宽度/排布、FFMA:LDS 指令比、
   unroll 深度、blockIdx 运算模式（**rasterization/swizzle 判定**——直接验证 FR2 假设）、
   launch config（从 nsys sqlite 的 grid/block 维度）；
4. 输出 `profile/cublas_disasm/report.md` + SASS 原文 + 与我方 deep（245-247 regs、
   LDS.128、dbuf 双缓冲）的逐项对照表。

**异常处理**：nsys 未装 → 检查 CUDA 12.5 安装组件，缺则记负结果；dll 无可提取
fatbin/cubin section → 降级为 kernel 名 + nsys launch config 归因（仅维度层）；
全链路失败 → 负结果归档，不阻塞 FR2/FR3（二者不依赖 E-B 结论）。

**验收标准**：
- Given cuBLAS bench 可运行，When 反汇编流水线执行，Then report.md 含 ≥1 个尺寸的
  tile 形状 + launch config + swizzle 判定结论（或有证据的降级/负结果记录）。

### FR2 L2 块序 swizzle（P1 主攻 G6@2048³）

**描述**：deep/dsk/streamk 增加分组光栅化块序重排，改变同波活跃 block 的 A/B 面板
足迹，实测裁决 L2 命中假说。

**触发条件**：T001 旋钮注册后。

**期望行为**：
1. 旋钮 `--swz {0,1}`（0=线性现状，1=分组列序）+ `--swzg {4,8,16}`（组宽 n-tiles，
   默认 8）注册进 common.h CLI + main.cu（不落 auto，消融专用，非默认参数分流
   `SGEMM_CSV`）；
2. deep/dsk：kernel 头部 remap（`bx,by` → 分组坐标；主循环零扰动，ptxas 寄存器
   增量 ≤2 硬门）；streamk：1D `b` → 分组坐标（cover/solo 逻辑不变，仅 tile 归属
   重排）；
3. 数值口径：deep/dsk tile 独立 → **bitwise 锚延续**（重排不改归约链序）；streamk
   若票据顺序变化 → rel≤1e-4 + 确定性双跑锚；
4. 消融矩阵：`--swz {0,1} × --swzg {4,8,16} × {2048³, 4096³}` like-for-like
   （同会话同轮 v4+ 协议）+ canonical 全尺寸回归；
5. 判定门 **G7**：swz1 vs swz0 @2048³/4096³ ≥ +1.0% 才吸收进 auto v5；
   512³/1024³ 不回退 >0.5pp；否则负结果归档（L2 足迹假说在本负载证伪）。

**异常处理**：remap 引发 ptxas 扰动（寄存器/spill 变化超限）→ 修正或按军规 4
记录取舍；bitwise 失败且无法归因 → rel 锚降级 + 披露。

**验收标准**：
- Given swz=1 kernel 编译 0 spill + 测试全绿，When 2048³ like-for-like 对测，
  Then G7 判定结论（正/负）+ 数据落盘 `results/swizzle_ar012.csv`。

### FR3 1024³ 2-blocks/SM 混合（P2 主攻 G1@1024³）—— **设计修订：SK_RED separate（design §4.1 问题 1 裁定）**

> **修订注（2026-10-07，设计期）**：原设想 half-K cover（2 blocks/SM）被双重证伪
> ——96 blocks 必然 U=43 ≡ AR011 W=2 结构（sweep 已证单调负）；且 243regs×256thr×2
> =124K>64K 寄存器文件，2 块/SM 物理不可达。替代机制 **SK_RED=separate**：融合
> 票据归并剥离为 cover-only 独立归约 kernel（对症 1024³ 输因：赢家块串行归并 +
> 早到块空转），预期 ~314μs ≈ 7000 GF（83%），与融合模式同链 → bitwise 一致锚。
> 目标与验收标准不变（≥6288 GF 翻 G1）。

**描述**：Stream-K W=1 的 cover 工作按 96 blocks（2 块/SM）组织，恢复 MLP 同时
保 tile 聚合免尾波，冲击 1.05μs 刀锋。

**期望行为**：
1. 结构（design 定稿）：cover tile 的 k 区间二分由 2 blocks 分担（half-K cover），
   其一持 P 回合；solo 快路径不动；
2. 1024³ TOT=64 tiles：48 solo + 16 cover → 32 half-K blocks → U=80 blocks，
   96 SM-slot 占用 83%（vs 现 86 blocks 单块/SM）→ 2 块/SM 区段 MLP 恢复；
3. 消融 @1024³/512³（like-for-like vs dsk/streamk_w1/deep 同会话）；
4. 判定：1024³ ≥ 6288 GF（= 75.00% × cuBLAS 同会话锚）翻 G1；< dsk 6279.5 则
   负结果归档，G1 终态维持 74.90%（门线不放宽）。

**异常处理**：P 流量税超预期（512³ 12 片/tile 灾难先例）→ 设计层钳制半分片数，
负则归档；bitwise/rel 锚按 FR2 同规。

**验收标准**：
- Given hybrid kernel 0 spill + 正确性全绿，When 1024³ 消融，Then 判定结论落盘
  `results/hybrid_ar012.csv`（翻门或诚实负结果）。

### FR4 auto v5 dispatch（P2，条件吸收）

**描述**：若 FR2 G7 判定正或 FR3 翻门，auto 表吸收胜者（verbose 决策表更新 +
canonical/补充尺寸配对回归）；双负则 auto 保持 v4，仅记录。

**验收标准**：auto 决策表 10/10 符合区带设计；G5 保真 ≤2pp（v5 协议）。

### FR5 门 v6 终判 + 交付（P1）

**描述**：run_paired v6（v5 协议继承：cublas 首末锚定 + 47°C 冷却 + GPU 时钟
spin + tools\env.cmd 包裹 + 追加前删旧 + 稳态末锚分母）× 全 kernel × 六尺寸；
硬门全量复判（memcheck 新 kernel / racecheck 抽查 / 0 spill / 套件全绿 /
图 fig35+ ≤72KB）；文档回填（report §14 / bottleneck 闭环 #9 / paper / interview /
详设 Kernel 11 / AGENTS 旋钮）。

## 3. 验收门（v6，v5 协议继承）

| 门 | 判据 | AR011 基线 | 主攻 |
|----|------|-----------|------|
| G1@1024³ | ≥75.00% cuBLAS 同会话锚 | 74.90%（miss 1.05μs） | FR3 |
| G6@2048³ | ≥85.00% cuBLAS 稳态末锚 | 84.85%（miss 3.2μs） | FR2 |
| G7 swizzle | swz1 vs swz0 ≥+1.0% @2048³/4096³ | 无（新门） | FR2 |
| G2@256³ 回归 | 反超保持 ≥120% | 122.2% | — |
| G3@4096³ 回归 | %peak ≥78.0 | 78.0 | — |
| G4 增量 | auto v5 vs v4 ≥+2% 尺寸数 4/6（若 auto 变更） | 1/6 MISS | FR4 |
| G5 保真 | canonical dispatch ≤2pp | PASS | FR4 |
| 硬门 | 套件全绿（163+新增）/ bitwise 锚扩展 / 0 spill（豁免清单不变）/ memcheck·racecheck 干净 | 全 PASS | 全 |

比例门同会话锚纪律、%peak 口径（peak=6144×实测钟频 GHz）、K=4096 双制度纪律
全部继承 AR011。

## 4. 需求范围

**In Scope**：FR1-FR5；`tools/cublas_disasm.py` 新增；deep/dsk/streamk swizzle；
streamk hybrid 变体；auto v5（条件）；旋钮 --swz/--swzg；测试与锚扩展；文档六处。

**Out of Scope**：
- ncu 计数器 / E-A 复核（P0 admin/TCC 待用户，解锁后按 e_a_alternative.md §4 另行）
- E-C 锁频五门复判（P0 依赖）
- 多波 Stream-K 重开（AR011 已证死）；tensor core / TF32 / FP16（军规死线）
- 512³/小尺寸新结构（75.01% 已边缘 PASS，非刀锋）

## 5. 约束与假设

**约束**：
- 军规全套（AGENTS §1-§9）：严格 FP32、CUDA events、负结果归档、门不放宽、
  BOM、图 ≤72KB、非默认参数 SGEMM_CSV 分流、git commit 溯源。
- swizzle/hybrid 均为消融旋钮起步，默认值 = 现状（swz=0），翻门前不进 auto。
- E-B 反汇编对象为闭源二进制：只做结构归因引用，不做逐行"抄袭"声明——结论
  以我方实测消融为准。

**假设**：
- nsys/cuobjdump/nvdisasm 随 CUDA 12.5.40 安装可用（T002 实测裁定）；
- cuBLAS canonical 首末锚稳定性延续 AR011 实测（±0.1pp 内）；
- 1024³ hybrid 的 P 税可控（半分片 = 每 cover tile 2 片 << 512³ 的 12 片）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| 块序 swizzle / 分组光栅化 | blockIdx → tile 坐标的重排映射，使同波活跃 blocks 覆盖窄列带 × 宽行（A/B 面板足迹最小化） |
| 足迹（footprint） | 一个波内活跃 blocks 引用的 A/B 面板字节数总和（vs L2 4MB） |
| half-K cover | cover tile 的 k 区间二分、2 blocks 各算半段的 Stream-K 变体 |
| E-B | cuBLAS 二进制静态反汇编归因（零权限路径，AR011 T007 误捆 P0 的解捆） |
| like-for-like | 同会话同协议同轮位对照（v4+ 协议，免 ramp 伪影） |
