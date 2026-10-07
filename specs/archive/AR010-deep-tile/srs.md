# AR010 srs — 128-acc 深分块攻坚（G1@1024³ / G2@256³ Gate-Closing）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR010 |
| 名称 | deep-tile：Kernel 9 deep/dsk（TM16×TN8 深寄存器分块）+ G2 精调 + auto v3 |
| 前置 | AR009（wide/wsk 负结果归档：占用率假说证伪，真墙 = LDS.128 带宽） |
| 日期 | 2026-10-06 |

## 1. 背景与问题（AR009 五门 v3 遗留）

| 门 | AR009 结果 | 根因诊断（实测数据支撑） |
|----|-----------|------------------------|
| G1@512³ ≥75% cuBLAS | **PASS 75.03%**（刀锋，swsk_sk3 4267） | 守成：不允许回退 |
| G1@1024³ ≥75% cuBLAS | FAIL 64.20%（swsk_sk3 5377 vs cuBLAS 8377，缺 906 GF） | **LDS.128 带宽墙**：swpipe 系 64 acc/4 LDS.128 = 16 acc-per-LDS → 54.5% peak 封顶；cuBLAS 84% peak 证明机器上限远高于此（其深寄存器分块 128+ acc/thread 设计点） |
| G2@256³ ≥1618.2 GF | FAIL 95.4%（swsk_sk6 1544.3；超同会话 cuBLAS 1.240×） | ① sk6 切片不均（32 k-tiles/6 = 6,6,6,6,6,2 尾片失衡）；② split-K 归约 + P 写出开销占比高（256³ 全程仅 21.7μs）；③ smem1d 历史最优 1618.2（AR007 会话）未纳入 auto 候选 |
| G3@4096³ ≥7.0 TF | PASS 102.6%（swpipe 7185.3） | 守成；deep 若胜出则增收 |

**核心科学假说（本 AR 检验对象）**：AR009 三重证据建立的 **acc-per-LDS.128 → %peak
经验律**（wide 8 acc/LDS → 31% peak；swpipe 16 → 46-54%；cuBLAS ~21+ → 84%）预测：
**TM16×TN8（128 acc/thread，6 LDS.128/128 FFMA = 21.3 acc-per-LDS）可把 1024³
抬升至 60-70% peak = 6.0-7.0 TF ≥ G1 门 6283 GF**。同时 128 条独立 FMA 链以
**ILP 替代 TLP**（占用率 50%→25%），直接检验 AR009 结论"ILP 深度而非占用率
决定 sm_75 FP32 吞吐"的可推广性——与 cuBLAS 的低占用高 ILP 设计点对齐。

## 2. 解决方案（FR）

### FR1 Kernel 9 deep：128-acc 深寄存器分块（本 AR 核心）
- tile **BM=256 × BN=128 × BK=8**，block 256 线程（16×16），每线程
  **TM=16 × TN=8 = 128 累加器**（第四代布局继承：A 转置 + PAD4 / B XOR
  swizzle / 零守卫 / 寄存器预取）
- **几何选择依据（TM>TN 的必然性）**：TN=16 会使 B 片段产生 stride-4 quad
  访问（16 连续列/线程），任意 16B XOR swizzle 均无法消解（mod-8 双残留类）
  → 8-way bank conflict；TM=16 时 B 片段退化为 **stride-2 quad（与 swpipe
  逐位同型，AR006 ncu 实测 0 冲突模式直接继承）**，A 片段变 warp 级广播
  （0 冲突），且 A loader STS 列连续（0 冲突）
- LDS:FFMA = 6 LDS.128 : 128 FFMA = **1:21.3**（swpipe 1:16、wide 1:8→恶化、
  deep 减 25% LDS 压力）；寄存器预算 ~184（128 acc + 12 预取 + 24 片段 +
  ~20 寻址），`__launch_bounds__(256,1)` 上限 255，**0 spill 硬门**
- 占用率 1 block/SM = 8 warp = 25%——**科学对照点**：与 AR009（占用率证伪）
  构成完整 ILP/TLP 二维消融矩阵
- **--dbuf {0,1} 消融**（默认 0，数据裁定后固化）：双缓冲 smem 实例
  （As[2]+Bs[2] = 24832B ≤ 48KB 静态上限），单 tile 同步 2→1——25% 占用下
  warp 级延迟覆盖薄弱，双缓冲为对冲路径

### FR2 dsk：deep 的 split-K 变体
- 主体经 `detail::deep_tile_grid` 参数化复用（z 切片 + Out_base，镜像
  swsk/wsk 既定模式）；归约复用 `detail::swsk_reduce`
- 波几何：1024³ grid 4×8=32 blocks × sk3 = 96 = **2 精确波**（1 block/SM ×
  48 slots）；512³ 8 blocks × sk6 = 48 精确波；2048³ 128 blocks（2.67 波）；
  4096³ 512 blocks（饱和）
- `--sk` 旋钮对 swsk/wsk/dsk 三生效

### FR3 归约 ILP2 升级（G2 攻坚其一）
- `detail::swsk_reduce` 内部双实现：v1（1 float4/线程，现状）vs v2
  （2 float4/线程，2× DRAM 延迟覆盖）；`--rv2 {0,1}` 全局旋钮 A/B，
  数据裁定后默认固化；**逐位等价保证**：每元素 z 升序累加链不变
- swsk/wsk/dsk 三者同时受益（单一数值路径军规不变）

### FR4 G2@256³ 精调（其二三）
- swsk 细粒度 sk ∈ {4,5,6,7,8}（sk5 = 7,7,7,7,4 切片均衡假说；sk6 现状
  6,6,6,6,6,2 尾片失衡）；dsk sk 探针；smem1d 同会话中继（历史 1618.2）
- 优胜者回填 auto v3 dispatch（数据真实性军规：表值必须来自本 AR 实测行）

### FR5 auto v3 + 五门 v4 复判 + 文档
- dispatch 表按 AR010 实测胜出者回填（deep/dsk 预期接管 512³-2048³ 计算区）
- 五门 v4：G1@512³（守成 75.03% 刀锋不许回退）/ G1@1024³（≥6283 GF 主攻）/
  G2@256³（≥1618.2 主攻）/ G3@4096³（守成 + deep 增收）/ G4''（auto v3 对
  v2 ≥4/6 尺寸 ≥+2% 且无 <-2% 回退）/ G5''（dispatch 保真 ≤2pp + 轮间极差
  median <2pp）
- **CCF-A 级实验报告**（results/paper_sgemm_turing.md）：研究问题 RQ1-4 /
  假说-实验-证据闭环 / 方法学（稳态测量纪律）/ threats to validity；
  **面试叙事文档**（results/interview_narrative.md）：全工程价值面向求职的
  STAR 化提炼；详设/AGENTS/report/README/bottleneck 闭环 #5 同步

## 3. 验收门（v4）

| 门 | 判定 | 通过条件 |
|----|------|---------|
| G1 | 512³、1024³ 最优自研（deep/dsk/auto v3） | 各 ≥ 同会话 cuBLAS 75%（1024³ ≥ 6283 GF 为主攻目标） |
| G2 | 256³ 最优自研 | ≥ 1618.2 GF（绝对门；同会话 cuBLAS 对照并报） |
| G3 | 4096³ 最优自研 | ≥ 7.0 TF（守成；deep 胜出则刷新） |
| G4'' | 6 尺寸 auto v3 vs v2 | ≥4/6 尺寸 delta ≥ +2% 且无尺寸 < -2% |
| G5'' | 方法学 | dispatch 保真 ≤2pp + paired 轮间极差 median <2pp |

## 4. 约束（不变军规）
- 严格 FP32 / 冻结签名 / CUDA events / median 判据 / 稳态测量纪律（1620MHz
  持续态 + 热身 spin + 冷却门 47C）/ 数据真实性（负结果归档、禁凭空填表）/
  racecheck-memcheck 硬门 / ptxas 审计 0 spill / CSV 经 SGEMM_CSV 分流
- deep 数值序锚点：主路径与 swpipe **逐位一致**（同 k 升序 FMA 链）；
  dsk 与 swsk **逐位一致**（同 k-tile 切分 + 单一归约路径）——均为 UT 专项
