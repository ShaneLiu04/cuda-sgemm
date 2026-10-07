# [AR008] 需求设计说明书 — 尺寸自适应 SGEMM（split-K + warp 专属化 + 配对测量）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR008 |
| AR 主题 | adaptive-sgemm（swsk 分裂-K / K7 ws warp 专属化 / auto 选核 / thermal-paired 协议） |
| 关联 SR | 无独立 SR（用户直接发起，2026-10-05："创新架构解决 G-大尺寸/G-全线两门，提升仓库创新性与价值性，auto 模式 harnessX 深度优化"） |
| 日期 | 2026-10-05 |
| 状态 | Confirmed（用户 auto 模式指令：各阶段不逐项询问，ST 集中回报——沿用 AR007 授权惯例） |

## 1. 背景与目标

AR007 ST 判定四门 2/4：G-K6 与 G-小尺寸 PASS（256³ 绝对反超 cuBLAS 117.4%），
G-大尺寸与 G-全线 FAIL。对失败数据的根因分析揭示**两个不同性质的架构空洞**，
均可在严格 FP32 军规内攻击，且各自对应一类真实 GEMM 库的核心机制——本 AR
以"创新架构 + 教学祛魅"双目标将其补齐：

1. **wave 饥饿（G-全线真凶，中/小尺寸）**：swpipe/vec4 的 128×128 tile 在
   512³ 仅产生 16 blocks（48 SM 的 33%）、1024³ 64 blocks（1.33 波，尾波半空转）、
   256³ 仅 4 blocks（8%——这正是 smem1d 32×32 tile 在 256³ 反超的结构性原因）。
   证据：512³ swpipe 2686.9 GF ≈ 6585 × (16/48)，波数比例与 GF 比例吻合。
   **对策：split-K**（grid = tiles × K-slices，部分积落 workspace，确定性归约）——
   cuBLAS 小尺寸高能的内部机制；配 `--kernel auto` 实测驱动选核表（库 heuristic 复刻）。
2. **issue-slot 竞争（G-大尺寸真凶，4096³）**：稳态循环每 64 FFMA 混入 ~16 LDS +
   2 STS + 2 LDG + 寻址 IMAD + 2 barrier，FFMA 发射槽占比 ~55%——与实测
   6585/11980 = 55% 峰值一致（间接证据；ncu 计数器仍受权限阻塞）。
   **对策：K7 ws（warp 专属化）**：2 producer warp 专职 LDG→STS 灌 3 级 smem 环，
   10 consumer warp 纯 LDS+FFMA；sm_75 无 cp.async，以 PTX named barriers
   （`bar.sync id, count`）实现生产者/消费者同步——pre-Ampere 架构上的
   "软件版 warp specialization"，本仓库核心创新点。
3. **热漂移掩蔽（G-全线假象成分）**：AR007 矩阵会话 10 分钟热浸没致系统性 -5%
   （cuBLAS 自身 -2～-12%），绝对值对比跨会话不可靠。
   **对策：thermal-paired 测量协议**：基线/挑战者**交替成对**测量 + 轮间冷却门控 +
   **对内 delta** 判定（共模热漂移在对内相消）——WDDM 动态时钟环境的方法学贡献。

**目标**：G-全线 6 尺寸中 ≥4 尺寸真实刷新（配对协议下）；512³/1024³ 大幅收窄
（力争 ≥ cuBLAS 的 75%）；4096³ ≥ 7.0 TF（力争 7.2 TF）；256³ 保持反超。
全部在严格 FP32 军规内（不解除 TF32 禁令——AR007 ST 已裁定维持军规）。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- `sgemm_swpipe_sk`（CLI 名 `swsk`）：swpipe 的 split-K 变体——3D grid、
  部分积 workspace、确定性固定序归约 kernel、CLI `--sk {1,2,4,8,16}` 旋钮
- `sgemm_ws`（CLI 名 `ws`，Kernel 7）：warp 专属化 producer/consumer 软件流水
  （named barriers + 多级 smem 环），非对齐回退路径
- `--kernel auto`：实测驱动的尺寸自适应选核表（dispatch table）
- thermal-paired 测量协议：run_matrix v2（交替配对 + 冷却门控）+ compare v2（对内 delta）
- swsk/ws 消融（sk 扫描、warp 配比、环深度）、全矩阵刷新、四门 v2 判定、
  图表/报告/README/详设/AGENTS 同步

**Out of Scope（本 AR 不做的）：**
- 解除严格 FP32 军规（TF32/tensor core/低精度）——AR007 ST 已裁定维持
- 修改冻结签名 `sgemm_x(A,B,C,M,N,K)`（swsk 的 workspace 走内部管理，见 §5）
- ncu 硬件计数器采集（权限仍阻塞，维持降级方法学；ws 的 issue-slot 假说以消融间接验证）
- cuBLAS 之外的第三方库对比；批量 GEMM/strassen 类算法重构

## 3. 功能需求

### 3.1 swsk：swpipe 的 split-K 变体

**描述**：新增注册 kernel `swsk`，把 K 维切为 SK 片并行，解除中/小尺寸 wave 饥饿。

**触发条件**：`--kernel swsk [--sk N]`（默认 sk=4；sk=1 时退化为单波 swpipe 路径）。
主路径条件同 swpipe（N%4==0、K%(4·SK)==0 感知、16B 对齐；不满足回退 tile2d 标量路径）。

**期望行为**：
- grid = (M/128) × (N/128) × SK 三维；每个 block 累加自己 K 片的部分积，
  写入 workspace P[SK][M][N]（fp32）；
- 随后启动确定性归约 kernel：固定顺序求和 SK 片 → C（run-to-run 逐位可复现）；
- 归约 kernel 与主 kernel 均在 bench 计时区内（诚实测量 split-K 全部工作）。

**异常处理**：workspace 分配失败报错退出；K 切片不整除时尾片守卫（零填充语义）；
非对齐尺寸回退 tile2d（--verbose 打印路径）。

**验收标准**：
- Given 测试矩阵 9 尺寸, When sgemm_test 运行, Then 11 kernel × 9 尺寸 + 2 自校验全 PASS
- Given 512³/1024³, When swsk(最优 sk) vs swpipe, Then GF 显著提升且对内 rel 判据达标

### 3.2 ws：Kernel 7 warp 专属化软件流水

**描述**：新增注册 kernel `ws`——producer/consumer 分工，把全局加载与 smem 搬运
从 FFMA 关键路径上剥离。

**触发条件**：`--kernel ws`；主路径条件同 swpipe，非对齐回退 tile2d。

**期望行为**：
- 384 线程/block：2 producer warp（LDG→寄存器→STS，灌 3 级 smem 环）+
  10 consumer warp（LDS→FFMA，128×128×BK 输出块）；
- 同步用 PTX named barriers（`bar.sync <id>, <count>`，sm_75 合法），
  producer/consumer 各持独立 barrier id；
- consumer 稳态循环仅 LDS + FFMA + 环形缓冲下标（issue-slot 占比最大化）。

**异常处理**：named barrier 死锁防护（环满/空条件谓词）；非对齐回退；
racecheck 0 hazards 为发布硬门。

**验收标准**：
- Given 测试矩阵, When sgemm_test 运行, Then ws 全尺寸 PASS
- Given 4096³ 冷态配对, When ws vs swpipe, Then 判定输出（正/负结果均如实归档——
  issue-slot 假说的实验检验）

### 3.3 auto：尺寸自适应选核

**描述**：`--kernel auto` 按 (M,N,K) 从**实测驱动的 dispatch 表**选择注册 kernel
（含 swsk 的 sk 参数），复刻 cuBLAS heuristic 选核机制。

**触发条件**：`--kernel auto --m --n --k`。

**期望行为**：查表规则 = 尺寸区间 → (kernel, 参数)；表内容来自本 AR 消融实测数据，
**禁止凭感觉填表**；未覆盖区间走最近邻规则并在 --verbose 打印选择依据。

**异常处理**：表未命中 → 回退 swpipe（安全默认）+ verbose 说明。

**验收标准**：
- Given 6 基准尺寸, When auto 运行, Then 每尺寸选中 kernel 与该尺寸实测最优一致
  （容差 ±2% 内允许并列者）
- Given 任意 M/N/K（含 17×33×65）, When auto 运行, Then 正确性 PASS

### 3.4 thermal-paired 测量协议

**描述**：run_matrix v2 + compare v2——交替配对测量 + 轮间冷却门控 + 对内 delta，
消除 WDDM 热漂移对绝对值对比的污染。

**触发条件**：`bench/run_matrix.ps1 -Paired`（或 v2 参数开关）。

**期望行为**：
- 每个尺寸：基线（swpipe）与挑战者（swsk/ws/auto）**交替**成对执行
  （A/B/A/B × rounds），对内 delta 即共模漂移相消后的真实差值；
- 配对组之间冷却门控（GPU 温度回落到阈值内才开下一组），GPU 状态逐行落 CSV；
- compare v2：输出对内 delta 表 + 四门 v2 判定。

**异常处理**：冷却超时 → 如实标注该组数据 `thermal-contaminated` 并保留。

**验收标准**：
- Given 配对协议会话, When compare v2 运行, Then 对内 delta 的跨轮一致性
  优于 AR007 跨会话绝对对比（对内 delta 轮间波动 < 2%）

### 3.5 消融与文档同步

**描述**：swsk 的 sk 扫描、ws 的 warp 配比/环深度消融；详设 §4.1（新 kernel 声明）、
§4.3（--sk 语义）、§4.6（旋钮表）、§5（Kernel 7 段）、AGENTS 消融旋钮段、
report/README/figures 全量刷新（含 auto 选核表与配对协议说明）。

**验收标准**：
- Given 全部开发完成, When 文档审阅, Then 无悬空引用、默认值与代码一致（AR007 review 教训）

## 4. 非功能需求（四门 v2 + 军规）

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能-G1 中尺寸 | 512³ 与 1024³ 自研最优 | 各 ≥ 同会话 cuBLAS 的 75%（512³ 现 41%） |
| 性能-G2 小尺寸守成 | 256³ auto/swsk | ≥ 1618.2 GF 且保持 > cuBLAS（反超不可丢） |
| 性能-G3 大尺寸 | 4096³ 最优（ws/swpipe） | 冷态配对 ≥ 7.0 TF（58.4% 峰值），力争 7.2 TF；vs swpipe 6584.8 ≥ +5% 或负结果归档 |
| 性能-G4 全线 | 6 尺寸配对刷新 | ≥ 4/6 尺寸对内 delta ≥ +2% |
| 性能-G5 方法学 | 配对协议 | 对内 delta 轮间波动 < 2%（vs AR007 热浸没有 ~5% 掩蔽） |
| 正确性 | 测试套件 | 11×9+2 = 101 例全 PASS；racecheck 抽查 ws + swsk |
| 资源 | ptxas 审计 | 0 spill 硬门；ws/swsk 资源与取舍记录 |
| 精度 | 严格 FP32 | 全程不变（军规继承） |
| 可复现 | 数据链 | CSV 逐行 git sha + GPU 状态；配对数据成组可溯 |

## 5. 约束与假设

**约束：**
- 冻结签名不变 → swsk workspace 走**内部 grow-only 缓存**（匿名 namespace，
  warmup 期完成分配，计时区零 alloc；进程退出统一释放，满足"成对释放"军规）；
  该取舍必须在详设与 kernel 头注释中显式记录；
- 归约确定性：固定顺序归约，禁 atomicAdd 非确定序（run-to-run 逐位可复现，
  延续本工程可复现学风；与单 kernel 累加序不同属正常，rel ≤ 1e-4 判据）；
- named barriers 允许 PTX 内联 asm（sm_75 合法指令），必须过 racecheck；
- auto 表条目必须来自本 AR 实测（数据真实性军规：禁凭空填表）；
- WDDM 时钟策略 B 沿用；CSV schema 冻结（新 kernel 即新行，列不变）；
- auto 模式：各阶段不逐项询问，ST 验收集中回报（含 ws 假说检验与四门 v2 裁定）。

**假设（待实验检验，诚实声明）：**
- 512³/1024³ 瓶颈主因 = wave 饥饿（证据：GF 比例 ≈ blocks/48 比例）——split-K 直击，
  G1 把握较高；
- 4096³ issue-slot 假说为**间接推断**（无 ncu 计数器）：ws 消融即假说检验；
  G3 的 7.2 TF 力争值**无把握承诺**，负结果（ws ≈ swpipe）则如实归档并构成
  "pre-Ampere 无 cp.async 下软件流水边界"的负结果贡献；
- workspace 额外流量（P 写+读）在中尺寸为净收益（估算：512³ sk=8 约 +17MB 流量
  vs 计算 100µs 量级，带宽占用 ~15%，利用率 33%→90%+ 的收益远盖过）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| split-K / swsk | K 维切分为 SK 片并行计算部分积，再归约——解除小尺寸 wave 饥饿 |
| wave | grid 的 block 总数除以 SM 数——尾波（wave 非整）即部分 SM 空转 |
| named barrier | PTX `bar.sync id, count`——按 barrier id 与线程数选择性同步，warp 专属化的同步原语 |
| producer/consumer | 专职加载 warp 与专职计算 warp 的分工流水（Hopper 时代的库标配，本工程在 sm_75 上软件复刻） |
| thermal-paired | 基线/挑战者交替成对测量、对内 delta 判定的热漂移免疫协议 |
| dispatch 表 | (M,N,K) 区间 → (kernel, 参数) 的实测驱动查表，cuBLAS heuristic 的祛魅复刻 |
