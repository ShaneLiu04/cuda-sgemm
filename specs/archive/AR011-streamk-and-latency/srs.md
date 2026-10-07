# AR011 srs — Stream-K 统一调度 + 主核延迟覆盖攻坚（G1@1024³ 翻门 / 2048³ 冲 85%）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR011 |
| 主题 | streamk-and-latency（Kernel 10 Stream-K + L2 钉 C + 延迟覆盖微优化 + auto v4） |
| 前置 | AR010（五门 v4 终态：G1@512³ 75.14% / G1@1024³ 74.77% 刀锋 / G2 钟态匹配 / G3 68.1% / auto v3 全绿） |
| 需求源 | `results/2026107.md`（AR010 收官优化分析报告，P0-P3 路线图） |
| 日期 | 2026-10-07 |
| 状态 | Draft（auto 模式：沿用 AR007-AR010 先例，req/design 门控跳过，ST 集中回报） |

## 1. 背景与问题（AR010 终态 + 2026107 归因）

| 门 | AR010 终态 | 2026107 归因（期望） |
|----|-----------|---------------------|
| G1@1024³ ≥75% cuBLAS | **FAIL 74.77%**（6279.2/8397.5，差 0.23pp） | 缺口折算**仅 1.05μs**（342.0 vs 341.0μs）；归约段 41μs——任何 >1.1μs 的归约侧节省即翻门（L2 钉 C 预计省 4~6μs，单独翻门） |
| G1@512³ | PASS 75.14% | 回归保持（swsk_sk3 守擂，不动） |
| G2@256³ | PASS（钟态匹配） | 回归保持（123.7% 反超 cuBLAS） |
| G3@4096³ ≥7.0TF | PASS 7970.8（68.1% peak） | 主核 12-16pp 延迟覆盖缺口是全尺寸共同主战场 |
| 2048³（无独立门） | deep 8568.5 = 82.6% cuBLAS | **波量化尾波 -11%**（128 blocks/48 SM = 2.67 波，尾波效率 88.9%）——Stream-K 直接吃掉 |

**三个核心事实（2026107 §0/§1/§2，全部有数据支撑）**：

1. **1024³ 是"一微秒的差距"**：main 297μs + direct 归约 41μs + 启动 ~3μs = 342.0μs，
   门线 341.0μs。归约已在流量地板（355 GB/s = 实测可达带宽 375.7 的 94.5%），
   ILP/提示词类优化物理无效——只有结构性消除（融合/省流量）有效。
2. **"LDS.128 带宽墙"需重新解读**：deep 设计点 LSU 吞吐利用率仅 37.5%
   （192 vs 512 clk/kk 步），72.1% 与 100% FFMA 屋顶之间的 28pp 是**延迟覆盖/调度气泡**
   而非带宽。acc/LDS 经验律的机制 = 比率是延迟覆盖能力的代理变量。
   此修正将优化方向从"加 acc"（寄存器堆已顶 247/255）转向"消灭气泡"
   （同相停顿 / barrier 歪斜 / 步首 LDS 依赖 / 尾 tile 裸延迟四候选，需 ncu 裁决）。
3. **两项结构性浪费可被 Stream-K 统一消除**：波量化（2048³ -11%）+ 归约独立
   第二 kernel（1024³ 归约 41μs + 启动 + P 自片 DRAM 往返）。
   `deep_tile_grid` 的 t0/t1 参数化已是 Stream-K 块映射雏形，主体可复用。

## 2. 功能需求（FR）

### FR1 测量体系解锁与归因（条件任务，P0）

**描述**：申请 admin → Quadro RTX 5000 切 TCC 模式，解锁 ncu 计数器与 `nvidia-smi -lgc` 锁频；
随后执行三组归因实验（E-A 主核 stall 四分类分解 / E-B 反向测 cuBLAS@1024³ launch config /
E-C 锁频重跑五门）。

**条件语义（数据真实性军规）**：
- admin/TCC **可得** → E-A/E-B/E-C 全量落盘（profile/ 目录 + environment.md 更新）；
- **不可得** → 落盘环境声明（沿用 WDDM 口径），E-A 用现有分段计时 + 消融间接归因替代，
  E-C 以 %peak 不变量 + v4 配对协议继续；**不得因环境缺失虚构计数器数据**。

**验收标准**：
- Given TCC 可得，When `ncu --query-metrics` 正常返回，Then E-A stall 分解落盘且四候选
  （同相停顿/barrier 歪斜/步首依赖/尾 tile）至少排除或确认两个；
- Given TCC 不可得，When 归因实验执行，Then 替代协议数据落盘 + 环境声明更新，实验可复现。

### FR2 L2 persistence window 钉 C（最快收益，P2.3）—— **T003 判 N/A（能力墙）**

> **T003 收口（2026-10-07，负结果归档）**：runtime 验证否定了 SS7 风险项的
> 乐观假设——TU104 sm_75 **不支持** L2 persistence（persistingL2CacheMaxSize=0、
> MaxAccessPolicyWindowSize=0、SetLimit 报 "not supported on this architecture"；
> Ampere+ 特性）。FR2 判 **N/A**（环境能力墙，非实现取舍）；证据与工程落点见
> environment.md §8。G1@1024³ 翻门（缺口 1.05μs）的机制支路去掉
> "钉 C 预计省 4~6μs"，责任转移至 Stream-K W sweep / FR3 消融 / auto v4，
> **门线不放宽**。--persist 1 → [note] 优雅降级（exit 0）。

**描述**：`cudaStreamSetAttribute` accessPolicyWindow 将 C 区域（1024³ = 4MB，恰等于
TU104 L2 容量）设 persisting，归约读 C 从 DRAM 挪入 L2。

**要点**：
- `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)` 上限设置 + hitRatio {0.6, 0.8, 1.0} 消融；
- dsk 已有 main/reduce 分段 verbose 计时基建（sgemm_deep_sk.cu:99-136），净效应
  （main 段 A/B L2 正常命中率是否被挤压）必须同会话对照裁决；
- 预期归约 41μs 中省 4~6μs——**单此一项越过 1.05μs 门缺口**。

**验收标准**：
- Given 1024³ dsk_sk3 直写路径，When persistence on/off 配对（v4 协议），
  Then 归约段分段计时差 ≥2μs 且 main 段退化 <1% 时判定净收益成立；
- 同会话比例门复判：G1@1024³ ≥75%（预期翻门）。

### FR3 主核延迟覆盖微优化消融（P2.1/P2.2）

**描述**：两个独立消融实验裁决 §1 事实 2 的延迟覆盖假说。

- **FR3a B 片段 kk 预取**：compute 循环内 B 片段(kk+1) 先 LDS 入第二组寄存器，
  在 kk 的 128 FFMA 期间覆盖延迟；`template <int BPF>` 双实例（on/off），
  成本 +8 regs（dbuf1 实例 243→251/255 刀口）。**ptxas 逐实例审计硬门**
  （AR010 寄存器重排教训直接适用：0 spill + 寄存器数逐实例记录）。
- **FR3b kk 轮转错相**：warp w 的 kk 处理序轮转偏移（`kk' = (u + w) & 7`），
  打破 8 warp 经屏障后的同相停顿；**零寄存器成本**。

**数值口径分级（srs 预声明，保护既有锚链解释力）**：
- FR3a 不改变加法链序 → **bitwise 锚链适用**（deep==swpipe 等 21 项专项可延续）；
- FR3b 改变每元素 k 加法序 → **bitwise 不适用**；正确性走 rel≤1e-4 双参考门 +
  run-to-run 逐位确定性门（双跑 memcmp 全位一致）。

**验收标准**：
- Given 消融构建（模板实例 0 spill），When v4 配对协议 @1024³/2048³，
  Then 每项给出净效应（+/-/0）与 ptxas 审计记录，胜者进入 auto v4 候选池；
- 任一实验若 main +≥2%，G1@1024³ 预期以余量越门。

### FR4 Stream-K Kernel 10（结构性主菜，P1）

**描述**：把 (tile_x × tile_y × k-step) 三维迭代空间切成恰好 48×W 个连续块
（变长 K 区间），消灭波量化；每 C-tile 最后完成块持 **per-tile 票据**（atomic ticket）
以固定 z 序读其他块部分积、累加自己寄存器累加器直写 C——归约第二 kernel、
启动开销、P 自片 DRAM 往返三者一并消除。

**设计要点（详见 design.md）**：
- 块映射：迭代空间连续切分（tile-major / k-major 混合序，design 定稿），
  复用 `deep_tile_grid` 的 t0/t1 变长区间参数化与 compute/流水/布局主体；
- 票据：每 C-tile 一个 atomicInc 计数器（workspace，RAII），最后完成块执行归并；
  **固定 z 序求和**——归并链序与 dsk direct 归约对齐（design.md 论证可否延续 bitwise）；
- 可见性：`__threadfence` + acquire 语义，racecheck 专项硬门；
- 翻案声明：AR008 否决的是"全局单块串行化"方案；**per-tile 票据**（96 块各自归并
  自己的 256×128 区域）不存在串行化，是该否决的正确变体，本 AR 重开。
- 尾块不均衡（票据赢家多干归并活 → kernel 尾部 ~5μs）如实披露并计入实测。

**验收标准**：
- Given 任意 M/N/K（含边界/回退），When 全矩阵回归 + 确定性双跑，
  Then 146/146 + run-to-run 逐位一致 + memcheck/racecheck 0；
- Given 1024³，When v4 配对协议，Then 总时间 ≤315μs（≥77% cuBLAS，越门且余量）；
- Given 2048³，When v4 配对协议，Then ≥85% cuBLAS（G6 新门）；
- 波填充率：任意尺寸 grid 块数 = 48 的整倍数（尾波浪费 ≤1/48）。

### FR5 auto v4 + 几何 sk 公式泛化

**描述**：dispatch 表吸收本 AR 胜者（streamk / L2 钉 C / FR3 胜者）；矩形/极端长宽比
尺寸的 sk 选择从固定带判升级为几何公式 `sk = clamp(ceil(48·W / blocks), 2, 16)`。

**验收标准**：
- Given 六尺寸 + 长宽比 ≥4:1 的补充尺寸（如 256×4096、4096×256），
  When A-B-A-B 保真验证，Then 10/12 对 ≤0.6pp 级保真（沿用 G5'' 口径），
  补充尺寸全量首测并回填。

### FR6 工程卫生（P3，半天量级）

- `sgemm_deep_sk.cu:15-17,109` 过时 `__stwt` 注释修正（与已回退实测矛盾，违反注释契约）；
  同文件 "归约 45→~33μs" 与实测 43μs 口径一并修正；
- ws/wsk 4×8B spill：根治或显式豁免记录（教学阶梯卫生债）；
- 交付文档同步：详设 Kernel 10 条目 + 旋钮表、AGENTS 旋钮、report.md §13、
  bottleneck 闭环 #6、paper/interview 附录、st_report 集中回报。

## 3. 验收门（v5，v4 协议继承）

| 门 | 判定 | 通过条件 |
|----|------|---------|
| G1@1024³ | 翻门主目标 | best own ≥ 同会话 cuBLAS 75%（预期 77%+；~~Stream-K + 钉 C 叠加~~ → T003 后机制 = Stream-K W sweep + FR3 消融 + auto v4，见 FR2 修订注） |
| G6@2048³（新门） | Stream-K 波效率检验 | best own ≥ 同会话 cuBLAS 85%（现状 82.6% + 尾波 -11% 消除的兑现度） |
| G1@512³ / G2@256³ / G3@4096³ | 回归保持 | ≥ AR010 终态（75.14% / 钟态匹配 PASS / ≥7.0TF） |
| G4''' auto v4 vs v3 | 增量门 | 4/6 尺寸 %peak ≥ +2%，0 尺寸 < -2% |
| G5''' dispatch 保真 | 回归门 | ≤2pp + 分级极差 median <2pp（补充尺寸纳入） |
| 正确性 | 硬门 | 146/146；bitwise 21 项锚链延续（链序不变路径）+ 新变体 rel≤1e-4 + 确定性双跑全位一致 |
| 资源 | 硬门 | 新模板全实例 0 spill（-Xptxas -v 逐实例审计入 build.log） |
| sanitizer | 硬门 | memcheck 0；racecheck（票据/fence/persistence 路径专项）0 hazards |
| P0 条件门 | 若解锁 | 锁频口径五门复判落盘（与 WDDM 口径并行报告，不替换历史结论） |

## 4. 需求范围

**In Scope**：FR1-FR6（P0 条件任务、L2 钉 C、FR3a/b 消融、Stream-K、auto v4、工程卫生）。

**Out of Scope（明确不做）**：
- 16-warp 控制实验（P3.1 研究项）→ backlog（等 E-A 归因结果再定优先级）；
- 4096³ 专项攻坚 → 依赖 P0 锁频归因（E-C），本 AR 仅做条件复判，不做 kernel 改动；
- 大尺寸 L2 swizzle（N7 预防性负结果封存：4096³ 每波 DRAM 需求 ~21% 峰值，非瓶颈）；
- TF32/快数学、cp.async、warp 专属化、占用率追逐（N1-N6 历史封存区）；
- 跨硬件（结论限定 TU104/sm_75/48 SM 几何）。

## 5. 约束与假设

**约束**：
- 冻结签名 `void sgemm_x(const float*, const float*, float*, int, int, int, int)`；行主序；严格 FP32；
- 数据真实性军规：CUDA events、median、CSV 带 git sha/时钟/温度；非默认参数分流独立 CSV；
- WDDM 无锁频（P0 未解锁时）：%peak 不变量 + 同会话配对锚定 + 47°C 冷却门沿用；
- ptxas 重排纪律：性能相关参数一律编译期模板实例化 + 逐实例审计；
- 测量协议：run_paired v4 必须 tools\env.cmd 包裹；脚本追加写 CSV 前删旧文件；
- gitee 推送小包纪律（PNG ≤72KB，文本提交安全）。

**假设**：
- admin/TCC 权限申请结果未知 → FR1 全部设计为条件任务，不阻塞 FR2-FR5；
- Stream-K 归并链序可对齐 dsk direct 归约（design.md 论证；若不可对齐则降级为
  rel≤1e-4 + 确定性门，srs 已预声明分级口径）；
- L2 persistence 净收益为正（若 main 段退化抵消归约收益，FR2 判负结果如实归档，
  不影响 FR4 主线）；
- hitRatio ≤1.0 时 persisting 行为可预期（~~sm_75 支持 accessPolicyWindow，需 runtime 验证~~
  **已验证：sm_75 不支持**——T003 能力墙，见 FR2 修订注与环境 §8）。

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| Stream-K | 三维迭代空间 (tile×k-step) 连续切分为固定块数（48×W）的调度方式，变长 K 区间 + 融合归并，消灭波量化 |
| per-tile 票据 | 每 C-tile 一个 atomic 计数器，最后完成的块执行该 tile 的归并（与 AR008 否决的"全局单块"方案相区别） |
| L2 persistence window | `cudaStreamSetAttribute` accessPolicyWindow 将指定地址区间钉入 L2 的机制 |
| 延迟覆盖 | warp 在 LDS/LDG 返回延迟期间有可发射独立指令（ILP/他 warp TLP）的比例——2026107 对 28pp 缺口的机制解释 |
| kk 轮转错相 | 各 warp 以不同偏移轮转处理 kk 序，打破屏障后同相停顿（改变加法序，数值口径降级） |
| 波填充率 | grid 块数相对 48 SM 整波的占比；Stream-K 目标 = 恒 100%（块数恒为 48 整倍数） |
