# EXPERIMENT_DESIGN.md — 可解释性实验设计（实验有效性验证集）

> 目的：用**丰富、可解释、可复现**的实验证据链证明六版优化的每一步都
> "改对了地方、拿到了预期效果、机理可解释"。每个实验固定五要素：
> **目的 / 假设 H / 方法 / 指标与预期 / 解读指南（含判伪条件）**。
> 全部实验在 Quadro RTX 5000（sm_75，本机调优目标机，2026-10-04 改靶）主场景 4096³ 下执行，
> 时钟策略见 TEST_PLAN §1。
> 结果统一落盘 `results/experiments/E<xx>_*.md`（数据 + 结论 + 与假设的对照）。

---

## 实验总览

| # | 实验 | 类型 | 支撑的验收点 |
|---|------|------|-------------|
| E01 | 性能阶梯与加速比 | 主实验 | 全 AR 性能门 |
| E02 | cuBLAS 公平性（TF32 开关对照） | 对照实验 | 参考线可信度 |
| E03 | 访存合并效率（naive vs coalesced） | 机理实验 | AR002 |
| E04 | DRAM 流量 vs 理论最小值（复用演进） | 机理实验 | AR002→AR003 证据链 |
| E05 | BK 扫描消融（8/16/32） | 消融 | AR003 T003 |
| E06 | launch_bounds / PAD 消融 | 消融 | AR004 T003 |
| E07 | swizzle 开关消融 | 消融 | AR005 conflict≈0 门 |
| E08 | cp.async 布局方案甲/乙消融 | 消融 | AR006 T003 |
| E09 | Warp stall 六版演化 | 机理实验 | 逐版闭环核心 |
| E10 | Roofline 定位 | 分析 | 全链归因 |
| E11 | 精度与数值稳定性 | 正确性深化 | 详设 §4.4 |
| E12 | 尺度扫描与 wave 量化 | 扩展 | 健壮性 |
| E13 | 降频/稳态长跑 | 环境 | ±10% 条款 |
| E14 | cp.async 流水收益分解 | 机理实验 | AR006 归因 |

---

## E01 性能阶梯与加速比

- **目的**：验证六版优化形成 113 GF → 6.61 TF 的阶梯，每级加速来自设计预期。
- **假设 H**：各版 GFLOPS(median) 落入目标 ±10%；相邻版加速比 ≥ 表中"最低加速比"。
- **方法**：`./bench/run_bench.sh ladder`（warmup 20 / iters 100 / CSV 落盘）。
- **指标与预期**：

| 版本 | 目标 GFLOPS | 最低加速比（vs 前版） |
|------|------------|---------------------|
| naive | 113.55 | — |
| coalesced | 740 | 6.0× |
| smem1d | 1430 | 1.7× |
| tile2d | 3510 | 2.2× |
| vec4 | 5840 | 1.5× |
| cpasync | 6610 | 1.1× 且 ≥97% cuBLAS |

- **解读**：某级加速比低于最低值 → 不解释为"硬件波动"，先跑该版 ncu 对照
  TEST_PLAN §7 预期签名；签名不符 = 实现偏差，回查代码。
- **判伪**：若 naive 达到 >300 GF（合并已被编译器优化掉），重写映射确保基线成立。

## E02 cuBLAS 公平性（TF32 对照）

- **目的**：证明参考线确为严格 FP32（否则 97% 对比失真）。
- **假设 H**：`CUBLAS_DEFAULT_MATH` 实测 ≈ 6.7 TF；TF32 模式 ≫ 15 TF（Ada TC）。
- **方法**：临时补一个对照程序（或改 cublas_baseline.cu 加 env 开关），
  分别以 `CUBLAS_DEFAULT_MATH` / `CUBLAS_TF32_TENSOR_OP_MATH` 测 4096³。
- **预期**：两者差距 > 2×。若 DEFAULT_MATH 结果接近 TF32 → 检查 handle 复用与设置顺序。
- **落盘**：E02 数据表 + 两种模式的 correctness 对比（TF32 误差应 ≈1e-3 量级，FP32 ≈1e-6）。

## E03 访存合并效率

- **目的**：量化"只改线程映射"带来的访存事务效率变化（AR002 核心主张）。
- **假设 H**：naive 的 B 读/C 写 sectors/request ≫ 4（按行跨步拆事务）；
  coalesced 后 → ≈4；A 读从"跨步"变"同地址广播"。
- **方法**：`ncu` 对两版采集
  `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_{ld,st}.ratio`
  与 `dram__bytes.sum`。
- **解读**：预期 coalesced 的 DRAM 总流量**不降反接近不变**（无分块，重复读取仍在），
  但事务效率提升 → 性能 6.5×。这条"流量不变、效率提升"的组合证据是 AR002 的机理闭环。

## E04 DRAM 流量 vs 理论最小值（复用演进）

- **目的**：用一条曲线讲清"分块消除重复读取"的六版演进。
- **假设 H**（流量比 = dram__bytes.sum / theoretical_min，后者由
  `2·M·N·K 相关三项矩阵字节` 给出，common.h 已提供计算）：
  naive ≈ 数十倍 → coalesced 仍数十倍 → smem1d ≈ K/16 量级（tile 级重复）→
  tile2d ≈ K/8 量级 → vec4 同级 → cpasync ≈ 1.0-1.2（逼近最小）。
- **方法**：profile_all.sh 产出各版 dram__bytes.sum，与理论值列表对比。
- **解读**：流量比每级下降应与 tile 形状理论复用率一致；不一致 → 检查边界补零与 L2 命中。

## E05 BK 扫描消融（smem1d）

- **目的**：为 32×32+TM=8 结构选 BK，展示"同步开销 vs 数据局部性"权衡。
- **方法**：`./build/sgemm_bench --kernel smem1d --bk {8,16,32} --csv`。
- **预期**：16 或 32 最优；8 因同步频率翻倍而劣化 >10%。
- **落盘**：三数据点 + 选择理由（写入 AR003 闭环）。

## E06 launch_bounds / PAD 消融（tile2d）

- **目的**：寄存器压力-占用率权衡的实证。
- **方法**：`--kernel tile2d --lb {1,2}`；对照 build.log 的 regs/spill 与 ncu occupancy。
- **预期**：lb=2 时 regs ≤128、无 spill、occupancy ↑；性能两种可能（占用胜 or ILP 胜），
  **以实测为准记录**（D5/D7 决策：报告数据而非套结论）。

## E07 swizzle 开关消融（vec4）

- **目的**：证明 XOR swizzle 消除了 b-frag 的 4-way bank conflict。
- **方法**：临时把 `sgemm_vec4_kernel` 中 swizzle 掩码改为 0（`sw=0` 常量，
  即逻辑=物理布局）重编译为 `vec4_noswiz`，ncu 对比
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 与 GFLOPS。
- **预期**：noswiz 版 conflict 计数显著非零（≈4-way 特征）、GFLOPS 下降 ≥3%；
  swizzle 版 ≈0。若两版无差 → 冲突推导有误，以 ncu 为准修订注释与文档。

## E08 cp.async 方案甲/乙消融

- **目的**：回答"cp.async 不能转置"的取舍（srs AR006 §3.1）。
- **方法**：`--kernel cpasync`（甲：全直拷，a-frag 标量+2-way 冲突）
  vs `--kernel cpasync2`（乙：A float4+转置+寄存器预取，B cp.async+swizzle）。
- **指标**：GFLOPS、a-frag 相关 conflict 计数、stall 分布（wait vs short scoreboard）。
- **预期**：乙的 conflict 更低、a 路径延迟被寄存器预取隐藏；总性能差异可能很小（<3%）——
  **以实测定稿**，数据与选择理由写入 AR006 闭环；落败方案的代码保留在仓库（军规：失败实验同样记录）。

## E09 Warp stall 六版演化

- **目的**：用 stall 主因迁移佐证"瓶颈逐版被消灭"的叙事（本工程分析部分的核心图）。
- **方法**：从 profile/<kernel>/*.csv 提取五类 stall 指标（long/short scoreboard、
  barrier、wait、mio_throttle），画堆叠占比图（脚本随 results 归档）。
- **预期签名**：long scoreboard（naive/coalesced 主导）→ barrier/short（smem1d）→
  short/wait（tile2d/vec4）→ barrier/wait（cpasync）。偏离签名 → 对应版本实现存疑。

## E10 Roofline 定位

- **目的**：把六版放进同一张 roofline，解释"从带宽界到计算界"的迁移。
- **方法**：x = 实测 DRAM GB/s（dram__bytes.sum / 时长），y = GFLOPS；
  屋顶 = 实测峰值带宽（用大尺寸 copy kernel 标定）与 FP32 峰值（E01 的 cuBLAS 上界近似）。
- **预期**：naive/coalesced 远离两条屋顶；cpasync 贴近计算屋顶（SM ~85%+ 利用率）。

## E11 精度与数值稳定性

- **目的**：证明严格 FP32 下各版误差一致且受控；识别"哪一版改变了累加顺序"。
- **方法**：sgemm_test 全矩阵 max_abs/max_rel 数据汇总；另跑 K ∈ {512, 1024, 2048, 4096}
  观察误差随 K 增长（FP32 累加误差 ~ O(√K) 相对量级）。
- **预期**：所有版本 rel ≤ 1e-4（对 cuBLAS）；与 CPU double 的 rel 随 K 缓增；
  任一版本误差显著高于同侪 → 疑似低精度路径（违反军规，立即排查）。

## E12 尺度扫描与 wave 量化

- **目的**：验证优化在非主场景尺寸下不崩、并解释 wave 量化效应。
- **方法**：`./bench/run_bench.sh sweep`（含 8192³，注意 8192²×4B=256MB/矩阵，8GB 显存内安全）。
- **解读**：4096³ = 1024 blocks / 48 SM ≈ 21.3 wave（对 2 block/SM 并发为 10.7）——
  尾部量化小；小尺寸（256³=4 blocks）下高 occupancy 版本受 launch/wave 影响大，
  属预期现象，记录即可。

## E13 降频 / 稳态长跑

- **目的**：量化笔记本功耗墙对测量的影响（±10% 条款的实证边界）。
- **方法**：解锁时钟下连跑 10 轮阶梯（每轮 iters=100），记录每轮 GFLOPS 与
  `nvidia-smi dmon` 频率/温度曲线。
- **预期**：前 1-2 轮最高，随后因温度降频下降 3-8% 并趋稳；
  报告取稳态段中位数，并给出"峰值 vs 稳态"两个数字。

## E14 cp.async 流水收益分解

- **目的**：拆解 Kernel 5 相对 Kernel 4 的 +13% 来自哪里。
- **方法**：构造一个"半流水"对照：把 cpasync(甲) 的双缓冲退化为单缓冲
  （`issue_tile(t)` 移到 compute 之后同步执行，即 load-compute 串行），
  重编译为对照版，比较三者：vec4（无 cp.async）< 单缓冲 ≈ vec4 < 双缓冲。
- **解读**：双缓冲-单缓冲 差值 = 流水重叠收益；单缓冲-vec4 ≈ cp.async 指令路径收益
  （释放寄存器 + cg 不污染 L1）。两个数字分别归因写入 AR006 闭环。

---

## 结果落盘规范

每实验一个文件 `results/experiments/E<xx>_<name>.md`，固定结构：

```markdown
# E<xx> <名称>
- 执行日期 / git commit / 时钟策略 / GPU 状态
- 数据表（原始数字 + 来源文件指针）
- 与假设 H 的对照结论：支持 / 部分支持 / 推翻（推翻时给出修正解释）
- 对应 AR 闭环文档的回填指针
```

**总原则**：实验结论必须能被第三方用本仓库代码 + 本手册命令复现；
任何与预期不符的数据都是最有价值的发现，如实记录。
