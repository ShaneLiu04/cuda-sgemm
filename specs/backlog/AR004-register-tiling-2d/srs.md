# [AR004] 需求设计说明书 — Kernel 3：二维寄存器分块（128×128×8，8×8）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR004 |
| AR 主题 | register-tiling-2d |
| 关联 SR | SR-SGEMM-OPT |
| 日期 | 2026-10-04 |
| 状态 | Draft（待用户确认） |

> 设计输入：AR003 闭环（预期证据：load:FMA 比例高、barrier 开销显著、单方向复用不足）。

## 1. 背景与目标

引入 Block Tile **BM×BN×BK = 128×128×8** 与 Thread Tile **TM×TN = 8×8**：每线程 64 个 fp32 累加器，
内层每 k 步从 smem 读 `a[8]`、`b[8]` 片段做 64 次外积 FMA，最大化寄存器复用与 ILP。这是本工程
计算密度的关键跃迁（目标 2.46×），也是寄存器压力风险最高的一版（64 累加器 + 片段 + 寻址 ≈ 90+ regs）。

## 2. 需求范围

**In Scope：** `sgemm_2d_tile`（256 线程 = 16×16、smem `[128][8+pad]`/`[8][128+pad]`、`__launch_bounds__`、
循环展开/ILP 策略）；正确性/bench/ncu 闭环；无 spill 硬门。
**Out of Scope：** float4 与转置布局（AR005）；cp.async（AR006）。

## 3. 功能需求

### 3.1 Kernel 3：sgemm_2d_tile

**描述：** 外积累加 `c[8][8] += a[i]·b[j]`；smem 加 padding 消 bank conflict（pad 值实测取舍）；
`__launch_bounds__(256, minBlocks)` 显式声明（minBlocks 实测 1 vs 2 取舍并记录）。
**异常处理：** 边界 tile 谓词正确；1023×1024×511 必须 PASS。
**验收标准：**
- Given 全测试矩阵，When `make test`，Then 2d_tile 全 PASS；
- Given 4096³，When benchmark，Then GFLOPS ≥ 3.16 TFLOPS（目标 ~3.51，±10%）。

### 3.2 寄存器资源硬门

**验收标准：**
- Given `-Xptxas -v` 输出，When 检查，Then **spill stores/loads = 0**（非零即缺陷：调 launch_bounds/
  片段缓存策略/循环结构，修复过程记录）；regs/thread 与 minBlocks 取舍数据存档 CSV。

### 3.3 ncu 闭环（#4）

**验收标准：**
- Given ncu 数据，When 分析，Then bottleneck_analysis.md 追加闭环：① FP32 pipe 利用率跃升证据；
  ② smem 吞吐与 conflict 计数；③ stall 分布（对比 AR003）；④ 剩余瓶颈归因
  （global→reg→smem 两次搬运、标量访存、无流水重叠）→ AR005 验证计划（float4 降指令数与事务数）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 4096³ GFLOPS | 目标 ~3.51 TFLOPS（2.46× vs AR003）；验收下限 ≥ 3.16 TFLOPS |
| 资源 | spill | **0**（硬门） |
| 资源 | regs/thread | 记录；目标 ≤ 128（支持 2 block/SM），实测取舍可放宽但须记录理由 |
| 正确性 | 全测试矩阵 | 全 PASS |

## 5. 约束与假设

**约束：** 禁止 float4/cp.async（职责边界）；launch_bounds 与 pad 的每个取舍必须有实测数据支撑。
**假设：** AR003 的 smem 加载/同步骨架可复用为本版基础。
