# [AR002] 需求设计说明书 — Kernel 1：访存合并优化

| 字段 | 内容 |
|------|------|
| AR 编号 | AR002 |
| AR 主题 | coalesced-access |
| 关联 SR | SR-SGEMM-OPT |
| 日期 | 2026-10-04 |
| 状态 | Draft（待用户确认） |

> 设计输入：`results/bottleneck_analysis.md` 中 AR001 产出的 Naive 瓶颈闭环（预期：访存事务低效 +
> long scoreboard 主导）。全局契约见组件详设 §4。

## 1. 背景与目标

Naive 的 warp 内 32 线程沿行方向映射时对 B 的读取与 C 的写回产生低效/重复内存事务。本 AR 仅做一件事：
调整线程→输出的映射，使 warp 内相邻线程访问相邻列，形成 128B 对齐合并事务——**不引入任何分块/复用**，
保持 memory-bound 特性，为 AR003 的"重复读取 O(N)/O(M) 次"证据链服务。

## 2. 需求范围

**In Scope：** `sgemm_coalesced` kernel + 正确性测试扩展 + benchmark + ncu 对比分析（vs Naive）。
**Out of Scope：** shared memory、任何 tile 复用、向量化（float4 属 AR005）。

## 3. 功能需求

### 3.1 Kernel 1：sgemm_coalesced

**描述：** 仅改线程映射：warp 内 32 线程**连续读 B、连续写 C**；A 的访问模式（同行广播/同地址）须分析并在 design.md 写明。
**触发条件：** AR001 已归档。
**期望行为：** 全测试矩阵 PASS（含边界尺寸）。
**验收标准：**
- Given 全测试矩阵，When `make test`，Then coalesced 全 PASS；
- Given 4096³，When benchmark，Then GFLOPS ≥ 666（目标 ~740.44，±10% 条款）。

### 3.2 ncu 对比分析（闭环 #2）

**验收标准：**
- Given ncu 数据（naive vs coalesced），When 分析，Then bottleneck_analysis.md 追加闭环，含：
  ① global load/store 的 sectors/request 数值对比（预期趋近 4）；② DRAM 流量 vs 理论最小流量比值
  （证明仍是数十倍重复读取）；③ stall 分布变化；④ AR003 验证计划（用 smem 分块消除重复读取）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 4096³ GFLOPS | 目标 ~740.44（6.52× vs Naive）；验收下限 ≥ 666 |
| 正确性 | 全测试矩阵 | 全 PASS |
| 资源 | regs/smem/occupancy | 记录到 CSV；smem 预期为 0 |

## 5. 约束与假设

**约束：** 禁止引入 smem/tiling（那是 AR003 的职责边界，本版必须保持"无分块"以支撑证据链）。
**假设：** AR001 的 harness 与参考线无需改动即可复用。
