# [AR003] 需求设计说明书 — Kernel 2：Shared Memory 分块 + 1D Thread Tiling（TM=8）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR003 |
| AR 主题 | smem-1d-tiling |
| 关联 SR | SR-SGEMM-OPT |
| 日期 | 2026-10-04 |
| 状态 | Draft（待用户确认） |

> 设计输入：AR002 闭环（预期证据：合并后 sectors/request≈4，但 DRAM 流量仍 ≈ 理论最小值数十倍，
> A/B 重复读取是主要浪费）。全局契约见组件详设 §4。

## 1. 背景与目标

用 shared memory 分块消除 block 内的重复全局读取，并以 1D thread tiling（TM=8）提升单线程计算密度、
摊薄同步开销。本版是"从访存优化走向计算密度优化"的桥梁，也是首次引入 `__syncthreads` 与 smem 布局问题。

## 2. 需求范围

**In Scope：** `sgemm_smem_1d`：32×32 输出分块、BK 扫描（8/16/32 实测取舍）、128 线程/块、每线程 TM=8；
正确性/bench/ncu 闭环。
**Out of Scope：** 2D thread tile（TM×TN，属 AR004）、float4（AR005）、A 转置布局（AR005）。

## 3. 功能需求

### 3.1 Kernel 2：sgemm_smem_1d

**描述：** block 协作将 A/B tile 装入 smem（含边界判断与补零/谓词），`__syncthreads` 后从 smem 累加；
每线程沿一个方向计算 8 个输出。线程映射方案（如 4×32 或 32×4 排布）在 design.md 定稿，
**硬要求：warp 级写回 C 合并**。
**异常处理：** M/N/K 非 tile 倍数时边界谓词正确（1023×1024×511 必须 PASS）。
**验收标准：**
- Given 全测试矩阵，When `make test`，Then smem_1d 全 PASS；
- Given 4096³，When benchmark，Then GFLOPS ≥ 1.29 TFLOPS（目标 ~1.43，±10%）；
- Given `-Xptxas -v`，Then smem/block 与 regs 记录入 CSV。

### 3.2 BK 扫描取舍与 ncu 闭环（#3）

**验收标准：**
- Given BK ∈ {8,16,32} 的实测数据，When 对比，Then design.md/报告记录最优 BK 及理由；
- Given ncu 数据，When 分析，Then bottleneck_analysis.md 追加闭环：① smem bank conflict 计数；
  ② barrier/short scoreboard stall 占比上升证据；③ load:FMA 比例仍高（单方向复用不足）；
  ④ AR004 验证计划（2D 寄存器分块提高每线程 FMA 密度）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 4096³ GFLOPS | 目标 ~1.43 TFLOPS（1.94× vs AR002）；验收下限 ≥ 1.29 TFLOPS |
| 正确性 | 全测试矩阵 | 全 PASS |
| 资源 | smem/block | ≤ 48KB；bank conflict 数值记录（允许非零，AR005 治理） |

## 5. 约束与假设

**约束：** 禁止 2D thread tile 与向量化（职责边界）；BK 取舍须以 4096³ 实测为准。
**假设：** harness 的 kernel 注册机制支持新增 kernel 无需改动核心。
