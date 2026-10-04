# [AR005] 需求设计说明书 — Kernel 4：float4 向量化

| 字段 | 内容 |
|------|------|
| AR 编号 | AR005 |
| AR 主题 | float4-vectorization |
| 关联 SR | SR-SGEMM-OPT |
| 日期 | 2026-10-04 |
| 状态 | Draft（待用户确认） |

> 设计输入：AR004 闭环（预期证据：标量访存指令数多、global→smem 搬运吞吐受限、部分 conflict 残留）。

## 1. 背景与目标

在 AR004 的 2D 分块骨架上做三处 16B 向量化：① global→smem 加载用 float4；② A tile 以转置布局写入
smem（`[BK][BM]` 侧），配合 padding/swizzle 同时消除"转置写"与"片段读"的 bank conflict，
使内层 `a[8]` 读取可 2×float4；③ C 回写每线程 8 个连续 N 输出 = 2×float4，warp 级合并。

## 2. 需求范围

**In Scope：** `sgemm_vec4`（三处向量化 + 非 4 倍数尺寸的谓词化回退路径）；正确性（含回退路径专项用例）/
bench/ncu 闭环。
**Out of Scope：** cp.async 与双缓冲（AR006）；转置写与直拷的最终取舍若与 AR006 冲突，可在 AR006 复审。

## 3. 功能需求

### 3.1 Kernel 4：sgemm_vec4

**描述：** 主路径假设 M/N/K 及 leading dimension 满足 16B 对齐（入口断言）；不满足走谓词化标量回退。
**异常处理：** 1000×1016×1024 与 1023×1024×511 必须 PASS（后者应触发回退路径，测试须断言路径覆盖）。
**验收标准：**
- Given 全测试矩阵（含回退路径专项用例），When `make test`，Then vec4 全 PASS；
- Given 4096³，When benchmark，Then GFLOPS ≥ 5.26 TFLOPS（目标 ~5.84，±10%）。

### 3.2 ncu 闭环（#5）

**验收标准：**
- Given ncu 数据，When 分析，Then bottleneck_analysis.md 追加闭环：① global load/store sectors 效率
  与指令数下降对比（vs AR004）；② bank conflict 计数 ≈ 0（转置写 + 片段读两处分别核对）；
  ③ 吞吐提升归因；④ 剩余瓶颈（同步串扰、无预取重叠、load-use 距离不足）→ AR006 验证计划（cp.async 双缓冲）。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 4096³ GFLOPS | 目标 ~5.84 TFLOPS（1.66× vs AR004）；验收下限 ≥ 5.26 TFLOPS |
| 资源 | bank conflict | smem ld/st conflict ≈ 0（ncu 计数为准，非零须归因） |
| 资源 | spill | 0（延续 AR004 硬门） |
| 正确性 | 全测试矩阵 + 回退路径 | 全 PASS |

## 5. 约束与假设

**约束：** 16B 对齐假设必须显式断言；padding/swizzle 取舍以 ncu 计数为准（AGENTS.md §6/D7）。
**假设：** AR004 骨架的累加器/外积结构不变，仅改数据搬运与布局。
