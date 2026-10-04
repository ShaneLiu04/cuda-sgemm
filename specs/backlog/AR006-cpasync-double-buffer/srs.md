# [AR006] 需求设计说明书 — Kernel 5：cp.async 双缓冲流水 + 工程终验

| 字段 | 内容 |
|------|------|
| AR 编号 | AR006 |
| AR 主题 | cpasync-double-buffer |
| 关联 SR | SR-SGEMM-OPT |
| 日期 | 2026-10-04 |
| 状态 | Draft（待用户确认） |

> 设计输入：AR005 闭环（预期证据：同步边界使全局加载与计算无法重叠、load-use 距离不足）。
> 本 AR 兼具**工程收口**职责：性能阶梯图、README、最终验收报告。

## 1. 背景与目标

用 sm_89 的 `cp.async`（`cuda::memcpy_async`/pipeline 原语或内联 PTX `cp.async.cg.shared.global [smem],[gmem],16`）
实现 global→smem 直拷（不占寄存器、cg 路径不污染 L1），并以 **2-stage 双缓冲**在计算 tile k 时预取
tile k+1，消除加载-计算串行。终态目标：**6.61 TFLOPS ≈ 58.24× vs Naive ≈ cuBLAS FP32 实测的 98.52%**，
验收下限 ≥ 6.3 TFLOPS 且 ≥ 97% × 本机实测 cuBLAS。

## 2. 需求范围

**In Scope：** `sgemm_cpasync`（双缓冲 + 谓词化边界 + 布局消融）；racecheck 抽查；最终验收
（全 kernel 全矩阵回归 + 完整性能阶梯 + 阶梯图 + README + 环境报告收口 + AR 归档）。
**Out of Scope：** 3+ 级流水、CUDA Graph、多流（记入未来 AR 候选）。

## 3. 功能需求

### 3.1 Kernel 5：sgemm_cpasync

**描述：** `commit_group`/`wait_group`（或 `__pipeline_commit`/`__pipeline_wait_prior`）与双缓冲交替
正确配合：预取 k+1 → 计算 k → 同步交换；边界 tile 谓词化（cp.async 4B/8B 变体或 smem 补零方案，
design.md 定稿）。**cp.async 不能转置**：A/B 布局须做消融——
方案甲：A/B 均 cp.async 直拷 + padding/swizzle 消冲突；方案乙：B 用 cp.async、A 保留 float4+转置写入
混合。两方案 4096³ 实测，取最优并记录数据与理由。
**异常处理：** 任意 M/N/K 正确；1023×1024×511 走回退且 PASS。
**验收标准：**
- Given 全测试矩阵，When `make test`，Then cpasync 全 PASS（含全 kernel 终验回归）；
- Given 4096³，When benchmark，Then GFLOPS ≥ 6.3 TFLOPS 且 ≥ 97% × 本机 cuBLAS FP32 实测；
  达到 6.61/98.52% 记录为达标，未达 6.61 但过 97% 门：验收通过 + 差距归因（频率/温度/调参空间）。

### 3.2 数据竞争验证

**验收标准：**
- Given 双缓冲 kernel，When `compute-sanitizer --tool racecheck` 抽查（至少主场景 + 一个边界尺寸），
  Then 无竞争报告（或报告为已知良性模式并逐条归因存档）。

### 3.3 ncu 终态闭环（#6）

**验收标准：**
- Given ncu 数据，When 分析，Then bottleneck_analysis.md 收口：① DRAM 吞吐逼近峰值的证据；
  ② stall 主因从 long scoreboard 转为 barrier/依赖等待的对比（跨六版 stall 演化表）；
  ③ 六版"瓶颈→证据→对策→验证"总闭环表。

### 3.4 工程收口交付

**验收标准：**
- Given 全部实测数据，When 汇总，Then 交付：① `results/performance.csv` 完整阶梯（六 kernel + cuBLAS，
  4096³，含全部元数据列）；② 性能阶梯图（GFLOPS vs 版本，含 cuBLAS 参考线，脚本可重生成）；
  ③ `README.md`（一键 build/test/bench/profile 复现命令 + 结果摘要表 + 时钟策略声明）；
  ④ `results/environment.md` 收口（频率曲线/方差）；⑤ PROMPT.md §5 全局验收清单逐项勾选并附证据指针。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 4096³ GFLOPS | 目标 ~6.61（+13.3% vs AR005；58.24× vs Naive）；硬门 ≥ 6.3 且 ≥ 97% cuBLAS 实测 |
| 资源 | spill / conflict | spill = 0；conflict 以实测归因（消融后最优方案） |
| 可靠性 | racecheck | 无未归因竞争 |
| 复现 | README | 同型号机器可复现全部结果 |

## 5. 约束与假设

**约束：** cp.async 16B 拷贝要求 16B 对齐；谓词路径不得拖慢主路径（对齐主场景不进回退）；
消融必须两方案都有 4096³ 实测数据，禁止凭直觉选择。
**假设：** AR001 时钟策略持续有效；cuBLAS 参考线复用 AR001 实测（同会话复测一次以防环境漂移）。
