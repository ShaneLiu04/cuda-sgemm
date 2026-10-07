# [AR001] 需求设计说明书 — 评测框架 + 环境基线 + cuBLAS 基线 + Naive Kernel

| 字段 | 内容 |
|------|------|
| AR 编号 | AR001 |
| AR 主题 | harness-naive-baseline |
| 关联 SR | SR-SGEMM-OPT（六版 SGEMM 逐层优化总需求） |
| 日期 | 2026-10-04 |
| 状态 | Draft（待用户确认） |

> 全局契约（kernel 签名、计时协议、CSV schema、正确性判据、ncu 指标集）见
> `specs/component-detail-design/cuda_sgemm_spec.md` §4，本 srs 不重复，仅引用。

## 1. 背景与目标

本工程的一切优化结论都必须建立在**可信的测量**之上。本 AR 从零建立统一评测框架（构建、正确性、
计时、落盘、profile），确立环境基线（硬件/驱动/时钟策略）与 cuBLAS FP32 参考线，并实现刻意不优化的
Naive kernel 作为加速比基准（1×）。本 AR 完成后，后续五个 AR 才有可对比、可复现的地基。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- 环境探测与回填（详设 §2 占位符 → `results/environment.md`）；
- `include/common.h` + `Makefile` + `src/main.cu` CLI 路由骨架；
- 正确性测试框架（CPU double 参考 + cuBLAS FP32 参考 + 误差判据 + 测试矩阵，详设 §4.4）；
- Kernel 0：`sgemm_naive`（教科书式，不优化）；
- cuBLAS FP32 基线接入（`CUBLAS_DEFAULT_MATH`，详设 §4.2 含自校验义务）；
- benchmark 落盘 `results/performance.csv`（schema 详设 §4.3），产出 naive 与 cublas 的 4096³ 首批数据；
- ncu 采集脚本 + Naive 首份瓶颈分析（`results/bottleneck_analysis.md` 建档）。

**Out of Scope（本 AR 不做的）：**
- Kernel 1-5 的任何优化（访问合并也不做——Naive 必须保持"教科书式"）；
- 非测试矩阵内的尺寸适配性保证；
- CUDA Graph / 多流。

## 3. 功能需求

### 3.1 环境探测与基线记录

**描述：** 探测硬件与工具链，回填详设 §2 占位符，确立时钟策略。
**触发条件：** AR001 开发启动时。
**期望行为：** `nvidia-smi -q`（含 POWER/CLOCK 段）、`nvcc --version`、`ncu --version`、
`compute-sanitizer --version` 探测；确定并记录时钟策略（优先 `nvidia-smi -lgc`，无权限则稳态预热）。
**异常处理：** 工具缺失时明确报告缺失项与降级方案（如无 ncu 则本 AR 阻塞——ncu 是硬依赖）。
**验收标准：**
- Given 本机 GPU 与工具链，When 运行探测脚本，Then `results/environment.md` 生成，
  含 GPU/TGP/驱动/CUDA/ncu 版本、时钟策略与实测 SM 频率，且详设 §2 占位符已回填。

### 3.2 基础设施（common.h + Makefile + main.cu）

**描述：** 建立 AGENTS.md §2/§3 约定的构建与 CLI 骨架。
**触发条件：** 3.1 完成。
**期望行为：** `make` 编译通过（含 `-Xptxas -v` 日志落 `build.log`）；`--kernel/--m/--n/--k/--warmup/--iters/--check`
参数解析正确；CUDA_CHECK 全覆盖。
**异常处理：** 非法参数/尺寸 ≤ 0 报错退出。
**验收标准：**
- Given 骨架代码，When `make && ./sgemm_bench --help`，Then 编译零警告且 CLI 可用；
- Given 任意非法参数，When 运行，Then 非零退出码与明确错误信息（非崩溃）。

### 3.3 正确性测试框架

**描述：** CPU double 参考、cuBLAS FP32 参考（自校验）、误差判据（≤1e-4 相对判据）、
一键脚本跑全测试矩阵（详设 §4.4）输出 PASS/FAIL 表。
**触发条件：** 3.2 完成（TDD：本条测试先于 naive 实现编写）。
**期望行为：** `make test` 对每个已接入 kernel × 全矩阵运行并汇总。
**异常处理：** 误差超限输出实际 max abs / max rel 值与失败尺寸。
**验收标准：**
- Given cuBLAS 参考实现，When 在 256³ 与 CPU double 参考比对，Then 通过（自校验，详设 §4.2 义务）；
- Given 测试脚本，When naive 尚未实现，Then 测试失败/无法链接（Red 状态可观察）。

### 3.4 Kernel 0：sgemm_naive

**描述：** 一维线程映射，每线程一个 `C[i][j]`，K 维串行累加；**不采取任何优化**。
**触发条件：** 3.3 的 Red 确认。
**期望行为：** 全测试矩阵 PASS；4096³ 可运行不越界。
**验收标准：**
- Given 全测试矩阵，When `make test`，Then naive 全部 PASS；
- Given 4096³，When benchmark（详设 §4.3 协议），Then 产出中位耗时与 GFLOPS 落盘 CSV。

### 3.5 cuBLAS FP32 基线

**描述：** 按详设 §4.2 接入 `cublasSgemm`（含行主序换算与 `CUBLAS_DEFAULT_MATH`）。
**验收标准：**
- Given 256³，When 与 CPU double 比对，Then 误差 ≤ 1e-4（换算正确性证据存档）；
- Given 4096³，When benchmark，Then cuBLAS GFLOPS 落盘，作为后续所有 AR 的 100% 参考线。

### 3.6 Benchmark 与首批数据落盘

**验收标准：**
- Given 4096³ 与时钟策略生效，When `make bench`，Then `results/performance.csv` 追加 naive 与 cublas
  两行完整数据（含频率/温度/功耗/git commit），并报告多轮 RSD。

### 3.7 Naive 首份瓶颈分析（建档）

**描述：** 用 ncu（详设 §4.5 指标集）采集 Naive，建立 `results/bottleneck_analysis.md`。
**验收标准：**
- Given ncu 报告，When 分析，Then 文档包含：① 实测 DRAM 流量 vs 理论最小流量对比数值；
  ② 访存事务效率；③ warp stall 分布（预期 long scoreboard 主导）；
  ④ "瓶颈 → 证据 → 对策 → AR002 验证计划"闭环段落。

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | Naive 4096³ GFLOPS | 目标 ~113.55；验收下限 ≥ 102（±10% 条款，详设 §7） |
| 性能 | cuBLAS 4096³ GFLOPS | 实测并记录（预期 ~6.71 TFLOPS 量级，以本机为准） |
| 计时质量 | iters ≥ 100 的 RSD | RSD ≤ 5%，否则报告频率曲线并归因 |
| 正确性 | 全测试矩阵 | 全 PASS，误差 ≤ 1e-4 |
| 资源 | 寄存器/smem | 记录到 CSV（无硬性上限） |
| 复现 | 数据可复现 | CSV 行含 git commit；README 章节骨架建立 |

## 5. 约束与假设

**约束：**
- 遵守 AGENTS.md 全部军规；编译禁止 `-use_fast_math`；计时只认 CUDA events；
- Naive 保持无优化状态（这是加速比基准的完整性要求，不得"顺手优化"）。

**假设：**
- 本机 ncu / compute-sanitizer 可用；`nvidia-smi -lgc` 若无权限则采用稳态预热策略并声明。

## 6. 术语说明

见组件详设 §9。
