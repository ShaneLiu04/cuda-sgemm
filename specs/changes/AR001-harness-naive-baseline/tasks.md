# [AR001] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR001 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-04 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | 环境探测：GPU/TGP/驱动/CUDA/ncu/compute-sanitizer 版本 + 时钟策略确定，回填详设 §2 占位符，生成 results/environment.md | - | pending | srs §3.1 |
| T002 | 基础设施：git init + include/common.h（CUDA_CHECK/CLI/事件计时/GFLOPS）+ Makefile + src/main.cu 路由骨架，build.log 落盘 | T001 | pending | srs §3.2 |
| T003 | Red：正确性测试框架——CPU double 参考 + cuBLAS FP32 参考（§4.2 自校验）+ 误差判据 + 全矩阵一键脚本；确认 naive 接入前测试为 Red | T002 | pending | srs §3.3；标准任务 |
| T004 | Green：实现 sgemm_naive（教科书式，零优化）并通过全矩阵测试 | T003 | pending | srs §3.4；标准任务 |
| T005 | cuBLAS FP32 基线接入（行主序换算 + CUBLAS_DEFAULT_MATH），与 CPU double 交叉验证并存档证据 | T003 | pending | srs §3.5 |
| T006 | benchmark 链路：4096³ 实测 naive + cublas，落盘 performance.csv（含频率/温度/功耗/commit），报告 RSD | T004, T005 | pending | srs §3.6 |
| T007 | ncu 采集 naive（详设 §4.5 指标集）→ profile/naive/ 归档；创建 results/bottleneck_analysis.md 首份闭环（DRAM 流量/事务效率/stall 分布 → AR002 验证计划） | T006 | pending | srs §3.7 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。

### 2026-10-04 · 无 GPU 预构建会话（预构建，未验证）

- **代码侧预完成**：T002–T005 的全部实现产物已在暂存目录写毕并搬运至工程根——
  `include/common.h`（CUDA_CHECK/CLI/events 计时/GFLOPS/CSV/GPU 状态）、
  `include/sgemm_kernels.h`（8 kernel 注册表 + 消融旋钮）、`src/main.cu`（CLI 路由/环境头/--check/CSV）、
  `src/sgemm_naive.cu`（刻意不合并映射）、`src/cublas_baseline.cu`（行主序换算 + DEFAULT_MATH）、
  `tests/test_correctness.cu`（CPU double + cuBLAS 双参考，9 尺寸 × 8 kernel）、
  `CMakeLists.txt`/`Makefile`/`bench/`/`profile/`（sh+ps1 双平台）。
- **T003 Red 说明**：无 nvcc 环境无法运行 Red；测试框架与 kernel 已同步写毕，
  Red/Green 顺序在 GPU 环境以"注释掉 kernel 接入"方式补验（见 docs/TEST_PLAN.md §3）。
- **无法执行项**：T001（环境探测）、T006（实测落盘）、T007（ncu 首份闭环）依赖 GPU。
- **状态纪律**：依数据真实性军规，全部任务保持 `pending`，代码状态为"预构建待验证"；
  GPU 环境执行顺序 = docs/TEST_PLAN.md §1（环境预检）→ §2（构建+资源审计）→ §3（正确性）。
- design.md 已生成（本目录），其中记录了 events 计时、行主序换算等实现决策。
