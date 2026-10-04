# [AR006] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR006 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-04 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | Red：为 cpasync 扩展测试（任意尺寸 + 边界回退专项），确认 Red | - | pending | 标准任务 |
| T002 | Green：实现 sgemm_cpasync（双缓冲、commit/wait 配合、谓词化边界），全矩阵 PASS | T001 | pending | 标准任务 |
| T003 | 布局消融：方案甲（全直拷+padding）vs 方案乙（B 直拷 + A float4 转置混合）4096³ 实测，定稿并记录数据 | T002 | pending | 严禁无数据选择 |
| T004 | racecheck 抽查（主场景 + 边界尺寸）+ memcheck 全量；竞争逐条归因存档 | T003 | pending | |
| T005 | benchmark 终验：4096³ 全 kernel 阶梯 + cuBLAS 同会话复测，核对硬门（≥6.3 TFLOPS 且 ≥97% cuBLAS）；不达标走归因流程 | T004 | pending | |
| T006 | ncu 终态闭环 #6：DRAM 吞吐 vs 峰值、六版 stall 演化表、总闭环表，收口 bottleneck_analysis.md | T005 | pending | |
| T007 | 工程收口：性能阶梯图（可重生成脚本）、README 复现命令与摘要表、environment.md 收口、PROMPT §5 清单逐项勾选 + 证据指针 | T006 | pending | |

## 状态说明

- `pending` / `in_progress` / `passing` / `failed`

## 进度记录

> 每个开发会话结束后追加。

## 阶段门控记录

> 由 sdd-phase-gate 在门控审查后追加。

### 2026-10-04 · 无 GPU 预构建会话（预构建，未验证）

- `src/sgemm_cpasync.cu` 已实现：双方案共存——
  方案甲（`cpasync`）：A/B 全 cp.async 直拷（A PAD=4，a-frag 2-way 冲突为已知保留项）+ B XOR swizzle；
  方案乙（`cpasync2`）：A float4+转置 smem+寄存器预取一拍，B cp.async+swizzle。
  2-stage 双缓冲骨架：issue(t+1)→commit→wait_prior(1)（末 tile wait_prior(0)）→双 syncthreads；
  回退谓词同 vec4。
- **已应用修复**：zfill=16 边界 quad 的 src 指针越界问题——统一传全局矩阵基地址保证可寻址
  （zfill=size 时源不被读取，仅要求指针合法）。
- **待 GPU 验证**：正确性回归 + 回退断言、racecheck 硬门（两方案 × {4096³, 1023×1024×511}）、
  memcheck 边界、E08 甲/乙消融定稿、E14 流水收益分解、6.61 TF ±10% + ≥97% cuBLAS 终验门。
  全部状态保持 `pending`；败者方案代码保留（军规：失败实验同样记录）。
- design.md 已生成（本目录），含"cp.async 不能转置"的方案取舍论证。
