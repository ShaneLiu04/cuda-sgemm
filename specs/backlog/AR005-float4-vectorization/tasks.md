# [AR005] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR005 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-04 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | Red：为 vec4 扩展测试，新增回退路径专项用例（断言非对齐尺寸走谓词路径且结果正确），确认 Red | - | pending | 标准任务 |
| T002 | Green：实现 sgemm_vec4 三处向量化（global float4 加载 / A 转置 smem 写入 + padding/swizzle / C 2×float4 回写）+ 谓词回退，全矩阵 PASS | T001 | pending | 标准任务 |
| T003 | benchmark：4096³ 落盘 CSV，核对性能门 ≥ 5.26 TFLOPS | T002 | pending | |
| T004 | ncu 采集 + 闭环 #5：sectors 效率与指令数对比（vs AR004）、bank conflict ≈ 0 核对、吞吐归因、AR006 验证计划 | T003 | pending | |

## 状态说明

- `pending` / `in_progress` / `passing` / `failed`

## 进度记录

> 每个开发会话结束后追加。

## 阶段门控记录

> 由 sdd-phase-gate 在门控审查后追加。

### 2026-10-04 · 无 GPU 预构建会话（预构建，未验证）

- `src/sgemm_vec4.cu` 已实现：三处 float4（全局读 A/B、smem→寄存器 Frag8、C 写回）、
  A 转置 smem `[BK][BM+4]`、B XOR swizzle（physical = logical ^ (kk&7)）、
  主路径谓词 `N%4==0 && K%4==0 && 16B 对齐`，否则回退 `sgemm_2d_tile`（--verbose 打印）。
- **待 GPU 验证**：正确性回归 + 回退路径覆盖断言（1023×1024×511 / 130×257×66 必须走回退）、
  E07 swizzle 消融（conflict≈0 门）、5.84 TF ±10% 性能门。全部状态保持 `pending`。
- design.md 已生成（本目录），含转置写代价与 swizzle 公式的取舍论证。
