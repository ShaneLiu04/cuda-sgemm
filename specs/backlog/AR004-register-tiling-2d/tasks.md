# [AR004] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR004 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-04 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | Red：为 2d_tile 扩展测试（边界尺寸谓词路径重点覆盖），确认 Red | - | pending | 标准任务 |
| T002 | Green：实现 sgemm_2d_tile（128×128×8、8×8 外积、smem padding、`__launch_bounds__`、展开/ILP），全矩阵 PASS | T001 | pending | 标准任务 |
| T003 | 资源硬门：`-Xptxas -v` 核对 spill=0；minBlocks 1 vs 2 与 pad 值消融实测，记录取舍 | T002 | pending | spill 非零即缺陷 |
| T004 | benchmark：4096³ 落盘 CSV，核对性能门 ≥ 3.16 TFLOPS | T003 | pending | |
| T005 | ncu 采集 + 闭环 #4：FP32 pipe 利用率、smem 吞吐/conflict、stall 分布对比、AR005 验证计划 | T004 | pending | |

## 状态说明

- `pending` / `in_progress` / `passing` / `failed`

## 进度记录

> 每个开发会话结束后追加。

## 阶段门控记录

> 由 sdd-phase-gate 在门控审查后追加。

### 2026-10-04 · 无 GPU 预构建会话（预构建，未验证）

- `src/sgemm_2d_tile.cu` 已实现：128×128×8、block(32,8)、每线程 8×8 外积、
  PAD_A=5（a-frag 0 冲突推导入注释）/PAD_B=4（b-frag 4-way 刻意保留）、
  `__launch_bounds__(256, lb)` lb∈{1,2} 双实例，`--lb` 运行期选择。
- **待 GPU 验证**：正确性回归、spill=0 硬门（build.log 审计）、E06 lb 消融、
  3.51 TF ±10% 性能门、b-frag conflict 计数实测复核（D7：实测与推导冲突时以实测修订文档）。
  全部状态保持 `pending`。
- design.md 已生成（本目录），含 PAD 推导与寄存器预算。
