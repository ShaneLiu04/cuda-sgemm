# [AR003] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR003 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-04 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | Red：为 smem_1d 扩展测试（重点：边界尺寸的 tile 谓词路径），确认 Red | - | pending | 标准任务 |
| T002 | Green：实现 sgemm_smem_1d（32×32 tile、TM=8、128 线程、BK 扫描框架、warp 合并写回），全矩阵 PASS | T001 | pending | 标准任务 |
| T003 | BK ∈ {8,16,32} 消融实测，定稿最优 BK 并记录理由 | T002 | pending | 数据入报告 |
| T004 | benchmark：4096³ 落盘 CSV，核对性能门 ≥ 1.29 TFLOPS | T003 | pending | |
| T005 | ncu 采集 + 闭环 #3：bank conflict 计数、barrier/short scoreboard 占比、load:FMA 比、AR004 验证计划 | T004 | pending | |

## 状态说明

- `pending` / `in_progress` / `passing` / `failed`

## 进度记录

> 每个开发会话结束后追加。

## 阶段门控记录

> 由 sdd-phase-gate 在门控审查后追加。

### 2026-10-04 · 无 GPU 预构建会话（预构建，未验证）

- `src/sgemm_smem_1d.cu` 已实现：`template<int BK>` 单实现三实例（8/16/32），
  运行期 `--bk` 分发；block(8,4)、TM=8、smem sA[32][BK]/sB[BK][32]；
  K 尾块与 M/N 越界守护写回已覆盖。
- **待 GPU 验证**：正确性回归、E05 BK 扫描（默认 BK=16 待实测定稿）、
  1.43 TF ±10% 性能门、ncu bank conflict "Before" 基线记录。全部状态保持 `pending`。
- design.md 已生成（本目录）。
