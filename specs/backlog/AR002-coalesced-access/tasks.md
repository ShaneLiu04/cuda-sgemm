# [AR002] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR002 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md |
| 创建日期 | 2026-10-04 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | Red：为 coalesced 扩展正确性测试（含边界尺寸 1023×1024×511），确认 Red | - | pending | 标准任务 |
| T002 | Green：实现 sgemm_coalesced（warp 连续读 B/写 C 的线程映射），全矩阵 PASS | T001 | pending | 标准任务 |
| T003 | benchmark：4096³ 实测落盘 CSV，核对性能门 ≥ 666 GFLOPS，报告 vs_naive/vs_cublas | T002 | pending | 不达标→调映射重测并记录 |
| T004 | ncu 采集 + 闭环 #2：sectors/request 对比、DRAM 流量 vs 理论最小值、stall 变化、AR003 验证计划，追加 bottleneck_analysis.md | T003 | pending | |

## 状态说明

- `pending` / `in_progress` / `passing` / `failed`

## 进度记录

> 每个开发会话结束后追加。

## 阶段门控记录

> 由 sdd-phase-gate 在门控审查后追加。

### 2026-10-04 · 无 GPU 预构建会话（预构建，未验证）

- `src/sgemm_coalesced.cu` 已实现：与 naive 完全同 block(16,16) 形状，仅交换
  row/col→threadIdx.x/y 映射（唯一变量原则，保证 E3 的 A/B 可比性）。
- 索引统一 `(long long)` 强转，为 8192³ 尺寸扫描留余量。
- **待 GPU 验证**：全矩阵正确性回归、E3（sectors/request 对比）、E04（流量比不变证据）、
  740 GF ±10% 性能门。全部状态保持 `pending`。
- design.md 已生成（本目录）。
