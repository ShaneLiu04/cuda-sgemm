# [AR007] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR007 |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-05 |

## 测试用例列表

### ST-001：swpipe 全尺寸正确性

**关联需求：** srs.md §3.1（验收：9 kernel × 9 尺寸 + 2 自校验全 PASS）
**测试类型：** 正常路径　**优先级：** High

**前置条件：** AR007 最终源码构建成功（含 review 修复后的 sgemm_smem_1d.cu 注释修订）
**测试步骤：** 1. `cmake --build build -j`（-Xptxas -v 全程输出）2. `build\sgemm_test.exe`
**期望结果：** 83/83 PASS，exit=0
**实际结果：** 2026-10-05 ST 会话重新构建 + 复跑：**83/83 PASS**（swpipe 主场景 4096³ rel=0.000e+00，与 vec4 同累加序；9 尺寸含边界全 PASS）
**状态：** PASS

---

### ST-002：swpipe 非对齐回退路径

**关联需求：** srs.md §3.1 异常处理（非对齐尺寸回退 tile2d，--verbose 打印路径）
**测试类型：** 异常处理　**优先级：** High

**前置条件：** swpipe 已注册，N%4!=0 / K%4!=0 用例在套件内
**测试步骤：** 1. 套件内 1023×1024×511、130×257×66、17×33×65、1×1×1、64×64×1 用例 2. `--kernel swpipe --m 130 --n 257 --k 66 --verbose` 路径打印抽查（review 子 Agent 已执行）
**期望结果：** 回退路径结果正确（vs CPU double rel ≤ 1e-4），verbose 打印 "scalar fallback (tile2d)"
**实际结果：** ST-001 套件内上述 5 用例全 PASS（vs CPU 参考交叉验证）；review 阶段 verbose 打印已确认
**状态：** PASS

---

### ST-003：G-K6 门（swpipe > vec4）

**关联需求：** srs.md §3.1 验收 / §1 四门之 G-K6
**测试类型：** 业务场景（性能门）　**优先级：** High

**前置条件：** AR007 矩阵会话 CSV 落盘（同会话背靠背测量）
**测试步骤：** 1. `python bench\compare.py <基线快照> results\performance.csv` 2. 核对 G-K6 表
**期望结果：** 6 尺寸 swpipe vs vec4 判定输出（正/负结果均如实记录）
**实际结果：** **6/6 全胜**（256³ +4.2% / 512³ +5.6% / 1000×1016×1024 +0.9% / 1024³ +0.9% / 2048³ +0.7% / 4096³ +0.1%）→ 判定 PASS；ST 会话复跑 compare.py 输出与归档 compare_ar007.md 逐值一致
**状态：** PASS

---

### ST-004：--rounds 多轮统计与 RSD 门控

**关联需求：** srs.md §3.2（跨轮 RSD>5% 自动追加轮次 ≤3 次，聚合轮间 median）
**测试类型：** 正常路径　**优先级：** High

**前置条件：** bench 二进制含 MultiRoundStats 路由
**测试步骤：** 1. `build\sgemm_bench.exe --kernel swpipe --m 1024 --n 1024 --k 1024 --rounds 3 --warmup 20 --iters 100 --check`
**期望结果：** 逐轮 median 与跨轮 RSD 报告；RSD>5% 触发重试（≤3 次）；最终值取轮间 median
**实际结果：** ST 会话现场执行：correctness PASS；rounds 3 → 首轮冷启动抬高跨轮 RSD 至 8.80% → 门控重试至 6 轮，最终 median 0.3838 ms（5595.6 GF，GPU 升至 1950 MHz 后稳定）；per-round medians 与 RSD 全部如实打印。rounds=1 逐位可比性已在 T004 实测记录
**状态：** PASS

---

### ST-005：--rounds 0 非法值拒绝

**关联需求：** srs.md §3.2 / 详设 §4.6（0 → CLI_ERROR）
**测试类型：** 异常处理　**优先级：** Medium

**测试步骤：** 1. `build\sgemm_bench.exe --kernel swpipe --rounds 0` 2. 检查退出码
**期望结果：** `[CLI_ERROR]` 且非零退出码
**实际结果：** ST 会话现场执行：`[CLI_ERROR] --rounds must be >= 1 (got 0)`，exit code = 1
**状态：** PASS

---

### ST-006：自动实验矩阵一键执行

**关联需求：** srs.md §3.3a（run_matrix.ps1：9 kernel × 6 尺寸 + 消融分流）
**测试类型：** 正常路径　**优先级：** High

**测试步骤：** 1. `bench/run_matrix.ps1`（T007 已执行，产物核验）
**期望结果：** 54 cells 追加主 CSV；消融数据分流独立文件；自动基线快照
**实际结果：** T007 执行记录：54 cells（全 RSD≤1.7%）+ 消融 4 行入 ablation_ar007.csv（SGEMM_CSV 分流）+ 基线快照 performance_preAR007_20261005_095539.csv 生成；主 CSV 未被消融污染（时间戳 2026-10-05 09:55–10:05 矩阵行）
**状态：** PASS

---

### ST-007：回归对比 delta 表与手工核算一致

**关联需求：** srs.md §3.3b（compare.py delta 表与手工核算一致）
**测试类型：** 正常路径　**优先级：** High

**测试步骤：** 1. 复跑 `python bench\compare.py results\performance_preAR007_20261005_095539.csv results\performance.csv`
**期望结果：** 输出完整 delta 表（每 kernel×尺寸：新旧 GFLOPS、变化 %、领先标记）+ 四门判定，与归档一致
**实际结果：** ST 会话复跑输出与 results/compare_ar007.md 逐值一致（含 G-全线 1/6 明细表、G-K6 6/6 WIN 表、基线/新会话 git 与默认值差异备注）；review 子 Agent 已抽查 delta 与首行/末行取数语义核算一致
**状态：** PASS

---

### ST-008：默认参数固化与文档一致性

**关联需求：** srs.md §3.4（BK 默认 32 + 详设 §4.6/CLI 文档/AGENTS 消融旋钮同步）
**测试类型：** 正常路径　**优先级：** Medium

**测试步骤：** 1. 核对 g_smem1d_bk 源码值 2. 核对 CLI usage / 详设 §4.6 表 / AGENTS.md §3 消融旋钮段 3. 默认参数下全套件运行
**期望结果：** 默认 32 生效且四处文档一致；全 PASS
**实际结果：** `src/sgemm_smem_1d.cu:27` g_smem1d_bk=32；头注释/CLI usage/详设 §4.6（2026-10-05 review 修复补建）/AGENTS.md §3 均已同步；ST-001 默认构建 83/83 PASS（smem1d 三实例 8/16/32 均编译，运行时选 32）
**状态：** PASS

---

### ST-009：性能四门集中判定（含"超越 cuBLAS"可达性回报）

**关联需求：** srs.md §1 落实方式 + §4 NFR + design §6.3
**测试类型：** 业务场景（性能门）　**优先级：** High

**测试步骤：** 1. 以 compare_ar007.md + cool_probe_ar007.csv 判定四门
**期望结果：** 逐门判定成立或负结果如实归档
**实际结果：**
| 门 | 判定 | 数据 |
|----|------|------|
| G-小尺寸 | **PASS** | 256³ smem1d(bk32) 1618.2 > cuBLAS 1377.9（**117.4%**，绝对反超）；力争项 512³ FAIL（2934.6 < 6558.7，如实记录） |
| G-大尺寸 | **FAIL** | 4096³ swpipe 热态 6306.5 / 冷态探针 6584.8 GF = cuBLAS 冷态 10136.2（本机历史最高）的 64.9%，< 7.2 TF 且 < 75% 门 |
| G-全线 | **FAIL** | 仅 1/6 尺寸刷新 ≥2%（256³ +28.4%）；归因：10 分钟热浸没有会话系统性降频 ~5%（尾段 1815–1860 MHz vs 单跑 1875–1950），cuBLAS 自身同会话 -2～-12% |
| G-K6 | **PASS** | 6/6 尺寸全胜 vec4（ST-003） |
**可达性论证回执（srs §1 要求 ST 显式回报）**：严格 FP32 军规下（禁 TF32/tensor core/快速 intrinsic），NVCC/ptxas 边界内手写 kernel 现实上限 7–8.5 TF；实测冷态 6584.8 GF 已逼近该带下沿（差距构成见 report.md §7：SASS 级调度/更深 K 向 ILP/功耗墙）。"全尺寸超越 cuBLAS"须解除 FP32 军规（TF32/分割精度），与工程身份冲突，**未获用户授权**——若授权另开 AR。
**状态：** PASS（判定完成：2/4 门达标；2 门负结果已如实归档并附归因与论证，符合数据真实性军规；最终裁定提交用户 Go/No-Go）

---

### ST-010：ptxas 资源审计（0 spill 硬门）

**关联需求：** srs.md §4 资源行 / design §6.4 资源异常
**测试类型：** 边界条件　**优先级：** High

**测试步骤：** 1. 检查 ST-001 构建的 -Xptxas -v 输出（已归档 build.log）
**期望结果：** 全 kernel 0 spill；K6 占用/spill 取舍有消融记录
**实际结果：** fresh build + 归档 build.log：naive/coalesced 50r、smem1d 72r、tile2d 114r、vec4 128r、cpasync 139r(20KB)、cpasync2 125r(16.6KB)、swpipe **127r(LB=1)/128r(LB=2)、8320B smem，全部 0 spill**；LB 消融记录于 T003（0.3% 并列，lb=1 定稿）
**状态：** PASS

---

### ST-011：严格 FP32 精度军规

**关联需求：** srs.md §4 精度行
**测试类型：** 边界条件　**优先级：** High

**测试步骤：** 1. 核对构建链无 -use_fast_math 2. 核对 cuBLAS DEFAULT_MATH 3. 数值量级交叉验证
**期望结果：** 全程严格 FP32
**实际结果：** CMakeLists 无 fast-math（有显式禁用声明）；cuBLAS CUBLAS_DEFAULT_MATH（详设 §4.2 契约）；ST-001 全部 rel ≈ 1e-6~1e-7（TF32 会是 ~1e-3 量级）→ 未发生精度降级
**状态：** PASS

---

### ST-012：数据可复现链

**关联需求：** srs.md §4 可复现行
**测试类型：** 边界条件　**优先级：** Medium

**测试步骤：** 1. 抽查 CSV 行字段 2. 核对基线快照机制
**期望结果：** 逐行 git sha + GPU 状态（时钟/温度/功耗）；对比实验同会话背靠背
**实际结果：** performance.csv 14 字段 schema（含 gpu_state 三元组与 git_sha）；四门判定数据全部来自同会话矩阵 + 冷态探针；基线快照独立留存。**已知缺口**：AR002–AR007 代码未提交（HEAD=84e261f），新 CSV 行 sha 指向 AR001 代码——复现链在提交前不闭合（提交处置列入 Go/No-Go 询问）
**状态：** PASS（机制完备；提交缺口列为遗留事项）

---

### ST-013：K6 racecheck

**关联需求：** srs.md §4 正确性行 / design §6.4（单缓冲写读交替 0 hazards）
**测试类型：** 异常处理　**优先级：** High

**测试步骤：** 1. `compute-sanitizer --tool racecheck` 抽查 swpipe（T002 已执行于 512³）
**期望结果：** 0 hazards
**实际结果：** T002 执行记录：swpipe@512³ racecheck 0 hazards（单缓冲双 __syncthreads 写读隔离验证）
**状态：** PASS

---

### ST-014：边界退化守卫

**关联需求：** design §6.4（K=1 / 1×1×1 / 17×33×65 对 swpipe 的守卫路径，零填充不越界）
**测试类型：** 边界条件　**优先级：** High

**测试步骤：** 1. ST-001 套件内对应用例
**期望结果：** 守卫路径正确，无越界（sanitizer memcheck 干净）
**实际结果：** 套件内 64×64×1、1×1×1、17×33×65 全 PASS（vs CPU double 交叉验证）；memcheck 干净（T002）
**状态：** PASS

---

### ST-015：WDDM 抖动下 cuBLAS 多轮门控行为

**关联需求：** srs.md §3.2 验收 / design §6.4 第一条
**测试类型：** 异常处理　**优先级：** Medium

**测试步骤：** 1. cuBLAS@4096³ --rounds 3（T004 已执行）2. swpipe 同机制现场复现（ST-004）
**期望结果：** 跨轮 RSD 报告，最终值稳定性优于单轮（或如实报告不可稳定及原因）
**实际结果：** T004 实测：cuBLAS --rounds 3 门控重试 3→6 轮，RSD 如实报告；ST-004 现场复现同机制（冷启动轮识别与门控重试正确触发）。WDDM 无 admin 锁频下轮间漂移已归因记录（environment.md §6）
**状态：** PASS

## 执行摘要

| 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|
| 15 | 15 | 0 | 0 |

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-05 |
| 执行结果 | PASS（性能四门 2/4 达标，负结果如实归档，提交用户裁定） |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 | K6 软件流水 kernel | ST-001, ST-002, ST-003 | PASS |
| §3.2 | 多轮统计与 RSD 门控 | ST-004, ST-005, ST-015 | PASS |
| §3.3 | 自动矩阵 + 回归对比 | ST-006, ST-007 | PASS |
| §3.4 | 参数固化与 spec 修订 | ST-008 | PASS |
| §4-NFR | 四门 / 正确性 / 资源 / 精度 / 可复现 | ST-009, ST-010, ST-011, ST-012, ST-013, ST-014 | PASS（四门内部 2/4 达标，见 ST-009） |

**需求覆盖率：** 4/4 功能需求（100%）；NFR 5 项全覆盖

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 7 | 7 | 0 | 0 |
| 边界条件 | 4 | 4 | 0 | 0 |
| 异常处理 | 4 | 4 | 0 | 0 |
| 回归测试 | 83 例套件（含于 ST-001/002/014） | 83 | 0 | 0 |
| **合计（ST 用例）** | **15** | **15** | **0** | **0** |

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | G-大尺寸 / G-全线 两门未达标（负结果 + 归因 + 可达性论证完备，见 ST-009） | 用户裁定（解除 FP32 军规另开 AR / 接受现状） |
| Minor | AR002–AR007 代码与数据未提交，CSV git_sha 复现链不闭合（ST-012） | 用户裁定提交时机（AGENTS §8 要求任务完成即提交） |
| Minor | ncu 硬件计数器 ERR_NVGPUCTRPERM（无 admin）持续阻塞 | 待权限；解锁步骤已记录 bottleneck_analysis.md 尾节 |
| Cosmetic | fig1–fig8 PNG 视觉效果需用户抽查（模型无图像预览能力） | 用户抽查 |

### 结论

> **建议 Conditional-Go**：15/15 ST 用例 PASS、需求覆盖 100%、无 Critical/Major 缺陷、83/83 回归无回归；性能四门 2/4 达标——G-小尺寸（256³ 绝对反超 cuBLAS 117.4%）与 G-K6（6/6 全胜）达成，G-大尺寸/G-全线 为如实归档的负结果（严格 FP32 边界 + 热浸没降频，论证回执完备）。srs §1 明示该偏差需 ST 时显式回报并由用户裁定。

**用户裁定（2026-10-05）**：Conditional-Go 确认——AR007 归档；G-大尺寸/G-全线 负结果接受为最终结论（维持严格 FP32 军规，未授权 TF32 路径）；git 提交按用户指示暂缓（完全不提交，工作树 dirty 状态与 CSV git_sha 复现链缺口延续，责任由用户承接）。
