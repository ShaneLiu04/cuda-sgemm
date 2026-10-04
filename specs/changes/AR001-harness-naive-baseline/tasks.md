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
| T001 | 环境探测：GPU/TGP/驱动/CUDA/ncu/compute-sanitizer 版本 + 时钟策略确定，回填详设 §2 占位符，生成 results/environment.md | - | passing | 2026-10-04 完成；**GPU 偏差：Quadro RTX 5000 (sm_75) 替代 RTX 4060 (sm_89)**，用户批准继续，绝对门改相对门 |
| T002 | 基础设施：git init + include/common.h（CUDA_CHECK/CLI/事件计时/GFLOPS）+ Makefile + src/main.cu 路由骨架，build.log 落盘 | T001 | passing | 2026-10-04 构建闭环；免管理员工具链组装（environment.md §4）；全 kernel 0 spill |
| T003 | Red：正确性测试框架——CPU double 参考 + cuBLAS FP32 参考（§4.2 自校验）+ 误差判据 + 全矩阵一键脚本；确认 naive 接入前测试为 Red | T002 | passing | 2026-10-04 Red 补验：注释 naive 注册 → 65/74 FAILED / exit 1（可观察 Red） |
| T004 | Green：实现 sgemm_naive（教科书式，零优化）并通过全矩阵测试 | T003 | passing | 2026-10-04：74/74 ALL PASS（8 kernel × 9 尺寸 + 2 自校验），rel≤1e-4；证据 results/test_correctness_2026-10-04.log |
| T005 | cuBLAS FP32 基线接入（行主序换算 + CUBLAS_DEFAULT_MATH），与 CPU double 交叉验证并存档证据 | T003 | passing | 2026-10-04：行主序自校验 256³/100×253×61 双 PASS（rel≤1.3e-07）；DEFAULT_MATH 生效（vs CPU 误差量级证实非 TF32） |
| T006 | benchmark 链路：4096³ 实测 naive + cublas，落盘 performance.csv（含频率/温度/功耗/commit），报告 RSD | T004, T005 | passing | 2026-10-04：naive 155.22 GF（RSD 0.5%✓）/ cublas 9.64 TF（RSD 8.8–9.6% 超 5% 门，已归因功耗墙+WDDM 无锁频并如实记录，跨轮 median 一致）；时钟策略 B（2×100 预跑稳态） |
| T007 | ncu 采集 naive（详设 §4.5 指标集）→ profile/naive/ 归档；创建 results/bottleneck_analysis.md 首份闭环（DRAM 流量/事务效率/stall 分布 → AR002 验证计划） | T006 | in_progress | **降级执行（用户批准）**：ncu 硬件计数器被 ERR_NVGPUCTRPERM 阻塞（无管理员）；首份闭环以理论+计时证据建档（bottleneck_analysis.md），计数器三指标待权限解锁后回补（步骤已写入该文档） |

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

### 2026-10-04 · GPU 验证会话（本地 Quadro RTX 5000，全自动模式）

- **T001–T006 全部完成，T007 降级（用户批准）**。本会话为预构建代码的首次 GPU 验证。
- **环境偏差处理**：本机 GPU = Quadro RTX 5000（sm_75 Turing）而非详设的 RTX 4060（sm_89）；
  用户批准继续 → 构建 `-arch=sm_75`，绝对性能门作废改相对门，偏差全量记录于
  results/environment.md §0 与详设 §2。
- **免管理员工具链组装**（本机无 nvcc/MSVC/ncu/sanitizer/git，无 winget，无管理员）：
  官方 redist zip 组装 nvcc 12.5.40 + cudart + cuBLAS 12.5.2.13；PortableBuildTools 装
  MSVC 14.44 + WinSDK；MinGit 2.50.1；ncu 2024.2.0 + sanitizer 2024.2.0；junction 绕过
  非 ASCII 路径；入口 tools/env.cmd（明细 environment.md §4）。
- **首构建修复记录**（TEST_PLAN §10 预期内的预构建缺陷，全部修复）：
  1. CMakeLists：check_cxx_compiler_flag 未 include 模块（删 no-op）；`-Xptxas -v` 被
     CMake 拼成 `-Xptxas-v`（改 `-Xptxas=-v`）；加 `/utf-8` + `/wd4828`；
  2. 源码 UTF-8 无 BOM → nvcc EDG 按 GBK(936) 误读中文注释吞行（identifier undefined
     假错误）→ 全部 .cu/.h 加 BOM；
  3. cuda_cccl 组件缺失（cuda_fp16.h → nv/target）→ pip cuda_cccl 并入 toolkit 树；
  4. cublas_baseline.cu 漏 `#include "common.h"`；
  5. vec4/cpasync 共 3 处 kernel 漏 `void` 返回类型；cpasync 2 处 `float4*`→`float*` 类型错。
  最终：**零 error 零 warning**，全 kernel 0 spill（tile2d=114r/vec4=128r/cpasync=139r/cpasync2=125r）。
- **T003 Red 补验**：注释 naive 注册 → `65/74 FAILED`，exit 1（run_case 增加 fn=nullptr 优雅 FAIL 防护）。
- **T004 Green**：74/74 ALL PASS；路径断言：vec4/cpasync/cpasync2 在 1023×1024×511、
  130×257×66 → scalar fallback 触发；4096³/1024³/256³/1000×1016×1024 无 fallback。
- **T005**：cuBLAS 行主序自校验双 PASS（rel 1.13e-07 / 1.27e-07）→ 换算正确；
  vs CPU double 误差量级证实 DEFAULT_MATH 生效（TF32 会是 ~1e-3 量级）。
- **T006**（时钟策略 B：2×100 预跑稳态，3 轮正式 ×100 iters）：
  naive 155.22 GF（RSD 0.47–0.60% ✓）；cublas 9.64 TF（RSD 8.76–9.55% 超 5% 门 →
  归因功耗墙振荡 185–225 W≈230 W cap + WDDM 无锁频，median 跨轮一致，如实记录未修饰）。
  相对线：naive = cuBLAS 的 1.61%。
- **§8 sanitizer 附加完成**：memcheck 全套件 0 errors；racecheck cpasync/cpasync2 0 hazards。
- **T007 降级**：ncu 报 ERR_NVGPUCTRPERM（GPU 性能计数器需管理员策略，HKLM 写入被拒）；
  bottleneck_analysis.md 首份闭环以理论+计时证据建档（155.22 GF=1.39% 理论峰值 +
  0.23 GB/s 有效带宽 vs E10 实测 375.7 GB/s → 延迟受限结论），计数器三指标待权限
  解锁后回补（解锁步骤已写入文档）。**T007 保持 in_progress，ST 验收待其完成后进行。**
- 会话 commits：9ba3f77（T001 产出）→ 35e44ef（T001+T002）→ 6eb197e（T003–T005）→ 本条（T006+T007 降级）。
