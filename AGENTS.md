# AGENTS.md — cuda-sgemm 组件开发约定

> 本文件是本工程所有 AI Agent 与开发者必须遵守的开发约定。与 `specs/component-detail-design/cuda_sgemm_spec.md`
> （组件详设，全局基准）配合使用。任何与本文件冲突的临时决定必须先修订本文件再执行。

## 1. 项目身份

- **组件**：`cuda-sgemm` — Quadro RTX 5000（sm_75, Turing, 48 SM）上严格 FP32 的 SGEMM 七版 kernel（naive→swpipe）逐层优化与性能分析工程。
  （原设计目标机 RTX 4060 Laptop（sm_89）已于 2026-10-04 改靶本机，偏差与决策记录见
  `results/environment.md` §0 与 `specs/component-detail-design/cuda_sgemm_spec.md` §2。）
- **语言/标准**：CUDA C++，C++17；CUDA Toolkit ≥ 11.8（实测 12.5.40）。
- **数据真实性**是本工程第一军规（见 PROMPT.md §6），性能数字一律实测、可复现。

## 2. 目录约定

```
cuda-sgemm/
├── AGENTS.md / PROMPT.md / README.md(最终交付)
├── Makefile                  # 一键构建（AR001 建立；make / make test / make bench / make profile）
├── include/common.h          # CUDA_CHECK、CLI 解析、计时工具、GFLOPS 计算
├── src/                      # main.cu + 六版 kernel（每版一个 .cu，自包含）
├── tests/                    # 正确性测试（一键脚本：全 kernel × 全尺寸 → PASS/FAIL 表）
├── bench/                    # 批量 benchmark 脚本，输出 CSV
├── profile/                  # ncu 采集脚本 + 各版本指标导出
├── results/                  # performance.csv、瓶颈分析、阶梯图、environment.md
└── specs/                    # SDD 文档（详设/changes/backlog/archive）
```

## 3. 构建与测试命令（AR001 建立 target 后必须保持可用）

| 命令 | 作用 |
|------|------|
| `make` / `make build` | 编译全部 kernel + tests + bench：`nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo -Xptxas -v` |
| `make test` | 运行全部正确性测试，输出 PASS/FAIL 表格 |
| `make bench [M= N= K= KERNEL=all]` | benchmark，追加写入 `results/performance.csv` |
| `make profile` | ncu 批量采集（指标集见详设 §4.5），导出到 `profile/<kernel>/` |

- **消融旋钮**（AR008 起冻结，默认值与依据见详设 §4.6）：`--bk`（smem1d，默认 32）、
  `--lb`（tile2d/ws，默认 1；swpipe/swsk 固化 128-reg 封顶，AR008 实测收窄）、
  `--sk`（swsk split-K 片数，默认 4，1..16）、
  `--stages`（ws smem 环深度，默认 3，2/3）、
  `--wp`（ws producer warp 数，默认 2，1/2）、
  `--rounds N`（多轮门控统计，默认 1，0 → CLI_ERROR）。
  非默认参数跑出的数据必须经 `SGEMM_CSV` 环境变量分流到独立 CSV，禁止污染主 `performance.csv`。

- **禁止 `-use_fast_math`**；FMA 合并默认允许（IEEE 合规范围）。
- `-Xptxas -v` 输出必须保留到构建日志（`build.log`），寄存器/smem 用量是验收数据。
- 代码风格：无独立 linter 时，保持 4 空格缩进、kernel 内关键块注释（见 §5）。

## 4. CUDA 编码军规

1. 所有 CUDA API 调用包裹 `CUDA_CHECK()`；kernel 启动后立即 `CUDA_CHECK(cudaGetLastError())`；
   程序退出前 `CUDA_CHECK(cudaDeviceSynchronize())` 兜底；所有资源 RAII 或成对释放。
2. 统一 kernel 接口：`void sgemm_x(const float* A, const float* B, float* C, int M, int N, K)`；
   矩阵**行主序**，计算 `C = A·B`（A: M×K，B: K×N，C: M×N）。禁止改变该签名。
3. 严格 FP32：禁止 TF32（不启用 `--use_fast_math`、不使用 tensor core 路径）、
   禁止 FP16/BF16 中间类型、禁止 `__fdividef` 等快速数学 intrinsic。
4. 显式资源声明：寄存器敏感 kernel 必须 `__launch_bounds__(<threads>, <minBlocks>)`，
   以控制 ptxas 分配；出现 spill（`-Xptxas -v` 报 `spill stores/loads`）即视为缺陷，必须修复或记录取舍。
5. 注释契约：每个 kernel 文件头部注释必须写明——tile 尺寸、每线程工作分配（TM/TN）、
   smem 布局与 padding/swizzle 理由、（若有）流水阶段与缓冲数量。缺注释不通过 review。
6. 边界处理：所有 kernel 必须支持任意 M/N/K；向量化主路径 + 谓词化回退路径并存时，
   回退路径的正确性必须有专项测试用例。
7. 对齐假设必须显式：依赖 16B 对齐的访存，入口处断言指针与步长条件，不满足走回退路径。

## 5. 性能测量纪律（所有 benchmark 必须遵守）

1. 计时只用 **CUDA events**（kernel 执行时间），禁止 host 墙钟；warmup ≥ 20 次，正式 ≥ 100 次；
   报告 **median / min / max** 与多轮方差。
2. `GFLOPs = 2·M·N·K / t / 1e9`；同时记录等效 DRAM 带宽（= 3 次读写矩阵的总字节 / t，另以 ncu 实测为准）。
3. **时钟策略**：优先 `nvidia-smi -lgc <freq>` 固定时钟（需权限，记录所用频率）；无权限则跑至稳态。
   每次正式测量前后记录 SM 频率、温度、功耗（`nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv`），
   写入 CSV 元数据与 `results/environment.md`。
4. 同一对比实验内，所有 kernel 与 cuBLAS 必须在**同一时钟策略、同一会话**下测量。
5. 每次实验记录 git commit hash，保证"任何一个历史数据点能用当前代码复现"。

## 6. 分析纪律（ncu / compute-sanitizer）

- ncu 指标集与命令模板见详设 §4.5；导出原始数据（`.ncu-rep` + csv）到 `profile/<kernel>/`，不得只留截图。
- 每版 kernel 的瓶颈结论必须构成闭环：**瓶颈 → 证据（指标数值）→ 对策 → 下一版验证结果**，
  追加到 `results/bottleneck_analysis.md`。
- cp.async 版本必须抽查 `compute-sanitizer --tool racecheck`；所有版本发布前 `--tool memcheck` 干净。
- **实验与图表纪律（每个 AR 强制）**：每个 AR 收尾前必须补充成组对比实验（消融/配对/回归，
  数量以"足以支撑结论"为准，宁多勿少）增强说服力；实验结果必须沉淀为**可解释、信息丰富、
  美观**的图表（`make_figures.py` 从 CSV 可复现生成，无手工修饰），并回填
  `results/report.md` 与 README。每个开发任务在完成时就明确自己的实验数据与图表交付物
  （见各 AR tasks.md 的交付物列），禁止"做完代码、图表最后凑数"。

## 7. TDD 规则（CUDA 特化）

- 每版 kernel 先写/扩展正确性测试（Red：测试因 kernel 未接入而失败），再实现（Green），再重构。
- 正确性判据、参考实现与测试矩阵见详设 §4.4；测试框架自研轻量即可（断言 + 表格输出），不引入重依赖。
- 性能门是 ST 验收内容，不进 TDD 循环；但 benchmark 接入（能跑、能落盘）是开发任务。

## 8. Git 规范

- 每个任务完成即提交：`feat(AR00x): T00x 任务简述`；实验性调参分支不合并主干，但数据可引用。
- `results/` 下 CSV 与报告必须与产生它的代码 commit 同步提交。
- 禁止提交 `*.ncu-rep` 之外的大体积二进制；禁止提交任何凭空生成的性能数据。

## 9. 已知风险速查（详见详设 §7）

| 风险 | 一句话对策 |
|------|-----------|
| 笔记本功耗墙降频 | 固定时钟或稳态预热；报告频率/温度/方差 |
| cuBLAS 偷开 TF32 | `cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH)` |
| cuBLAS 行主序陷阱 | 按详设 §4.2 的换算调用，并用 CPU 参考交叉验证 |
| 8×8 分块寄存器溢出 | `__launch_bounds__` 控制；spill 为缺陷 |
| float4/cp.async 对齐 | 谓词化回退路径 + 断言 |
| bank conflict 凭感觉 | 以 ncu conflict 计数为准，padding/swizzle 实测取舍 |
