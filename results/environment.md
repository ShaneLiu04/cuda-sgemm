# results/environment.md — 环境基线报告（AR001/T001 实测）

> 本文件是全部性能数据的环境上下文，**缺失本文件的 benchmark 数据不可引用**。
> 探测时间：2026-10-04；探测人：AI Agent（用户批准的免管理员工具链组装 + RTX 5000 继续方案）。

## 0. ⚠️ 环境偏差声明（必读）

**本机 GPU 不是详设假设的 RTX 4060 Laptop（sm_89, Ada），而是 Quadro RTX 5000（sm_75, Turing）。**
经用户批准（2026-10-04），本工程在以下偏差条件下继续，处理原则 = 数据真实性军规（如实记录，禁止美化学数字）：

| 偏差项 | 详设假设（RTX 4060） | 本机实测（RTX 5000） | 影响与处理 |
|--------|---------------------|---------------------|-----------|
| 架构 | sm_89（Ada） | **sm_75（Turing）** | 构建改 `-arch=sm_75`；详设 §4.6 cp.async 路径在 sm_75 **无硬件支持**（cp.async 指令需 sm_80+），AR006 的 `__pipeline_memcpy_async` 将由 CUDA 头文件退化为同步拷贝——正确性可验、性能特征不同，届时如实归档 |
| 绝对性能门 | 按 RTX 4060 推导（naive 113.55 GF 等） | 本机理论 FP32 峰值 11.15 TF（≠13.3 TF） | **绝对门作废，改用相对门**：① 各版本 vs 本机 cuBLAS FP32 实测百分比；② 相邻版本加速比。绝对值仅作参考记录 |
| L2 | 32MB（Ada） | 4MB（Turing） | AR003+ 的 tile 复用收益可能更明显（L2 命中率基线更低），以 ncu 实测为准 |
| 显存 | 8GB | 16GB | 4096³ 无影响 |
| 时钟控制 | 假设可 `-lgc` | WDDM 模式 + 无管理员权限，`-lgc` 不可用 | 时钟策略强制走 **方案 B（稳态预热）**，见 §3 |

## 1. 硬件与驱动（`nvidia-smi -q` 摘录）

| 项 | 值 |
|----|-----|
| GPU 型号 | NVIDIA Quadro RTX 5000（Turing TU104 GL，桌面工作站卡） |
| TGP 档位 | 固定 230.00 W（Default=Max=230 W，Min 可调下限 125 W；无 Dynamic Boost） |
| 显存 | 16 GB GDDR6，256-bit @ 7001 MHz（等效 14 Gbps）→ 理论带宽 **448.1 GB/s** |
| 驱动版本 | 556.18（CUDA 12.5 runtime 兼容） |
| 驱动模式 | WDDM（Windows 显示驱动模式，非 TCC） |
| CUDA Toolkit | nvcc release 12.5, V12.5.40（Build cuda_12.5.r12.5/compiler.34177558_0） |
| ncu / compute-sanitizer | Nsight Compute 2024.2.0.0 (build 34181891) / 2024.2.0.0 (build 34165569) |
| OS | Windows（PowerShell 5.1，无管理员权限，无 winget/choco） |

## 2. 实测规格（deviceQuery 实测，2026-10-04）

| 项 | 值 | 测量方法 |
|----|-----|---------|
| SM 数量 | 48 | cudaDeviceProp::multiProcessorCount |
| 架构 | sm_75，每 SM 64 FP32 lanes（3072 cores 总计） | cudaDeviceProp + 架构手册 |
| boost 时钟 | 1815 MHz（cudaDeviceProp::clockRate=1815000 kHz；nvidia-smi 最大 SM 时钟 2100 MHz） | deviceQuery |
| 空闲时钟 | SM 300 MHz / Mem 405 MHz（探测时 37°C, 13.5W, P8） | nvidia-smi |
| 理论 FP32 峰值 | **11.15 TFLOPS**（2 × 3072 × 1.815 GHz） | 按实测 boost 频率计算 |
| 理论 DRAM 带宽 | 448.1 GB/s | 按显存规格计算 |
| 实测可达 DRAM 带宽 | **375.7 GB/s**（83.8% 理论值） | E10 标定（2026-10-04）：1 GiB D2D 拷贝 ×20 取最优，CUDA events 计时 |
| L2 cache | 4 MB | deviceQuery |
| smem/block | 48 KB（每 SM 64 KB 可配） | deviceQuery |
| 寄存器 | 64K × 32bit / block / SM | deviceQuery |
| warp / 最大线程/SM | 32 / 1024 | deviceQuery |

## 3. 时钟策略（AGENTS.md §5.3）

- [x] **方案 B：稳态预热**（唯一可行：WDDM + 无管理员权限，`nvidia-smi -lgc` 不可用）
  - 预热协议：每次正式测量前 warmup ≥ 20 次（bench 程序内置），并以 **2 × 100 次预跑** 驱动 GPU 进入稳态；
  - 稳态判据：RSD ≤ 5%（AGENTS.md §5.1/§5.2）；超限复测 3 次取中位并记录频率/温度；
  - 无锁频声明：本机所有数字均为**未锁频 WDDM 测量**，绝对值可比性弱于锁频环境；同会话内相对比较（kernel vs kernel）有效。
- 监控方式：每次测量前后 `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv` 落入 CSV `gpu_state` 列；正式 benchmark 会话另用 `nvidia-smi dmon -s puc -d 1` 采样归档到 `results/dmon_<date>.log`（T006 执行）。

## 4. 工具链组装记录（免管理员用户空间方案，2026-10-04）

本机无系统级 CUDA Toolkit / MSVC / git（曾有 CUDA 12.5 安装但仅剩 version.json 与 MSBuild 残留）。
经用户批准，以下组件**全部以官方发行包组装到用户目录**，未触碰系统：

| 组件 | 版本 | 来源 | 位置 |
|------|------|------|------|
| nvcc（含 cicc/ptxas/nvlink/nvvm/libdevice） | 12.5.40 | NVIDIA 官方 redist zip（cuda_nvcc-windows-x86_64-12.5.40-archive.zip） | `C:\Users\l30086046\csg-tools\cuda-toolkit\` |
| cudart（头/lib/dll） | 12.5.39 | 同上（cuda_cudart-…-archive.zip） | 同上（合并树） |
| cuBLAS（头/lib/dll，含 cublas.lib 导入库） | 12.5.2.13 | 同上（libcublas-…-archive.zip） | 同上（合并树） |
| MSVC Build Tools（cl.exe 14.44.35207 + WinSDK 10.0.26100） | VS 2022 工具链 | PortableBuildTools v2.10.2（微软官方包源，免管理员解包） | `C:\Users\l30086046\csg-tools\msvc\`（入口 `devcmd.bat`） |
| Nsight Compute（ncu.exe） | 2024.2.0.0 | NVIDIA 官方 redist zip | `C:\Users\l30086046\csg-tools\cuda\ncu-root\...\target\windows-desktop-win7-x64\` |
| compute-sanitizer | 2024.2.0.0 | NVIDIA 官方 redist zip | 同上目录 |
| git | 2.50.1.windows.1 | MinGit portable（git-for-windows GitHub release） | `C:\Users\l30086046\csg-tools\git\` |
| cmake | 4.4.2 | 系统已有 | PATH |
| 构建 junction | — | mklink /J（绕过项目路径非 ASCII 字符【】对 nvcc/MSVC 的兼容风险） | `C:\Users\l30086046\csg-work` → 项目根 |

> 环境引导：项目内 `tools/env.cmd` 一键设置上述 PATH/INCLUDE/LIB；所有构建与测试命令通过它执行。
> 冒烟证据：hello.cu（sm_75, -O3, FP32 kernel）编译执行输出正确；ncu/sanitizer --version 正常。

## 5. -Xptxas -v 资源审计表（build.log 实测摘录，sm_75 / CUDA 12.5.40 / MSVC 19.44，2026-10-04）

实测全部 **spill = 0**（AR004+ 硬门通过）。与 TEST_PLAN §2 预期对照：

| kernel | regs/thread（实测） | spill (st/ld) | smem/block（实测） | TEST_PLAN 预期 |
|--------|-------------|---------------|-----------|------|
| naive | 50 | 0/0 | 0 | ≤24（实测偏高：64 位索引乘法 + 谓词；无 spill 无 stack，非缺陷，记录取舍） |
| coalesced | 50 | 0/0 | 0 | ≤24（同上） |
| smem1d (bk=8/16/32) | 72 | 0/0 | 2688 / 4864 / 9216 B | ≤32 regs, ~4.9KB（bk16 实测 4.75KB ✓；regs 72 高于预期但无 spill） |
| tile2d (lb=1 和 2) | 114 | **0/0** | 10880 B | 100–168 regs ✓，~10.9KB ✓ |
| vec4 | 128 | **0/0** | 8320 B | 100–168 regs ✓，~8.4KB ✓ |
| cpasync (甲) | 139 | **0/0** | 20480 B | 100–168 regs ✓，~16.6KB（实测 20KB，padding/双缓冲布局差异，如实记录） |
| cpasync2 (乙) | 125 | **0/0** | 16640 B | ~16.9KB ✓ |

> 注：naive/coalesced/smem1d 的 regs 高于 TEST_PLAN 预估，因寄存器上限（64K/SM）远未触顶且
> spill=0，按"记录取舍"处理（AGENTS.md §4.4 只把 spill 判为缺陷）。tile2d/vec4/cpasync 系列
> `__launch_bounds__(256,1)` 生效，全部落在预期区间。

## 6. 环境漂移记录（每次正式测量会话追加）

| 日期 | 会话目的 | 室温/机况 | 频率范围 | 备注 |
|------|---------|----------|---------|------|
| 2026-10-04 | T001 探测 + 工具链组装 | 37°C 空闲 / P8 | SM 300 MHz（空闲） | 探测会话，无正式测量 |
| 2026-10-04 | T006 正式 benchmark（4096³ naive+cublas 各 3 轮 ×100 iters） | 60–76°C / 66–225 W | SM 1845–1920 MHz（稳态） | naive RSD 0.47–0.60% ✓；cublas RSD 8.76–9.55% 超标，归因：功耗墙振荡（185–225 W≈230 W cap）+ WDDM 无锁频（策略 B 已知代价），median 跨轮一致（13.94/14.26/14.37 ms），取中位 14.26 ms 有效 |
| 2026-10-04 | §8 sanitizer（memcheck 全套件 + racecheck cpasync×2） | — | — | memcheck 0 errors（8 kernel × 9 尺寸）；racecheck 0 hazards |
| 2026-10-04 | E10 带宽标定 | D2D 拷贝稳态 | — | 375.7 GB/s（20 轮最优 5.715 ms/GiB） |
| 2026-10-04 | 改靶后对比实验（E1 全 kernel×6 尺寸 + E2 消融，git=84e261f） | 43–82°C / 15–226 W | SM 375–1950 MHz（动态加速超标称 1815） | 53 行 CSV；WDDM 单点提交抖动致部分 RSD 虚高（median 鲁棒，min/max 保留）；小尺寸冷启动（390 MHz）由 warmup 拉起 |
| 2026-10-05 | AR007 矩阵会话（9 kernel × 6 尺寸 × rounds=3 + 4 消融，10 分钟连续热浸没） | 57–85°C / 52–227 W | SM 1815–1950 MHz（尾段 1815–1860，热降频 ~5%） | 54 cells 全 RSD≤1.7%（--rounds 门控生效：跨轮 RSD 0.02–0.35%）；热浸没致绝对值系统性偏低（cuBLAS 4096³ 9780 vs 冷态 10136）；消融行分流 ablation_ar007.csv；冷态确认探针分流 cool_probe_ar007.csv（swpipe 6584.8 / vec4 6506.3 / cuBLAS 10136.2 GF，cuBLAS=本机历史最高=峰值 84.6%） |
