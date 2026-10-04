# cuda-sgemm — RTX 4060 Laptop 上的 SGEMM 逐层优化与性能分析

严格 FP32（禁 TF32/Tensor Core/低精度）的 SGEMM 六版 kernel 逐层优化工程，
目标：**cp.async 双缓冲版 ≥ 6.61 TFLOPS ≈ 58.24× vs Naive ≈ 98.5% cuBLAS FP32**（4096³）。

> **当前状态**：无 GPU 环境预构建（pre-built, unverified）。
> 代码已完整实现，编译/正确性/性能验证按 `docs/TEST_PLAN.md` 在 RTX 4060 环境执行。
> 全部性能数字在 GPU 实测前均为**目标值**，以 `results/performance.csv` 实测数据为准。

## 性能阶梯（目标值，±10% 条款见详设 §7）

| # | Kernel | 目标 | vs Naive | 核心技术 |
|---|--------|------|---------|---------|
| - | cuBLAS FP32（禁 TF32） | ~6.71 TF | ~59× | 参考线 |
| 0 | naive | ~114 GF | 1× | 教科书基线（刻意不合并映射） |
| 1 | coalesced | ~740 GF | 6.5× | warp 级合并访存 |
| 2 | smem1d | ~1.43 TF | ~12.6× | 32×32 smem 分块 + 1D TM=8 |
| 3 | tile2d | ~3.51 TF | ~30.9× | 128×128×8 分块 + 8×8 寄存器外积 |
| 4 | vec4 | ~5.84 TF | ~51.4× | float4×3 处 + A 转置 + B XOR swizzle |
| 5 | cpasync | ~6.61 TF | **58.24×** | cp.async + 2-stage 双缓冲流水 |
| 5' | cpasync2 | 消融用 | - | 方案乙：A 寄存器预取 + B cp.async |

## 快速开始（GPU 环境）

```bash
# 构建（Linux；Windows 见下）
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
# 或: make build

make test                      # 全 kernel x 全矩阵正确性回归（含 CPU double 参考）
make bench KERNEL=all          # 4096³ 性能阶梯，落盘 results/performance.csv
./bench/run_bench.sh sweep     # 阶梯 + 尺寸扫描
./profile/profile_all.sh       # ncu 全量采集（指标集 = 详设 §4.5）
./profile/sanitize.sh memcheck && ./profile/sanitize.sh racecheck
```

Windows（PowerShell + VS 生成器）：

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release -j
.\build\Release\sgemm_test
.\bench\run_bench.ps1
.\profile\profile_all.ps1
.\profile\sanitize.ps1 -Tool racecheck -Kernels cpasync,cpasync2
```

CLI 示例：

```bash
./build/sgemm_bench --kernel cpasync --m 4096 --n 4096 --k 4096 \
                    --warmup 20 --iters 100 --check --csv --verbose
# 消融旋钮
./build/sgemm_bench --kernel smem1d --bk 32      # BK ∈ {8,16,32}
./build/sgemm_bench --kernel tile2d  --lb 2      # __launch_bounds__ minBlocks
./build/sgemm_bench --kernel cpasync2            # cp.async 方案乙
```

## 目录结构

```
cuda-sgemm/
├── PROMPT.md / AGENTS.md                    # SDD 总入口 / 开发军规
├── CMakeLists.txt / Makefile                # 构建（禁 -use_fast_math）
├── include/common.h                         # CUDA_CHECK/CLI/events 计时/CSV/元数据
├── include/sgemm_kernels.h                  # 统一接口 + kernel 注册表
├── src/                                     # main.cu + 六版 kernel + cublas 基线
├── tests/test_correctness.cu                # 全矩阵回归（CPU double + cuBLAS 双参考）
├── bench/  profile/                         # 批量跑分 / ncu / sanitizer 脚本（sh+ps1）
├── docs/TEST_PLAN.md                        # GPU 环境严格测试手册（四门验收）
├── docs/EXPERIMENT_DESIGN.md                # E01-E14 可解释性实验设计
├── results/                                 # performance.csv / environment.md / 闭环分析
└── specs/                                   # SDD 文档（详设 / AR001-006 / backlog / archive）
```

## 文档导航

| 文档 | 内容 |
|------|------|
| `PROMPT.md` | SDD 流程契约：六阶段生命周期、四门验收、AR 流转 |
| `AGENTS.md` | CUDA 编码军规、测量纪律、数据真实性军规 |
| `specs/component-detail-design/cuda_sgemm_spec.md` | 组件详设：统一契约（接口/计时/CSV/判据/ncu 指标） |
| `specs/changes/AR001-*/` + `specs/backlog/AR00x-*/` | 各 AR 的 srs / design / tasks |
| `docs/TEST_PLAN.md` | GPU 迁移后的完整测试执行手册 |
| `docs/EXPERIMENT_DESIGN.md` | 14 组实验：目的/假设/方法/预期/判伪 |

## 上传 Gitee

```bash
git init
git add .
git commit -m "feat: cuda-sgemm pre-built (code complete, GPU verification pending)"
# 在 Gitee 建仓后：
git remote add origin https://gitee.com/<user>/cuda-sgemm.git
git branch -M master
git push -u origin master
```

迁移到 4060 环境后：`git clone` → 按 `docs/TEST_PLAN.md` §1-§9 顺序执行；
每完成一个 AR 的 ST 验收即提交一次（代码 + CSV + 闭环文档同步，AGENTS.md §8）。

## 核心约束（数据真实性军规）

1. 所有性能数字实测可复现（CSV 含 git sha / GPU 状态 / RSD）；
2. cuBLAS 对比显式 `CUBLAS_DEFAULT_MATH`（禁 TF32，E02 提供对照证据）；
3. ncu 结论必须引用具体指标数值；失败实验同样记录；
4. 禁止 `-use_fast_math` 与一切低精度路径冒充 FP32。
