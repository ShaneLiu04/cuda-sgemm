# [AR001] design.md — 评测框架 + 环境 + cuBLAS 基线 + Naive

| 组件名称 | cuda-sgemm |
| --- | --- |
| AR系统流水号 | AR001 |
| AR描述 | 统一评测框架（构建/正确性/计时/落盘/profile）+ 环境基线 + cuBLAS FP32 参考线 + Naive 基准 kernel |

# 2 动态行为

```plantuml
@startuml
participant User as U
participant "sgemm_bench\n(main.cu)" as M
participant "kernels/cublas" as K
participant "results/performance.csv" as R
U -> M: --kernel X --m --n --k [--check] --csv
M -> M: 环境信息头（GPU/driver/git/状态）
opt --check
  M -> K: cuBLAS 参考计算
  M -> K: 待测 kernel
  M -> M: max_abs/rel <= 1e-4 ?
end
M -> K: warmup x20 + events 计时 x100
M -> R: 追加行（median/min/max/RSD/GFLOPS/状态）
@enduml
```

# 3 功能点分解

| 序号 | 功能点名称 | 功能点描述 | 实现 |
| --- | --- | --- | --- |
| 1 | 环境探测 | GPU/TGP/驱动/CUDA/ncu 版本 + 时钟策略 → results/environment.md | TEST_PLAN §1 流程 + environment.md 模板 |
| 2 | 公共基础设施 | CUDA_CHECK/CLI/events 计时/GFLOPS/CSV/GPU 状态 | include/common.h |
| 3 | 构建系统 | CMake+Make，-O3/-lineinfo/-Xptxas -v，禁 fast_math | CMakeLists.txt / Makefile |
| 4 | kernel 注册表 | 统一签名 + 名称/ID/函数映射 + 消融旋钮 | include/sgemm_kernels.h |
| 5 | 正确性框架 | CPU double + cuBLAS 双参考、≤1e-4 判据、全矩阵一键回归 | tests/test_correctness.cu |
| 6 | Naive kernel | 刻意不合并映射的一输出/线程基线 | src/sgemm_naive.cu |
| 7 | cuBLAS 基线 | 行主序换算 + CUBLAS_DEFAULT_MATH + 自校验 | src/cublas_baseline.cu |
| 8 | benchmark 链路 | warmup/iters/median/CSV/环境头 | src/main.cu |
| 9 | profile 骨架 | ncu 指标集采集脚本（sh/ps1）+ sanitizer 脚本 | profile/ |

# 4 实现设计

## 4.1 功能实现思路

- **计时**：per-iteration cudaEvent 对（iters=100 → 200 events，一次终态同步后统一取时），
  统计 median/min/max/mean/RSD；禁止 host 墙钟。
- **对齐实验变量**：naive 与 coalesced 用完全相同的 block(16,16) 形状，唯一差异是
  线程映射方向（E3 的 A/B 可比性）。
- **Naive 的"刻意不合并"**：row→threadIdx.x（行方向跨步访存）。这是基线完整性的
  一部分，注释中明确禁止"顺手优化"。

## 4.2 功能实现设计（关键决策）

1. **events 数量**：不用 2 events 循环复用（需逐次同步，引入 host 噪声），
   用 iters 对 events 终态一次性 `cudaEventElapsedTime`。
2. **cuBLAS 行主序**：C'=B'·A' 换算（cublas_baseline.cu 文件头有推导），
   必须过 test_correctness 的 CPU double 交叉验证（自校验义务）。
3. **GPU 状态采集**：`popen(nvidia-smi ...)` best-effort，失败写 "n/a"，
   不阻塞 benchmark。
4. **CSV**：append + 首次自动写表头/版本注释；git sha 由构建系统注入 `-DGIT_SHA`。

## 4.3 接口描述

```cpp
// 统一 kernel 接口（冻结，详设 §4.1）
void sgemm_naive(const float* A, const float* B, float* C, int M, int N, int K);
void sgemm_cublas(const float* A, const float* B, float* C, int M, int N, int K);
// CLI：--kernel --m --n --k --warmup --iters --check --csv --bk --lb --verbose --list-kernels
```

## 4.4 代码设计

模块边界 = 目录边界（common.h 无 CUDA kernel 代码；每版 kernel 一个 .cu 自包含；
main.cu 只做路由与编排，不含算法）。构建产物：`sgemm_bench`（编排）与
`sgemm_test`（回归）两个独立可执行文件，共享 kernel 源文件列表。

# 6 测试设计

## 6.1 单元测试（UT）

- test_correctness.cu：9 个尺寸用例 × 全 kernel + cuBLAS 自校验 2 例；
  判据 rel ≤ 1e-4；输出 PASS/FAIL 表格 + 退出码。

## 6.3 业务场景测试

- `--check`：bench 前置抽查（与全量回归互补，防止"测错对象"）。

## 6.4 异常场景测试

- 非法参数（M/N/K≤0、未知 kernel、缺参数值）→ 非零退出码 + 明确错误信息；
- K=1、1×1×1、17×33×65 退化用例；
- `--verbose` 路径打印：供回退路径覆盖断言（grep）。

## 预构建状态记录

本 design 在无 GPU 环境定稿并完成代码实现；TDD Red/Green 执行、环境回填、
ncu 首份闭环全部按 docs/TEST_PLAN.md 在 GPU 环境补做。
