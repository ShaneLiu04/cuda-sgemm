# results/environment.md — 环境基线报告（模板）

> 由 AR001/T001 在 GPU 环境执行 `docs/TEST_PLAN.md` §1 后填写；
> 填写后同步回填 `specs/component-detail-design/cuda_sgemm_spec.md` §2 占位符。
> 本文件是全部性能数据的环境上下文，**缺失本文件的 benchmark 数据不可引用**。

## 1. 硬件与驱动（`nvidia-smi -q` 摘录）

| 项 | 值 |
|----|-----|
| GPU 型号 | `<NVIDIA RTX 4060 Laptop GPU>` |
| TGP 档位 | `<nvidia-smi -q -d POWER：如 35W-115W Dynamic Boost>` |
| 显存 | `<8GB GDDR6, ...bit @ ...GB/s>` |
| 驱动版本 | `<>` |
| CUDA Toolkit | `<nvcc --version>` |
| ncu / compute-sanitizer | `<版本>` |

## 2. 实测规格

| 项 | 值 | 测量方法 |
|----|-----|---------|
| SM 数量 | `<>` | sgemm_bench 启动头自动打印 |
| boost 时钟 | `<MHz>` | cudaDeviceProp::clockRate |
| 实测稳态时钟 | `<MHz>` | nvidia-smi dmon 长跑采样（E13） |
| 理论 FP32 峰值 | `<TFLOPS = 2 × cores × freq>` | 按实测频率计算 |
| 实测峰值 DRAM 带宽 | `<GB/s>` | 大拷贝 kernel 标定（E10） |

## 3. 时钟策略（AGENTS.md §5.3）

- [ ] 方案 A：固定时钟 `nvidia-smi -lgc <freq>`，锁频值 = `<MHz`
- [ ] 方案 B：稳态预热（说明预热轮数与稳态判据）
- 监控方式：`nvidia-smi dmon -s puc -d 1` 采样归档路径：`<>`

## 4. -Xptxas -v 资源审计表（build.log 摘录）

| kernel | regs/thread | spill (st/ld) | smem/block |
|--------|-------------|---------------|-----------|
| naive | `<>` | `<>` | `<>` |
| coalesced | `<>` | `<>` | `<>` |
| smem1d (bk=16) | `<>` | `<>` | `<>` |
| tile2d (lb=1) | `<>` | `<>` | `<>` |
| vec4 | `<>` | `<>` | `<>` |
| cpasync | `<>` | `<>` | `<>` |
| cpasync2 | `<>` | `<>` | `<>` |

## 5. 环境漂移记录（每次正式测量会话追加）

| 日期 | 会话目的 | 室温/机况 | 频率范围 | 备注 |
|------|---------|----------|---------|------|
| `<>` | `<>` | `<>` | `<>` | `<>` |
