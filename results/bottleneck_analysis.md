# results/bottleneck_analysis.md — 逐版瓶颈闭环分析

> 格式契约（TEST_PLAN §7）：瓶颈 → 证据 → 对策 → 下一版验证。
> **⚠️ 本机 ncu 硬件计数器被 ERR_NVGPUCTRPERM 阻塞（WDDM + 无管理员权限，2026-10-04）**，
> AR001 首份闭环以「理论推导 + CUDA events 计时证据 + ptxas 资源审计」建立，
> 计数器证据（dram__bytes / sectors-per-request / stall 分布）待获得管理员权限后回补
> （解锁方式见 environment.md §4 与下方待办）。

---

## Kernel 0 — naive（AR001，2026-10-04 实测）

- **瓶颈**：非合并全局访存（A 读与 C 写按行跨步 16 KB）+ 零数据复用 + 每线程串行 K 累加（ILP=1），
  访存延迟无法被并行计算隐藏 → **访存延迟受限**（非吞吐受限）。
- **证据**（全部实测，本会话 sm_75 / 1890 MHz 稳态）：
  1. **计时**：4096³ median **885.42 ms = 155.22 GFLOPS**
     = 理论 FP32 峰值 11.15 TFLOPS 的 **1.39%**；
     = 本机 cuBLAS FP32（9.64 TFLOPS）的 **1.61%**（62.1× 差距）。
     数据源：results/performance.csv（3 轮 × 100 iters，RSD 0.47–0.60%）。
  2. **延迟受限的直接证据**（免计数器推理）：理论最小 DRAM 流量（三矩阵各读写一次）
     = 201.3 MB → naive 有效带宽仅 **0.23 GB/s**，远低于 E10 实测可达带宽
     **375.7 GB/s**（1 GiB D2D 拷贝 ×20 取最优，本机标定）——
     若 naive 是吞吐受限，应能观察到接近带宽上限的流量；实际只用到 0.06%，
     说明瓶颈在**事务拆分（32 请求拆散）+ 延迟排队**，而非 DRAM 带宽耗尽。
  3. **访问模式推导**（源码 sgemm_naive.cu:23-24，刻意不合并映射）：
     warp = (x:0-15, y:0-1)，A[row*K+kk] 相邻线程地址差 K×4B=16 KB
     → 每 warp 请求拆成 32 个独立 32B 事务（sectors/request = 32，FP32 合并理想 = 4，**8× 放大**）；
     B[kk*N+col] 同列广播（2 列/warp，近似无害）；C[row*N+col] 同样 16 KB 跨步。
  4. **资源审计**（build.log）：50 regs / **0 spill** / 0 smem；block(16,16)，
     1024 threads/SM 满占用——占用率不是瓶颈，延迟隐藏能力才是。
- **缺失项（待补）**：dram__bytes.sum、l1tex sectors/request、smsp long_scoreboard stall 比例
  —— `ncu` 报 ERR_NVGPUCTRPERM（性能计数器需管理员策略
  `HKLM\…\nvlddmkm\Global\NVTweak → RmGpuProfilerSupport=1`，本会话无权限写入）。
- **对策**：AR002 访问合并——线程映射改为 col→threadIdx.x（相邻线程读 B 相邻列、写 C 相邻列，
  128B 事务合并；A 读变为 warp 内广播模式）。块形状保持 (16,16) 不变（唯一变量 = 映射方向，
  保证 E3 实验可比性，见 AR001 design.md §4.1）。
- **验证**：[AR002 回填] 预期：① sectors/request(lu_mem_global_op_ld) 从 32 → ~4；
  ② GFLOPS 提升约 6×（RTX 4060 详设参考比 6.52×，本机以实测为准）；
  ③ long scoreboard stall 占比显著下降（待计数器解锁后量化）。

---

## 待办：ncu 计数器解锁步骤（获得管理员权限后）

1. 管理员执行：
   `reg add "HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" /v RmGpuProfilerSupport /t REG_DWORD /d 1`
   （或在 NVIDIA 控制面板 → 桌面 → Manage GPU Performance Counter → Allow access to all users）
2. 重启驱动（重启系统或禁用/启用显卡）；
3. 重跑 `profile\profile_all.ps1 -Kernels naive`（其余 kernel 随各 AR 采集）；
4. 本文档回补「缺失项」三指标 + 修订推导数值。
