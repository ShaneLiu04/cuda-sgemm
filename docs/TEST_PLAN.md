# TEST_PLAN.md — GPU 环境严格测试执行手册

> 状态：代码已在无 GPU 环境预构建（pre-built, unverified-on-GPU）。
> 本手册是迁移到 RTX 4060 Laptop（sm_89）环境后的**唯一测试执行依据**，
> 覆盖：环境预检 → 构建 → 正确性 → 基准 → 性能门 → ncu → sanitizer → 归档。
> 对应验收契约：PROMPT.md §3 阶段 5「四门验收」（正确性/性能/资源/分析）。
>
> **诚实性声明**：预构建代码未经任何编译或运行验证。首次构建出现编译错误属预期内，
> 修复记录须写入对应 AR 的 tasks.md 进度记录，不得隐瞒。

---

## 0. 前置条件与责任矩阵

| 项 | 要求 | 检查命令 |
|----|------|---------|
| GPU | Quadro RTX 5000（sm_75, Turing, 48 SM，本工程调优目标机）；其他卡需改 `-DCMAKE_CUDA_ARCHITECTURES` | `nvidia-smi` |
| 驱动 | ≥ CUDA Toolkit 对应版本要求 | `nvidia-smi --query-gpu=driver_version` |
| CUDA Toolkit | ≥ 11.8（建议 12.x），含 nvcc | `nvcc --version` |
| Nsight Compute | 与 Toolkit 配套 | `ncu --version` |
| compute-sanitizer | 与 Toolkit 配套 | `compute-sanitizer --version` |
| CMake + 构建器 | ≥ 3.18；Ninja/Make/VS 均可 | `cmake --version` |
| git | 记录 commit 与数据关联 | `git --version` |

**测试执行顺序强制**：环境预检(§1) → 构建(§2) → 正确性(§3) → 基准与性能门(§4-§5)
→ 资源审计(§6) → ncu 分析(§7) → sanitizer(§8) → 归档(§9)。**任何一步失败，停止后续步骤并修复。**

---

## 1. 环境预检（对应 AR001/T001）

```bash
nvidia-smi                                        # GPU 型号/驱动
nvidia-smi -q -d POWER,CLOCK                      # TGP 档位/频率范围
nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv
nvcc --version && ncu --version && compute-sanitizer --version
```

**时钟策略决策**（二选一，写入 results/environment.md）：
- A（推荐）：`sudo nvidia-smi -lgc <freq>`（选稳态可达频率，如 2100 附近的稳定档）；
  测后 `sudo nvidia-smi -rgc` 恢复。
- B（无权限）：跑 2×100 次预热至稳态再正式测量，全程 `nvidia-smi dmon -s puc -d 1` 记录。

**产出**：填写 `results/environment.md`（模板见该文件），回填
`specs/component-detail-design/cuda_sgemm_spec.md` §2 全部占位符。

---

## 2. 构建验证

```bash
# Linux
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
# Windows（PowerShell）
cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release -j
```

**通过判据**：
1. 编译零 error；新引入 warning 必须清零（AGENTS.md §4b）；
2. `grep -i fast_math` 构建命令为空（严格 FP32 军规）；
3. 产出 `sgemm_bench` 与 `sgemm_test` 两个可执行文件；
4. `./build/sgemm_bench --list-kernels` 列出 8 个 kernel。

**寄存器/smem 审计**（AR004+ 硬门数据来源）：

```bash
make ptxas_log          # 或: cmake --build build -- VERBOSE=1 > build.log 2>&1
grep -E "Compiling entry|registers|spill|shared" build.log
```

预期登记表（写入 results/environment.md 附录）：

| kernel | regs/thread 预期 | spill 预期 | smem/block 预期 |
|--------|-----------------|-----------|----------------|
| naive / coalesced | ≤ 24 | 0 | 0 |
| smem1d (BK=16) | ≤ 32 | 0 | ~4.9KB |
| tile2d (lb=1) | 100-168 | **0（硬门）** | ~10.9KB |
| vec4 | 100-168 | **0（硬门）** | ~8.4KB |
| cpasync / cpasync2 | 100-168 | **0（硬门）** | ~16.6KB / ~16.9KB |

> tile2d/vec4/cpasync 的 `__launch_bounds__(256,1)` 允许 ptxas 用到 255 regs；
> 若出现 `spill stores/loads` 非零 → **缺陷**，按 AR004 srs §3.2 处理
> （调 `--lb`、减片段缓存、改循环结构），修复过程记录进 tasks.md。

---

## 3. 正确性验收（四门之一；每 AR 开发阶段必须全绿）

```bash
./build/sgemm_test                     # 全 kernel x 全测试矩阵（含 CPU double 参考）
./build/sgemm_test --verbose           # 额外打印 vec4/cpasync 的回退路径选择
```

**通过判据**：
1. 末行 `SUMMARY: N / N PASS -> ALL PASS`，退出码 0；
2. 每格 `rel ≤ 1e-4`（判据详设 §4.4），实际 max_abs/max_rel 数值抄录归档；
3. `cublas row-major self-check` 两行必须 PASS（行主序换算正确性，详设 §4.2 义务）；
4. **路径覆盖断言**：`--verbose` 输出中，
   - `1023x1024x511` 与 `130x257x66` 两尺寸必须出现
     `[vec4] ... -> scalar fallback` 与 `[cpasync] ... -> scalar fallback`；
   - `4096x4096x4096` 等对齐尺寸**不得**出现 fallback。
   ```bash
   ./build/sgemm_test --verbose 2>&1 | grep fallback
   ```

**失败排查表**：

| 症状 | 首查 |
|------|------|
| cublas self-check FAIL | 行主序换算（cublas_baseline.cu 文件头）参数序 |
| 仅大尺寸 FAIL | 索引 int 溢出（查 (long long) 强转）、tile 边界 |
| 仅边界尺寸 FAIL | 谓词/回退路径；`compute-sanitizer --tool memcheck` 定位越界 |
| vec4/cpasync 全 FAIL | 对齐条件判断；Frag8 union 的对齐假设 |
| 随机偶发 FAIL | 竞争：racecheck（cpasync 双缓冲）；未初始化 smem |

---

## 4. 基准测试规程（四门之二的数据来源）

```bash
# 单 kernel
./build/sgemm_bench --kernel cpasync --m 4096 --n 4096 --k 4096 \
                    --warmup 20 --iters 100 --check --csv
# 全阶梯（推荐脚本）
./bench/run_bench.sh ladder          # Linux
.\bench\run_bench.ps1                # Windows
```

**规程（AGENTS.md §5，全部强制）**：
1. warmup ≥ 20、iters ≥ 100；报告 median/min/max 与 RSD；
2. **RSD > 5% 判为无效测量**：检查时钟策略与温度，稳态后重测；
3. 同一 AR 的所有对比数字必须同一会话、同一时钟策略产出；
4. `--check` 先行：正确性不过，性能数字作废；
5. CSV 自动落盘（含 git sha / GPU 状态）；测量前后 `gpu_state` 差异过大须记录。

---

## 5. 性能门验收（四门之二；±10% 条款见详设 §7）

主场景 4096³，判据 = `gflops(median)`：

| Kernel | 目标 | 验收下限（目标×0.90） | 终验附加门 |
|--------|------|----------------------|-----------|
| naive | 113.55 GF | 102 GF | — |
| coalesced | 740.44 GF | 666 GF | ≥ 6.0× vs naive |
| smem1d | 1.43 TF | 1.29 TF | ≥ 1.7× vs coalesced |
| tile2d | 3.51 TF | 3.16 TF | ≥ 2.2× vs smem1d |
| vec4 | 5.84 TF | 5.26 TF | ≥ 1.5× vs tile2d |
| cpasync | 6.61 TF | 5.95 TF | **≥ 6.3 TF 且 ≥ 97% × 本机 cuBLAS FP32** |
| cublas | 实测 | — | 100% 参考线 |

**不达标处理流程**（PROMPT.md §6 数据真实性军规）：
1. 复测 3 次取中位，记录频率/温度曲线；
2. 若频率 < 策略频率 5% 以上 → 功耗墙归因，冷却后重测；
3. 仍不达标 → 如实记录实测值 + ncu 证据 + 差距归因，写入 bottleneck_analysis.md，
   **禁止**调整数字或选择性汇报；
4. 相邻版本加速比异常（如 < 1.3×）→ 先查 ncu 指标是否符合该版"预期证据"
   （见 §7 预期签名表），定位实现偏差而非直接调参。

---

## 6. 资源门验收（四门之三）

数据源：`build.log`（-Xptxas -v）+ ncu `launch__registers_per_thread` /
`launch__shared_mem_per_block_static` / `sm__warps_active.avg.pct_of_peak_sustained_active`。

| 检查 | 判据 |
|------|------|
| spill | tile2d/vec4/cpasync 必须 = 0 |
| regs | 记录实际值；`--lb 2` 消融时 ≤ 128 |
| bank conflict | vec4/cpasync2：smem ld/st conflict ≈ 0；cpasync(甲)：a-frag 允许 2-way（设计已知项，须实测确认 ≤2-way） |
| occupancy | 记录 achieved；对照理论值解释差异 |

---

## 7. Nsight Compute 分析验收（四门之四）

```bash
./profile/profile_all.sh              # 全 kernel，指标集 = 详设 §4.5
# Windows: .\profile\profile_all.ps1
```

产出：`profile/<kernel>/{*.ncu-rep, *.csv, *.details.txt}`。

**各版本"预期指标签名"**（bottleneck_analysis.md 闭环的对照基线；
偏差过大 = 实现有问题，先查代码再下结论）：

| kernel | DRAM bytes vs 理论最小 | sectors/req (ld) | 主要 stall | bank conflict |
|--------|----------------------|-----------------|-----------|---------------|
| naive | ≫ 数十倍 | ≫ 4 | long scoreboard | — |
| coalesced | 仍数十倍（无分块） | → 4 | long scoreboard（下降） | — |
| smem1d | 降至 tile 级 | ~4 | barrier / short scoreboard 上升 | 少量 |
| tile2d | 进一步降 | ~4 | short scoreboard（smem）+ wait | a 0-way；b 4-way（已知项） |
| vec4 | 接近最小 | ≈4（128B 满） | wait / short scoreboard | **≈0（swizzle 生效证据）** |
| cpasync | ≈ 最小，DRAM 吞吐近峰 | ≈4 | **barrier / 依赖等待为主** | b ≈0；a ≤2-way（甲） |
| cpasync2 | 同上 | ≈4 | 同上 | ≈0（全消） |

**闭环文档**：每 kernel 追加一段到 `results/bottleneck_analysis.md`，格式固定：
```
## Kernel N — <名称>
- 瓶颈：<一句话>
- 证据：<指标名=数值>（来源 profile/<kernel>/<kernel>.csv）
- 对策：<下一版做了什么>
- 验证：<下一版对应指标的变化数值>  [由下一版回填]
```

---

## 8. Sanitizer 验收

```bash
./profile/sanitize.sh memcheck                 # 全 kernel × {1024³, 1023×1024×511}
./profile/sanitize.sh racecheck cpasync        # AR006 硬门
./profile/sanitize.sh racecheck cpasync cpasync2   # 双方案都查
```

**判据**：memcheck 零 error；racecheck 零 hazard（若报 cp.async pipeline 相关
hazard，逐条核对是否为 `__pipeline_wait_prior` + 双 `__syncthreads` 结构性误报，
确认后归因记录，不得静默忽略）。

---

## 9. 结果归档（每 AR ST 通过后执行）

1. `results/performance.csv` 提交（与代码 commit 同步）；
2. `profile/<kernel>/*.csv, *.details.txt` 提交（.ncu-rep 本地留存，git 忽略）；
3. `results/bottleneck_analysis.md` 追加闭环段并提交；
4. tasks.md 勾选对应任务为 passing，附证据指针（文件+关键数值）；
5. AR 目录移入 `specs/archive/`，从 `specs/backlog/` 取下一 AR。

---

## 10. 预构建代码的已知风险清单（首次 GPU 会话必读）

以下问题因无 GPU 未验证，按风险排序：

| # | 风险点 | 位置 | 验证方法 |
|---|--------|------|---------|
| 1 | 编译错误（模板/语法/头文件） | 全部 .cu | §2 构建；预期首轮流为小修 |
| 2 | ncu 指标名在不同版本间差异 | profile_all.* | `ncu --query-metrics \| grep <name>`，缺项用 --set full 等价页签 |
| 3 | `__pipeline_memcpy_async` zfill 语义（zfill=16 时 src 不读） | sgemm_cpasync.cu | 边界用例 correctness + memcheck；如有问题改为"越界时先 cp.async 合法地址再补零覆盖" |
| 4 | smem 声明尺寸超限 / 静态 smem 上限 | cpasync（2 缓冲 ×2 矩阵 ≈ 16.6KB） | §2 构建（超限会编译错） |
| 5 | bank conflict 实测与注释推导不符 | smem_1d/2d_tile/vec4/cpasync | §7 ncu conflict 计数；以实测为准修订注释（D7 决策） |
| 6 | naive 的"刻意不合并"映射在部分编译器下被优化 | sgemm_naive.cu | E3 实验核对 sectors/request ≫ 4 |
| 7 | Windows 下 `_popen` nvidia-smi 路径 | common.h | 运行期 best-effort，失败仅影响元数据列 |
