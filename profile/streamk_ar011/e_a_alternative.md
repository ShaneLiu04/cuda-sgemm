# E-A 替代归因协议（P0 不可得分支）— AR011 T007 收口
# 日期：2026-10-07 | 状态：P0（admin/TCC + ncu 权限）未解锁，按 design §T007
# 不可得分支收口；若后续解锁 ncu（ERR_NVGPUCTRPERM 消除），本文件结论可用
# stall 四分类实测直接复核（复核路径见文末）。

## 0. 协议依据

- design T007：admin/TCC 可得 → ncu stall 四分类；不可得 → 环境声明落盘 +
  E-A 替代协议（分段计时 + T004 消融间接归因）+ %peak 不变量口径延续。
- 权限状态：ncu 实测 ERR_NVGPUCTRPERM（计数器权限缺失，WDDM + 无 admin），
  申请记录见 `results/environment.md` §7（2026-10-07 发起，用户侧待办）。

## 1. 1024³ 主核损失的四分类假设 vs 实测证据（G1 缺口 1.05μs@341μs）

dsk sk3 分段计时（本会话 3 rep 重锚，1024³，WDDM 稳态）：
**main 298.2-307.2μs + reduce 42.5-43.4μs**（复现 AR010 297+41 基线，
reduce share ~12.4%）。cuBLAS 同协议 255.9μs（canonical 8397.5 GF）。
缺口 = (298.2+42.5) − 255.9 ≈ 84.8μs，其中 reduce 归并占 ~43μs。

| # | 假设（design 四分类） | 判据/实验 | 结果 |
|---|---------------------|-----------|------|
| 1 | 同相停顿（warp 齐步 B 同 kk，multicast/依赖耦合） | T004 PHASE 错相消融（lat_cover_ar011.csv）：错相 **-0.8%~-2.1% 负效应** → 同相不是损失源，反而是 multicast 红利 | **排除** |
| 2 | 步首依赖（kk 步进 LDS→FFMA 未解耦） | T004 BPF 预取消融：+0.37%/-0.05% 噪声内零增益 → dbuf 双缓冲已覆盖加载延迟 | **排除** |
| 3 | 尾 tile / 波量化（部分波 SM 空转） | T005 sweep 实测：2048³（t=128, 2.67 波）deep→streamk W=1 **+2.1%**；1024×2048（t=64, 1.33 波）deep→streamk **+28%**；1024³（dsk 96 块恒 2 整波）无尾波税但 1 块/SM MLP 减半 → streamk W=1 反输 13% | **确认**（在非整波几何成立；1024³ 本身无此项损失） |
| 4 | barrier 歪斜（__syncthreads 等待最长 warp） | 无 ncu 无法直接分解；间接证据：dbuf1（单同步）全尺寸 +3~14% > dbuf2（双同步，AR010 T004）→ 同步次数本身是税，但单同步内歪斜幅度未知 | **未决**（需 ncu 复核） |

结论：四分类中 **2 项排除、1 项确认、1 项未决**（超出 design "至少排除或
确认两个" 的下限）。1024³ G1 缺口的构成（84.8μs）：reduce 归并 ~43μs
（结构性，split-K 代价）+ main 相对 cuBLAS ~42μs（其中假设 1/2 已排除、
假设 3 在 1024³ 不适用 → 主因收窄至假设 4 barrier 歪斜与 cuBLAS 调度
差异，未决待 ncu）。

## 2. %peak 不变量口径延续

- G2@256³ %peak 口径（钟态匹配 PASS）与 G3@4096³（≥7.0TF）沿用 AR010
  判定口径：同会话 cublas 锚 + 钟态列核对（run_paired_v4 先例），本 AR
  不引入新口径。
- WDDM 无锁频下的钟态纪律：v4 协议（47°C 冷却 + cublas 每轮首锚 +
  gpu_state 列留证）+ T006 §9 补充尺寸同制度配对原则（AR011 新增）。

## 3. E-B / E-C（P0 关联项，未执行）

- E-B cuBLAS@1024³ launch config 反测（cublasLt heuristic 查询）：非
  不可得分支必需项，未执行；若 G1 终判 miss 需归因 cuBLAS 调度优势，
  可在后续 AR 补做（纯 runtime API，无需 admin）。
- E-C 锁频五门复判：需 `nvidia-smi -lgc` 权限，未执行；WDDM 口径结论
  不因此降级（AGENTS §5.3 无权限则稳态协议，已遵守）。

## 4. ncu 解锁后的复核路径（backlog）

1. `ncu --set full --kernel-name regex:sgemm_deep sgemm_bench --m 1024 ...`
   → `smsp__pcsamp_warps_issue_stalled_*` 四分类计数，验证 §1 表；
2. racecheck 票据路径专项（AR011 硬门已过，ncu 解锁后可加 persist 项）；
3. E-B：cublasLt `cublasLtMatmulAlgoGetHeuristic` 枚举 @1024³；
4. E-C：`nvidia-smi -lgc 1815` 后五门复判，与 WDDM 口径并行报告。
