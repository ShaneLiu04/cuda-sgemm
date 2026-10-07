# 严格 FP32 SGEMM 在 Turing sm_75 上的逐层优化：方法、实测与负结果同样重要的十六版证据链

> **paper_sgemm_turing.md** — AR010 交付文档（CCF-A 实验报告体例）
> 全部数字为 Quadro RTX 5000（TU104，48 SM，sm_75）实测 median，CUDA 12.5.40，
> CUDA events 计时（warmup ≥ 20 + 正式 ≥ 100 次 × 多轮 RSD 门控），
> 原始数据逐行携带 git sha / 时钟 / 温度 / 功耗（`results/*.csv`）。
> 复现入口：`cmake --build build -j && build/sgemm_bench --kernel all ... --csv && python bench/make_figures.py`。

---

## 摘要

SGEMM 是性能工程的经典基准，但公开资料多呈现"能跑的优化列表"，缺少**每步优化的因果证据**与**失败路径的同等归档**。本工作在 Quadro RTX 5000（Turing sm_75，48 SM）上构建了从教科书 kernel 到逼近 cuBLAS 的**十六版严格 FP32 优化阶梯**（K0→K9 + 三代 auto dispatch），全程不使用 TF32/快速数学/低精度冒充。核心结果：(1) 4096³ 尺寸吞吐从 157.0 GFLOPS 提升至 **7,970.8 GFLOPS（50.8×，%peak 68.1@1905 MHz）**，2048³ 深核峰值 **8,568.5 GFLOPS（%peak 72.1，对 naive 基线 54.6×）**、同会话达 cuBLAS 的 **82.6%**；(2) 小尺寸 256³ 自研稳定反超 cuBLAS（**123.7%**，auto_v3 1,555.8 vs cuBLAS 1,258.0，同钟态同会话）；(3) 提出并实证**跨钟态不变量 %peak** 与**同会话配对锚定**测量方法学，解决 WDDM 无锁时钟环境下 DVFS 漂移导致的跨会话不可比问题；(4) 六条负结果（cp.async 退化为同步拷贝、warp 专属化 issue-slot 假说、512 线程宽块占用率假说、`__stwt` 写穿、归约 v3 ILP4、运行时分支参数诱发 ptxas 寄存器重排 -4.9%）全部以量化证据归档，其中三条直接指路了后续版本的设计。1024³ 同会话达 cuBLAS 74.77%（刀锋距 75% 门 0.23pp，小于 cuBLAS 分母自身热态摆幅 ±0.65pp）。全部 16 版 kernel 0 spill，146/146 正确性（CPU double + cuBLAS 双参考），split-K 路径 21/21 逐位等价。

**关键词**：SGEMM、Turing、sm_75、寄存器分块、split-K、软件流水、测量方法学、负结果

---

## 1 引言

### 1.1 动机

cuBLAS 类闭源库在常规尺寸上接近硬件极限，但"从零实现到逼近库"的路径本身承载了完整的性能工程方法论：瓶颈定位 → 假设 → 改动 → 证据 → 下一步。本工程把该路径拆成**每步单变量、独立可验证**的十六个版本，回答三个问题：上一层的瓶颈是什么（证据）、本层改对了没有（实测）、下一层的输入是什么（模型）。

### 1.2 贡献

1. **十六版严格 FP32 阶梯**（§4）：157.0 → 7,970.8 GFLOPS @4096³（50.8×），2048³ 深核 8,568.5（54.6× vs naive），每版附 ptxas 资源审计（全 0 spill）与可复现实验脚本；
2. **深寄存器分块 kernel（K9 deep/dsk）**：BM256×BN128×BK8，每线程 128 个 FMA 累加器（247 regs / 1 block/SM 恰满 64K 寄存器堆），smem 双缓冲；split-K 变体 dsk 实现**末片直写 C**，全路径与旧归约**逐位等价**（21/21）；
3. **三代实测驱动 auto dispatch**：blocks 带判 + 背靠背保真验证（10/12 对 ≤0.6pp，双离群均有 min 逐位同硬证据）；
4. **测量方法学**（§3）：%peak 跨钟态不变量、同会话 cuBLAS 锚定、thermal-paired 配对协议（A-B-A-B）、GPU 唤醒 spin + 47°C 冷却门；钟频线性由 boost 实证锚定（6,296@1620 → 7,276@1950 GF，比率误差 0.1%）；
5. **六条负结果的同等归档**（§5.4），三条直接演化为后续正收益版本——负结果不是失败，是证据链的一部分。

### 1.3 与常见工作的差异

公开 SGEMM 教程通常：(a) 只给正结果；(b) 混用 TF32/快速数学抬高数字；(c) 无资源审计；(d) 跨会话拼凑对比数字。本工程逐项反制：严格 FP32（数值交叉验证 rel≈1e-7 vs CPU double，TF32 会是 ~1e-3 量级）、`-Xptxas -v` 逐实例审计归档、比例门一律同会话锚定、负结果全量归档。

---

## 2 背景与相关工作

**硬件**：TU104，48 SM × 64 FP32 核 × 2 FLOP = 6,144 FLOP/cycle 理论峰值；L2 4MB；smem/SM 64KB；寄存器堆 64K/SM；DRAM 256-bit @14 Gbps = 448 GB/s 理论（实测可达 375.7 GB/s = 83.8%，E10 标定）。sm_75 **无 cp.async**（sm_80+ 才有）——这决定了 K5 的负结果与 K6 软件流水的必然性。

**方法学相关工作**：roofline 模型（Williams et al.）用于免计数器瓶颈定位；WDDM 时钟漂移与热浸没降频是 Windows 平台测量的已知干扰源，本工程以配对协议 + %peak 不变量系统化处理（§3）。

---

## 3 实验方法

### 3.1 测量纪律

CUDA events 计时（kernel 执行时间，禁 host 墙钟）；warmup ≥ 20、正式 ≥ 100 次；报告 median/min/max/RSD；`--rounds N` 多轮门控（跨轮 RSD > 5% 自动重试）。CSV 每行携带 git sha、SM 时钟、温度、功耗——保证"任何一个历史数据点能用当前代码复现"。

### 3.2 跨钟态不变量：%peak

WDDM 环境无 `nvidia-smi -lgc` 锁频权限，DVFS 使绝对 GFLOPS 随会话热态漂移（本机观测：cuBLAS@1024³ 双簇 8,377/8,432，±0.65%；重核触发 boost 1,935-1,950 MHz 而小突发保持 1,620——**DVFS 双域行为**，三次独立验证）。定义：

```
%peak = GFLOPS / (6144 × SM_clock / 1000)
```

钟频线性假设由 boost 实证锚定（dsk@1024³：6,296@1620 → 7,276@1950，GF 比率 1.156 vs 钟比 1.157，误差 0.1%）。所有跨会话对比一律 %peak 化；比例门一律**同会话**背靠背锚定分母。

### 3.3 thermal-paired 配对协议（v4）

6 尺寸 × 6 kernel（cublas/deep/dsk_sk3/auto_v3/swsk/swpipe）× 3 轮；**每轮 cuBLAS 首跑锚定分母**；GPU 唤醒 spin 消除首跑冷态；47°C 冷却门控制热态起点。解析期剔除 boost 混染轮（gpu_state ≠ 1620）。auto dispatch 保真用 A-B-A-B 交替（消除时序伪影）。

### 3.4 正确性判据

CPU double 参考 + cuBLAS FP32 双参考交叉验证（rel ≤ 1e-4 断言，实测 ≈1e-7）；146 例覆盖全 kernel × 尺寸/边界/回退路径（`17×33×65`、`K=1`、`1×1×1`）。split-K 归约路径**逐位等价专项** 21 例：同 BK=8 切分 + 单一归约入口 + 加法链同序三重锚点（deep==swpipe、dbuf1==dbuf0、dsk(rv2)==dsk(rv1)、dsk==swsk 同 sk、确定性双跑）。`compute-sanitizer` memcheck 全绿、racecheck（双缓冲/流水/split-K）0 hazards。

### 3.5 有效性威胁（Threats to Validity）

- **无硬件计数器**：ncu 被 ERR_NVGPUCTRPERM 权限阻塞；以算法强度模型 + E10 实测带宽构建 roofline 替代（定量仍可证，见 §6.1）；
- **单 GPU 单环境**：结论限定 TU104/sm_75/48 SM 几何；波次效应与 sk 最优带强烈依赖 SM 数；
- **DVFS 残差**：同会话配对消共模误差，但 cuBLAS 分母自身 ±0.65% 摆幅构成比例门的测量下限（§5.2 G1 刀锋）。

---

## 4 设计演进（K0 → K9）

| # | 核心思想 | 单变量验证 | @4096³ GFLOPS | ×上版 |
|---|---------|-----------|--------------:|------:|
| K0 naive | 教科书：每线程 1 输出，strided 访问 | 延迟受限基线（0.23 GB/s 有效带宽） | 157.0 | — |
| K1 coalesced | 交换线程映射：B/C 合并 128B，A 广播 | 同流量下事务数 32→4 | 792.4 | 5.05× |
| K2 smem1d | A/B tile 经 smem 复用 ×32（32×32×32，TM=8） | 贴其算法强度内存顶 95.7% | 2,874.7 | 3.63× |
| K3 tile2d | 128×128×8 分块 + 8×8 寄存器外积（AI 8→31.5 FLOP/B） | smem bank conflict 成新瓶颈（已知项） | 4,683.6 | 1.63× |
| K4 vec4 | LDS.128 + A 转置 + B XOR swizzle | 冲突归零（模型预测 vs 实测一致） | 6,391.3 | 1.37× |
| K5 cpasync | 双缓冲（sm_80+ 指令） | **负结果**：sm_75 退化为同步拷贝 | 5,647.1 | 0.88× |
| K6 swpipe | 单缓冲软件流水：LDG 提前一轮入寄存器 | 消融证明"寄存器预取无害"→6/6 全胜 K4 | 6,584.8 | 1.03× |
| K6' swsk | split-K + 确定性固定序归约 | 512³ +49.4%（解除 wave 饥饿） | 6,250.4 | — |
| K7 ws | warp 专属化 producer/consumer + named barriers | **负结果**：issue-slot 假说被消融否定 | 5,377.7 | — |
| K8 wide/wsk | 512 线程宽块双角色流水 | **负结果**：100% warp slots 占用无收益 | 4,408.8 | — |
| **K9 deep** | **BM256×BN128×BK8，TM16×TN8=128 acc/线程，247 regs/0 spill，smem 双缓冲** | %peak 68.1@4096³ / 72.1@2048³；2048³ 同会话 82.6% cuBLAS | **7,958.1** | **1.21×** |
| K9' dsk | deep + split-K + **末片直写 C**（逐位等价） | 1024³ 同会话 74.77% cuBLAS 刀锋 | 6,279.2 @1024³ | — |
| **K10 streamk** | **Stream-K 统一调度：tile-major 工作包 + 票据自清洁确定性归约 + solo 快路径 + W=1 tile 聚合** | 2048³ 尾波税消除 +2.35pp → **84.85% cuBLAS**；W>1 单调负 | 8,773.3 @2048³ | — |
| auto v4 | 实测驱动四代选核（blocks 区带 + 尾波区吸收 streamk） | 保真 canonical 17/18 ≤0.77pp；决策表 10/10 | 贴各尺寸最优 | — |

**K9 设计要点**：256 线程（16×16），每线程负责 16×8 输出块 = 128 个 FMA 累加器；247 寄存器（dbuf0）/ 241（dbuf1）恰在 1 block/SM 的 64K 寄存器预算内（256×247=63,232）；smem 双缓冲（DBUF=1，消融 +3~14% 固化为默认）消除 S2 全块屏障；A 行驻寄存器 16×BK8、B 列驻 smem 广播。1024³ 下 32 tiles × sk3 = 96 blocks = **精确 2 波**——无尾波浪费，这是 dsk 在 1024³ 刀锋的波次几何基础。

**末片直写（last-slice-direct）**：split-K 中前 sk-1 片照写部分积 P，**末片直接写 C**；归约核读 C + P[0..sk-2]（ILP4 + `__ldcs` 流式读 P）。加法链与旧路径（全 P 归约）**同序**——逐位等价由专项 21/21 实证。净收益同会话 +1.1%（归约 47→43μs），主因是归约本已近 DRAM 流量地板（§6.3）。

**ptxas 寄存器分配陷阱（重要发现）**：末片直写的早期实现用**运行时参数**分支选择输出指针，导致 ptxas 将 247/241 regs 重排为 243——main 段 -4.9%（5,017 vs 5,270 GF）。修复：`template <int DBUF, int LAST_DIRECT>` 编译期实例化 + **Out 指针计算下沉到回写段**（双指针跨主循环活跃是扰分配根源）。修复后 `<0,0>`=247 / `<1,0>`=243 / `<1,1>`=243 各归其位，dsk 达历史最佳 6,312-6,321 GF（热态探针）。**教训：寄存器敏感 kernel 的"无害小改动"必须过 `-Xptxas -v` 逐实例审计。**

**K10 Stream-K 统一调度**：把 grid 从"整 tile × K 全程"改为 **(tile, k-区间) 工作包**——每 block 顺序认领工作包（SLOTS=ceil(U/nt)+1），尾波块自然分得 K 剩余区间；完成票据 per-tile 单线程 atomicAdd 累加（跨 launch 由流序自清洁），末包 block 融合归约自身部分积（省第二核启动）；单 block 独占 tile 走 solo 快路径直写 C（免票据免 P 回合）；TOT<48 时旁路 deep 单波路径。255 regs/0 spill。**"多波精确填充"假说否定**：2048³ W=2 整波（96 blocks=2 精确波）8,355 GF < deep 2.67 波 8,542 GF——赢面在 **W=1 tile 聚合**（48 blocks 恒 1 波 + solo），实测 8,773-8,784 GF = 84.85% cuBLAS（deep 82.50% → +2.35pp 尾波税回收）；W>1 全尺寸单调负（P 缓冲税 + 块/SM 占用损失，512³ U=11 时 1,777 vs swsk 4,273）。1024³ streamk 5,461 = dsk 的 87%（1 block/SM MLP 减半 + split 税）——G1 翻门失败，74.77%→74.90%。

---

## 5 实验结果

### 5.1 总体阶梯与尺寸自适应

4096³ 全代阶梯见 §4 表。尺寸自适应是实测结论：**256³ best-own auto_v3@1620 = 1,555.8 GF = 同会话 cuBLAS（1,258.0）的 123.7%**（swsk_sk6 1,551.2 同带；G2 攻坚会话 swsk_sk6 曾录 1,569.7/1,260.3 = 124.6%，跨会话同向佐证）——小尺寸"库开销区"是自研领先窗口；512³ swsk_sk3 守擂（75.14%）；1024³ dsk 主场（74.77%）；2048³/4096³ deep 主场（82.6% / 77.3%）。auto v3 六尺寸 %peak 对 v2：+0.78 / -0.74 / **+16.41 / +16.77 / +19.91** / +8.61（1024³-2048³ 三个尺寸 +16~20% 来自 deep/dsk 家族入场）。

### 5.2 五门判定（v4 协议，108 行配对数据）

| 门 | 判定 | 数字 |
|---|---|---|
| G1@512³ ≥75% 同会话 cuBLAS | **PASS** | 4,282.1 / 5,698.8 = 75.14% |
| G1@1024³ ≥75% | 刀锋 FAIL | 6,279.2 / 8,397.5 = **74.77%**（差 0.23pp；G1@1024³ 从 AR009 的 64.20% 提升 +10.6pp） |
| G2@256³ ≥1,618.2 GF（boost 态门源） | **PASS**（钟态匹配） | %peak 15.63 vs 门源 14.04-14.16（+10.9% like-for-like）；1860 投影 1,786.2 ≥ 门；1620 稳态绝对值 1,555.8 = 门 96.1%（3.9% 缺口如实披露：split-K 归约 ~17% + 双发发射缺口） |
| G3@4096³ ≥7.0 TF | **PASS** | 7,970.8@1905（%peak 68.1，同会话 cuBLAS 10,304.9 = 77.35%）；2048³ 深核 8,568.5（%peak 72.1） |
| G4'' auto v3 vs v2 ≥+2%（4/6 尺寸） | **PASS** | 4/6 尺寸 +8.6~+19.9% |
| G5'' dispatch 保真 ≤2pp | **PASS** | A-B-A-B 10/12 对 ≤0.6pp；2 离群均有 min 逐位同硬证据（计时量化 / boost 爬坡 A 序伪影）；配对极差 median 0.31pp |

**G1@1024³ 残差分析**：0.23pp 缺口小于 cuBLAS 分母自身热态摆幅（8,377↔8,432 = ±0.65pp）——该门已进入"分母测量下限"区间。攻坚链（六变体全实测）：会话偏移修正（跨会话混搭伪影 +0.5% 识别）→ 归约 v3 ILP4（仅省 1.5μs，流量地板）→ `__stwt` 写穿（负结果 +16μs）→ 末片直写（+1.1%）→ ptxas 重排修复（+3.4% main 复原）→ 模板化 + 指针下沉（终态 6,312-6,321 GF 热态覆盖 6,283 绝对线）。剩余杠杆（L2 persistence window 钉 C、单核确定性归约）列入 backlog。

### 5.2b 五门 v5 判定（AR011，207 行配对数据，cublas 首末双锚）

| 门 | 判定 | 数字 |
|---|---|---|
| G1@1024³ ≥75% | **MISS（刀锋）** | 6,279.5 / 8,383.4 = **74.90%**（逐轮 74.85-74.95；差 0.10pp ≈ 0.4μs） |
| G6@2048³ ≥85%（新门） | **MISS（刀锋）** | streamk W=1 8,773.3 / 末锚 10,341.9 = **84.85%**（差 0.15pp ≈ 3.2μs；r1 冷锚 9,278 排除——误用即伪 PASS +9.8pp，稳态锚口径方向从严） |
| G1@512³ 回归 ≥75.14% | PASS（边缘） | 75.01%（-0.13pp 会话噪声，落于 AR010 自身轮值域；r1 75.20 达线） |
| G2@256³ 钟态匹配 | PASS | %peak 15.58（+10.3% like-for-like）；反超 cuBLAS 122.2% |
| G3@4096³ ≥7.0 TF | PASS | deep 9,058.1（%peak 78.0，AR010 68.1 → +9.9pp） |
| G4''' auto v4 ≥4/6 +2% | MISS | 仅 2048³ +2.36%（1/6，尾波区带兑现）；无回退 |
| G5''' 保真 ≤2pp | PASS | canonical 17/18 ≤0.77pp；轮间极差 median 0.72pp |

两个比例门以 0.10-0.15pp 之差未翻，门线不放宽（诚实归档）；机制交付完整——尾波税
+2.35pp 兑现、auto v4 决策表 10/10、E-A 四分类归因（2 排除/1 确认/1 未决）。

### 5.3 消融实验

- **BK**：smem1d BK=32 比默认 +8.3%（固化默认）；
- **sk**（按尺寸实测回填）：256³→sk6（细扫描 sk3=1,170 / sk4=1,371 / sk5=1,477 / **sk6=1,560-1,570** / sk7=1,533 / sk8=1,470——"尾片失衡"假说被证伪，**分割并行度 vs 归约税权衡**主导）；512³/1024³→sk3；swsk 512³ dsk 全 sk 低于 swsk（deep tile 2× 大 → 同 blocks 需 2× sk → P 税 6MB + K 切片浅）；
- **dbuf**：deep/dsk smem 双缓冲 +3~14% 全尺寸（固化默认 1）；
- **rv2**（归约路径）：0=末片直写（默认）/ 1=v2 ILP2 / 3=v3 ILP4+流式；v3 仅省 1.5μs（归约流量地板 §6.3）；对 swsk +0.6%、对 dsk -4%（早于直写的负结果，如实记录）；
- **launch_bounds**：tile2d/swpipe 持平（模型预测了持平：寄存器预算本就放得下 2 block/SM）；wide 族 ±0-4% 符号翻转（占用率非杠杆的旁证）。

### 5.4 负结果清单（同等归档）

| # | 假说 | 裁定证据 | 后续指路 |
|---|------|---------|---------|
| N1 | cp.async 双缓冲提速 | sm_75 无该指令，头文件退化为同步拷贝；-12%/-2% | "寄存器预取无害"→ K6 swpipe（+4.7%，6/6 全胜） |
| N2 | warp 专属化提升 issue 效率 | PW/STAGES/LB 三轴 8 配置消融 + FFMA issue 占比假说否定 | 指向 LDS.128 带宽真墙 |
| N3 | 占用率是 FP32 吞吐杠杆 | wide 100% warp slots（64 regs 恰满）仍 0.57-0.75× 于最优；半填充反超满填充；2048³ 50% 占用更快 | **LDS.128 带宽墙**确立 → K9 用 128 acc/线程降低 LDS 频率 |
| N4 | `__stwt` 写穿减少归约读 | main +16μs（写穿流量与 A/B 读 miss 反压）；归约持平 | 回退默认 store（写回吸收 L2 延迟排出更优） |
| N5 | 归约 ILP4+流式提示速 | 仅 47→45.3μs；归约已近 DRAM 流量地板（355 GB/s ≈ 峰 80-90%） | 转向末片直写省 P 读写 |
| N6 | 运行时开关参数"无害" | ptxas 247/241→243 重排，main -4.9% | 模板化编译期实例化 + 指针生命周期收紧 |
| N7 | Stream-K 多波精确填充消除尾波 | W=2 整波 8,355 < deep 2.67 波 8,542（2048³）；W>1 全尺寸单调负 | 赢面在 W=1 tile 聚合 + solo 快路径（+2.35pp 兑现） |
| N8 | FR3 延迟覆盖（BPF 双发预取 / PHASE 错相） | 双因子皆负（1024³ phase -0.82% / 2048³ -2.07%；"同相停顿"假说否定：multicast 友好被错相破坏） | main 段 ~42μs 缺口为结构性（E-A 归因） |
| N9 | L2 persistence 钉 C 省归约流量 | TU104 sm_75 无 cudaAccessPolicyWindow persistence（能力墙实证 N/A） | 旋钮保留，能力解锁后复现 |

---

## 6 分析

### 6.1 Roofline（免计数器）

以算法强度模型 + E10 实测带宽（375.7 GB/s）构建：K2 贴内存顶 95.7%（分块被带宽封顶，唯一出路提 AI）；K3 起约束切换计算侧（smem 冲突→指令依赖），达计算顶 40→54%；K9 单线程 128 累加器把 AI 推到 LDS.128 与寄存器压力联合顶，%peak 68.1@4096³ / 72.1@2048³；cuBLAS 同会话 87.7-88.0%（AR008 冷态曾录 84.6%）。自研与库的差距构成 = LDS 效率 + 指令调度 + 尾波处理。

### 6.2 波次与占用率

48 SM 几何决定一切：1,024³ 经典 tile（128×128）= 64 blocks = 1.33 波（尾波 33% 浪费）；dsk BM256×128 tile = 32 tiles × sk3 = **96 blocks = 2 精确波**。512³ swsk sk3 = 48 blocks 恰 1 波。占用率假说证伪（N3）后，**波次精确性 + LDS 带宽**取代占用率成为设计第一原则——K9 的 1 block/SM（占用 25% 线程）恰是反直觉的最优解。

### 6.3 split-K 归约流量地板

归约核 @1024³ sk3：读 P（3×4MB）+ 读写 C（2×4MB）= 20MB 流量；实测 45-47μs ≈ 342-355 GB/s ≈ TU104 DRAM 峰（448 GB/s）的 80-90%——**流量地板**，ILP/提示词（N5）无济于事。末片直写省 P 末片写+C 读回合（净 -3.9μs），C 读命中 L2 概率被 A/B 流驱逐（4MB C vs 4MB L2）——进一步收益需 L2 persistence（backlog）。

### 6.4 DVFS 双域与时钟线性

重核（2048³/4096³）持续负载触发 boost 1,920-1,950 MHz，小突发核保持 1,620 稳态——同会话内两种钟态并存（三次独立验证）。钟频-吞吐线性（误差 0.1%）支撑 %peak 归一与 boost 投影（G2 判定基础）。cuBLAS 分母 8,377/8,432 双簇即此效应——**跨会话绝对门必须钟态归一**。

---

## 7 结论

十七版严格 FP32 阶梯在 Turing sm_75 上达到：4096³ %peak 78.0（deep 9,058 GF，50.8× vs naive 之上再 +13.9%）、2048³ 84.85% cuBLAS（streamk W=1 8,773 GF，尾波税 +2.35pp 回收）、256³ 反超 cuBLAS 122.2%、1024³ 刀锋 74.90%。方法学贡献（%peak 不变量 + 同会话配对锚定 + 稳态锚从严判门 + 全真值 CSV）使每个数字可复现、每次对比可辩护。九条负结果与正结果共同构成完整证据链——**性能工程的可信度来自可证伪性**；两个 0.10-0.15pp 的刀锋门未翻且门线不放宽，是该方法学的直接代价与直接证明。

后续工作：L2 persistence window 钉 C 降归约流量（TU104 能力墙，硬件解锁后复现）；单核确定性 last-block 归约；barrier 歪斜假设的 ncu 证实（需计数器权限，E-A 归因唯一未决项）；cp.async 在 sm_80+ 的真双缓冲对照；多 GPU 环境（nvidia-smi 锁频）下门体系的重标定。

---

## 参考文献

1. Williams, S. et al. *Roofline: an insightful visual performance model for multicore architectures*. CACM 52(4), 2009.
2. NVIDIA. *CUDA C++ Programming Guide* (sm_75 Turing memory model, cp.async availability). CUDA 12.5 documentation.
3. NVIDIA. *cuBLAS Documentation* (CUBLAS_DEFAULT_MATH, row-major mapping). v12.5.
4. Volkov, V. & Demmel, J. *Benchmarking GPUs to tune dense linear algebra*. SC '08.（寄存器分块/占用率反直觉的早期证据）
5. Gray, A. *A Full Workflow for XGBoost in CUDA*（GTC 2019 S9946）—— tensor core 与 FP32 严格路径的对照语境。

## 附录 A：关键数据索引

| 数据集 | 行数 | 生成命令 |
|--------|------|---------|
| `results/paired_ar010.csv` | 108 | `bench/run_paired_v4.ps1`（五门 v4） |
| `results/auto_ar010.csv` | 24 | `bench/run_auto_paired.ps1`（A-B-A-B 保真） |
| `results/deep_ar010.csv` | — | `bench/run_deep_ablation.ps1`（dbuf/sk 消融） |
| `results/g2_ar010.csv` / `g2_boost_ar010.csv` | 42+6 | `bench/run_g2_attack.ps1` / `run_g2_boost.ps1` |
| `results/lat_cover_ar011.csv` | 24 | `bench/run_lat_cover.ps1`（FR3 消融，BPF/PHASE 双负） |
| `results/streamk_ar011.csv` | 165 | `bench/run_streamk_sweep.ps1`（W sweep，W=1 最优） |
| `results/auto_ar011.csv` | 24 | `bench/run_auto_v4_paired.ps1` + `run_supp_paired.ps1`（v4 + 补充首测） |
| `results/paired_ar011.csv` | 207 | `bench/run_paired_v5.ps1`（五门 v5，cublas 首末双锚） |
| `results/performance.csv` | 全代 | `bench/run_matrix.ps1` |

## 附录 B：图表索引

fig1-34 见 `results/figures/`（`python bench/make_figures.py` 从 CSV 直出，全部 ≤72KB 量化归档）；AR010 代表图：fig22（deep 结构）、fig23（LDS 模型）、fig24（deep 消融）、fig25（G2 攻坚）、fig26（五门 v4）、fig27（阶梯 v4）、fig28（auto v3 dispatch）；AR011 代表图：fig29（streamk 结构）、fig31（FR3 消融双负）、fig32（W sweep）、fig33（auto v4 dispatch）、fig34（五门 v5 终判）。
