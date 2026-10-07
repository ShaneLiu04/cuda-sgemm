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

## Kernel 6 — swpipe 软件流水（AR007，2026-10-05 实测）：瓶颈闭环 + 负结果翻正

- **上游瓶颈（继承 K5 消融结论）**：vec4（K4）主循环中全局 LDG 与 LDS→FMA 同拍串行，
  全局延迟（~400-800 cycle 量级）未被计算覆盖；K5 cp.async 双缓冲本应对症，
  但 sm_75 无硬件指令退化为同步拷贝（-12%/-2%，负结果）。K5 方案乙消融证明
  **"A 转置 + 寄存器预取"本身几乎无害（-1.8%）**——该结论即 K6 设计依据。
- **对策（K6）**：单缓冲软件流水——每 tile 的 1×A-float4 + 1×B-float4 提前一轮
  LDG 入寄存器，store→smem 与 compute 交替，双 `__syncthreads` 隔离写读；
  布局/搬运划分与 vec4 完全一致（bank 冲突结论继承）。
- **证据**：
  - 正确性：83/83（主场景 rel=0，与 vec4 同累加顺序）；racecheck 0 hazards（单缓冲写读隔离验证）；
  - 资源：128/127 regs（lb 消融两实例）0 spill，8.1KB smem，同 vec4 的 2 block/SM；
  - 性能：6/6 尺寸全胜 vec4（256³ +4.2%、512³ +5.6%、4096³ 冷态 +1.2%）；
    4096³ 冷态 6584.8 GF = cuBLAS（10136.2）的 64.9%。
- **新瓶颈（下一版输入）**：swpipe 后达计算顶 ~55%（6585/11980），单级 LDG 预取已做满；
  剩余差距归因 SASS 级调度/更深 K 向 ILP/warp 专用装卸（报告 §7 表），
  在严格 FP32 + ptxas 工具链边界内不可再收敛——**本机自研 FP32 天花板已逼近**。
- **缺失项**：ncu 计数器（smsp stall 构成、inst_executed 比例）——同 K0 受阻于
  ERR_NVGPUCTRPERM，解锁步骤见下节。

---

## 待办：ncu 计数器解锁步骤（获得管理员权限后）

1. 管理员执行：
   `reg add "HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" /v RmGpuProfilerSupport /t REG_DWORD /d 1`
   （或在 NVIDIA 控制面板 → 桌面 → Manage GPU Performance Counter → Allow access to all users）
2. 重启驱动（重启系统或禁用/启用显卡）；
3. 重跑 `profile\profile_all.ps1 -Kernels naive`（其余 kernel 随各 AR 采集）；
4. 本文档回补「缺失项」三指标 + 修订推导数值。


---

## AR008 闭环 #1：wave 饥饿 → swsk split-K（证实）

- **瓶颈**：swpipe 128×128 tile 在中小尺寸 blocks < 48 SM——512³ 仅 16 blocks（33% 占用面）、
  1024³ 64 blocks=1.33 波（尾波空转），SM 大量闲置。
- **证据**：同会话配对 512³ swpipe 2370.3 vs swsk(sk4) **3542.5（+49.4%，3 轮极差 0.42pp）**；
  256³ swpipe 560.4 vs swsk(sk12) 1379.7（+146%）；sk 扫描（fig9a）显示最优 sk 随 blocks 变化
  （256³: 4→48 恰满单波；2048³: 256 blocks 饱和后 sk↑ 单调反噬）。
- **对策**：split-K 沿 K 切片抬波数 + 确定性固定序归约（workspace + 独立 kernel，逐位可复现）。
- **验证**：G4 全线 4/6 尺寸 delta ≥+2%（256/512/1024/1000×1016 全 HIT）；
  auto dispatch（swsk 带：≤64 blocks）背靠背对内 delta ≤0.9% vs 实测最优。
- **边界**：4096³ swsk -3.5%（P 写出反噬）→ auto 分界线 >64 blocks 回 swpipe，实测闭环。

## AR008 闭环 #2：issue-slot 竞争 → ws warp 专属化（否定，负结果归档）

- **假说**：大尺寸稳态循环 FFMA 发射槽占比 ~55%（AR007 推断）→ 剥离 LDG/STS 给专职
  producer warp 可显著抬升消费者发射槽利用率。
- **证据（消融矩阵 fig12 + 配对 fig13/14）**：ws 全 8 配置（PW{1,2}×STAGES{2,3}×LB{1,2}，
  升降序 2-pass）在 1024³/2048³/4096³ 全部不敌 swpipe（-7.9%/-20.4%/-15.0%）；4096³ 配对
  delta -16.00%（3 轮极差 0.33pp，结论稳健）；LB=2 双 block 驻留（62.5% 占用）3/4 尺寸劣于
  31.3%；PW=1 单 producer warp 全面更差（搬运吞吐不足）。
- **归因**：把搬运剥离出计算 warp 后，消费者 warp 内只剩 LDS→FFMA 依赖链，ILP 上限反而
  降低；swpipe 的寄存器预取混合流在同一 warp 内同时提供"内存指令填充发射槽 + 覆盖 DRAM
  延迟"两个收益——软件 warp specialization 在 sm_75（无 cp.async/mbarrier/专属化硬件）上
  得不偿失。与 AR006 cp.async 负结果（-12%）构成同族证据："硬件缺席时软件复刻"边界。
- **对策**：ws 保留为教学阶梯 Kernel 7（结构完整、racecheck 0 hazards、110/110），性能定位
  如实标注"负结果"；auto dispatch 不选 ws（实测败于 swsk/swpipe 全尺寸）。
- **验证**：G3 门（4096³ ws ≥7.0TF）FAIL——按数据真实性军规归档，fig13 即负结果证据图。

## AR008 方法论闭环 #3：WDDM 热漂移 → thermal-paired 协议（证实）

- **问题**：跨会话绝对值漂移（AR007→AR008 同 kernel -5%~-14%）掩蔽真实优化效果；
  会话内 WDDM 动态时钟双峰（1620↔1935MHz）致 swpipe 单组 benchmark 双峰分布。
- **对策**：交替配对（A,B）×3 轮 + 冷却门控，对内 delta 相消共模漂移。
- **验证**：全 24 组 delta 轮间极差 median **0.53pp**（G5 PASS）；对照 AR007 跨会话漂移
  ~5-14%。污染组（冷却超时 48C）按协议剔除重测，原始行备份
  `paired_ar008_thermal_bak.csv`——协议诚实性自证。

## AR009 闭环 #4：占用率墙 → wide/wsk 512 线程宽块 → LDS.128 带宽墙确证（假说证伪，负结果归档）

- **问题**（AR008 遗留）：swpipe 128 regs → 2 block/SM = 50% warp slots；三个 FAIL 门
  （G1/G2/G3）初步归因占用率不足。假说：100% 占用可显著抬升吞吐。
- **对策**：Kernel 8 wide（64×256×8 tile，TM4×TN8，512 线程双角色流水：搬运期 A/B loader
  分工 256+256，计算期全 512 线程；单缓冲双同步）+ wsk（wide+split-K，detail::swsk_reduce
  共享归约）。寄存器预算工程（基址预计算 + Out 指针延迟物化）达成 **LB=2 = 64 regs/0 spill，
  2×512×64 = 65536 恰满 64K → 100% warp slots 占用判据达成**（build.log ptxas 审计）。
- **验证（三重独立证据，全部否定占用率假说）**：
  1. T002 smoke：wide LB=1(50%) vs LB=2(100%) 全尺寸同速（fig16c）；
  2. T004 波消融（170 行，同会话双 kernel）：**半填充 sk3（48 blocks = 1/SM）反超满填充
     sk6（96 = 2/SM）**——wsk 2917 vs 2479（+18%）、swsk 4273 vs 3413（+25%）@512³（fig17）；
     split 开销（P 写出+归约+浅切片）> 额外 warp 并行收益；
  3. T005 LB 消融（42 行）：LB 效应 ±0~4% 且符号随配置翻转——wsk sk3 LB1 +4.1%@512³、
     sk12 LB2 +3.5%、**2048³ wide 50% 占用反而 +2.2% 更快**（fig18）。
- **结论**：sm_75 FP32 的真墙 = **LDS.128 带宽**（wide 4 LDS/64 FFMA = 1:16 vs swpipe
  平衡点 3:32；wide/wsk best-vs-best 全尺寸 0.57-0.75× 于 swsk 最优——fig17f/21）。
  swpipe 已处 LDS/FFMA Pareto 平衡点。wide/wsk 保留为教学阶梯 Kernel 8（128/128 正确、
  memcheck/racecheck 0、与 swpipe/swsk 数值逐位同源），性能定位如实标注负结果。
- **意外正收益**：①swsk sk3/sk6 调优点（512³ +21.5%/1024³ +13.3%/256³ sk6 最优）→ auto v2
  dispatch 回填（G4'/G5' PASS）；②**稳态测量纪律**：1620 MHz 持续态为无 -lgc 权限时的
  诚实基准（swsk_sk3@1024³ 连续 8 探针 8/8 = 5405.03 GFLOPS 丝毫不差；boost 1875-1950 行
  为瞬态彩票）；③4096³ 批间双峰披露（同 declared 时钟 6.5T vs 7.3T，核内有效时钟不可经
  行间查询观测）；④G3 首过（swpipe@4096³ 3 轮中位 7185.3 = 7.0T 门 102.6%）。
- **证据链**：fig16（结构+资源包络+LSU 墙预测）、fig17（pre-wave 扫描+波几何注记）、
  fig18（LB 消融三探针裁决）、fig19c（稳态纪律）、fig20（三门 v3 判定）、fig21（全 kernel
  阶梯）；CSV：ablation_ar009/ablation_wlb_ar009/auto_ar009(+dispatchA)/boost_lottery/
  paired_ar009/smoke_wide/smoke_wsk；判定文书 compare_ar009_paired.md。

## AR010 闭环 #5：LDS.128 带宽墙 → deep/dsk 128 累加器深分块 + 末片直写 → 归约流量地板（残差 0.23pp 归档）

- **问题**（AR009 遗留）：LDS.128 带宽墙确证后，swpipe 家族 54.5% peak @1024³ vs cuBLAS 84%——
  1:16 LDS:FFMA 比例下继续加 warp/块已无意义，必须降每 FFMA 的 LDS 次数。
- **对策**：Kernel 9 deep——BM256×BN128×BK8、256 线程、每线程 16×8 = **128 FMA 累加器**
  （LDS:FFMA 恶化换取计算密度），247 regs/0 spill 恰满 64K 寄存器堆（1 block/SM = 25% 线程占用，
  与 AR009 结论自洽）；A 行驻寄存器 16×BK8、B 列 smem 广播、--dbuf 双缓冲（+3~14% 固化默认）。
  dsk = deep + split-K（--sk）+ **末片直写 C**：前 sk-1 片写 P、末片直写 C、归约读 C+P[0..sk-2]，
  加法链与全 P 归约同序 → **bitwise 21/21 逐位等价**。1024³ 波次几何：32 tiles × sk3 =
  96 blocks = **精确 2 波**（经典 128 tile 为 1.33 波，尾波 33% 浪费）。
- **验证**：deep@2048³ **8568.5 GF = %peak 72.1**（同会话 cuBLAS 82.6%）；@4096³ auto_v3
  7970.8 = 68.1%（G3 PASS）；dsk@1024³ 同会话 **74.77%**（AR009 64.20% → +10.6pp）；
  G1@512³ 75.14% PASS；256³ 同会话反超 cuBLAS **123.7%**；auto v3 六尺寸 %peak
  +16.4/+16.8/+19.9/+8.6（四个尺寸 ≥+2%，G4'' PASS）。
- **残差分析（0.23pp 刀锋）**：六变体攻坚链全实测——①跨会话混搭伪影修正（+0.5% 识别）；
  ②归约 v3 ILP4+__ldcs 仅省 1.5μs：归约流量地板 **355 GB/s ≈ DRAM 峰 80-90%**
  （20MB 流量 @1024³sk3 物理决定，ILP/提示词无效）；③__stwt 写穿**负结果**（main +16μs，
  写穿流与 A/B 读 miss 反压；回退默认 store）；④末片直写 +1.1%（归约 47→43μs）；⑤ptxas
  **寄存器重排陷阱**：运行时分支参数诱发 247/241→243 重排（main -4.9%，0 spill——"合法但更差"
  的分配），	emplate <DBUF, LAST_DIRECT> 编译期实例化 + Out 指针下沉回写段修复，热态探针
  **6312-6321 GF** 历史最佳；⑥终态缺口 0.23pp < cuBLAS 分母热态摆幅 ±0.65pp——**分母测量
  下限**，门判 FAIL 不放宽，L2 persistence/单核确定性归约列 backlog。
- **证据链**：fig22（deep 结构+资源包络）、fig23（LDS 模型 vs 实测）、fig24（dbuf/sk 消融）、
  fig25（G2 钟态考古）、fig26（五门 v4）、fig27（阶梯 v4）、fig28（auto v3 保真）；
  CSV：deep_ar010/g2_ar010/g2_boost_ar010/paired_ar010(108 行)/paired_ar010_predirect/
  auto_ar010(24 行)；判定文书 compare_ar010_paired.md；论文体 paper_sgemm_turing.md。
- **新瓶颈移交**：归约段已到 DRAM 流量地板 + G1@1024³ 分母测量下限——后续杠杆为系统级
  （L2 persistence window 钉 C、单核确定性 last-block 归约），kernel 级单变量优化空间收尽。

## AR011 闭环 #6——FR3 延迟覆盖消融（假说否定：BPF/PHASE 双因子皆负，2026-10-07 实测）

- **假说**：2026107.md 需求 P2——deep/dsk 主核稳态循环存在"同相停顿"（全 warp 同 kk
  取 B 片段时 FFMA 依赖链暴露）→ 预取 B 片段 kk+1 一拍（BPF）或轮转错相取 kk（PHASE）
  可覆盖。
- **证据**：results/lat_cover_ar011.csv（git=9eaf90a，v4 协议 3 轮中位，dsk --sk 3 主路径）：
  - 1024³：base 6280.4 / BPF 6303.7（+0.37%）/ PHASE 6228.8（-0.82%）/ both 6240.4（-0.64%）
  - 2048³：base 8406.1 / BPF 8401.9（-0.05%）/ PHASE 8232.3（-2.07%）/ both 8240.4（-1.97%）
  - 正负判定阈值 ±0.1%（协议噪声内为持平）→ PHASE 双尺寸显著负 / both 由 PHASE 主导 /
    BPF 持平
  - ptxas（build.log T004 段）：10 实例全 0 spill（BPF +2r=245、PHASE -13r=230）
- **机理归因**：单 warp 全体线程同 kk 取 B 列时 smem multicast 友好；PHASE 错相使各 warp
  取 B 落入不同 multicast 组——smem 扇出恶化（sm_75 无 ncu 计数器，multicast 计数无法
  直接证实，机理为间接推断，T007 P0 解锁后可复核 LSU 侧计数）；BPF 无效 = dbuf 双缓冲
  已覆盖预取需求（下一拍 B 已在备用缓冲）。
- **负结果处置**：两因子均不进 auto v4 候选池；--bpf/--phase 旋钮保留作消融复现（沿
  AGENTS §3 --wlb 先例）；G1@1024³ 缺口兑现路径收窄（+0.31% = 1.05μs 无着落）→ 转向
  Stream-K W sweep（T005）。
- **证据链**：套件 163/163（BPF bitwise 锚 + PHASE 确定性/rel 锚）；fig31；闭环 #6。

## AR011 闭环 #7——Stream-K W sweep（假说否定：多波精确填充；确认：W=1 tile 聚合最优，2026-10-07 实测）

- **假说**：design §4.1 尾波消除预期"多波精确填充"：1024³ W=2 可拟 dsk sk3 波效率
  77-79%、2048³ W=5 可达 90-95%。
- **证据**：results/streamk_ar011.csv（git=9eaf90a 构建，v4 协议 3 轮中位，165 行）：
  - **W=1 全尺寸最优、W>1 单调负**：1024³ W=1..8 = 5461→3432 GF 单调降；2048³
    W=1..8 = 8721→7408；4096³ W=1..6 = 8919→8666（auto 公式曾设 W=5@1024³ /
    W=8@2048³——实测全部错误）。
  - **"多波精确填充"假说否定**：2048³ W=2（96 块 = 2 精确波）8355 < deep（128 块
    2.67 波）8542——多波反而引入 split P 税与占用损失；**唯一赢面 = W=1 tile 聚合**：
    48 块恒 1 波 + cover 5.3 片的 tile 聚合 + solo 快路径（免票据免 P 回合）：
    8721 = +2.1% over deep = **尾波税兑现 2.1%** = dsk 的 104.4% = cuBLAS 的 85.4%
    （G6@85% 边缘；T008 v5 终判 84.85% MISS 0.15pp）。
  - **1024³ G1 翻门失败**：streamk_w1 5461 = dsk 6279 的 87%（融合 393μs vs dsk
    297+41=338μs 二段——同 48 SM 下 1 块/SM（dsk 96 块 = 2 块/SM）MLP 减半 +
    split 税；T003（FR2 N/A）+ T004（FR3 双负）+ T005（streamk 13% 差距）三线闭合
    ——**G1 责任链终态 74.77%，miss 1.05μs**，门线不放宽）。
  - **小尺寸负对照**：512³ swsk_sk3 4273 >> streamk_w1 1777（TOT=512 需 U=11 覆盖、
    每 tile 切 ~12 k 片——P 流量税爆炸）；256³ TOT<48 wrapper 旁路 deep 生效
    （292.7 vs deep 309.8）。
  - 4096³：deep 8994 > streamk_w1 8919 > dsk 8666 → dispatch 表守 deep。
- **尾块不均衡量化**：W=1 时 U=ceil(TOT/48)：1024³ U=86 每块 54/86=63%（尾块仅 32 块
  做满 2 tile，其余 1 tile——非满载波）；2048³ U=683 每块 667/683=97.7%；4096³
  U=5462 每块 5430/5462=99.4%——尾块不均衡解释 1024³ 差距形态：波载不均叠加
  1 块/SM 结构税。
- **对 auto 公式的裁定（T006 实现）**：auto W 公式校准 **W:=1 恒定**；TOT≥48 才进
  dispatch 候选（2048³ 区带弃 dsk 取 streamk W=1，swsk/deep/dsk 各守其带）。
- **证据链**：fig32 三联（W 扫描 / W 单调性 / 1024³ 差距归因）；套件 163/163；
  streamk W=1 memcheck 0/0（T008 @1024³/2048³）。
- **测量学教训**：cublas@1024³ sweep 会话曾录 boost 混染 9906.8 vs canonical 8397.5
  （同会话锚原则再实证）——比例门一律以同会话锚为准（T008 v5 协议强化）。

## AR011 闭环 #8——auto v4 dispatch 进化 + 增量门诚实负 + K=4096 时钟制度发现（2026-10-07 实测）

- **auto v4 进化**：sgemm_auto.cu（commit 7d9a117）：blocks>64 区带细分（deep 整波
  t%48==0 或 t>192 → deep；其余尾波区 → streamk W=1）；17..64 区带 sk 几何公式
  sk=clamp(ceil(96/t),2,16)（t=32 → sk3 向后兼容）；streamk auto W 校准恒 1；
  canonical 十尺寸 verbose 决策全符合（2048³ 区带弃 deep 取 streamk W=1）。
- **G4''' 增量门 MISS（诚实负结果）**：v4 vs v3 = 2048³ +2.36% 唯一达标（≥+2%）；
  256³ +0.08% / 512³ +0.05% / 1024³ -0.01% / 1000×1016 -0.05% / 4096³ +0.17% 均
  pick 同核 delta 持平 → 1/6 达标 vs 门要求 4/6。根因 = AR011 的 FR2/FR3 皆负后
  streamk 只在 2048³ 一带兑现（2048³ auto=streamk W=1 8784-8788 GF = 89.7% cublas
  该会话锚）；门判不放宽。
- **G5 dispatch 保真 PASS**：canonical 11/12 对 ≤0.6pp（唯一 miss 256³ r2 -2.12pp
  同路径噪声——auto 与 w4_swsk_sk6 同 kernel 同参，计时涨落）；补充尺寸干净对
  -0.45/-0.61/-0.03/-0.69/0.00pp 全 ≤2pp。
- **补充尺寸首测（FR6）**：256×4096 与 4096×256 → dsk sk3 双制度最优（warm
  8919/8894 vs strk 8438/8508；cold 7766/7723 vs strk 7258/7337）——auto 选 dsk
  正确；1024×2048 → streamk W=1 略胜 dsk（warm 8398/8339 vs 8338/8306，+0.7%）
  >> deep 5958-6490（t=64 = 1.33 波尾波税 ~25% 实证）——区带规则验证。
- **K=4096 时钟制度发现（协议级）**：WDDM 下 K=4096 长核（~1-2.5ms）存在两种制度：
  逐行冷却（47°C 门后单测）稳态 1620 MHz vs in-sequence 连续 7 行升温至 1920-1935
  MHz，dsk@256×4096 跨制度差 -13%（8919 vs 7766）——cublas/deep 亦随制度漂移
  不可混比；**run-1 auto@4096×256 = 7668 GF "盲区" 为降频伪影（gpu_state 列
  1740/1620 MHz vs 同行前序 1920-1935），已撤回**；协议结论 = 同轮同制度才可比、
  保真须逐行冷却；v5（T008）补充协议改 in-sequence warm + auto 居中
  （run_supp_paired.ps1 双制度对照）。
- **证据链**：verbose 10 尺寸决策表；套件 163/163 零回归；fig33。
