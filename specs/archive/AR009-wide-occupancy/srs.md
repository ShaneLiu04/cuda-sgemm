# AR009 srs — 宽块占用率攻坚（Gate-Closing）

| 字段 | 内容 |
|------|------|
| AR 编号 | AR009 |
| 名称 | wide-occupancy：Kernel 8 wide + 精确波 split-K + auto v2 |
| 前置 | AR008（swsk/ws/auto/paired 协议已交付） |
| 日期 | 2026-10-05 |

## 1. 背景与问题（AR008 四门 v2 遗留）

| 门 | AR008 结果 | 根因诊断（实测数据支撑） |
|----|-----------|------------------------|
| G1 512³/1024³ ≥75% cuBLAS | FAIL（62.2%/58.1%） | **占用率墙**：swpipe 系 256 线程×128 regs → 2 block/SM=50% 封顶；512³(sk4) 64 blocks→1.33/SM=33% 占用，实测 36% of peak；1024³ 同理 33%→46%。效率≈占用率×1.2（2048³ 50% 占用→60% peak 交叉验证） |
| G2 256³ ≥1618.2 GF | FAIL（1517.5） | swsk(sk12) 48 blocks×8 warps=25% 占用；且跨会话漂移 -14%（绝对门环境敏感） |
| G3 4096³ ≥7.0 TF | FAIL（swpipe 6474） | 4096³ 稳态恰为 50% 占用上限（2 block/SM），效率 55% peak；ws 专属化已被消融否定（AR008 T008）——唯一出路是抬占用率本身 |

**共同根因**：TM=TN=8（64 acc）的寄存器需求 → 128 regs/thread → 2 block/SM（64K regs/SM）
占用天花板 50%。AR008 ws 负结果证明"在 50% 占用内重排指令流"无益——必须破墙。

## 2. 解决方案（FR）

### FR1 Kernel 8 wide：512 线程宽块（本 AR 核心）
- block = 512 线程（16 warp），tile 仍 128×128×BK8；每线程 **TM=4×TN=8（32 acc）**
- 搬运划分：warps 0-7 载 A（256 quads，每线程 1×float4）、warps 8-15 载 B（256 units）——
  预取寄存器从 8 降到 4
- 寄存器预算（设计目标）：32 acc + 12 frag + 4 prefetch + ~14 寻址/杂项 ≈ 62；
  `__launch_bounds__(512, 2)` 上限 64 regs——**ptxas 0 spill 达标 → 512×64×2 = 65536
  = 100% 寄存器占用 = 2 block/SM = 32 warp = 100% 占用**（sm_75 线程上限 1024/SM 恰好满足）
- 继承 swpipe 全部布局结论（A 转置+PAD4、B XOR swizzle、零守卫、单缓冲双同步、
  寄存器预取混合流——AR008 证实的正确结构）
- 若 64 regs 不可达（spill）：LB=1 实例（≤128 regs，1 block/SM=50% 占用）仍因
  **单块 warp 数翻倍**改善中尺寸聚合占用（1024³：64 blocks×16 warps/1536 = 67% vs 现状 33%），
  为保底收益；spill 取舍如实记录（AGENTS §4 军规）

### FR2 wsk：wide 的 split-K 变体
- 参数化复用 wide kernel 主体（z 切片 + Out_base，镜像 swpipe 参数化模式）
- 归约复用 swsk 的确定性固定序归约（跨编译单元 detail 共享，消除复制漂移）
- `--kernel wsk --sk N`；`--sk` 旋钮对 swsk/wsk 双生效

### FR3 精确波对齐消融（wsk/swsk sk 扩展扫描）
- 满驻留波 = 96 blocks（2/SM×48 SM）：512³ sk6（96）、1024³ sk3（192=2 波）、
  256³ sk12（48）/sk16（64）——AR008 扫描集 {1,2,4,8,12,16} 缺 {3,6}，本 AR 补齐
- P 流量 vs 占用率权衡实证（P 尺寸 vs L2 4MB）

### FR4 auto v2 dispatch 更新
- wide/wsk 实测胜出尺寸回填 dispatch 表（数据真实性军规：表值必须来自本 AR 实测行）

### FR5 文档/图表/协议同步
- 详设/AGENTS/report/README/figures；paired 协议复测三门 v2 重新判定（沿用 AR008 协议，
  污染组剔除重测规则不变）；负结果如实归档军规不变

## 3. 验收门（三门 v2 复判 + 守成）

| 门 | 判定 | 通过条件 |
|----|------|---------|
| G1 | 512³、1024³ 最优自研（wide/wsk/auto） | 各 ≥ 同会话 cuBLAS 75% |
| G2 | 256³ 最优自研 | ≥ 1618.2 GF（绝对门；同会话 cuBLAS 对照同时报告） |
| G3 | 4096³ 最优自研（wide/swpipe） | ≥ 7.0 TF（力争 7.2） |
| G4' | 全线守成 | 6 尺寸 auto v2 对内 delta ≥ AR008 auto 水平（不允许回退；≥4/6 尺寸 delta ≥ +2% 维持） |
| G5' | 方法学 | paired delta 轮间极差 median < 2pp 维持 |

## 4. 约束（不变军规）
- 严格 FP32 / 冻结签名 / CUDA events 计时 / median 判据 / 数据真实性（禁凭空填表、
  负结果归档）/ racecheck-memcheck 硬门 / ptxas 审计入 build.log / spill 即缺陷或记录取舍
