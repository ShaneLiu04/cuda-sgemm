# [AR012] 任务清单

| 字段 | 内容 |
|------|------|
| AR 编号 | AR012 |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成） |
| 创建日期 | 2026-10-07 |
| 模式 | auto（用户指令"按照推断深度优化"，沿用 AR007-AR011 先例） |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 完成记录 |
|----|---------|------|------|---------|
| T001 | Red：`--swz {0,1}` / `--swzg {4,8,16}` / `--skred {0,1}` 旋钮注册（common.h CLI + main.cu + 跨 kernel note 语义：--swz 限 deep/dsk/streamk、--skred 限 streamk、--swzg 随 --swz 1）+ suite 注册变体槽位（deep_swz1/dsk_swz1/streamk_swz1/streamk_skred1，fn=nullptr 红观察）+ CLI 校验用例（--swz 2/--skred 7 → CLI_ERROR） | - | passing | 2026-10-07。变体槽位改为 **wrapper marker 探针**方案（沿 AR011 --waves T001 绑定先例，避免 17→21 注册表膨胀）：g_swz/g_swzg（deep.cu 定义）/g_skred（streamk.cu 定义）+ g_launch_swz/g_launch_skred 快照探针（sgemm_kernels.h extern，wrapper launch 处写入）+ main.cu [note] 接线 + suite 2 探针（marker 快照断言）+ 2 旋钮中立锚（swz1 vs swz0 / skred1 vs skred0 逐位——T003/T005 Green 后 remap 链序不变/归约链≡F2 → 锚恒成立，永久保留）。Red：改前二进制 `--swz 1` = `[CLI_ERROR] unknown argument`。Green IT：--swz 2/--swzg 5/--skred 7 → CLI_ERROR ✓；naive 跨域 → 双 [note] ✓；--swzg 16 单用 → [note]（effective with --swz 1 only）✓。构建 0 spill（deep 243-247/streamk 255 regs 不变，marker 纯 host 侧零寄存器税）；套件 **167/167 ALL PASS**（163+2 探针+2 锚，9 bitwise/rel 锚） |
| T002 | FR1 E-B 反汇编流水线：nsys profile（trace=cuda）跑 cuBLAS bench @1024³/2048³/4096³ 提 kernel 名与 launch config → `cuobjdump -xelf` 提 cubin → SASS → `tools/cublas_disasm.py` 特征提取（寄存器/smem/LDGSTS 排布/FFMA:LDS/blockIdx 运算模式=swizzle 判定/unroll）→ `profile/cublas_disasm/report.md` 逐项对照我方 deep；失败链路逐级降级并记负结果（不阻塞主线） | - | passing | 2026-10-07。**降级链**：L0 nsys 双份二进制负结果（无 admin 环境 CLI→后端 IPC 静默失败，whoami 最小用例亦不可采集）→ L1 进程内 CUPTI activity API（tools/cupti_trace.cu，新版 RegisterCallbacks 缓冲对 API；与 nsys 同数据源、零权限）✓ → L2 cuobjdump+nvdisasm 组件包官方 redist 下载（sha256 校验）✓。**Level-1 结构**（CUPTI 实测 4 尺寸）：512³=volta_sgemm_64x64_nn(8,8,z3,64thr,126regs)；1024³=128x64_nn(8,16,z3,128thr,122regs)；2048³=128x128_nn(16,16,z2,256thr,118regs)；4096³=128x64_nn(32,64,z1)。z=K 分片（kernel 内信号量串行归并，无独立 reduce launch、无 atomics）；CTAID 线性映射无 swizzle；每线程恒 64 输出（缩线程不缩 tile 深度）；全家族 LDGSTS=0（无 cp.async）；118-126 regs → 2 blocks/SM（我方 deep 243regs 恒 1）。**Level-2 指令构成**：cuBLAS FFMA=520/LDS=60（FFMA:LDS 8.67，连段 58）vs 我方 deep 1024/48（21.33，连段 171）——我方计算密度优、cuBLAS 占用优；4096³ 双方满波差距 4-6%（epilogue STG 32 vs 9 为残差候选）。**主线结论**（report §5）：cuBLAS 无块序 swizzle（G7 消融照做）；1024³ 差距主因=波填充/K 分片非 FFMA 效率（支持 T005 SK_RED、反对缩 tile）；工具坑（PS 重定向 UTF-16LE/无 BOM 中文注释 CP936 幻影错/LNK1104 非 ASCII 输出路径）与 nsys 负结果已沉淀 environment.md §4 与 report §1 |
| T003 | Green：deep/dsk swizzle（SWZ 头部 remap，主循环零扰动；ptxas 增量 ≤2 寄存器硬门 + 0 spill）+ bitwise 锚（tile 独立 → 链序不变）+ 任意 M/N/K 边界谓词路径回归 | T001 | passing | 2026-10-07。**Red**：suite 新增 host UT 区引用 `sgemm::detail::swizzle_tile` → 编译红（`namespace "sgemm::detail" has no member "swizzle_tile"`，test_correctness.cu:388/391）。**Green**：①纯函数 `swizzle_tile`（__host__ __device__ inline，sgemm_kernels.h detail 域；分组列序闭式：g=l/(gm·G)、r=l-g·gm·G（r<gm·G_eff 恒成立）、m'=r%gm、n'=g·G+r/gm；swz=0 恒等）——host UT 与 device 同源；②deep kernel 模板 +SWZ 维（实例 ×2：10→20）：头部 `if(SWZ)` remap（组宽 G 经运行时参数 swz_g=g_swzg 传入，每 block 头部一次整除，主循环零扰动；z 维不参与）；SWZ=0 实例编译期剔除 remap + swz_g 不被引用 → 现役 codegen 不变（AR011 T004 手法）；③wrapper deep_tile_grid 按 g_swz 选实例（dsk 经咽喉点自动继承）。**ptxas 审计**（build.log 20 实例）：SWZ=0 全族与 AR011 逐值一致（DBUF1=243/DBUF0=247/BPF=245/PHASE=230——现役路径 codegen 零扰动实证）；SWZ=1 生产默认 <1,0,0,0> 243→239（−4，优于 ≤2 门）；消融组合 PHASE+SWZ 230→251/254（ptxas 重分配扰动，非生产路径，仍 ≤255 包络）+ **全部 20 实例 0 spill**。**套件 189/189 ALL PASS**（167+22：host UT 18 组合=6 网格{(8,4),(16,8),(32,16),(5,3),(7,11),(1,1)}×G{4,8,16} 双射+组序+恒等 PASS + dsk swz 锚@1024³ + 边界锚 deep/dsk@1000×1016×516（M/N 非 tile 倍数+K%BK≠0 尾 k-tile+尾组 G_eff 收窄）+ 回退锚@999×1001×997（N%4≠0→tile2d）全 BITWISE-PASS；T001 中立锚 deep(swz1)==deep(swz0)@1536×1024×512 转**真 remap 锚**（marker g_launch_swz=1 + 寄存器差证 SWZ=1 独立实例，bitwise 证 device remap 纯置换）；9 bitwise/rel 锚保持）。CLI 冒烟：deep --swz 1 --swzg 4 @1024³ / dsk --swz 1 @2048³ 端到端跑通（冷态 RSD 高，数字不作结论，正式消融属 T004 配对协议） |
| T004 | streamk swizzle + 消融数据：`--swz {0,1} × --swzg {4,8,16} × {2048³,4096³}` like-for-like（v4+ 协议，SGEMM_CSV 分流）+ canonical 全尺寸回归 + **G7 判定**（≥+1.0% @2048³/4096³ 且 512³/1024³ 不回退 >0.5pp）+ fig35（swizzle 消融三联） | T003 | passing | 2026-10-07。**Green**：streamk c-decode 重排（模板 +SWZ 维，实例 2→4；切点/票据/P 槽位保持线性 c 空间，仅物理 tile 坐标变；wrapper 实例选择 + g_launch_swz 探针含 streamk）；ptxas：SWZ=0 现役 255 regs 不变、SWZ=1 254（重分配 −1）、全 0 spill。**实测裁定 design §4.2.1 笔误**：初稿"每 tile 链不变 → bitwise"错误——切点 t0=b·U−c·nt 是 **c 的函数**（c·nt mod U 轮转），remap 使同物理 tile 的 K 分段括号序改变（首版 bitwise 锚实测 FAIL 抓获）→ 按 srs 预声明③口径改 rel≤1e-4 + 双跑确定（PHASE 锚同型）；design.md 修订注回填。套件 **193/193 ALL PASS**（+4：streamk swz 双跑确定 BITWISE-PASS + rel=8.9e-07@1024³ + 边界 rel=7.3e-07@1000×1016×516 + marker 探针）。**消融**（run_swizzle_ablation.ps1，lat_cover 协议同型：47°C 冷却门 + 每轮 cublas 首锚 + 3 轮中位；**代码先 commit e87141a 再重 configure/重建/重跑**——首次跑于未提交树被丢弃，CSV git_sha 溯源纪律 e87141a）：{deep,dsk,streamk}×{swz0,g4,g8,g16}×{2048,4096,1024,512,256}³。**G7 判定 = streamk-only 吸收**：deep 全负（−2.9%@2048/−2.2%@4096/−8.8%@1024——2D 线性光栅 bx=n 最快已优，重排破坏 B 复用）；dsk 全负（−1.9%/−1.0%）；**streamk 正**：+3.91%@2048³（8803→9147）、**+9.11%@4096³**（7981→8708）、+1.60%@1024³（5490→5578）、512³ −0.05pp（门内）——1D c 空间 tile-major 线性化默认块序 L2 复用差，分组列序修正；G=4 全尺寸最优或并列（@4096 8708>g8 8624）→ T006 auto v5 仅 streamk 路径吸收 `--swz 1 --swzg 4`。**副产物**：streamk swz1g4@2048³ = 88.3% cuBLAS → **G6（≥85%）翻门**（AR011 曾 miss 84.85%）；@4096 85.2%。fig35 三联（G 扫描/波足迹机理/判定矩阵热图）已出（≤72KB） |
| T005 | FR3 SK_RED separate（design §4.1 问题 1 方案 B；half-K cover 已证伪弃选）：streamk 双 epilogue——`--skred 1` 下 cover tile 免票据免归并直退 + cover-reduce kernel（冷 C = 0+P[b_lo..b_hi] 升序 __ldcs，链≡F2）+ host cover 清单 + bitwise 锚（skred0 vs skred1 同 C 逐位）+ 1024³/512³ 消融 vs dsk/streamk_w1/deep/cublas 同会话 + **G1 判定**（≥6288 GF 翻门或诚实负结果）+ fig36 | T001 | pending | |
| T006 | FR4 auto v5：吸收胜者（G7 正 → swz 进 deep/dsk/streamk 路径；G1 翻 → hybrid 进 1024³ 区带；双负 → auto 保持 v4 仅记录）+ canonical 十尺寸 verbose 决策表 + 补充尺寸配对（256×4096/4096×256/1024×2048，双制度纪律）+ G4/G5 预判 | T004,T005 | pending | |
| T007 | FR5 门 v6 终判：run_paired v6（v5 协议继承）× deep/dsk/streamk/auto v5-or-v4/hybrid/swsk/cublas 全 kernel × 六尺寸 + 硬门全量复判（套件全绿/bitwise 锚/0 spill/memcheck 新 kernel/racecheck 抽查）+ judge 解析 + **G1/G6/G7/G2/G3/G4/G5 七门终判表** + fig37（七门总览） | T006,T002 | pending | |
| T008 | 工程卫生 + 交付收尾：详设同步（Kernel 11 条目/旋钮表/AR 表/术语）+ AGENTS 旋钮段 + report §14 + bottleneck 闭环 #9 + paper/interview 回填 + E-B 报告交叉引用 + st_report.md + 全量自检（--clean-first 重建/套件/图 ≤72KB/BOM）+ git 溯源 + 归档建议 | T007 | pending | |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加。

- 2026-10-07（T001 完成会话）：T001 旋钮注册全链路落地——common.h（字段/usage/parse/校验
  四区）→ sgemm_kernels.h extern（g_swz/g_swzg/g_skred + 探针 g_launch_swz/g_launch_skred）
  → deep.cu/streamk.cu 定义与 wrapper 快照 → main.cu [note] 接线（--swz/--swzg 限
  deep/dsk/streamk；--skred 限 streamk；--swzg 单用 note）→ suite AR012 探针区（2 探针
  + 2 中立锚）。绑定先行纪律披露：Red 阶段 device 侧 remap/独立归约未接入，旋钮仅被
  快照、行为仍现状（中立锚即此事实的显式断言）；T003/T004/T005 Green 接入后生效。
  变体槽位方案由注册表膨胀改为 marker 探针（注册表保持 17，跨 kernel bitwise 走
  run_dump 旋钮就地设置，AR011 锚区同模式）。
- 2026-10-07（T002 完成会话）：E-B 全链路落地——nsys 负结果（环境不可用，非 ASCII
  路径 protobuf 失败 + ASCII 下不拉起 app）→ 进程内 CUPTI activity（cupti_trace.cu，
  签到 cublasLt volta_sgemm 家族）→ cuobjdump/nvdisasm 官方组件包下载（redist json
  索引 + sha256 校验，装配 csg-tools\cuda\）→ SASS 特征提取（cublas_disasm.py）→
  report.md 六节（降级链/Level-1/Level-2/对照解读/主线结论/复现清单）。工具链增补
  与 nsys 负结果同步 environment.md §4。**E-B 核心情报**：cuBLAS 恒 64 输出/线程 +
  118-126 regs + 2 blk/SM + z 维 K 分片（kernel 内信号量归并）+ 无 cp.async + 无
  swizzle；我方 deep 计算密度 2.5×（FFMA:LDS 21.33 vs 8.67）但恒 1 blk/SM。1024³
  差距归因波填充/K 分片而非发射效率——T005 SK_RED 主攻方向获得外部证据支持。
  交付物：tools/cupti_trace.cu、tools/build_cupti_trace.cmd、tools/cublas_disasm.py、
  profile/cublas_disasm/{report.md, 4×.sass, cupti_trace_log.txt}。
- 2026-10-07（T003 完成会话，含 T002 推送善后）：T002 原单笔 commit（1.3MB，含
  116KB pack）遭企业代理 403（HIS Proxy 页）——按 AGENTS §8 拆分预案：剥离 SASS
  控制编码注释列（1.0MB→365KB，原始输出可由 report §0 命令 1:1 再生，裁剪声明
  入库 report §0）+ 拆两笔 commit（工具+报告 / SASS 证据）逐笔推送成功
  （50edda3/25b465b）；push_min_pack.py 修两 bug（PS5.1 GBK stdout 崩溃 →
  reconfigure；内层 PS 凭据 `$u:` 作用域解析 → `${u}` 双大括号转义——修复后
  直读 403 诊断，凭据文件崩溃残留风险清零）。T003 主体：swizzle_tile 纯函数
  + deep/dsk SWZ 模板接入 + 20 实例 ptxas 审计（SWZ=0 现役 codegen 逐值不变，
  全 0 spill）+ 套件 189/189（host UT 18 组合 + dsk/边界/回退锚全 BITWISE，
  T001 中立锚转真 remap 锚）。冒烟通过；正式消融留 T004。
- 2026-10-07（T004 完成会话）：streamk swizzle Green（c-decode 重排，193/193）
  + **实测抓获 design §4.2.1 数值口径笔误**（bitwise → rel：切点 t0=b·U−c·nt
  是 c 的函数，remap 改变同物理 tile 的 K 分段括号序；首版 bitwise 锚实测 FAIL
  证伪设计断言，按 srs 预声明③改 PHASE 锚同型口径）——TDD 红锚的价值实例。
  消融溯源纪律修正：首次消融跑于未提交树（CSV git_sha 列为陈旧 configure 值）
  → 数据丢弃重跑（commit e87141a → 重 configure → 重建 → 重跑），CSV 溯源
  列正确。**G7 = streamk-only 吸收**（deep/dsk 全负；streamk +3.9%@2048³/
  +9.1%@4096³/+1.6%@1024³，512³ −0.05pp 门内；G=4 全尺寸最优或并列）；
  **G6 副产物翻门**（streamk swz1g4@2048³ = 88.3% cuBLAS，AR011 曾 miss
  84.85%）。交付物：run_swizzle_ablation.ps1（UTF-8 BOM 军规复验）、
  results/swizzle_ar012.csv（git=e87141a 溯源）、fig35_swizzle_ablation.png
  （三联：G 扫描/波足迹机理/判定矩阵，≤72KB）。

## 门控记录

> 每任务每会话可自由追加，记录关键决策。

## 阶段门关键记录

- 2026-10-07：用户指示（"按照你的推断，使用自动化的 harnessX，进行深度的优化"）
  → 开启 AR012。req/design 阶段门控跳过（AR007-AR011 auto 模式先例），ST 验收
  集中回报。需求源 = 本会话"如何全尺寸反超 cuBLAS"推断分析（三杠杆：E-B 零权限
  反汇编 / L2 块序 swizzle / 1024³ 2-blocks/SM 混合）+ AR011 st_report §4 终态。
- 2026-10-07：srs.md 关键口径预声明——①E-B 为独立主线（全链路失败不阻塞 FR2/FR3，
  降级逐级记录）；②swizzle/hybrid 均为消融旋钮起步（默认 swz=0=现状），翻门前
  不进 auto；③数值口径分级（deep/dsk swizzle 链序不变 → bitwise 锚；streamk
  swizzle/hybrid 链序可能变 → rel≤1e-4 + 确定性双跑）；④G1/G6 判定锚 = 同会话
  稳态末锚（AR011 v5 纪律继承，r1 冷锚一律排除）。
