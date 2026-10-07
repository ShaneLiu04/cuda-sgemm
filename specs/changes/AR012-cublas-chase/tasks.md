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
| T002 | FR1 E-B 反汇编流水线：nsys profile（trace=cuda）跑 cuBLAS bench @1024³/2048³/4096³ 提 kernel 名与 launch config → `cuobjdump -xelf` 提 cubin → SASS → `tools/cublas_disasm.py` 特征提取（寄存器/smem/LDGSTS 排布/FFMA:LDS/blockIdx 运算模式=swizzle 判定/unroll）→ `profile/cublas_disasm/report.md` 逐项对照我方 deep；失败链路逐级降级并记负结果（不阻塞主线） | - | pending | |
| T003 | Green：deep/dsk swizzle（SWZ 头部 remap，主循环零扰动；ptxas 增量 ≤2 寄存器硬门 + 0 spill）+ bitwise 锚（tile 独立 → 链序不变）+ 任意 M/N/K 边界谓词路径回归 | T001 | pending | |
| T004 | streamk swizzle + 消融数据：`--swz {0,1} × --swzg {4,8,16} × {2048³,4096³}` like-for-like（v4+ 协议，SGEMM_CSV 分流）+ canonical 全尺寸回归 + **G7 判定**（≥+1.0% @2048³/4096³ 且 512³/1024³ 不回退 >0.5pp）+ fig35（swizzle 消融三联） | T003 | pending | |
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
