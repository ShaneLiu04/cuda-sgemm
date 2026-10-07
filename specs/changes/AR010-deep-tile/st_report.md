# AR010 ST 验收报告（deep-tile）— 2026-10-07

> 集中回报（auto 模式：开发各阶段不询问，ST 统一汇报）。全部数字实测可复现，
> 原始 CSV 行级 gpu_state 逐行可核。

## 1. 验收范围

srs FR1（Kernel 9 deep 深寄存器分块）+ FR2（dsk split-K + 末片直写）+ FR3（auto v3）+
FR4（五门 v4 复判）+ FR5（交付文档 paper/interview/report §12）。
任务 T001-T008 全数 passing（specs/changes/AR010-deep-tile/tasks.md）。

## 2. ST 正确性门（全绿）

| 检查 | 结果 |
|------|------|
| 回归套件（16 kernel × 尺寸/边界/回退） | **146/146 ALL PASS**（T008 终检复跑确认） |
| 逐位等价专项（bitwise memcmp） | **21/21 YES**：deep==swpipe、dbuf1==dbuf0、dsk(rv2)==dsk(rv1)、dsk==swsk 同 sk、确定性双跑——同 BK=8 切分 + 单一归约入口 + 加法链同序三重锚点 |
| 确定性 | dsk 归约固定 z 序 + 直写路径加法链与全 P 归约同序（逐位实证） |
| memcheck | deep/dsk 主路径 + 130×257×66 回退 0 errors |
| racecheck | deep 双实例 0 hazards（DBUF=1 单同步屏障契约）；dsk workspace/归约 0 hazards |
| ptxas 审计（build.log） | deep\<0\>=247r/12416B、deep\<1\>=241r/24832B、dsk 四模板实例 247/247/243/243 regs——**全 0 spill**；ws/wsk 4×8B spill 为既有非回归基线 |

## 3. 五门 v4 判定（compare_ar010_paired.md，paired_ar010.csv 108 行）

| 门 | 判定 | 关键数据 |
|----|------|---------|
| G1@512³ ≥75% cuBLAS | **PASS（75.14%）** | swsk_sk3 4282.1 / cuBLAS 5698.8 |
| G1@1024³ ≥75% | **刀锋 FAIL（74.77%）** | dsk_sk3 6279.2 / cuBLAS 8397.5（AR009 64.20% → +10.6pp；残差 0.23pp < 分母热态摆幅 ±0.65pp） |
| G2@256³ ≥1618.2 | **PASS（钟态匹配）** | %peak 15.63 vs 门源 14.04-14.16（+10.9%）；1860 投影 1786.2 ≥ 门；同会话反超 cuBLAS **123.7%** |
| G3@4096³ ≥7.0TF | **PASS** | auto_v3 7970.8@1905（%peak 68.1；cuBLAS 同会话 77.35%）；2048³ deep 8568.5（%peak 72.1） |
| G4'' auto v3 ≥4/6 +2% | **PASS** | %peak：1024³ +16.41 / 1000×1016 +16.77 / 2048³ +19.91 / 4096³ +8.61 |
| G5'' dispatch 保真 ≤2pp | **PASS** | A-B-A-B 10/12 对 ≤0.6pp（双离群有 min 逐位同硬证据）；配对极差 median 0.31pp |

G1@1024³ 六变体攻坚链全实测（会话偏移修正→归约 v3→`__stwt` 负结果→末片直写→ptxas 重排修复→
模板化终态热态 6312-6321 GF）后仍差 0.23pp——**按军规如实 FAIL 不放宽**，残差归因
"分母测量下限"，剩余杠杆（L2 persistence、单核确定性归约）列 backlog。

## 4. 核心科学结论

1. **%peak 经验律**：acc-per-LDS 10.7→~35%、16→55-60%、21.3→**72.1%**（deep@2048³）；
   LDS.128 带宽墙（AR009 确证）的直接兑现——降 LDS 频率换计算密度。
2. **ptxas 寄存器重排陷阱**：运行时分支参数诱发 247/241→243 重排（0 spill 仍 main -4.9%）；
   对策 = 编译期模板实例化 + 指针使用点下沉。寄存器敏感 kernel 改动必过逐实例审计。
3. **归约流量地板**：355 GB/s ≈ DRAM 峰 80-90%（20MB @1024³sk3 物理决定）；ILP/提示词/写穿
   全部无效（N4/N5 负结果）；末片直写省流量 +1.1% 为唯一正解。
4. **%peak 跨钟态不变量 + DVFS 双域**：钟频线性 boost 实证（误差 0.1%）；重核 1920-1950 /
   小突发 1620 同会话并存（三次独立验证）——跨会话绝对门必须钟态归一。
5. **逐位等价工程**：性能优化（归约三版迭代）与数值不变性同时成立，21/21 专项实证。

## 5. 图表交付（fig22-28，make_figures.py 可复现）

| 图 | 内容 | 数据 |
|----|------|------|
| fig22 | deep 结构（DBUF 流水时空 + ILP-vs-TLP 资源矩阵 + 冒烟） | smoke_deep + build.log |
| fig23 | LDS 模型 vs 实测（%peak 经验律主线） | deep_ar010.csv（105 行） |
| fig24 | deep/dsk 消融（dbuf/sk 全扫描 + 波几何注记） | deep_ar010.csv |
| fig25 | G2 攻坚（门源钟态考古 + sk 细扫描 + boost 实证） | g2_ar010.csv（42 行）+ g2_boost |
| fig26 | 五门 v4 判定 | paired_ar010.csv（108 行） |
| fig27 | 全 kernel 阶梯 v4（六尺寸 winner 高亮） | paired_ar010.csv |
| fig28 | auto v3 dispatch 保真（A-B-A-B） | auto_ar010.csv（24 行） |

## 6. 事故与披露记录（军规执行）

- run_paired_v4 裸 powershell 唤醒 spin 失败（必须 tools\env.cmd 包裹）
- 脚本追加写 CSV 重跑前未删旧文件 → 216 行事故，修复后核验 108 行
- T004 cuBLAS@1024³ 8432 行系 boost 混染（p15/p16 钉死稳态 8374-8381）
- T005 跨会话混搭伪影识别（AR010 会话全核 +0.44~0.65% 同移 → 75.16% 修正为同会话 74.57%）
- rv2 对 dsk -4%（T005 负结果）；`__stwt` 写穿 +16μs（T007 负结果）——均归档
- 直调 nvcc 编译 bitwise 工具缺 `-Xcompiler=/utf-8` 首试失败（补旗标通过）

## 7. 结论

AR010 以 **Kernel 9 deep/dsk（%peak 72.1 峰值 + 1024³ 刀锋 74.77%）+ auto v3 + 五门 v4
（4 PASS / 1 刀锋 FAIL）+ 测量方法学（%peak 不变量/流量地板/ptxas 重排）+ 交付文档
（paper_sgemm_turing.md / interview_narrative.md / report.md §12）**完成交付。
十六版严格 FP32 阶梯收官：4096³ 50.8×、2048³ 54.6×（%peak 72.1）、256³ 反超 cuBLAS 123.7%、
146/146 + bitwise 21/21 + 全 0 spill + sanitizer 全清。
建议归档：specs/changes/AR010-deep-tile → archive（含 7 份 CSV + 7 图 + 判定文书 + 双交付文档）。
