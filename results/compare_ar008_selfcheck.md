# AR008 thermal-paired 对内 delta 报告

- paired csv : `results\paired_ar008_selfcheck.csv`
- baseline   : swpipe（run_paired.ps1 交替配对产出）
- 语义       ：对内 delta = 挑战者 vs 基线（同轮同热状态，共模漂移相消）

## 对内 delta 明细

| size | challenger | rounds | baseline GF | challenger GF | delta%（逐轮） | median | 极差(pp) |
|---|---|---:|---:|---:|---|---:|---:|
| 512x512x512 | vec4 | 3 | 2349.8 | 2222.7 | -5.43, -5.30, -5.41 | -5.41 | 0.13 |
| 512x512x512 | cublas | 3 | 2350.4 | 5698.8 | +142.22, +142.46, +142.66 | +142.46 | 0.44 |
| 1024x1024x1024 | vec4 | 3 | 4785.1 | 4629.9 | -0.98, -1.03, -3.29 | -1.03 | 2.31 |
| 1024x1024x1024 | cublas | 3 | 4678.4 | 8439.8 | +80.46, +80.42, +80.29 | +80.42 | 0.17 |

## 四门 v2 判定（srs AR008 §4）

### G1 中尺寸（512^3/1024^3 >= 75% cuBLAS）

| size | 自研最优 GF | kernel | cuBLAS GF | 比值 | 判定 |
|---|---:|---|---:|---:|---|
| 512^3 | 2222.7 | vec4 | 5698.8 | 39.0% | FAIL |
| 1024^3 | 4629.9 | vec4 | 8439.8 | 54.9% | FAIL |

- **判定：FAIL**（0/2 达标）

### G2 小尺寸守成（256^3 auto/swsk >= 1618.2 GF）

- （本 CSV 无 256^3 auto/swsk 组，跳过）

### G3 大尺寸（4096^3 ws：对内 delta + 绝对值 >= 7.0 TF）

- （本 CSV 无 4096^3 ws 组，跳过）

### G4 全线（>= 4/6 尺寸对内 delta >= +2%）

| size | 最佳挑战者 | delta | 判定 |
|---|---|---:|---|
| 512x512x512 | vec4 | -5.41% | miss |
| 1024x1024x1024 | vec4 | -1.03% | miss |

- 命中 0/2（门槛 4/6）→ **判定：FAIL**

### G5 方法学（对内 delta 轮间极差 < 2pp）

- 全组 delta 极差：median = 0.30pp，最大 = 2.31pp（4 组）
- **判定：PASS**（对照：AR007 热浸没有会话绝对值漂移 ~5%）

## 备注

- 协议：run_paired.ps1 交替配对（A,B）×rounds，组间冷却门控；冷却超时组由脚本标注 thermal-contaminated（本表不区分，见执行日志）。
- delta 为 GF 比值；cublas 作为"挑战者"运行时其组值即同协议 cuBLAS 参考（G1 分母），其 delta 行 = swpipe/cuBLAS 相对关系。
