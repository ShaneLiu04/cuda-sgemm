# AR012 T002（FR1 E-B）：cuBLAS SGEMM 零权限反汇编报告

- 日期：2026-10-07 | AR012-cublas-chase | 任务：T002
- 采集机：Quadro RTX 5000（TU104, sm_75, 48 SM, 16 GiB）| driver 12.50 | runtime 12.5.40
- 反汇编对象：`cublasLt64_12.dll`（cuBLAS 12.5.2.13 内部经 cublasLt 派发；见 §2 证据链）
- 证据文件（本目录）：`cupti_trace_log.txt`、`volta_sgemm_{64x64,128x64,128x128}_nn.sass`、
  `sgemm_deep_default.sass`（我方对照，`<DBUF=1,BPF=0,PH=0>` 默认实例）
- 结构归因声明：本报告对闭源二进制仅做**结构特征归因**（tile/占用/staging/指令构成），
  不复刻其代码；所有结论可由本目录 SASS 原文复核。

## 0. 工具链与采集协议（零权限，可复现）

| 组件 | 来源 | 校验 |
|---|---|---|
| cupti 组件包 | `developer.download.nvidia.com/compute/cuda/redist/cuda_cupti/windows-x86_64/cuda_cupti-windows-x86_64-12.5.39-archive.zip` | sha256 `9d9e23d1…e02a1d73` ✓ |
| cuobjdump 组件包 | 同源 `cuda_cuobjdump-…-12.5.39-archive.zip` | sha256 `39ca9868…70dd6d9` ✓ |
| nvdisasm 组件包 | 同源 `cuda_nvdisasm-…-12.5.39-archive.zip` | sha256 `2b378494…ca0cd98` ✓ |

组件包解压于 `C:\Users\l30086046\csg-tools\cuda\`（工具链装配目录，不入库）。
复现命令：

```text
# Level-1（launch 结构 + device 时长）：
cmd /c "call tools\env.cmd && tools\build_cupti_trace.cmd && build\cupti_trace.exe"
# Level-2（SASS 特征提取）：
#   先 cuobjdump -xelf sm_75 cublasLt64_12.dll（定位 cublasLt64_12.3007.sm_75.cubin）
#   再 cuobjdump --dump-sass --function volta_sgemm_<t>_nn <cubin>（需 nvdisasm 在 PATH）
#   最后 python tools\cublas_disasm.py <sass...> --trace cupti_trace_log.txt
```

> **证据裁剪声明（数据真实性）**：入库 `.sass` 文件已删除每指令后的控制编码注释列
> （形如 `/* 0x000fe40000000f00 */` 的整行，纯二进制编码冗余，分析不使用），指令地址、
> 调词、操作数完整保留；原始全量输出可由上述 §0 命令 1:1 再生。裁剪动机：企业代理对
> git push 请求体上限（AGENTS §8），入库体积 1.0MB→365KB。

## 1. 降级链记录（逐级，含负结果）

| 级 | 通道 | 结果 |
|---|---|---|
| L0 | **nsys profile**（设计首选） | **负结果**：本机两份 nsys CLI（系统 Nsight Systems 2024.2.3 与 Nsight Compute 归档附带版）在无 admin 企业环境下均无法工作——非 ASCII 工作目录触发 protobuf `Message parsing failed`；ASCII 暂存目录下 exit 0 但**不拉起 app、不产 report**（连 `whoami.exe` 最小用例亦失败）。判定为 CLI→后端服务 IPC 被环境阻断，与 CUDA 内容无关 |
| L1 | **进程内 CUPTI activity API**（本报告采纳） | ✓ 成功。与 nsys CUDA trace 同一数据源（CUPTI activity 记录），不依赖注入/后端服务/性能计数器权限（区别于 ncu 的 ERR_NVGPUCTRPERM P0 阻塞）。采集器 `tools/cupti_trace.cu`（新版 2024.x SDK 缓冲回调对 API，`cuptiActivityRegisterCallbacks`；旧 `cuptiActivitySubscribe` 已从该 SDK 移除） |
| L2 | **cuobjdump + nvdisasm 静态 SASS** | ✓ 成功（组件包下载，见 §0）。`cuobjdump --dump-sass` 内部 shell 调 nvdisasm，需将 nvdisasm bin 前置 PATH |

工程坑记录（复现者注意）：① PowerShell `>` 重定向产出 UTF-16LE，SASS 导出必须经
`cmd /c "… > file"` 原始字节；② 无 BOM UTF-8 中文注释被 MSVC host 按 CP936 误解析
（C4819 连锁幻影语法错），`cupti_trace.cu` 必须 UTF-8 **带 BOM**；③ nvcc 直连 link.exe
对非 ASCII 输出路径报 LNK1104，链接须在 ASCII 暂存目录完成再拷回。

## 2. Level-1：cuBLAS SGEMM 实际 launch 结构（CUPTI activity 实测）

行主序调用协议与 `src/cublas_baseline.cu` 逐字一致（`CUBLAS_DEFAULT_MATH` 显式禁
TF32；种子 7/8）。每尺寸 warmup 3 + 5 轮，共 8 次 launch，全部落同一配置：

| 尺寸 | kernel（cublasLt 符号） | grid (x,y,z) | block | regs/thr | smem | dur_med |
|---|---|---|---|---|---|---|
| 512³ | volta_sgemm_64x64_nn | (8,8,**3**) | (64,1,1) | 126 | 8448 B | 46.5 μs |
| 1024³ | volta_sgemm_128x64_nn | (8,16,**3**) | (128,1,1) | 122 | 12544 B | 253.9 μs |
| 2048³ | volta_sgemm_128x128_nn | (16,16,**2**) | (256,1,1) | 118 | 16640 B | 1871.5 μs |
| 4096³ | volta_sgemm_128x64_nn | (32,64,**1**) | (128,1,1) | 122 | 12544 B | 14586.7 μs |

（`cupti_trace_log.txt` 原文；dur 为快节奏小会话中位值，未锁频——结构数据为本节
载荷，时长仅作指示。同协议 steady-state 参照：`results/performance.csv` cuBLAS 行
512³=40.96μs / 1024³=214-237μs / 2048³=1661-1735μs / 4096³=13.9-14.4ms。）

**结构解码**（tile 映射按 `cublasSgemm(N,M,K, B ldb=N, A lda=K)` 换算，x=m 维=
我方 N、y=n 维=我方 M）：

1. **tile 缩放律**：三变体每线程输出数恒 **64**（64×64/64thr = 128×64/128thr =
   128×128/256thr = 64）——cuBLAS 以固定每线程工作量（≈1/2 我方 deep 的 128）
   换取低寄存器压力，用线程数缩放 tile，我方 deep 反其道（256thr 恒定、tile 深
   128 输出/线程）。1024³ 用 128×64 小 tile 是**主动选择**（同 DLL 含 128×128）。
2. **z 维 = K 分片**（split-K 家族）：SASS 入口 `S2R SR_CTAID.Z` + `IMAD.U32 R2, R1,
   c[0x1e0]`（z × 片宽）+ 4 个 per-z 工作区指针（c[0x198/0x19c/0x1a0/0x1a4]）；
   **全 trace 只见单 kernel、无独立 reduce launch**、SASS 无 RED/ATOMG → 归并为
   **kernel 内串行信号量**（z 片序工作区累加，单条 `@P5 MEMBAR.GPU` 0x4c60 为片间
   fence）。1024³ z=3 → 384 CTA；2048³ z=2 → 512 CTA；4096³ z=1（K 足长无需分片）。
3. **块映射无 swizzle**：CTAID.X/Y 直接进线性 `IMAD`（`IMAD.U32 R8, R6, c[0xc], R84`），
   无 XOR/LOP3 remap；y-z 平面有线性重编号（`IMAD.U32 R85, R11, -0x3, R85`，z 主
   序展平）。**cuBLAS 自己不用块序 swizzle**——与 AR012 F2 的诚实先验（swizzle 对
   总 DRAM 流量不变）相容，G7 判定仍待我方消融数据。
4. **占用结构**（sm_75：64K regs/SM、64KB smem/SM）：118-126 regs × 64-256 thr →
   **全部可 2 blocks/SM**（如 128×128：256thr×118regs×2=60.4K regs、2×16.6KB=33.2KB
   smem ✓）。我方 deep：243regs×256thr 恒 1 block/SM。

## 3. Level-2：SASS 指令构成对照（cuobjdump/nvdisasm 实测）

| kernel | insns | FFMA | LDS | STS | LDGSTS | LDG | STG | BAR | IMAD | LOP3 | FFMA:LDS | 最长FFMA连段 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| volta_sgemm_64x64_nn | 1328 | 520 | 60 | 35 | **0** | 71 | 9 | 7 | 26 | 46 | 8.67 | 58 |
| volta_sgemm_128x64_nn | 1248 | 520 | 60 | 27 | **0** | 59 | 9 | 7 | 26 | 46 | 8.67 | 58 |
| volta_sgemm_128x128_nn | 1208 | 520 | 60 | 19 | **0** | 47 | 9 | 7 | 26 | 46 | 8.67 | 58 |
| **sgemm_deep（我方默认）** | 1688 | 1024 | 48 | 9 | **0** | 6 | 32 | 1 | 69 | 16 | **21.33** | 171 |

（FFMA/LDS/IMAD 跨 cuBLAS 三变体恒定=同一内层模板；差异只在 staging（STS/LDG）与
launch 形状。我方 deep 为 `<DBUF=1,BPF=0,PH=0>`，243 regs / 24832B smem。）

## 4. 逐项对照解读（E-B 主问题：13pp@2048³ 差距的发射结构归因）

1. **FFMA 发射密度不是当前差距来源**。我方 FFMA:LDS=21.33、FFMA 连段 171，
   计算指令效率显著优于 cuBLAS（8.67/58）；但在 cuBLAS 领先的 1024³/2048³，
   其优势来自**占用与波填充**而非指令效率。4096³（双方满波、z=1 纯 tiling）
   我方 deep 9058 GF vs cuBLAS 9572-9881 GF（差距收窄至 4-6%），佐证满波区
   我方结构占优、残差在别处（cuBLAS 128×64 大网格 tile 复用 + epilogue STG=9
   vs 我方 32——我方 128 输出/线程的 epilogue 写出与尾延迟是可攻击点）。
2. **寄存器经济学**：cuBLAS 118-126 regs（2 blk/SM、8 warps×2）vs 我方 243-247
   （1 blk/SM、8 warps）。同 warp 数下 cuBLAS 用**两份独立 tile**覆盖延迟，
   我方用**单 tile 内深 ILP**（连段 171）。二者延迟 hiding 机制不同；AR009 已
   实测我方占用率非杠杆（--wlb 消融），故**不建议**向 cuBLAS 方向缩 tile——
   1024³ 差距应走 SK_RED（F3）/K 分片路径而非缩 tile。
3. **staging 无 cp.async**：cuBLAS sm_75 FP32 路径 LDGSTS=0——纯 LDG→STS 同步
   staging（LDG 47-71 条小粒度 + 7×BAR）vs 我方 deep LDG=6（vec4 宽载）+ STS=9
   + dbuf 单同步（BAR=1）。**我方 smem 流水指令效率更高**；cuBLAS 以更多阻塞
   点换 staging 简单性。这关闭了"cuBLAS 靠 cp.async 赢"的猜想。
4. **K 分片是 cuBLAS 小尺寸胜负手**：1024³ 我方最佳 6288 GF（74.9% miss）时
   cuBLAS 用 z=3 → 384 CTA 填满 48 SM（8 CTA/SM 驻留 2/SM × 4 波），我方 deep
   32 CTA 仅 2/3 波 + streamk W=1 96 CTA 仅 2 CTA/SM。**fr3 SK_RED separate
   （T005）的 cover-reduce 剥离方向与 cuBLAS 证据一致**：其归并也在 kernel 外
   结构（工作区+信号量）而非赢家线程串行 384KB。区别：cuBLAS 单 kernel 信号量
   串行，我方 T005 双 kernel（main 免票据直退 + cover-reduce kernel）——后者
   epilogue 更薄（预期 main 免 P 流量 + reduce 冷 C 流式），G1 判定数据说话。
5. **512³ 区带**：cuBLAS 64×64（64thr 小块、z=3）仅 5773-6554 GF（%peak 47-53，
   全场最低区）——小尺寸 cuBLAS 自身让出大量空间，我方 swsk/dsk 在此区带已
   领先（G2 122%@256³ 先例），auto v5 无需动此区带。
6. **对我方 IMAD=69 vs 26**：deep 的手动地址算术多 2.6×，绝对量小（<4% 指令），
   非当前瓶颈，不动。

## 5. 对 AR012 主线的可执行结论

| 结论 | 去向 |
|---|---|
| cuBLAS 无块序 swizzle（线性光栅+z 分片） | F2/G7 假说不受反面冲击，消融照做（T004）；若 G7 正，机制只能是波界残留/扇区粒度（cuBLAS 未用它≠无效，其 tile 复用模式与我方不同） |
| 1024³ 差距主因=波填充/K 分片/占用，非 FFMA 效率 | 支持 T005 SK_RED 主攻方向；**不支持**缩 tile 路线（AR009 负结果先例） |
| cuBLAS 归并走 kernel 外工作区+信号量（单 kernel） | T005 双 kernel 设计（链≡F2→bitwise）保留，epilogue 更薄的卖点成立 |
| 4096³ 残差 4-6% 聚焦 epilogue（STG 32 vs 9）+ tile 复用 | 供 G6/4096³ 后续（若 T005 后仍有余量，epilogue 向量化是下一候选，本 AR 不做） |
| cuBLAS 4096³ 选 128×64 而非 128×128（z=1） | 大尺寸 B 复用优先于单 tile 深度，与 §4.1 我方 deep 优势解读互洽（不冲突：我方 256×128 更大 tile 同向） |

## 6. 复现清单

1. `tools/build_cupti_trace.cmd && build\cupti_trace.exe` → Level-1 表（§2）
2. cuobjdump/nvdisasm 组件包就位后：§0 命令序列 → §3 表
3. 证据原文：本目录 4 个 `.sass` + `cupti_trace_log.txt`（git 同 commit 入库）
