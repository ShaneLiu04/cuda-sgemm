# [AR002] design.md — Kernel 1：访存合并

| 组件名称 | cuda-sgemm / sgemm_coalesced |
| --- | --- |
| AR系统流水号 | AR002 |
| AR描述 | 仅交换线程映射使 warp 连续读 B / 连续写 C；无分块、无复用（证据链要求） |

# 2 动态行为

同 AR001 bench 流程；kernel 内部无状态、无同步。

# 3 功能点分解

| 序号 | 功能点 | 描述 |
| --- | --- | --- |
| 1 | 合并映射 kernel | col→threadIdx.x、row→threadIdx.y；block(16,16) 与 naive 完全同形 |
| 2 | A 访问模式说明 | warp 内同地址 → 硬件广播（写入注释与闭环文档） |
| 3 | ncu 对比闭环 | sectors/request、DRAM 流量比、stall 变化 → bottleneck_analysis.md |

# 4 实现设计

## 4.1 思路

唯一变量原则：与 naive 的 diff 仅有两行（row/col 与 grid 维度交换），
保证 E3 实验的因果干净。

## 4.2 关键设计

- 索引用 `(long long)` 防大矩阵溢出（4096×4096 = 2^24 元素，行首字节偏移可达 2^26，
  int 安全但乘积中间量可能溢出 —— 统一强转，为 8192³ 扫描留余量）。
- 不引入 smem：本版必须保持"重复读取 O(N)/O(M)"事实，供 E04 流量比证据。

## 4.3 接口

```cpp
void sgemm_coalesced(const float* A, const float* B, float* C, int M, int N, int K);
```

# 6 测试设计

- 复用 AR001 全矩阵回归（tests 自动覆盖新 kernel，无需新增用例文件）；
- E3（ncu sectors/request 对比）+ E04（流量比不变证据）为验收分析门。

## 预构建状态记录

代码已实现（src/sgemm_coalesced.cu）；GPU 环境执行 E3/E04 并回填闭环。
