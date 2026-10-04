# specs/backlog — 未开工 AR 暂存区

此目录存放**尚未开工**的 AR（AR002–AR006）的预填 srs.md 与 tasks.md。

**流转规则**（见 PROMPT.md §4）：
1. `specs/changes/` 一次只保留一个进行中的 AR；
2. 当前 AR 过 ST 验收后，其目录移入 `specs/archive/`；
3. 将本目录下一个 AR（严格按编号顺序）移入 `specs/changes/`，开启新会话；
4. 开工时先读上一 AR 在 `results/bottleneck_analysis.md` 的闭环记录，作为本 AR 设计输入。

| AR | 移入 changes 的前置条件 |
|----|------------------------|
| AR002-coalesced-access | AR001 归档 |
| AR003-smem-1d-tiling | AR002 归档 |
| AR004-register-tiling-2d | AR003 归档 |
| AR005-float4-vectorization | AR004 归档 |
| AR006-cpasync-double-buffer | AR005 归档 |
