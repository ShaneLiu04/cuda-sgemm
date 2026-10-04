#!/usr/bin/env bash
# =====================================================================
# profile/profile_all.sh — Nsight Compute 批量采集（Linux/bash）
# 用法：./profile/profile_all.sh [kernel ...]  （默认全部自研 kernel）
# 产出：profile/<kernel>/<kernel>.ncu-rep + <kernel>.csv（指标导出）
# 指标集 = 详设 §4.5（SOL / DRAM 流量 / sectors / bank conflicts /
#         stall 分布 / occupancy / 寄存器 / FP32 pipe）
# =====================================================================
set -euo pipefail

BIN="${BIN:-build/sgemm_bench}"
if [[ ! -x "$BIN" && -x "build/Release/sgemm_bench" ]]; then
    BIN="build/Release/sgemm_bench"
fi

METRICS="gpu__time_duration.sum,
sm__throughput.avg.pct_of_peak_sustained_elapsed,
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,
dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,
smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,
sm__warps_active.avg.pct_of_peak_sustained_active,
launch__registers_per_thread,launch__shared_mem_per_block_static,
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active"

KERNELS=("${@:-}")
if [[ ${#KERNELS[@]} -eq 0 ]]; then
    KERNELS=(naive coalesced smem1d tile2d vec4 cpasync cpasync2)
fi

for kn in "${KERNELS[@]}"; do
    mkdir -p "profile/$kn"
    echo "== ncu profile: $kn =="
    ncu -k "regex:$kn" --launch-count 3 --metrics "$METRICS" \
        -o "profile/$kn/$kn" -f \
        "$BIN" --kernel "$kn" --m 4096 --n 4096 --k 4096 --warmup 3 --iters 3
    ncu --import "profile/$kn/$kn.ncu-rep" --csv \
        --page raw > "profile/$kn/$kn.csv"
    ncu --import "profile/$kn/$kn.ncu-rep" --page details \
        > "profile/$kn/$kn.details.txt"
done
echo "done. reports under profile/<kernel>/"
