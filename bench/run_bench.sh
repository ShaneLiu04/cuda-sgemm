#!/usr/bin/env bash
# =====================================================================
# bench/run_bench.sh — 批量 benchmark（Linux/bash）
# 用法：
#   ./bench/run_bench.sh                 # 性能阶梯：全 kernel x 主场景 4096^3
#   ./bench/run_bench.sh sweep           # 阶梯 + 全测试尺寸扫描
# 输出：终端摘要 + results/performance.csv（自动落盘）
# 时钟策略提醒（AGENTS.md §5.3）：跑分前建议
#   sudo nvidia-smi -lgc <freq>          # 固定时钟（需权限）
#   nvidia-smi dmon -s puc -d 1          # 另开终端监控频率/温度/功耗
# =====================================================================
set -euo pipefail

BIN="${BIN:-build/sgemm_bench}"
if [[ ! -x "$BIN" && -x "build/Release/sgemm_bench" ]]; then
    BIN="build/Release/sgemm_bench"
fi

main_scene=(--m 4096 --n 4096 --k 4096)

run_ladder() {
    echo "== performance ladder (4096^3, strict FP32) =="
    for kn in naive coalesced smem1d tile2d vec4 cpasync cpasync2 cublas; do
        "$BIN" --kernel "$kn" "${main_scene[@]}" --warmup 20 --iters 100 --csv
    done
}

run_sweep() {
    echo "== size sweep =="
    for sz in "4096 4096 4096" "1024 1024 1024" "256 256 256" \
              "1000 1016 1024" "1023 1024 511" "8192 8192 8192"; do
        set -- $sz
        for kn in naive coalesced smem1d tile2d vec4 cpasync cpasync2 cublas; do
            "$BIN" --kernel "$kn" --m "$1" --n "$2" --k "$3" --csv
        done
    done
}

case "${1:-ladder}" in
    ladder) run_ladder ;;
    sweep)  run_ladder; run_sweep ;;
    *)      echo "usage: $0 [ladder|sweep]"; exit 1 ;;
esac
