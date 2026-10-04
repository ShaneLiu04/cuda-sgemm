#!/usr/bin/env bash
# =====================================================================
# profile/sanitize.sh — compute-sanitizer 检查（Linux/bash）
#   ./profile/sanitize.sh memcheck [kernel]    # 默认 memcheck，全 kernel
#   ./profile/sanitize.sh racecheck cpasync    # cp.async 双缓冲竞争抽查（AR006 硬门）
# 说明：racecheck 至少覆盖主场景 + 一个边界尺寸（srs AR006 §3.2）
# =====================================================================
set -euo pipefail

TOOL="${1:-memcheck}"
BIN="${BIN:-build/sgemm_bench}"
if [[ ! -x "$BIN" && -x "build/Release/sgemm_bench" ]]; then
    BIN="build/Release/sgemm_bench"
fi

run_one() {  # kernel m n k
    echo "== $TOOL: $1 ${2}x${3}x${4} =="
    compute-sanitizer --tool "$TOOL" "$BIN" --kernel "$1" \
        --m "$2" --n "$3" --k "$4" --warmup 1 --iters 1
}

case "$TOOL" in
    memcheck)
        for kn in naive coalesced smem1d tile2d vec4 cpasync cpasync2; do
            run_one "$kn" 1024 1024 1024
            run_one "$kn" 1023 1024 511     # 边界路径
        done ;;
    racecheck)
        for kn in "${@:2}"; do
            [[ $# -gt 1 ]] || kn=cpasync
            run_one "$kn" 4096 4096 4096
            run_one "$kn" 1023 1024 511
        done ;;
    *) echo "unknown tool: $TOOL (memcheck|racecheck)"; exit 1 ;;
esac
echo "sanitize done: no error reported above means clean."
