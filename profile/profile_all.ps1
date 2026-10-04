# =====================================================================
# profile/profile_all.ps1 — Nsight Compute 批量采集（Windows/PowerShell）
# 用法：.\profile\profile_all.ps1 [-Kernels naive,coalesced,...]（默认全部）
# 产出：profile\<kernel>\<kernel>.ncu-rep + .csv + .details.txt
# =====================================================================
param([string[]]$Kernels = @('naive','coalesced','smem1d','tile2d','vec4','cpasync','cpasync2'))

$bin = "build\sgemm_bench.exe"
if (-not (Test-Path $bin) -and (Test-Path "build\Release\sgemm_bench.exe")) {
    $bin = "build\Release\sgemm_bench.exe"
}
if (-not (Test-Path $bin)) { Write-Error "binary not found: $bin"; exit 1 }

$metrics = "gpu__time_duration.sum," +
"sm__throughput.avg.pct_of_peak_sustained_elapsed," +
"gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed," +
"dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum," +
"l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio," +
"l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio," +
"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum," +
"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum," +
"smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio," +
"smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio," +
"smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio," +
"smsp__average_warps_issue_stalled_wait_per_issue_active.ratio," +
"smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio," +
"sm__warps_active.avg.pct_of_peak_sustained_active," +
"launch__registers_per_thread,launch__shared_mem_per_block_static," +
"sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active"

foreach ($kn in $Kernels) {
    New-Item -ItemType Directory -Force -Path "profile\$kn" | Out-Null
    Write-Output "== ncu profile: $kn =="
    & ncu -k "regex:$kn" --launch-count 3 --metrics $metrics `
        -o "profile\$kn\$kn" -f `
        $bin --kernel $kn --m 4096 --n 4096 --k 4096 --warmup 3 --iters 3
    if ($LASTEXITCODE -ne 0) { Write-Error "ncu failed: $kn"; exit 1 }
    & ncu --import "profile\$kn\$kn.ncu-rep" --csv --page raw `
        > "profile\$kn\$kn.csv"
    & ncu --import "profile\$kn\$kn.ncu-rep" --page details `
        > "profile\$kn\$kn.details.txt"
}
Write-Output "done. reports under profile\<kernel>\"
