# =====================================================================
# run_size256.ps1 — AR008 T005 一次性采集：256^3 全 kernel 同会话对比
# 输出：results\size256_ar008.csv（独立分流，不污染主 CSV / 消融 CSV）
# =====================================================================
$ErrorActionPreference = 'Stop'
$Bench = 'build\sgemm_bench.exe'
$Out = 'results\size256_ar008.csv'
$env:SGEMM_CSV = $Out

$kernels = @('naive','coalesced','smem1d','tile2d','vec4','cpasync','cpasync2','swpipe','cublas')
foreach ($k in $kernels) {
    & $Bench --kernel $k --m 256 --n 256 --k 256 --warmup 20 --iters 100 --rounds 1 --csv
    if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $k" }
}

# swsk 今会话最优片数 sk=12（T004 扫描结论），行名标记 swsk_sk12
& $Bench --kernel swsk --sk 12 --m 256 --n 256 --k 256 --warmup 20 --iters 100 --rounds 1 --csv
if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: swsk sk12" }
$lines = [System.Collections.Generic.List[string]](Get-Content $Out)
$last = $lines[$lines.Count - 1]
if ($last.StartsWith('swsk,')) {
    $lines[$lines.Count - 1] = 'swsk_sk12' + $last.Substring(5)
    Set-Content -Path $Out -Value $lines -Encoding ASCII
}

Remove-Item Env:SGEMM_CSV
Write-Host "[size256] DONE -> $Out"
