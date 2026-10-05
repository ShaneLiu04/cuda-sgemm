# =====================================================================
# run_auto_check.ps1 — AR008 T006 验证：auto 在 6 门尺寸的选核与性能
# 输出：results\auto_ar008.csv + 终端 dispatch 日志（--verbose）
# =====================================================================
$ErrorActionPreference = 'Stop'
$Bench = 'build\sgemm_bench.exe'
$Out = 'results\auto_ar008.csv'
$env:SGEMM_CSV = $Out

$sizes = @('256x256x256','512x512x512','1024x1024x1024','1000x1016x1024','2048x2048x2048','4096x4096x4096')
foreach ($s in $sizes) {
    $p = $s.Split('x')
    & $Bench --kernel auto --m $p[0] --n $p[1] --k $p[2] --warmup 20 --iters 100 --rounds 1 --csv --verbose
    if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: auto @ $s" }
}

Remove-Item Env:SGEMM_CSV
Write-Host "[auto-check] DONE -> $Out"
