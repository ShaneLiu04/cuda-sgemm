# =====================================================================
# profile/sanitize.ps1 — compute-sanitizer 检查（Windows/PowerShell）
#   .\profile\sanitize.ps1 -Tool memcheck        # 全 kernel memcheck
#   .\profile\sanitize.ps1 -Tool racecheck -Kernels cpasync,cpasync2
# =====================================================================
param(
    [ValidateSet('memcheck','racecheck')][string]$Tool = 'memcheck',
    [string[]]$Kernels = @('naive','coalesced','smem1d','tile2d','vec4','cpasync','cpasync2')
)

$bin = "build\sgemm_bench.exe"
if (-not (Test-Path $bin) -and (Test-Path "build\Release\sgemm_bench.exe")) {
    $bin = "build\Release\sgemm_bench.exe"
}
if (-not (Test-Path $bin)) { Write-Error "binary not found: $bin"; exit 1 }

function Invoke-One {
    param([string]$kn, [int]$m, [int]$n, [int]$k)
    Write-Output "== $Tool : $kn ${m}x${n}x${k} =="
    & compute-sanitizer --tool $Tool $bin --kernel $kn --m $m --n $n --k $k --warmup 1 --iters 1
    if ($LASTEXITCODE -ne 0) { Write-Error "sanitizer reported issues for $kn"; exit 1 }
}

if ($Tool -eq 'racecheck') {
    # racecheck 至少覆盖主场景 + 一个边界尺寸（srs AR006 §3.2）
    foreach ($kn in $Kernels) {
        Invoke-One $kn 4096 4096 4096
        Invoke-One $kn 1023 1024 511
    }
} else {
    foreach ($kn in $Kernels) {
        Invoke-One $kn 1024 1024 1024
        Invoke-One $kn 1023 1024 511     # 边界/回退路径
    }
}
Write-Output "sanitize done: no error reported above means clean."
