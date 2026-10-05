# =====================================================================
# run_matrix.ps1 — AR007 自动实验矩阵：9 kernel x 6 尺寸 + 消融一键执行
# ---------------------------------------------------------------
# 用法（需先构建，且经 tools\env.cmd 引导 CUDA 环境）：
#   cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_matrix.ps1"
# 可选参数：
#   -Kernels naive,vec4,...     默认全部 9 个
#   -Sizes 256x256x256,...      默认 6 个尺寸（含 1000x1016x1024 非方形）
#   -Rounds 3 -Warmup 20 -Iters 100
#   -SkipAblations              跳过消融（smem1d bk8/16、tile2d lb2、swpipe lb2 @4096^3）
# 行为：
#   1) 快照当前 results\performance.csv → performance_preAR007_<stamp>.csv（compare 基线）
#   2) 矩阵：每 cell 调 sgemm_bench --rounds --csv（主 CSV，一行一结果）
#   3) 消融：SGEMM_CSV 重定向到 results\ablation_ar007.csv（不污染主矩阵）
#   4) 输出 compare.py 调用提示
# =====================================================================
param(
    [string[]]$Kernels = @('naive','coalesced','smem1d','tile2d','vec4','cpasync','cpasync2','swpipe','cublas'),
    [string[]]$Sizes   = @('256x256x256','512x512x512','1024x1024x1024','2048x2048x2048','4096x4096x4096','1000x1016x1024'),
    [int]$Rounds = 3,
    [int]$Warmup = 20,
    [int]$Iters  = 100,
    [switch]$SkipAblations,
    [string]$Bench = 'build\sgemm_bench.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Bench)) {
    Write-Error "bench not found: $Bench (run: cmake --build build first)"
}

$ts0  = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$snapshot = "results\performance_preAR007_$stamp.csv"
if (Test-Path 'results\performance.csv') {
    Copy-Item 'results\performance.csv' $snapshot
    Write-Host "[matrix] baseline snapshot -> $snapshot"
} else {
    Write-Warning "[matrix] results\performance.csv not found; compare.py will have no baseline"
}

foreach ($k in $Kernels) {
    foreach ($s in $Sizes) {
        $p = $s.Split('x'); $m = [int]$p[0]; $n = [int]$p[1]; $kk = [int]$p[2]
        Write-Host "`n[matrix] $k  ${m}x${n}x${kk}  rounds=$Rounds warmup=$Warmup iters=$Iters"
        & $Bench --kernel $k --m $m --n $n --k $kk --warmup $Warmup --iters $Iters --rounds $Rounds --csv
        if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: kernel=$k size=$s" }
    }
}

if (-not $SkipAblations) {
    # 消融分流：SGEMM_CSV 重定向（main.cu 支持），主矩阵 CSV 保持一行一 cell
    $env:SGEMM_CSV = 'results\ablation_ar007.csv'
    $ablations = @(
        @{ k = 'smem1d'; extra = @('--bk','8');  note = 'smem1d bk=8  @4096^3' },
        @{ k = 'smem1d'; extra = @('--bk','16'); note = 'smem1d bk=16 @4096^3' },
        @{ k = 'tile2d'; extra = @('--lb','2');  note = 'tile2d lb=2  @4096^3' },
        @{ k = 'swpipe'; extra = @('--lb','2');  note = 'swpipe lb=2  @4096^3' }
    )
    foreach ($ab in $ablations) {
        Write-Host "`n[ablation] $($ab.note)"
        & $Bench --kernel $ab.k --m 4096 --n 4096 --k 4096 --warmup $Warmup --iters $Iters --rounds $Rounds --csv @($ab.extra)
        if ($LASTEXITCODE -ne 0) { Write-Error "ablation failed: $($ab.note)" }
    }
    Remove-Item Env:SGEMM_CSV
}

$ts1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "[matrix] DONE  start=$ts0  end=$ts1  cells=$($Kernels.Count * $Sizes.Count)"
Write-Host "[matrix] baseline snapshot : $snapshot"
Write-Host "[matrix] next              : python bench\compare.py $snapshot results\performance.csv -o results\compare_ar007.md"
