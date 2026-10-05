# =====================================================================
# run_paired.ps1 — AR008 thermal-paired 配对测量协议执行器
# ---------------------------------------------------------------
# 原理（design.md §4.2.4）：基线/挑战者【交替成对】背靠背执行（A,B,A,B...），
# 共模热漂移（WDDM 动态时钟、功耗墙）在对内 delta 中相消；配对组间冷却门控。
# 行为：
#   1) 每尺寸：冷却门控（GPU 温度 <= -CooldownTempC 或超时标注 thermal-contaminated）
#   2) 每挑战者：Rounds 轮 [bench(baseline) → bench(challenger) 背靠背]
#      两行连续追加到 -Out（14 列 schema 不变；行序即配对关系，compare --paired 消费）
#   3) 输出 compare.py --paired 调用提示
# 用法（需先构建，且经 tools\env.cmd 引导 CUDA 环境）：
#   cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_paired.ps1 -Challengers vec4,cublas -Sizes 512x512x512,1024x1024x1024"
# 挑战者带参数写法：'swsk:--sk 8'（冒号后为原样 CLI 参数）
# =====================================================================
param(
    [string]$Baseline = 'swpipe',
    [string[]]$Challengers = @('swsk','ws','auto'),
    [string[]]$Sizes = @('256x256x256','512x512x512','1024x1024x1024','2048x2048x2048','4096x4096x4096','1000x1016x1024'),
    [int]$Rounds = 3,
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [int]$CooldownTempC = 45,
    [int]$CooldownTimeoutSec = 180,
    [string]$Out = 'results\paired_ar008.csv',
    [string]$Bench = 'build\sgemm_bench.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Bench)) {
    Write-Error "bench not found: $Bench (run: cmake --build build first)"
}

# 数组参数经 cmd /c 转发时会被拼成单串（'a,b'）——统一规范化拆分
foreach ($paramName in @('Challengers', 'Sizes')) {
    $val = Get-Variable -Name $paramName -ValueOnly
    if ($val.Count -eq 1 -and $val[0] -match ',') {
        Set-Variable -Name $paramName -Value @($val[0].Split(','))
    }
}

function Get-GpuTemp {
    $t = & nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>$null
    if ($null -eq $t -or $t -eq '') { return -1 }
    return [int]$t
}

function Wait-Cooldown {
    $deadline = (Get-Date).AddSeconds($CooldownTimeoutSec)
    $temp = Get-GpuTemp
    if ($temp -ge 0 -and $temp -le $CooldownTempC) { return $true }
    Write-Host "[cooldown] temp=${temp}C > ${CooldownTempC}C, waiting (timeout ${CooldownTimeoutSec}s)..."
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        $temp = Get-GpuTemp
        if ($temp -ge 0 -and $temp -le $CooldownTempC) {
            Write-Host "[cooldown] settled at ${temp}C"
            return $true
        }
    }
    Write-Warning "[cooldown] TIMEOUT at ${temp}C — subsequent group marked thermal-contaminated"
    return $false
}

# 挑战者规格解析：'name' 或 'name:--arg val ...'
function Parse-Challenger([string]$spec) {
    $parts = $spec.Split(':', 2)
    $name = $parts[0].Trim()
    $extra = @()
    if ($parts.Count -gt 1 -and $parts[1].Trim() -ne '') {
        $extra = ($parts[1].Trim() -split '\s+')
    }
    return @{ name = $name; extra = $extra }
}

$ts0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$env:SGEMM_CSV = $Out
Write-Host "[paired] baseline=$Baseline challengers=$($Challengers -join ',') rounds=$Rounds -> $Out"

foreach ($s in $Sizes) {
    $p = $s.Split('x'); $m = [int]$p[0]; $n = [int]$p[1]; $kk = [int]$p[2]
    $clean = Wait-Cooldown

    foreach ($cspec in $Challengers) {
        $ch = Parse-Challenger $cspec
        for ($r = 1; $r -le $Rounds; $r++) {
            Write-Host ("[paired] {0}x{1}x{2} round {3}/{4}: {5} -> {6} {7}" -f
                        $m, $n, $kk, $r, $Rounds, $Baseline, $ch.name, ($ch.extra -join ' '))
            & $Bench --kernel $Baseline --m $m --n $n --k $kk --warmup $Warmup --iters $Iters --rounds 1 --csv
            if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $Baseline @ $s" }
            & $Bench --kernel $ch.name --m $m --n $n --k $kk --warmup $Warmup --iters $Iters --rounds 1 --csv @($ch.extra)
            if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $($ch.name) @ $s" }
        }
    }
}

Remove-Item Env:SGEMM_CSV
$ts1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "[paired] DONE  start=$ts0  end=$ts1  out=$Out"
Write-Host "[paired] next  : python bench\compare.py --paired $Out -o results\compare_ar008_paired.md"

