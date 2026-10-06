# =====================================================================
# run_smoke_deep_ar010.ps1 — AR010 T002 deep 冒烟（早期性能信号 + fig22 数据）
# ---------------------------------------------------------------
# 覆盖：deep dbuf{0,1} + swpipe 参照 × {256³,512³,1024³,1000x1016x1024,2048³,4096³}
# 纪律：GPU 唤醒 spin（256³ 冷启动伪影教训）+ 冷却门 47C + 行名改写
#       deep->deep_dbuf<N>；swpipe 同会话参照（跨会话不可比教训）。
# 用法：
#   cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_smoke_deep_ar010.ps1"
# 输出：results\smoke_deep_ar010.csv（冒烟分流，禁止污染主 performance.csv）
# =====================================================================
param(
    [int]$Warmup = 20,
    [int]$Iters = 60,
    [int]$CooldownTempC = 47,
    [int]$CooldownTimeoutSec = 300,
    [string]$Out = 'results\smoke_deep_ar010.csv',
    [string]$Bench = 'build\sgemm_bench.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Bench)) {
    Write-Error "bench not found: $Bench (run: cmake --build build first)"
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
    Write-Host "[cooldown] temp=${temp}C > ${CooldownTempC}C, waiting..."
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

function Rename-LastRow([string]$old, [string]$new) {
    $lines = [System.Collections.Generic.List[string]](Get-Content $Out)
    if ($lines.Count -eq 0) { Write-Error "csv empty: $Out" }
    $last = $lines[$lines.Count - 1]
    if ($last.StartsWith($old + ',')) {
        $lines[$lines.Count - 1] = $new + $last.Substring($old.Length)
        Set-Content -Path $Out -Value $lines -Encoding ASCII
    } else {
        Write-Error "last row does not start with '${old}': $last"
    }
}

# GPU 唤醒 spin（AR009 T006 教训：空闲降频后首轮 0.44x 伪影）
Write-Host "[smoke] GPU wake spin..."
& $Bench --kernel deep --m 512 --n 512 --k 512 --warmup 20 --iters 30 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "wake spin failed" }

$sizes = @(
    ,@('256x256x256', 256, 256, 256)
    ,@('512x512x512', 512, 512, 512)
    ,@('1024x1024x1024', 1024, 1024, 1024)
    ,@('1000x1016x1024', 1000, 1016, 1024)
    ,@('2048x2048x2048', 2048, 2048, 2048)
    ,@('4096x4096x4096', 4096, 4096, 4096)
)
# 每尺寸配置：swpipe 参照 / deep_dbuf0 / deep_dbuf1
$cfgs = @(
    ,@('swpipe', '',           'swpipe', '')
    ,@('deep',   '',           'deep',   '_dbuf0')
    ,@('deep',   '--dbuf 1',   'deep',   '_dbuf1')
)

$ts0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$env:SGEMM_CSV = $Out
Write-Host "[smoke] -> $Out"

foreach ($sz in $sizes) {
    $name = $sz[0]; $m = $sz[1]; $n = $sz[2]; $k = $sz[3]
    $clean = Wait-Cooldown
    if (-not $clean) { Write-Warning ("[smoke] {0} thermal-contaminated group" -f $name) }
    foreach ($c in $cfgs) {
        $kern = $c[0]; $extra = $c[1]; $pref = $c[2]; $suf = $c[3]
        Write-Host ("[smoke] {0} {1} {2}" -f $name, $kern, $extra)
        $args = "--kernel $kern $extra --m $m --n $n --k $k --warmup $Warmup --iters $Iters --rounds 1 --csv"
        & $Bench @($args -split ' ')
        if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $kern $extra @ $name" }
        if ($suf -ne '') { Rename-LastRow $pref ($pref + $suf) }
    }
}

Remove-Item Env:SGEMM_CSV
$ts1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "[smoke] DONE  start=$ts0  end=$ts1  out=$Out"
