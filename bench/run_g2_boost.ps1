# =====================================================================
# run_g2_boost.ps1 — AR010 T005b G2 boost 钟态实证（gate 门源钟态匹配）
# ---------------------------------------------------------------
# 背景：G2 绝对门 1618.2 GF 源自 AR007 会话 smem1d@256³（performance.csv
#       line 77: 1598.4 GF @ 1860 MHz——boost 钟态）；同核 1620 稳态仅
#       1260-1398 GF。绝对门跨钟态不可比 → 在 boost 钟态（冷态单发，
#       ~1935 MHz）直接复测候选，与门同钟态对判。
# 纪律：每行前冷却门 <=46C（冷态=boost 条件），无唤醒 spin（冷启动即
#       目的；warmup 20 内部拉钟），行级 gpu_state 记录实际钟态。
# 用法：
#   cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_g2_boost.ps1"
# 输出：results\g2_boost_ar010.csv
# =====================================================================
param(
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [int]$BoostGateC = 46,
    [int]$CooldownTimeoutSec = 400,
    [string]$Out = 'results\g2_boost_ar010.csv',
    [string]$Bench = 'build\sgemm_bench.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Bench)) {
    Write-Error "bench not found: $Bench"
}

function Get-GpuTemp {
    $t = & nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>$null
    if ($null -eq $t -or $t -eq '') { return -1 }
    return [int]$t
}

function Wait-Cold {
    $deadline = (Get-Date).AddSeconds($CooldownTimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $temp = Get-GpuTemp
        if ($temp -ge 0 -and $temp -le $BoostGateC) { return $temp }
        Start-Sleep -Seconds 5
    }
    return Get-GpuTemp
}

function Rename-LastRow([string]$old, [string]$new) {
    $lines = [System.Collections.Generic.List[string]](Get-Content $Out)
    if ($lines.Count -eq 0) { Write-Error "csv empty: $Out" }
    $last = $lines[$lines.Count - 1]
    if ($last.StartsWith($old + ',')) {
        $lines[$lines.Count - 1] = $new + $last.Substring($old.Length)
        Set-Content -Path $Out -Value $lines -Encoding ASCII
    } else {
        Write-Error "last row mismatch '${old}': $last"
    }
}

$cfgs = @(
    ,@('swsk',  '--sk 6',              'swsk',    '_sk6_boost')
    ,@('swsk',  '--sk 6 --rv2 1',      'swsk',    '_sk6_rv2_boost')
    ,@('smem1d', '',                   'smem1d',  '_boost')
    ,@('dsk',   '--sk 12 --dbuf 1',    'dsk',     '_sk12_dbuf1_boost')
    ,@('cublas', '',                   'cublas',  '_boost')
    ,@('swpipe', '',                   'swpipe',  '_boost')
)

$env:SGEMM_CSV = $Out
Write-Host "[g2-boost] -> $Out  (256x256x256, gate 1618.2 GF provenance ~1860 MHz)"

foreach ($c in $cfgs) {
    $kern = $c[0]; $extra = $c[1]; $pref = $c[2]; $suf = $c[3]
    $temp = Wait-Cold
    Write-Host ("[g2-boost] {0} {1} (start temp {2}C)" -f $kern, $extra, $temp)
    $argl = "--kernel $kern $extra --m 256 --n 256 --k 256 --warmup $Warmup --iters $Iters --rounds 1 --csv"
    & $Bench @($argl -split ' ')
    if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $kern $extra" }
    if ($suf -ne '') { Rename-LastRow $pref ($pref + $suf) }
    # 行后立即读钟态核对（应 ~1935）
    $smi = & nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader
    Write-Host ("[g2-boost]   post-run state: {0}" -f $smi)
}

Remove-Item Env:SGEMM_CSV
Write-Host ""
Write-Host "[g2-boost] DONE  out=$Out"
Write-Host "[g2-boost] check: every row's gpu_state clock >= 1900 MHz (boost regime);"
Write-Host "[g2-boost]        if any row shows 1620, that row is regime-mixed -> re-run."
