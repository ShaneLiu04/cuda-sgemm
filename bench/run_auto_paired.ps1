# =====================================================================
# run_auto_paired.ps1 — AR010 T006 auto v3 配对保真验证
# ---------------------------------------------------------------
# 目的：auto v3 dispatch vs 各尺寸实测冠军 背靠背交替（A-B-A-B），
#       |Δ| ≤ 2pp 为保真通过（同会话同钟态配对相消）。
# 冠军表（T004/T005 1620 稳态同会话实测）：
#   256³ swsk sk6 / 512³ swsk sk3 / 1024³+1000x1016 dsk sk3 dbuf1 /
#   2048³+4096³ deep dbuf1（sk1 单波）
# 输出：results\auto_ar010.csv
# 用法：cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_auto_paired.ps1"
# =====================================================================
param(
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [int]$CooldownTempC = 47,
    [int]$CooldownTimeoutSec = 300,
    [string]$Out = 'results\auto_ar010.csv',
    [string]$Bench = 'build\sgemm_bench.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Bench)) { Write-Error "bench not found: $Bench" }

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
            Write-Host "[cooldown] settled at ${temp}C"; return $true
        }
    }
    Write-Warning "[cooldown] TIMEOUT at ${temp}C"; return $false
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

# size -> 冠军配置：kernel, extra_args, 行名
$winners = @(
    @{ s = '256x256x256';      w = @('swsk', '--sk 6',           'winner_swsk_sk6') },
    @{ s = '512x512x512';      w = @('swsk', '--sk 3',           'winner_swsk_sk3') },
    @{ s = '1024x1024x1024';   w = @('dsk',  '--sk 3',           'winner_dsk_sk3') },
    @{ s = '1000x1016x1024';   w = @('dsk',  '--sk 3',           'winner_dsk_sk3') },
    @{ s = '2048x2048x2048';   w = @('deep', '',                 'winner_deep') },
    @{ s = '4096x4096x4096';   w = @('deep', '',                 'winner_deep') }
)

# GPU 唤醒 spin
Write-Host "[auto-paired] GPU wake spin..."
& $Bench --kernel deep --m 512 --n 512 --k 512 --warmup 20 --iters 30 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "wake spin failed" }

$env:SGEMM_CSV = $Out
Write-Host "[auto-paired] -> $Out  (A-B-A-B per size, |delta| <= 2pp pass)"

foreach ($g in $winners) {
    $p = $g.s.Split('x'); $m = [int]$p[0]; $n = [int]$p[1]; $kk = [int]$p[2]
    $wk = $g.w[0]; $wx = $g.w[1]; $wn = $g.w[2]
    $it = if ($kk -ge 4096) { 50 } else { $Iters }
    [void](Wait-Cooldown)
    foreach ($rep in 1..2) {
        # A: auto
        Write-Host ("[auto-paired] {0} rep{1} auto" -f $g.s, $rep)
        & $Bench @("--kernel","auto","--m","$m","--n","$n","--k","$kk",
                   "--warmup","$Warmup","--iters","$it","--rounds","1","--csv")
        if ($LASTEXITCODE -ne 0) { Write-Error "auto failed @ $($g.s)" }
        # B: winner
        Write-Host ("[auto-paired] {0} rep{1} {2} {3}" -f $g.s, $rep, $wk, $wx)
        $wargs = @("--kernel",$wk) + @($wx -split ' ' | Where-Object { $_ -ne '' }) +
                 @("--m","$m","--n","$n","--k","$kk",
                   "--warmup","$Warmup","--iters","$it","--rounds","1","--csv")
        & $Bench @wargs
        if ($LASTEXITCODE -ne 0) { Write-Error "winner failed @ $($g.s)" }
        Rename-LastRow $wk $wn
    }
}

Remove-Item Env:SGEMM_CSV
Write-Host ""
Write-Host "[auto-paired] DONE  out=$Out"
Write-Host "[auto-paired] next : parse pairs -> |delta| table (fig 附 auto v3 保真)"
