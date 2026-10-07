# =====================================================================
# run_auto_v4_paired.ps1 — AR011 T006 auto v4 配对验证
# ---------------------------------------------------------------
# 协议（v4 同型）：每尺寸 47°C 冷却 + cublas 每轮首锚 + 唤醒 spin +
# SGEMM_CSV 分流。每轮序列：
#   canonical 六尺寸：cublas -> v3_pick（AR010 dispatch）-> v4_pick
#                     （AR011 胜者，显式跑）-> auto（v4 dispatch）
#     -> v3 vs v4 delta = G4''' 增量证据；v4_pick vs auto delta ~ 0 =
#        G5''' dispatch 保真（A-B-A-B，2 轮 = 4 配对/尺寸）
#   补充尺寸（4:1 长宽比，全量首测）：cublas -> deep -> dsk_sk3 ->
#     swsk_sk3 -> streamk_w1 -> streamk_w2 -> auto（v4 首测回填）
# 行名：cublas / v3_<k> / w4_<k> / auto_v4 / deep / dsk_sk3 /
#       swsk_sk3 / streamk_w1 / streamk_w2
# 输出：results\auto_ar011.csv
# 用法：cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_auto_v4_paired.ps1"
# =====================================================================
param(
    [int]$Rounds = 2,
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [int]$CooldownTempC = 47,
    [int]$CooldownTimeoutSec = 300,
    [string]$Out = 'results\auto_ar011.csv',
    [string]$Bench = 'build\sgemm_bench.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Bench)) { Write-Error "bench not found: $Bench (run: cmake --build build first)" }

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

# 尺寸 -> 轮内序列（kernel, extra, rowname）。canonical：v3_pick / w4_pick
# / auto；补充尺寸：全候选首测 + auto。
#   v3（AR010）：256³ swsk6 / 512³ swsk3 / 1024³+1000x1016 dsk3 /
#               2048³+4096³ deep
#   v4（AR011）：仅 2048³ 变更为 streamk W=1（T005 裁决），其余承袭
$plan = @(
    @{ s = '256x256x256'; sup = $false; seq = @(
        @('cublas', '', 'cublas'),
        @('swsk', '--sk 6', 'v3_swsk_sk6'),
        @('swsk', '--sk 6', 'w4_swsk_sk6'),
        @('auto', '', 'auto_v4')) },
    @{ s = '512x512x512'; sup = $false; seq = @(
        @('cublas', '', 'cublas'),
        @('swsk', '--sk 3', 'v3_swsk_sk3'),
        @('swsk', '--sk 3', 'w4_swsk_sk3'),
        @('auto', '', 'auto_v4')) },
    @{ s = '1024x1024x1024'; sup = $false; seq = @(
        @('cublas', '', 'cublas'),
        @('dsk', '--sk 3', 'v3_dsk_sk3'),
        @('dsk', '--sk 3', 'w4_dsk_sk3'),
        @('auto', '', 'auto_v4')) },
    @{ s = '1000x1016x1024'; sup = $false; seq = @(
        @('cublas', '', 'cublas'),
        @('dsk', '--sk 3', 'v3_dsk_sk3'),
        @('dsk', '--sk 3', 'w4_dsk_sk3'),
        @('auto', '', 'auto_v4')) },
    @{ s = '2048x2048x2048'; sup = $false; seq = @(
        @('cublas', '', 'cublas'),
        @('deep', '', 'v3_deep'),
        @('streamk', '--waves 1', 'w4_streamk_w1'),
        @('auto', '', 'auto_v4')) },
    @{ s = '4096x4096x4096'; sup = $false; seq = @(
        @('cublas', '', 'cublas'),
        @('deep', '', 'v3_deep'),
        @('deep', '', 'w4_deep'),
        @('auto', '', 'auto_v4')) },
    @{ s = '256x4096x4096'; sup = $true; seq = @(
        @('cublas', '', 'cublas'),
        @('deep', '', 'deep'),
        @('dsk', '--sk 3', 'dsk_sk3'),
        @('swsk', '--sk 3', 'swsk_sk3'),
        @('streamk', '--waves 1', 'streamk_w1'),
        @('streamk', '--waves 2', 'streamk_w2'),
        @('auto', '', 'auto_v4')) },
    @{ s = '4096x256x4096'; sup = $true; seq = @(
        @('cublas', '', 'cublas'),
        @('deep', '', 'deep'),
        @('dsk', '--sk 3', 'dsk_sk3'),
        @('swsk', '--sk 3', 'swsk_sk3'),
        @('streamk', '--waves 1', 'streamk_w1'),
        @('streamk', '--waves 2', 'streamk_w2'),
        @('auto', '', 'auto_v4')) },
    @{ s = '1024x2048x2048'; sup = $true; seq = @(
        @('cublas', '', 'cublas'),
        @('deep', '', 'deep'),
        @('dsk', '--sk 3', 'dsk_sk3'),
        @('swsk', '--sk 3', 'swsk_sk3'),
        @('streamk', '--waves 1', 'streamk_w1'),
        @('streamk', '--waves 2', 'streamk_w2'),
        @('auto', '', 'auto_v4')) }
)

$ts0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$env:SGEMM_CSV = $Out
Write-Host "[auto-v4] rounds=$Rounds -> $Out"

Write-Host "[auto-v4] wake spin..."
& $Bench --kernel swpipe --m 512 --n 512 --k 512 --warmup 20 --iters 100 --rounds 1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "wake spin failed" }

foreach ($p in $plan) {
    $d = $p.s.Split('x'); $m = [int]$d[0]; $n = [int]$d[1]; $kk = [int]$d[2]
    $it = if ($kk -ge 4096) { 50 } else { $Iters }
    [void](Wait-Cooldown)

    for ($r = 1; $r -le $Rounds; $r++) {
        foreach ($c in $p.seq) {
            $kern = $c[0]; $extra = @($c[1] -split ' ' | Where-Object { $_ })
            Write-Host ("[auto-v4] {0} round {1}/{2}: {3} {4}" -f $p.s, $r, $Rounds, $kern, ($extra -join ' '))
            $argv = @("--kernel",$kern) + $extra +
                @("--m","$m","--n","$n","--k","$kk","--warmup","$Warmup","--iters","$it","--rounds","1","--csv")
            & $Bench @argv
            if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $kern @ $($p.s)" }
            if ($c[2] -ne $kern) { Rename-LastRow $kern $c[2] }
        }
    }
}

Remove-Item Env:SGEMM_CSV
$ts1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "[auto-v4] DONE  start=$ts0  end=$ts1  out=$Out"
Write-Host "[auto-v4] next : parse (G4''' v3vs v4 delta / G5''' w4 vs auto 保真) -> fig33 + tasks T006 回填"
