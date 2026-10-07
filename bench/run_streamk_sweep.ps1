# =====================================================================
# run_streamk_sweep.ps1 — AR011 T005 Stream-K W sweep + auto 公式校准
# ---------------------------------------------------------------
# 协议（run_lat_cover_ablation.ps1 / run_paired_v4.ps1 同型）：每尺寸
# 47°C 冷却 + cublas 每轮首跑锚点 + GPU 唤醒 spin + SGEMM_CSV 分流
# （不触主 performance.csv）。
# 尺寸：canonical 六尺寸（256³ TOT<48 → streamk wrapper 旁路 deep，单行
# 验证；512³ W 上限几何受限，扫 {1,2}；1024³/2048³ 主战场扫 {1..8}；
# 1000×1016×1024 / 4096³ 扫 {1..6}，4096³ iters=50 沿 v4 先例）。
# 参照（同会话）：cublas / deep / dsk --sk 3 / swsk（sk 按 v4 尺寸表）。
# 行名（Rename-LastRow）：cublas / deep / dsk_sk3 / swsk_skN /
#                          streamk_wW
# 输出：results\streamk_ar011.csv
# 用法：cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_streamk_sweep.ps1"
# =====================================================================
param(
    [int]$Rounds = 3,
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [int]$CooldownTempC = 47,
    [int]$CooldownTimeoutSec = 300,
    [string]$Out = 'results\streamk_ar011.csv',
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

# 尺寸 -> (W 集合, swsk 的 sk)。256³ TOT<48：streamk wrapper 旁路 deep（单行）。
$plan = [ordered]@{
    '256x256x256'      = @{ ws = @(1);             sk = 6 }
    '512x512x512'      = @{ ws = @(1, 2);          sk = 3 }
    '1024x1024x1024'   = @{ ws = @(1,2,3,4,5,6,7,8); sk = 3 }
    '1000x1016x1024'   = @{ ws = @(1,2,3,4,5,6);   sk = 3 }
    '2048x2048x2048'   = @{ ws = @(1,2,3,4,5,6,7,8); sk = 3 }
    '4096x4096x4096'   = @{ ws = @(1,2,3,4,5,6);   sk = 3 }
}

$ts0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$env:SGEMM_CSV = $Out
Write-Host "[sk-sweep] rounds=$Rounds -> $Out"

Write-Host "[sk-sweep] wake spin..."
& $Bench --kernel swpipe --m 512 --n 512 --k 512 --warmup 20 --iters 100 --rounds 1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "wake spin failed" }

foreach ($s in $plan.Keys) {
    $p = $s.Split('x'); $m = [int]$p[0]; $n = [int]$p[1]; $kk = [int]$p[2]
    $ws = $plan[$s].ws; $sks = $plan[$s].sk
    $it = if ($kk -ge 4096) { 50 } else { $Iters }
    [void](Wait-Cooldown)

    for ($r = 1; $r -le $Rounds; $r++) {
        # 每轮序列：cublas 锚点 -> deep -> dsk_sk3 -> swsk_skN -> streamk W 扫
        $seq = @(
            @('cublas',   '',                    'cublas'),
            @('deep',     '',                    'deep'),
            @('dsk',      "--sk 3",              'dsk_sk3'),
            @('swsk',     "--sk $sks",           "swsk_sk$sks")
        )
        foreach ($w in $ws) {
            $seq += ,@('streamk', "--waves $w",  "streamk_w$w")
        }
        foreach ($c in $seq) {
            $kern = $c[0]; $extra = @($c[1] -split ' ' | Where-Object { $_ })
            Write-Host ("[sk-sweep] {0} round {1}/{2}: {3} {4}" -f $s, $r, $Rounds, $kern, ($extra -join ' '))
            $argv = @("--kernel",$kern) + $extra +
                @("--m","$m","--n","$n","--k","$kk","--warmup","$Warmup","--iters","$it","--rounds","1","--csv")
            & $Bench @argv
            if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $kern @ $s" }
            if ($c[2] -ne $kern) { Rename-LastRow $kern $c[2] }
        }
    }
}

Remove-Item Env:SGEMM_CSV
$ts1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "[sk-sweep] DONE  start=$ts0  end=$ts1  out=$Out"
Write-Host "[sk-sweep] next : parse (W 最优表 + 波量化验证 + 时间预算) -> fig32 + tasks T005 回填"
