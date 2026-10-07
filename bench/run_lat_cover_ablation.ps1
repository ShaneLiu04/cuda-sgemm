# =====================================================================
# run_lat_cover_ablation.ps1 — AR011 T004 FR3 延迟覆盖消融（{on,off}² 因果分解）
# ---------------------------------------------------------------
# 协议（run_paired_v4.ps1 同型）：每尺寸 47°C 冷却门 + cublas 每轮首跑锚点 +
# GPU 唤醒 spin + SGEMM_CSV 分流（不触主 performance.csv）。
# 因子：--bpf（B 片段 kk+1 寄存器预取，FR3a）/ --phase（kk 轮转错相，FR3b）
# 载体：dsk --sk 3（1024³/2048³ 皆 AR010 实测最优配置；96/384 blocks 全整波）
# 序列：cublas -> dsk(00 基线) -> dsk --bpf 1 (10) -> dsk --phase 1 (01)
#       -> dsk --bpf 1 --phase 1 (11)
# 输出：results\lat_cover_ar011.csv
# 用法：cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_lat_cover_ablation.ps1"
# =====================================================================
param(
    [int]$Rounds = 3,
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [int]$CooldownTempC = 47,
    [int]$CooldownTimeoutSec = 300,
    [string]$Out = 'results\lat_cover_ar011.csv',
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

$sizes = @('1024x1024x1024', '2048x2048x2048')

$ts0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$env:SGEMM_CSV = $Out
Write-Host "[lat-cover] rounds=$Rounds -> $Out"

Write-Host "[lat-cover] wake spin..."
& $Bench --kernel swpipe --m 512 --n 512 --k 512 --warmup 20 --iters 100 --rounds 1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "wake spin failed" }

foreach ($s in $sizes) {
    $p = $s.Split('x'); $m = [int]$p[0]; $n = [int]$p[1]; $kk = [int]$p[2]
    $it = if ($kk -ge 4096) { 50 } else { $Iters }
    [void](Wait-Cooldown)

    for ($r = 1; $r -le $Rounds; $r++) {
        # 每轮序列：cublas 锚点 -> 基线 00 -> BPF 10 -> PHASE 01 -> 组合 11
        $seq = @(
            @('cublas', '',                     'cublas'),
            @('dsk',    '--sk 3',               'dsk_base'),
            @('dsk',    '--sk 3 --bpf 1',       'dsk_bpf'),
            @('dsk',    '--sk 3 --phase 1',     'dsk_phase'),
            @('dsk',    '--sk 3 --bpf 1 --phase 1', 'dsk_bpf_phase')
        )
        foreach ($c in $seq) {
            $kern = $c[0]; $extra = @($c[1] -split ' ' | Where-Object { $_ })
            Write-Host ("[lat-cover] {0} round {1}/{2}: {3} {4}" -f $s, $r, $Rounds, $kern, ($extra -join ' '))
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
Write-Host "[lat-cover] DONE  start=$ts0  end=$ts1  out=$Out"
Write-Host "[lat-cover] next : fig31 (make_figures.py) + tasks T004 回填"
