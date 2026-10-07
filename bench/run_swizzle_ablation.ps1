# =====================================================================
# run_swizzle_ablation.ps1 — AR012 T004 FR2 L2 块序 swizzle 消融（G7 判定）
# ---------------------------------------------------------------
# 协议（run_lat_cover_ablation.ps1 同型，AR011 v4+ 纪律）：每尺寸 47°C 冷却门
# + cublas 每轮首跑锚点 + GPU 唤醒 spin + SGEMM_CSV 分流（不触主 csv）。
# 因子：--swz {0,1} × --swzg {4,8,16}；载体：deep / dsk / streamk
#   （G7 正 → T006 auto v5 吸收路径全集；dsk 默认 --sk 4、streamk 默认 W=1）
# 尺寸：2048³/4096³ = G7 主判定（≥+1.0%）；512³/1024³ = 回退门（≤0.5pp）；
#   256³ = G2 守门（消融证伪区，swz 不应伤小尺寸）
# 序列（每轮）：cublas -> 每 kernel: swz0 -> swz1g4 -> swz1g8 -> swz1g16
# 输出：results\swizzle_ar012.csv（kernel 后缀 _swz0/_swz1g{G}）
# 用法：cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_swizzle_ablation.ps1"
# =====================================================================
param(
    [int]$Rounds = 3,
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [int]$CooldownTempC = 47,
    [int]$CooldownTimeoutSec = 300,
    [string]$Out = 'results\swizzle_ar012.csv',
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

# 尺寸次序：主判定先行（热斜坡下同轮互锚），回退门随后
$sizes = @('2048x2048x2048', '4096x4096x4096', '1024x1024x1024',
           '512x512x512', '256x256x256')
$kernels = @('deep', 'dsk', 'streamk')

$ts0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$env:SGEMM_CSV = $Out
Write-Host "[swz-abl] rounds=$Rounds -> $Out"

Write-Host "[swz-abl] wake spin..."
& $Bench --kernel swpipe --m 512 --n 512 --k 512 --warmup 20 --iters 100 --rounds 1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "wake spin failed" }

foreach ($s in $sizes) {
    $p = $s.Split('x'); $m = [int]$p[0]; $n = [int]$p[1]; $kk = [int]$p[2]
    $it = if ($kk -ge 4096) { 50 } elseif ($kk -ge 2048) { 100 } else { $Iters }
    [void](Wait-Cooldown)

    for ($r = 1; $r -le $Rounds; $r++) {
        # cublas 锚点（每轮首跑，热斜坡同轮互锚）
        Write-Host ("[swz-abl] {0} round {1}/{2}: cublas" -f $s, $r, $Rounds)
        & $Bench --kernel cublas --m $m --n $n --k $kk --warmup $Warmup --iters $it --rounds 1 --csv
        if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: cublas @ $s" }

        foreach ($kern in $kernels) {
            $cfgs = @(
                @('',                 'swz0'),
                @('--swz 1 --swzg 4', 'swz1g4'),
                @('--swz 1 --swzg 8', 'swz1g8'),
                @('--swz 1 --swzg 16', 'swz1g16')
            )
            foreach ($c in $cfgs) {
                $extra = @($c[0] -split ' ' | Where-Object { $_ })
                $tag = "${kern}_$($c[1])"
                Write-Host ("[swz-abl] {0} round {1}/{2}: {3}" -f $s, $r, $Rounds, $tag)
                $argv = @("--kernel", $kern) + $extra +
                    @("--m", "$m", "--n", "$n", "--k", "$kk",
                      "--warmup", "$Warmup", "--iters", "$it", "--rounds", "1", "--csv")
                & $Bench @argv
                if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $tag @ $s" }
                Rename-LastRow $kern $tag
            }
        }
    }
}

Remove-Item Env:SGEMM_CSV
$ts1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "[swz-abl] DONE  start=$ts0  end=$ts1  out=$Out"
Write-Host "[swz-abl] next : G7 判定解析 + fig35 (make_figures.py) + tasks T004 回填"
