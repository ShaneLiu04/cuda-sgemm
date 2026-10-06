# =====================================================================
# run_paired_v3.ps1 — AR009 T007 thermal-paired 配对测量协议 v3
# ---------------------------------------------------------------
# 与 AR008 run_paired.ps1 差异：
#   1) 挑战者按尺寸带 sk（T004 实测最优片数）：256³ wsk12/swsk6，
#      512³/1024³/1000x1016 wsk3/swsk3，2048³/4096³ 不带 swsk（sk1==基线）
#   2) 序列前热身 spin（T006 教训：空闲降频后首轮冷启动伪影 0.44x）
#   3) 冷却门 47C（本机闲置 46C，45C 门会全程空转超时）
#   4) 行名改写：wsk->wsk_sk<N> / swsk->swsk_sk<N> / auto->auto_v2
# 基线 swpipe 与挑战者交替背靠背 x 3 轮，共模热漂移对内相消。
# 输出：results\paired_ar009.csv
# =====================================================================
param(
    [string]$Baseline = 'swpipe',
    [int]$Rounds = 3,
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [int]$CooldownTempC = 47,
    [int]$CooldownTimeoutSec = 240,
    [string]$Out = 'results\paired_ar009.csv',
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

# 尺寸 -> 挑战者列表：@( kernel, extra_args, row_new_name 或 '' 保留原名 )
$plan = @(
    @{ s = '256x256x256'; chs = @(
        ,@('wide',  '',            'wide')
        ,@('wsk',   '--sk 12',     'wsk_sk12')
        ,@('swsk',  '--sk 6',      'swsk_sk6')
        ,@('auto',  '',            'auto_v2')
        ,@('cublas','',            'cublas') ) },
    @{ s = '512x512x512'; chs = @(
        ,@('wide',  '',            'wide')
        ,@('wsk',   '--sk 3',      'wsk_sk3')
        ,@('swsk',  '--sk 3',      'swsk_sk3')
        ,@('auto',  '',            'auto_v2')
        ,@('cublas','',            'cublas') ) },
    @{ s = '1024x1024x1024'; chs = @(
        ,@('wide',  '',            'wide')
        ,@('wsk',   '--sk 3',      'wsk_sk3')
        ,@('swsk',  '--sk 3',      'swsk_sk3')
        ,@('auto',  '',            'auto_v2')
        ,@('cublas','',            'cublas') ) },
    @{ s = '1000x1016x1024'; chs = @(
        ,@('wide',  '',            'wide')
        ,@('wsk',   '--sk 3',      'wsk_sk3')
        ,@('swsk',  '--sk 3',      'swsk_sk3')
        ,@('auto',  '',            'auto_v2')
        ,@('cublas','',            'cublas') ) },
    @{ s = '2048x2048x2048'; chs = @(
        ,@('wide',  '',            'wide')
        ,@('wsk',   '--sk 3',      'wsk_sk3')
        ,@('auto',  '',            'auto_v2')
        ,@('cublas','',            'cublas') ) },
    @{ s = '4096x4096x4096'; chs = @(
        ,@('wide',  '',            'wide')
        ,@('wsk',   '--sk 1',      'wsk_sk1')
        ,@('auto',  '',            'auto_v2')
        ,@('cublas','',            'cublas') ) }
)

$ts0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$env:SGEMM_CSV = $Out
Write-Host "[paired-v3] baseline=$Baseline rounds=$Rounds -> $Out"

# 热身 spin：把时钟/功耗态拉起（不落盘）
Write-Host "[paired-v3] warm spin..."
& $Bench --kernel swpipe --m 512 --n 512 --k 512 --warmup 20 --iters 100 --rounds 1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "warm spin failed" }

foreach ($group in $plan) {
    $p = $group.s.Split('x'); $m = [int]$p[0]; $n = [int]$p[1]; $kk = [int]$p[2]
    $clean = Wait-Cooldown
    if (-not $clean) { Write-Warning ("[paired-v3] {0} thermal-contaminated group" -f $group.s) }

    foreach ($c in $group.chs) {
        $kern = $c[0]; $extra = ($c[1] -split ' ') | Where-Object { $_ }
        for ($r = 1; $r -le $Rounds; $r++) {
            Write-Host ("[paired-v3] {0} round {1}/{2}: {3} -> {4} {5}" -f
                        $group.s, $r, $Rounds, $Baseline, $kern, ($extra -join ' '))
            & $Bench --kernel $Baseline --m $m --n $n --k $kk --warmup $Warmup --iters $Iters --rounds 1 --csv
            if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $Baseline @ $($group.s)" }
            & $Bench --kernel $kern --m $m --n $n --k $kk --warmup $Warmup --iters $Iters --rounds 1 --csv @($extra)
            if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $kern @ $($group.s)" }
            if ($c[2] -and $c[2] -ne $kern) { Rename-LastRow $kern $c[2] }
        }
    }
}

Remove-Item Env:SGEMM_CSV
$ts1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "[paired-v3] DONE  start=$ts0  end=$ts1  out=$Out"
