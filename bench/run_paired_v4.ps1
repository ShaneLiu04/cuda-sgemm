# =====================================================================
# run_paired_v4.ps1 — AR010 T007 五门 v4 复判配对协议
# ---------------------------------------------------------------
# 与 AR009 run_paired_v3.ps1 差异：
#   1) kernel 集 = cublas/deep/dsk(sk3)/auto(v3)/swsk(按尺寸 sk)/swpipe
#      —— deep/dsk 为 AR010 新家族；cublas 每轮首跑（G1 比例门分母锚点，
#      轮内核背靠背共模钟态相消）
#   2) --dbuf 默认已固化 1（T004/T006），deep/dsk 行即 DBUF=1 主路径
#   3) 解析期钟态过滤：256³-1024³ 门用 1620 稳态轮（gpu_state 列），
#      boost 混染轮丢弃；2048³/4096³ 重核轮内 cublas 与自研核同钟态，
#      比例门直接有效；跨钟态绝对比较一律 %peak 归一
#   4) 行名改写：dsk->dsk_sk3 / auto->auto_v3 / swsk->swsk_sk<N>
# 输出：results\paired_ar010.csv
# 用法：cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_paired_v4.ps1"
# =====================================================================
param(
    [int]$Rounds = 3,
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [int]$CooldownTempC = 47,
    [int]$CooldownTimeoutSec = 300,
    [string]$Out = 'results\paired_ar010.csv',
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

# 尺寸 -> swsk 片数（AR009/AR010 实测守擂/冠军配置）
$swsk_sk = @{ '256x256x256' = 6; '512x512x512' = 3; '1024x1024x1024' = 3;
              '1000x1016x1024' = 3; '2048x2048x2048' = 3; '4096x4096x4096' = 3 }
$sizes = @('256x256x256', '512x512x512', '1024x1024x1024',
           '1000x1016x1024', '2048x2048x2048', '4096x4096x4096')

$ts0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$env:SGEMM_CSV = $Out
Write-Host "[paired-v4] rounds=$Rounds -> $Out"

# 唤醒 spin：拉起时钟/功耗态（不落盘）
Write-Host "[paired-v4] wake spin..."
& $Bench --kernel swpipe --m 512 --n 512 --k 512 --warmup 20 --iters 100 --rounds 1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "wake spin failed" }

foreach ($s in $sizes) {
    $p = $s.Split('x'); $m = [int]$p[0]; $n = [int]$p[1]; $kk = [int]$p[2]
    $skn = $swsk_sk[$s]
    $it = if ($kk -ge 4096) { 50 } else { $Iters }
    [void](Wait-Cooldown)

    for ($r = 1; $r -le $Rounds; $r++) {
        # 每轮序列：cublas（分母锚点）-> deep -> dsk -> auto -> swsk -> swpipe
        $seq = @(
            @('cublas', '',   'cublas'),
            @('deep',   '',   'deep'),
            @('dsk',    '--sk 3', 'dsk_sk3'),
            @('auto',   '',   'auto_v3'),
            @('swsk',   "--sk $skn", "swsk_sk$skn"),
            @('swpipe', '',   'swpipe')
        )
        foreach ($c in $seq) {
            $kern = $c[0]; $extra = @($c[1] -split ' ' | Where-Object { $_ })
            Write-Host ("[paired-v4] {0} round {1}/{2}: {3} {4}" -f $s, $r, $Rounds, $kern, ($extra -join ' '))
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
Write-Host "[paired-v4] DONE  start=$ts0  end=$ts1  out=$Out"
Write-Host "[paired-v4] next : parse (1620-regime filter for 256-1024 gates) -> compare_ar010_paired.md"
