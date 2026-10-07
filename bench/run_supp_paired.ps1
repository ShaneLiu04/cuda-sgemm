# =====================================================================
# run_supp_paired.ps1 — AR011 T006 补充尺寸重测（逐行 47°C 冷却协议）
# ---------------------------------------------------------------
# 动因：run_auto_v4_paired.ps1 中 auto 固定在轮序列末位，K=4096 补充尺寸
# 连续 7 行重载后 GPU 降频（auto@4096x256 实测 1740/1620 MHz vs 其余行
# 1920-1935），末位行系统性偏低 → 逐行冷却重测。序列内 auto 居中
# （dsk 与 streamk 之间），防位置偏置。
# 输出：追加至 results\auto_ar011.csv（先由调用方剥离旧补充尺寸行）
# 用法：cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_supp_paired.ps1"
# =====================================================================
param(
    [int]$Rounds = 2,
    [int]$Warmup = 20,
    [int]$Iters = 50,
    [int]$CooldownTempC = 47,
    [int]$CooldownTimeoutSec = 300,
    [string]$Out = 'results\auto_ar011.csv',
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
    Write-Warning "[cooldown] TIMEOUT at ${temp}C — subsequent row marked thermal-contaminated"
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

# 序列：cublas -> dsk_sk3 -> auto（居中）-> streamk_w1 -> deep -> swsk_sk3
# -> streamk_w2；每行前 47°C 冷却。
$seq = @(
    @('cublas',   '',             'cublas'),
    @('dsk',      '--sk 3',       'dsk_sk3'),
    @('auto',     '',             'auto_v4'),
    @('streamk',  '--waves 1',    'streamk_w1'),
    @('deep',     '',             'deep'),
    @('swsk',     '--sk 3',       'swsk_sk3'),
    @('streamk',  '--waves 2',    'streamk_w2')
)
$sizes = @('256x4096x4096', '4096x256x4096', '1024x2048x2048')

$ts0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$env:SGEMM_CSV = $Out
Write-Host "[supp] rounds=$Rounds -> $Out"

Write-Host "[supp] wake spin..."
& $Bench --kernel swpipe --m 512 --n 512 --k 512 --warmup 20 --iters 100 --rounds 1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "wake spin failed" }

foreach ($s in $sizes) {
    $d = $s.Split('x'); $m = [int]$d[0]; $n = [int]$d[1]; $kk = [int]$d[2]
    for ($r = 1; $r -le $Rounds; $r++) {
        foreach ($c in $seq) {
            [void](Wait-Cooldown)
            $kern = $c[0]; $extra = @($c[1] -split ' ' | Where-Object { $_ })
            Write-Host ("[supp] {0} round {1}/{2}: {3} {4}" -f $s, $r, $Rounds, $kern, ($extra -join ' '))
            $argv = @("--kernel",$kern) + $extra +
                @("--m","$m","--n","$n","--k","$kk","--warmup","$Warmup","--iters","$Iters","--rounds","1","--csv")
            & $Bench @argv
            if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $kern @ $s" }
            if ($c[2] -ne $kern) { Rename-LastRow $kern $c[2] }
        }
    }
}

Remove-Item Env:SGEMM_CSV
$ts1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "[supp] DONE  start=$ts0  end=$ts1  out=$Out"
Write-Host "[supp] next : re-parse supplementary + fix fig33 (c) + tasks T006 回填"
