# =====================================================================
# run_paired_v5.ps1 — AR011 T008 门 v5 终判配对协议
# ---------------------------------------------------------------
# 与 AR010 run_paired_v4.ps1 差异：
#   1) kernel 集 + streamk W=1（AR011 新家族，G6 主候选）
#   2) cublas 首末双锚（每轮首行 + 末行；轮内漂移可检测）
#   3) auto -> auto_v4（AR011 dispatch）
#   4) 补充尺寸（256x4096/4096x256/1024x2048）纳入：in-sequence warm
#      序列 + auto 居中（T006 §9 时钟制度教训：末位行降频伪影防范）
#   5) 追加前删旧文件（v5 纪律）
#   6) 解析期钟态过滤沿 v4：256³-1024³ 门用 1620 稳态轮，boost 混染
#      轮丢弃；2048³/4096³ 轮内共模钟态相消；跨钟态绝对比较 %peak 归一
# 输出：results\paired_ar011.csv
# 用法：cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_paired_v5.ps1"
# =====================================================================
param(
    [int]$Rounds = 3,
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [int]$CooldownTempC = 47,
    [int]$CooldownTimeoutSec = 300,
    [string]$Out = 'results\paired_ar011.csv',
    [string]$Bench = 'build\sgemm_bench.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Bench)) { Write-Error "bench not found: $Bench (run: cmake --build build first)" }
if (Test-Path $Out) { Remove-Item $Out; Write-Host "[paired-v5] stale $Out removed (v5 discipline)" }

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

function Run-Row([string]$kern, [string[]]$extra, [int]$m, [int]$n, [int]$kk, [int]$it, [string]$rowname) {
    Write-Host ("[paired-v5] {0}x{1}x{2}: {3} {4}" -f $m, $n, $kk, $kern, ($extra -join ' '))
    $argv = @("--kernel",$kern) + $extra +
        @("--m","$m","--n","$n","--k","$kk","--warmup","$Warmup","--iters","$it","--rounds","1","--csv")
    & $Bench @argv
    if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $kern @ ${m}x${n}x${kk}" }
    if ($rowname -ne $kern) { Rename-LastRow $kern $rowname }
}

# canonical 六尺寸：swsk 片数沿用 v4 守擂配置
$swsk_sk = @{ '256x256x256' = 6; '512x512x512' = 3; '1024x1024x1024' = 3;
              '1000x1016x1024' = 3; '2048x2048x2048' = 3; '4096x4096x4096' = 3 }
$sizes = @('256x256x256', '512x512x512', '1024x1024x1024',
           '1000x1016x1024', '2048x2048x2048', '4096x4096x4096')
# 补充三尺寸（in-sequence warm，auto 居中）
$supp = @('256x4096x4096', '4096x256x4096', '1024x2048x2048')

$ts0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$env:SGEMM_CSV = $Out
Write-Host "[paired-v5] rounds=$Rounds -> $Out"

Write-Host "[paired-v5] wake spin..."
& $Bench --kernel swpipe --m 512 --n 512 --k 512 --warmup 20 --iters 100 --rounds 1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "wake spin failed" }

foreach ($s in $sizes) {
    $p = $s.Split('x'); $m = [int]$p[0]; $n = [int]$p[1]; $kk = [int]$p[2]
    $skn = $swsk_sk[$s]
    $it = if ($kk -ge 4096) { 50 } else { $Iters }
    [void](Wait-Cooldown)

    for ($r = 1; $r -le $Rounds; $r++) {
        # 每轮序列：cublas 首锚 -> deep -> dsk -> streamk W=1 -> auto -> swsk -> swpipe -> cublas 末锚
        Run-Row 'cublas'   @() $m $n $kk $it 'cublas'
        Run-Row 'deep'     @() $m $n $kk $it 'deep'
        Run-Row 'dsk'      @('--sk','3') $m $n $kk $it 'dsk_sk3'
        Run-Row 'streamk'  @('--waves','1') $m $n $kk $it 'streamk_w1'
        Run-Row 'auto'     @() $m $n $kk $it 'auto_v4'
        Run-Row 'swsk'     @('--sk',"$skn") $m $n $kk $it "swsk_sk$skn"
        Run-Row 'swpipe'   @() $m $n $kk $it 'swpipe'
        Run-Row 'cublas'   @() $m $n $kk $it 'cublas_end'
    }
}

foreach ($s in $supp) {
    $p = $s.Split('x'); $m = [int]$p[0]; $n = [int]$p[1]; $kk = [int]$p[2]
    [void](Wait-Cooldown)
    for ($r = 1; $r -le $Rounds; $r++) {
        # in-seq warm + auto 居中（T006 §9）：cublas -> dsk -> auto -> strk -> deep -> swsk -> cublas_end
        Run-Row 'cublas'   @() $m $n $kk 50 'cublas'
        Run-Row 'dsk'      @('--sk','3') $m $n $kk 50 'dsk_sk3'
        Run-Row 'auto'     @() $m $n $kk 50 'auto_v4'
        Run-Row 'streamk'  @('--waves','1') $m $n $kk 50 'streamk_w1'
        Run-Row 'deep'     @() $m $n $kk 50 'deep'
        Run-Row 'swsk'     @('--sk','3') $m $n $kk 50 'swsk_sk3'
        Run-Row 'cublas'   @() $m $n $kk 50 'cublas_end'
    }
}

Remove-Item Env:SGEMM_CSV
$ts1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "[paired-v5] DONE  start=$ts0  end=$ts1  out=$Out"
Write-Host "[paired-v5] next : parse (1620-regime filter) -> compare_ar011_paired.md + fig34"
