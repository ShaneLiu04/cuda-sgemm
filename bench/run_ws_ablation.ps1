# =====================================================================
# run_ws_ablation.ps1 — AR008 T008 ws 三轴消融扫描
# ---------------------------------------------------------------
# 轴：--wp {1,2} x --stages {2,3} x --lb {1,2} = 8 配置
# 尺寸：512/1024/2048/4096（issue-slot 假说主战场 = 大尺寸；512 为
#       wave 饥饿对照点）
# 行为（同 run_sk_ablation 协议）：
#   1) 每尺寸冷却门控（<= -CooldownTempC 或超时标注 thermal-contaminated）
#   2) 每尺寸 2 遍扫描（配置正序 + 逆序），同配置两行 GF 取 median——
#      线性热漂移在正/逆序配对中相消
#   3) cublas + swpipe 参考行同协议（fig 参考线：swpipe = 被挑战基线）
#   4) 行名标记：落盘后把末行 'ws' 改写为 'ws_pw<P>_st<S>_lb<L>'
# 用法（需先构建，且经 tools\env.cmd 引导 CUDA 环境）：
#   cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_ws_ablation.ps1"
# 输出：results\ablation_ar008.csv（消融分流，禁止污染主 performance.csv）
# =====================================================================
param(
    [string[]]$Sizes = @('512x512x512','1024x1024x1024','2048x2048x2048','4096x4096x4096'),
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [int]$CooldownTempC = 45,
    [int]$CooldownTimeoutSec = 300,
    [string]$Out = 'results\ablation_ar008.csv',
    [string]$Bench = 'build\sgemm_bench.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Bench)) {
    Write-Error "bench not found: $Bench (run: cmake --build build first)"
}

# 数组参数经 cmd /c 转发时会被拼成单串——统一规范化拆分
if ($Sizes.Count -eq 1 -and $Sizes[0] -match ',') {
    $Sizes = @($Sizes[0].Split(','))
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
    Write-Host "[cooldown] temp=${temp}C > ${CooldownTempC}C, waiting (timeout ${CooldownTimeoutSec}s)..."
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

# 末行改名：ws -> ws_pw<P>_st<S>_lb<L>（CSV 无旋钮列的标记方案）
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

# 8 配置（pw, stages, lb）：正序 + 逆序两遍
$cfgs = @()
foreach ($pw in 1,2) { foreach ($st in 2,3) { foreach ($lb in 1,2) {
    $cfgs += , @($pw, $st, $lb)
} } }
$cfgOrders = @(, $cfgs)
$rev = @($cfgs[($cfgs.Count - 1)..0])
$cfgOrders += , $rev

$ts0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$env:SGEMM_CSV = $Out
Write-Host "[ws-ablation] sizes=$($Sizes -join ' ') cfgs=8x2pass -> $Out"

foreach ($s in $Sizes) {
    $p = $s.Split('x'); $m = [int]$p[0]; $n = [int]$p[1]; $kk = [int]$p[2]
    $clean = Wait-Cooldown
    if (-not $clean) { Write-Warning ("[ws-ablation] {0} thermal-contaminated group" -f $s) }

    foreach ($cfgOrder in $cfgOrders) {
        foreach ($c in $cfgOrder) {
            $pw = $c[0]; $st = $c[1]; $lb = $c[2]
            Write-Host ("[ws-ablation] {0} pw={1} stages={2} lb={3}" -f $s, $pw, $st, $lb)
            & $Bench --kernel ws --wp $pw --stages $st --lb $lb --m $m --n $n --k $kk --warmup $Warmup --iters $Iters --rounds 1 --csv
            if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: ws pw=$pw st=$st lb=$lb @ $s" }
            Rename-LastRow 'ws' ("ws_pw{0}_st{1}_lb{2}" -f $pw, $st, $lb)
        }
        Write-Host ("[ws-ablation] {0} swpipe/cublas ref" -f $s)
        & $Bench --kernel swpipe --m $m --n $n --k $kk --warmup $Warmup --iters $Iters --rounds 1 --csv
        if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: swpipe @ $s" }
        & $Bench --kernel cublas --m $m --n $n --k $kk --warmup $Warmup --iters $Iters --rounds 1 --csv
        if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: cublas @ $s" }
    }
}

Remove-Item Env:SGEMM_CSV
$ts1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "[ws-ablation] DONE  start=$ts0  end=$ts1  out=$Out"
Write-Host "[ws-ablation] next : python bench\make_figures.py   (fig_ws_ablation / fig_isslot_hypothesis 从本 CSV 生成)"
