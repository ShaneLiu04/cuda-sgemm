# =====================================================================
# run_sk_ablation.ps1 — AR008 T004 swsk split-K 片数消融扫描
# ---------------------------------------------------------------
# 行为：
#   1) 每尺寸冷却门控（<= -CooldownTempC 或超时标注 thermal-contaminated）
#   2) 每尺寸 2 遍扫描（sk 升序 + 降序），同 sk 两行 GF 取 median——
#      线性热漂移在升/降序配对中相消（残余倾斜由行内 GPU state 列可查）
#   3) cublas 参考行同协议各尺寸 2 行（fig_splitk_sweep 参考线）
#   4) 行名标记：CSV 无旋钮列，落盘后把末行 'swsk' 改写为 'swsk_sk<N>'
#      （14 列 schema 不变；make_figures 按名分组取 median）
# 用法（需先构建，且经 tools\env.cmd 引导 CUDA 环境）：
#   cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_sk_ablation.ps1"
# 输出：results\ablation_ar008.csv（消融分流，禁止污染主 performance.csv）
# =====================================================================
param(
    [string[]]$Sizes = @('256x256x256','512x512x512','1024x1024x1024','1000x1016x1024','2048x2048x2048'),
    [int[]]$Sks = @(1,2,4,8,12,16),
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
foreach ($paramName in @('Sizes','Sks')) {
    $val = Get-Variable -Name $paramName -ValueOnly
    if ($val.Count -eq 1 -and $val[0] -match ',') {
        Set-Variable -Name $paramName -Value @($val[0].Split(',') | ForEach-Object { [int]$_ })
    }
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

# 末行改名：swsk -> swsk_sk<N>（CSV 无旋钮列的标记方案）
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

$ts0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$env:SGEMM_CSV = $Out
Write-Host "[sk-ablation] sizes=$($Sizes -join ' ') sks=$($Sks -join ',') -> $Out"

# 双向遍历：pass 1 升序 / pass 2 降序（线性热漂移配对相消）
$skOrders = @(, $Sks)
$rev = [int[]]($Sks | Sort-Object -Descending)
$skOrders += , $rev

foreach ($s in $Sizes) {
    $p = $s.Split('x'); $m = [int]$p[0]; $n = [int]$p[1]; $kk = [int]$p[2]
    $clean = Wait-Cooldown
    if (-not $clean) { Write-Warning ("[sk-ablation] {0} thermal-contaminated group" -f $s) }

    foreach ($skOrder in $skOrders) {
        foreach ($sk in $skOrder) {
            Write-Host ("[sk-ablation] {0} sk={1}" -f $s, $sk)
            & $Bench --kernel swsk --sk $sk --m $m --n $n --k $kk --warmup $Warmup --iters $Iters --rounds 1 --csv
            if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: swsk --sk $sk @ $s" }
            Rename-LastRow 'swsk' ("swsk_sk{0}" -f $sk)
        }
        Write-Host ("[sk-ablation] {0} cublas ref" -f $s)
        & $Bench --kernel cublas --m $m --n $n --k $kk --warmup $Warmup --iters $Iters --rounds 1 --csv
        if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: cublas @ $s" }
    }
}

Remove-Item Env:SGEMM_CSV
$ts1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "[sk-ablation] DONE  start=$ts0  end=$ts1  out=$Out"
Write-Host "[sk-ablation] next : python bench\make_figures.py   (fig_splitk_sweep 从本 CSV 生成)"
