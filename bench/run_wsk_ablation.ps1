# =====================================================================
# run_wsk_ablation.ps1 — AR009 T004 精确波消融扫描（wsk + swsk 同会话）
# ---------------------------------------------------------------
# 与 AR008 run_sk_ablation.ps1 的差异：
#   1) 双 kernel 扫描：wsk 与 swsk 在同一会话/同一钟态下成对扫描——
#      跨会话 swsk 不可比（本日实测 swsk@256^3 跨会话差 2.2x，钟态双峰），
#      fig17 的 delta 曲线必须同会话闭环
#   2) wsk 走 --sk（split-K 片数，经 wide tile 网格），swsk 走 --sk（swpipe tile）
#   3) 冷却门控 + 每尺寸 2 遍（sk 升序 + 降序），线性热漂移配对相消
#   4) 行名标记：wsk -> wsk_sk<N> / swsk -> swsk_sk<N>（CSV 无旋钮列）
# 用法：
#   cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_wsk_ablation.ps1"
#   （可加 -Sizes 256x256x256 -WskSks 1,2,4 -SwskSks 1,2,4 分尺寸分段执行）
# 输出：results\ablation_ar009.csv（消融分流，禁止污染主 performance.csv）
# =====================================================================
param(
    [string[]]$Sizes = @('256x256x256','512x512x512','1024x1024x1024','1000x1016x1024','2048x2048x2048'),
    [int[]]$WskSks = @(1,2,3,4,6,8,12,16),
    [int[]]$SwskSks = @(1,2,3,4,6,8,12,16),
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [int]$CooldownTempC = 45,
    [int]$CooldownTimeoutSec = 300,
    [string]$Out = 'results\ablation_ar009.csv',
    [string]$Bench = 'build\sgemm_bench.exe'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Bench)) {
    Write-Error "bench not found: $Bench (run: cmake --build build first)"
}

# 数组参数经 cmd /c 转发时会被拼成单串——统一规范化拆分
foreach ($paramName in @('Sizes','WskSks','SwskSks')) {
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

# 末行改名：<old> -> <new>（CSV 无旋钮列的标记方案）
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
Write-Host "[wsk-ablation] sizes=$($Sizes -join ' ') wsk_sks=$($WskSks -join ',') swsk_sks=$($SwskSks -join ',') -> $Out"

# 双向遍历：pass 1 升序 / pass 2 降序（线性热漂移配对相消）
$wskOrders = @(, [int[]]($WskSks | Sort-Object))
$wskOrders += , [int[]]($WskSks | Sort-Object -Descending)
$swskOrders = @(, [int[]]($SwskSks | Sort-Object))
$swskOrders += , [int[]]($SwskSks | Sort-Object -Descending)

foreach ($s in $Sizes) {
    $p = $s.Split('x'); $m = [int]$p[0]; $n = [int]$p[1]; $kk = [int]$p[2]
    $clean = Wait-Cooldown
    if (-not $clean) { Write-Warning ("[wsk-ablation] {0} thermal-contaminated group" -f $s) }

    foreach ($pass in 0..1) {
        foreach ($sk in $wskOrders[$pass]) {
            Write-Host ("[wsk-ablation] {0} pass{1} wsk sk={2}" -f $s, ($pass+1), $sk)
            & $Bench --kernel wsk --sk $sk --m $m --n $n --k $kk --warmup $Warmup --iters $Iters --rounds 1 --csv
            if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: wsk --sk $sk @ $s" }
            Rename-LastRow 'wsk' ("wsk_sk{0}" -f $sk)
        }
        foreach ($sk in $swskOrders[$pass]) {
            Write-Host ("[wsk-ablation] {0} pass{1} swsk sk={2}" -f $s, ($pass+1), $sk)
            & $Bench --kernel swsk --sk $sk --m $m --n $n --k $kk --warmup $Warmup --iters $Iters --rounds 1 --csv
            if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: swsk --sk $sk @ $s" }
            Rename-LastRow 'swsk' ("swsk_sk{0}" -f $sk)
        }
        Write-Host ("[wsk-ablation] {0} pass{1} cublas ref" -f $s, ($pass+1))
        & $Bench --kernel cublas --m $m --n $n --k $kk --warmup $Warmup --iters $Iters --rounds 1 --csv
        if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: cublas @ $s" }
    }
}

Remove-Item Env:SGEMM_CSV
$ts1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "[wsk-ablation] DONE  start=$ts0  end=$ts1  out=$Out"
Write-Host "[wsk-ablation] next : python bench\make_figures.py   (fig17_prewave_sweep 从本 CSV 生成)"
