# =====================================================================
# run_wlb_ablation.ps1 — AR009 T005 wide/wsk LB(占用率)消融
# ---------------------------------------------------------------
# 目的：形式收尾占用率假说 —— LB=2(64regs, 2blk/SM=100% warp slots) vs
#       LB=1(79regs, 1blk/SM=50%)，同会话同钟态配对；swpipe 参照行同协议。
# 覆盖：
#   wide x wlb{1,2}          x {512^3, 1024^3, 2048^3}
#   wsk  x wlb{1,2} x sk{3,12} x {256^3, 512^3, 1024^3}
#   swpipe（参照）            x {512^3, 1024^3, 2048^3}
# 每配置 2 pass（正/反序），行名改写：wide->wide_wlb<N> / wsk->wsk_sk<S>_wlb<N>
# 用法：
#   cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_wlb_ablation.ps1"
# 输出：results\ablation_wlb_ar009.csv（消融分流，禁止污染主 performance.csv）
# =====================================================================
param(
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [int]$CooldownTempC = 47,
    [int]$CooldownTimeoutSec = 300,
    [string]$Out = 'results\ablation_wlb_ar009.csv',
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

# 配置元组：size -> @( kernel, extra_args, row_prefix, row_suffix )
$plan = @(
    @{ s = '512x512x512';  cfgs = @(
        ,@('wide',   '--wlb 1', 'wide', '_wlb1')
        ,@('wide',   '--wlb 2', 'wide', '_wlb2')
        ,@('wsk', '--sk 3 --wlb 1', 'wsk', '_sk3_wlb1')
        ,@('wsk', '--sk 3 --wlb 2', 'wsk', '_sk3_wlb2')
        ,@('wsk', '--sk 12 --wlb 1', 'wsk', '_sk12_wlb1')
        ,@('wsk', '--sk 12 --wlb 2', 'wsk', '_sk12_wlb2')
        ,@('swpipe', '', 'swpipe', '') ) },
    @{ s = '1024x1024x1024'; cfgs = @(
        ,@('wide',   '--wlb 1', 'wide', '_wlb1')
        ,@('wide',   '--wlb 2', 'wide', '_wlb2')
        ,@('wsk', '--sk 3 --wlb 1', 'wsk', '_sk3_wlb1')
        ,@('wsk', '--sk 3 --wlb 2', 'wsk', '_sk3_wlb2')
        ,@('wsk', '--sk 12 --wlb 1', 'wsk', '_sk12_wlb1')
        ,@('wsk', '--sk 12 --wlb 2', 'wsk', '_sk12_wlb2')
        ,@('swpipe', '', 'swpipe', '') ) },
    @{ s = '2048x2048x2048'; cfgs = @(
        ,@('wide',   '--wlb 1', 'wide', '_wlb1')
        ,@('wide',   '--wlb 2', 'wide', '_wlb2')
        ,@('swpipe', '', 'swpipe', '') ) },
    @{ s = '256x256x256';  cfgs = @(
        ,@('wsk', '--sk 3 --wlb 1', 'wsk', '_sk3_wlb1')
        ,@('wsk', '--sk 3 --wlb 2', 'wsk', '_sk3_wlb2')
        ,@('wsk', '--sk 12 --wlb 1', 'wsk', '_sk12_wlb1')
        ,@('wsk', '--sk 12 --wlb 2', 'wsk', '_sk12_wlb2') ) }
)

$ts0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$env:SGEMM_CSV = $Out
Write-Host "[wlb-ablation] -> $Out"

foreach ($group in $plan) {
    $s = $group.s
    $p = $s.Split('x'); $m = [int]$p[0]; $n = [int]$p[1]; $kk = [int]$p[2]
    $clean = Wait-Cooldown
    if (-not $clean) { Write-Warning ("[wlb-ablation] {0} thermal-contaminated group" -f $s) }
    # 双 pass：正序 + 反序（线性热漂移配对相消）
    $orders = @(, $group.cfgs)
    $orders += , (@($group.cfgs)[(@($group.cfgs).Count - 1)..0])
    foreach ($pass in 0..1) {
        foreach ($c in $orders[$pass]) {
            $kern = $c[0]; $extra = $c[1]; $pref = $c[2]; $suf = $c[3]
            Write-Host ("[wlb-ablation] {0} pass{1} {2} {3}" -f $s, ($pass + 1), $kern, $extra)
            $args = "--kernel $kern $extra --m $m --n $n --k $kk --warmup $Warmup --iters $Iters --rounds 1 --csv"
            & $Bench @($args -split ' ')
            if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $kern $extra @ $s" }
            if ($suf -ne '') { Rename-LastRow $pref ($pref + $suf) }
        }
    }
}

Remove-Item Env:SGEMM_CSV
$ts1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "[wlb-ablation] DONE  start=$ts0  end=$ts1  out=$Out"
Write-Host "[wlb-ablation] next : python bench\make_figures.py   (fig18 从本 CSV 生成)"
