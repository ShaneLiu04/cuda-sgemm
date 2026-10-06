# =====================================================================
# run_deep_ablation.ps1 — AR010 T004 deep/dsk 消融（LDS 墙假说正式检验）
# ---------------------------------------------------------------
# 目的：①deep/dsk DBUF x sk x 尺寸全扫（G1@1024³ 主攻 + auto v3 输入）
#       ②五 kernel 家族同会话同钟态对照（swpipe/swsk/wide/wsk/deep/dsk
#         + cublas 门参考）——fig23 acc-per-LDS→%peak 经验律主图数据
# 纪律：GPU 唤醒 spin + 冷却门 47C（尺寸组间）+ 双 pass 正反序（组内
#       线性热漂移配对相消）+ 行级 gpu_state（逐行核钟态）
# 行名改写：deep->deep_dbuf<N> / dsk->dsk_sk<S>_dbuf<M> / swsk->swsk_sk<S>
# 用法：
#   cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_deep_ablation.ps1"
# 输出：results\deep_ar010.csv（消融分流，禁止污染主 performance.csv）
# =====================================================================
param(
    [int]$Warmup = 20,
    [int]$CooldownTempC = 47,
    [int]$CooldownTimeoutSec = 300,
    [string]$Out = 'results\deep_ar010.csv',
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

# 配置元组：size -> @( kernel, extra_args, iters, row_name )
# 家族覆盖：swpipe(16 acc/LDS) / swsk(16) / wide(10.7) / wsk(10.7) /
#           deep-dsk(21.3) / cublas(商业实现参考点)
$plan = @(
    @{ s = '256x256x256'; iters = 100; cfgs = @(
        ,@('swpipe', '', 'swpipe', '')
        ,@('swsk', '--sk 6', 'swsk', '_sk6')
        ,@('cublas', '', 'cublas', '')
        ,@('wide', '', 'wide', '')
        ,@('deep', '', 'deep', '_dbuf0')
        ,@('deep', '--dbuf 1', 'deep', '_dbuf1')
        ,@('dsk', '--sk 2 --dbuf 1', 'dsk', '_sk2_dbuf1')
        ,@('dsk', '--sk 4 --dbuf 1', 'dsk', '_sk4_dbuf1')
        ,@('dsk', '--sk 8 --dbuf 1', 'dsk', '_sk8_dbuf1')
        ,@('dsk', '--sk 12 --dbuf 1', 'dsk', '_sk12_dbuf1')
        ,@('dsk', '--sk 16 --dbuf 1', 'dsk', '_sk16_dbuf1')
        ,@('dsk', '--sk 8', 'dsk', '_sk8_dbuf0') ) },
    @{ s = '512x512x512'; iters = 100; cfgs = @(
        ,@('swpipe', '', 'swpipe', '')
        ,@('swsk', '--sk 3', 'swsk', '_sk3')
        ,@('cublas', '', 'cublas', '')
        ,@('wide', '', 'wide', '')
        ,@('deep', '', 'deep', '_dbuf0')
        ,@('deep', '--dbuf 1', 'deep', '_dbuf1')
        ,@('dsk', '--sk 3 --dbuf 1', 'dsk', '_sk3_dbuf1')
        ,@('dsk', '--sk 4 --dbuf 1', 'dsk', '_sk4_dbuf1')
        ,@('dsk', '--sk 6 --dbuf 1', 'dsk', '_sk6_dbuf1')
        ,@('dsk', '--sk 8 --dbuf 1', 'dsk', '_sk8_dbuf1')
        ,@('dsk', '--sk 12 --dbuf 1', 'dsk', '_sk12_dbuf1')
        ,@('dsk', '--sk 6', 'dsk', '_sk6_dbuf0') ) },
    @{ s = '1024x1024x1024'; iters = 100; cfgs = @(
        ,@('swpipe', '', 'swpipe', '')
        ,@('swsk', '--sk 3', 'swsk', '_sk3')
        ,@('cublas', '', 'cublas', '')
        ,@('wide', '', 'wide', '')
        ,@('wsk', '--sk 3', 'wsk', '_sk3')
        ,@('deep', '', 'deep', '_dbuf0')
        ,@('deep', '--dbuf 1', 'deep', '_dbuf1')
        ,@('dsk', '--sk 2 --dbuf 1', 'dsk', '_sk2_dbuf1')
        ,@('dsk', '--sk 3 --dbuf 1', 'dsk', '_sk3_dbuf1')
        ,@('dsk', '--sk 4 --dbuf 1', 'dsk', '_sk4_dbuf1')
        ,@('dsk', '--sk 6 --dbuf 1', 'dsk', '_sk6_dbuf1')
        ,@('dsk', '--sk 8 --dbuf 1', 'dsk', '_sk8_dbuf1')
        ,@('dsk', '--sk 3', 'dsk', '_sk3_dbuf0') ) },
    @{ s = '1000x1016x1024'; iters = 100; cfgs = @(
        ,@('swpipe', '', 'swpipe', '')
        ,@('swsk', '--sk 3', 'swsk', '_sk3')
        ,@('cublas', '', 'cublas', '')
        ,@('deep', '--dbuf 1', 'deep', '_dbuf1')
        ,@('dsk', '--sk 3 --dbuf 1', 'dsk', '_sk3_dbuf1')
        ,@('dsk', '--sk 4 --dbuf 1', 'dsk', '_sk4_dbuf1') ) },
    @{ s = '2048x2048x2048'; iters = 100; cfgs = @(
        ,@('swpipe', '', 'swpipe', '')
        ,@('cublas', '', 'cublas', '')
        ,@('wide', '', 'wide', '')
        ,@('deep', '', 'deep', '_dbuf0')
        ,@('deep', '--dbuf 1', 'deep', '_dbuf1')
        ,@('dsk', '--sk 2 --dbuf 1', 'dsk', '_sk2_dbuf1') ) },
    @{ s = '4096x4096x4096'; iters = 50; cfgs = @(
        ,@('swpipe', '', 'swpipe', '')
        ,@('cublas', '', 'cublas', '')
        ,@('deep', '', 'deep', '_dbuf0')
        ,@('deep', '--dbuf 1', 'deep', '_dbuf1') ) }
)

# GPU 唤醒 spin（AR009 T006 教训：空闲降频后首轮 0.44x 伪影）
Write-Host "[deep-ablation] GPU wake spin..."
& $Bench --kernel deep --m 512 --n 512 --k 512 --warmup 20 --iters 30 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "wake spin failed" }

$ts0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$env:SGEMM_CSV = $Out
Write-Host "[deep-ablation] -> $Out"

foreach ($group in $plan) {
    $s = $group.s
    $it = $group.iters
    $p = $s.Split('x'); $m = [int]$p[0]; $n = [int]$p[1]; $kk = [int]$p[2]
    $clean = Wait-Cooldown
    if (-not $clean) { Write-Warning ("[deep-ablation] {0} thermal-contaminated group" -f $s) }
    # 双 pass：正序 + 反序（线性热漂移配对相消）
    $orders = @(, $group.cfgs)
    $orders += , (@($group.cfgs)[(@($group.cfgs).Count - 1)..0])
    foreach ($pass in 0..1) {
        foreach ($c in $orders[$pass]) {
            $kern = $c[0]; $extra = $c[1]; $pref = $c[2]; $suf = $c[3]
            Write-Host ("[deep-ablation] {0} pass{1} {2} {3}" -f $s, ($pass + 1), $kern, $extra)
            $args = "--kernel $kern $extra --m $m --n $n --k $kk --warmup $Warmup --iters $it --rounds 1 --csv"
            & $Bench @($args -split ' ')
            if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $kern $extra @ $s" }
            if ($suf -ne '') { Rename-LastRow $pref ($pref + $suf) }
        }
    }
}

Remove-Item Env:SGEMM_CSV
$ts1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "[deep-ablation] DONE  start=$ts0  end=$ts1  out=$Out"
Write-Host "[deep-ablation] next : python bench\make_figures.py   (fig23/fig24 从本 CSV 生成)"
