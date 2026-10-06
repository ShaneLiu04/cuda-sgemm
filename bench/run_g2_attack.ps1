# =====================================================================
# run_g2_attack.ps1 — AR010 T005 G2@256³ 攻坚（绝对门 >= 1618.2 GF）
# ---------------------------------------------------------------
# 假说：①swsk sk6 切片失衡（32 k-tiles = 6,6,6,6,6,2）——均衡 sk4(8x)/sk8(4x)
#         或深 sk5(7,7,7,7,4) 更优；②split-K 归约占 dsk@256³ 16-17%——
#         rv2 归约 ILP2 直接补口；③smem1d 历史冠军 1618.2（AR007）——
#         32x32 小 tile = 64 blocks 波填充优势，同会话中继复现；
# 纪律：GPU 唤醒 spin + 47C 冷却门 + 双 pass 正反序 + 行级 gpu_state
# 用法：
#   cmd /c "call tools\env.cmd && powershell -ExecutionPolicy Bypass -File bench\run_g2_attack.ps1"
# 输出：results\g2_ar010.csv
# =====================================================================
param(
    [int]$Warmup = 20,
    [int]$Iters = 100,
    [int]$CooldownTempC = 47,
    [int]$CooldownTimeoutSec = 300,
    [string]$Out = 'results\g2_ar010.csv',
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
    Write-Warning "[cooldown] TIMEOUT at ${temp}C"
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

# 参考行 + swsk 细 sk 扫描（rv1/rv2）+ dsk 探针 + smem1d/tile2d 中继
$cfgs = @(
    ,@('cublas',  '',                    'cublas',   '')
    ,@('swpipe',  '',                    'swpipe',   '')
    ,@('smem1d',  '',                    'smem1d',   '')
    ,@('tile2d',  '',                    'tile2d',   '')
    ,@('deep',    '--dbuf 1',            'deep',     '_dbuf1')
    ,@('swsk',    '--sk 3',              'swsk',     '_sk3')
    ,@('swsk',    '--sk 4',              'swsk',     '_sk4')
    ,@('swsk',    '--sk 5',              'swsk',     '_sk5')
    ,@('swsk',    '--sk 6',              'swsk',     '_sk6')
    ,@('swsk',    '--sk 7',              'swsk',     '_sk7')
    ,@('swsk',    '--sk 8',              'swsk',     '_sk8')
    ,@('swsk',    '--sk 4 --rv2 1',      'swsk',     '_sk4_rv2')
    ,@('swsk',    '--sk 5 --rv2 1',      'swsk',     '_sk5_rv2')
    ,@('swsk',    '--sk 6 --rv2 1',      'swsk',     '_sk6_rv2')
    ,@('swsk',    '--sk 8 --rv2 1',      'swsk',     '_sk8_rv2')
    ,@('dsk',     '--sk 6 --dbuf 1',     'dsk',      '_sk6_dbuf1')
    ,@('dsk',     '--sk 8 --dbuf 1',     'dsk',      '_sk8_dbuf1')
    ,@('dsk',     '--sk 12 --dbuf 1',    'dsk',      '_sk12_dbuf1')
    ,@('dsk',     '--sk 16 --dbuf 1',    'dsk',      '_sk16_dbuf1')
    ,@('dsk',     '--sk 12 --dbuf 1 --rv2 1', 'dsk', '_sk12_dbuf1_rv2')
    ,@('dsk',     '--sk 16 --dbuf 1 --rv2 1', 'dsk', '_sk16_dbuf1_rv2')
)

# GPU 唤醒 spin
Write-Host "[g2-attack] GPU wake spin..."
& $Bench --kernel deep --m 512 --n 512 --k 512 --warmup 20 --iters 30 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "wake spin failed" }

$ts0 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$env:SGEMM_CSV = $Out
Write-Host "[g2-attack] -> $Out  (256x256x256, gate >= 1618.2 GF)"

$clean = Wait-Cooldown
if (-not $clean) { Write-Warning "[g2-attack] thermal-contaminated session" }

# 双 pass 正反序
$orders = @(, $cfgs)
$orders += , (@($cfgs)[(@($cfgs).Count - 1)..0])
foreach ($pass in 0..1) {
    foreach ($c in $orders[$pass]) {
        $kern = $c[0]; $extra = $c[1]; $pref = $c[2]; $suf = $c[3]
        Write-Host ("[g2-attack] pass{0} {1} {2}" -f ($pass + 1), $kern, $extra)
        $argl = "--kernel $kern $extra --m 256 --n 256 --k 256 --warmup $Warmup --iters $Iters --rounds 1 --csv"
        & $Bench @($argl -split ' ')
        if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $kern $extra" }
        if ($suf -ne '') { Rename-LastRow $pref ($pref + $suf) }
    }
}

Remove-Item Env:SGEMM_CSV
$ts1 = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "[g2-attack] DONE  start=$ts0  end=$ts1  out=$Out"
Write-Host "[g2-attack] next : 分解探针（--verbose dsk）+ fig25_g2_attack"
