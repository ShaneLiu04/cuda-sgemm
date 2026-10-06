# =====================================================================
# run_auto_paired_v2.ps1 — AR009 T006 auto v2 配对验证
# ---------------------------------------------------------------
# 每尺寸背靠背 [auto v1 选择] -> [auto v2 胜者] -> [auto] x 2 轮：
#   - v1 vs v2 同会话 delta = G4' 门证据（auto v2 相对 v1 的实测提升）
#   - v2 winner vs auto delta ~0 = dispatch 正确性（同代码路径）
# v1/v2 行名改写：swsk -> v1_swsk_sk<N> / v2_swsk_sk<N>；auto -> auto_v2；
# swpipe 行（v1==v2 同路径）保留原名。
# 输出：results\auto_ar009.csv
# =====================================================================
$ErrorActionPreference = 'Stop'
$Bench = 'build\sgemm_bench.exe'
$Out = 'results\auto_ar009.csv'
$env:SGEMM_CSV = $Out

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

# 尺寸 -> v1 选择 / v2 胜者（AR008 T004/T005 vs AR009 T004/T005+boost彩票
# 实测；稳态=1620MHz 持续（AGENTS §5 无权限跑至稳态）；boost 态行(1935/1950)
# 为瞬态彩票，仅作归一旁证不进 dispatch）
# 2048³/4096³：v1 与 v2 同选 swpipe，单行代表两者
$plan = @(
    @{ s = '256x256x256';    v1 = @('swsk', '--sk 12', 'v1_swsk_sk12');
                             v2 = @('swsk', '--sk 6',  'v2_swsk_sk6') },
    @{ s = '512x512x512';    v1 = @('swsk', '--sk 4',  'v1_swsk_sk4');
                             v2 = @('swsk', '--sk 3',  'v2_swsk_sk3') },
    @{ s = '1024x1024x1024'; v1 = @('swsk', '--sk 4',  'v1_swsk_sk4');
                             v2 = @('swsk', '--sk 3',  'v2_swsk_sk3') },
    @{ s = '1000x1016x1024'; v1 = @('swsk', '--sk 4',  'v1_swsk_sk4');
                             v2 = @('swsk', '--sk 3',  'v2_swsk_sk3') },
    @{ s = '2048x2048x2048'; v1 = @('swpipe', '',      'swpipe');
                             v2 = @('swpipe', '',      'swpipe') },
    @{ s = '4096x4096x4096'; v1 = @('swpipe', '',      'swpipe');
                             v2 = @('swpipe', '',      'swpipe') }
)

foreach ($p in $plan) {
    $d = $p.s.Split('x')
    for ($r = 1; $r -le 2; $r++) {
        foreach ($side in @('v1', 'v2')) {
            $c = $p.$side
            Write-Host ("[auto-paired-v2] {0} round {1}: {2} {3}" -f $p.s, $r, $c[0], $c[1])
            & $Bench --kernel $c[0] --m $d[0] --n $d[1] --k $d[2] --warmup 20 --iters 100 --rounds 1 --csv @(($c[1] -split ' ') | Where-Object { $_ })
            if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $($c[0]) @ $($p.s)" }
            if ($c[2] -ne $c[0]) { Rename-LastRow $c[0] $c[2] }
        }
        Write-Host ("[auto-paired-v2] {0} round {1}: auto" -f $p.s, $r)
        & $Bench --kernel auto --m $d[0] --n $d[1] --k $d[2] --warmup 20 --iters 100 --rounds 1 --csv
        if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: auto @ $($p.s)" }
        Rename-LastRow 'auto' 'auto_v2'
    }
}

Remove-Item Env:SGEMM_CSV
Write-Host "[auto-paired-v2] DONE -> $Out"
