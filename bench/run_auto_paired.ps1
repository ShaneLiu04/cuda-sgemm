# =====================================================================
# run_auto_paired.ps1 — AR008 T006 配对验证：auto 选核 vs 尺寸实测最优
# 每尺寸背靠背 [winner -> auto] x 2 轮（同钟态对内 delta；auto 与 winner
# 是同一代码路径，delta 应 ~0，非 0 即 dispatch 开销或选错核）
# 输出：追加 results\auto_ar008.csv
# =====================================================================
$ErrorActionPreference = 'Stop'
$Bench = 'build\sgemm_bench.exe'
$Out = 'results\auto_ar008.csv'
$env:SGEMM_CSV = $Out

# 尺寸 -> (winner kernel, winner 额外参数)：T004/T005 扫描实测最优
$plan = @(
    @{ s = '256x256x256';      k = 'swsk';   extra = @('--sk','12') },
    @{ s = '512x512x512';      k = 'swsk';   extra = @('--sk','4')  },
    @{ s = '1024x1024x1024';   k = 'swsk';   extra = @('--sk','4')  },
    @{ s = '1000x1016x1024';   k = 'swsk';   extra = @('--sk','4')  },
    @{ s = '2048x2048x2048';   k = 'swpipe'; extra = @()            },
    @{ s = '4096x4096x4096';   k = 'swpipe'; extra = @()            }
)

foreach ($p in $plan) {
    $d = $p.s.Split('x')
    for ($r = 1; $r -le 2; $r++) {
        Write-Host ("[auto-paired] {0} round {1}: {2} -> auto" -f $p.s, $r, $p.k)
        & $Bench --kernel $p.k --m $d[0] --n $d[1] --k $d[2] --warmup 20 --iters 100 --rounds 1 --csv @($p.extra)
        if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $($p.k) @ $($p.s)" }
        & $Bench --kernel auto --m $d[0] --n $d[1] --k $d[2] --warmup 20 --iters 100 --rounds 1 --csv
        if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: auto @ $($p.s)" }
    }
}

Remove-Item Env:SGEMM_CSV
Write-Host "[auto-paired] DONE -> $Out"
