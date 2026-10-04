# =====================================================================
# bench/run_bench.ps1 — 批量 benchmark（Windows/PowerShell）
# 用法：
#   .\bench\run_bench.ps1                 # 性能阶梯：全 kernel x 主场景 4096^3
#   .\bench\run_bench.ps1 -Mode sweep     # 阶梯 + 全测试尺寸扫描
# 输出：终端摘要 + results/performance.csv（自动落盘）
# 时钟策略提醒（AGENTS.md §5.3）：跑分前固定时钟（需管理员）：
#   nvidia-smi -lgc <freq>；另开终端 nvidia-smi dmon -s puc -d 1 监控
# =====================================================================
param([ValidateSet('ladder','sweep')][string]$Mode = 'ladder')

$bin = "build\sgemm_bench.exe"
if (-not (Test-Path $bin) -and (Test-Path "build\Release\sgemm_bench.exe")) {
    $bin = "build\Release\sgemm_bench.exe"
}
if (-not (Test-Path $bin)) { Write-Error "binary not found: $bin (run make first)"; exit 1 }

$kernels = @('naive','coalesced','smem1d','tile2d','vec4','cpasync','cpasync2','cublas')

function Invoke-Ladder {
    Write-Output "== performance ladder (4096^3, strict FP32) =="
    foreach ($kn in $kernels) {
        & $bin --kernel $kn --m 4096 --n 4096 --k 4096 --warmup 20 --iters 100 --csv
        if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $kn"; exit 1 }
    }
}

function Invoke-Sweep {
    Write-Output "== size sweep =="
    $sizes = @(
        ,(4096,4096,4096), (1024,1024,1024), (256,256,256),
        ,(1000,1016,1024), (1023,1024,511),  (8192,8192,8192))
    foreach ($s in $sizes) {
        foreach ($kn in $kernels) {
            & $bin --kernel $kn --m $s[0] --n $s[1] --k $s[2] --csv
            if ($LASTEXITCODE -ne 0) { Write-Error "bench failed: $kn @ $s"; exit 1 }
        }
    }
}

Invoke-Ladder
if ($Mode -eq 'sweep') { Invoke-Sweep }
