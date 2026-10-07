// =====================================================================
// main.cu — sgemm_bench 入口：CLI 路由 / 正确性抽查 / benchmark / CSV
// ---------------------------------------------------------------
// CLI（冻结于 AR001，详设 §3；AR007 扩展 swpipe/--rounds）：
//   --kernel naive|coalesced|smem1d|tile2d|vec4|cpasync|cpasync2|swpipe|cublas|all
//   --m --n --k --warmup --iters --rounds --check --csv --bk --lb --verbose
// 输出：终端摘要 + （--csv）追加 results/performance.csv（详设 §4.3 schema）
// =====================================================================
#include "common.h"
#include "sgemm_kernels.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// 一次性分配/释放的 RAII 帮助器
struct DeviceBuffers {
    float *A = nullptr, *B = nullptr, *C = nullptr;
    int M, N, K;
    DeviceBuffers(int m, int n, int k) : M(m), N(n), K(k) {
        CUDA_CHECK(cudaMalloc(&A, (size_t)m * k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B, (size_t)k * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&C, (size_t)m * n * sizeof(float)));
    }
    void fill(unsigned seed) {
        const size_t nA = (size_t)M * K, nB = (size_t)K * N;
        std::vector<float> h(nA > nB ? nA : nB);
        sgemm::init_matrix_host(h.data(), nA, seed);
        CUDA_CHECK(cudaMemcpy(A, h.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
        sgemm::init_matrix_host(h.data(), nB, seed + 1);
        CUDA_CHECK(cudaMemcpy(B, h.data(), nB * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~DeviceBuffers() {
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
    }
};

// 正确性抽查：与 cuBLAS FP32 对比（判据详设 §4.4：rel <= 1e-4）
bool quick_check(int kernel_id, int M, int N, int K) {
    DeviceBuffers buf(M, N, K);
    buf.fill(1234u);

    // 参考：cuBLAS FP32
    std::vector<float> ref((size_t)M * N);
    sgemm_cublas(buf.A, buf.B, buf.C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(ref.data(), buf.C, ref.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // 待测 kernel
    CUDA_CHECK(cudaMemset(buf.C, 0, ref.size() * sizeof(float)));
    sgemm::kernel_fn(kernel_id)(buf.A, buf.B, buf.C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("    [CHECK] kernel launch error: %s\n", cudaGetErrorString(err));
        return false;
    }
    std::vector<float> got(ref.size());
    CUDA_CHECK(cudaMemcpy(got.data(), buf.C, got.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    double max_abs = 0.0, max_ref = 0.0;
    const size_t total = ref.size();
    for (size_t i = 0; i < total; ++i) {
        double d = std::fabs((double)got[i] - (double)ref[i]);
        if (d > max_abs) max_abs = d;
        double r = std::fabs((double)ref[i]);
        if (r > max_ref) max_ref = r;
    }
    const double rel = max_abs / (max_ref > 1e-30 ? max_ref : 1e-30);
    const bool pass = rel <= 1e-4;
    std::printf("    [CHECK] max_abs=%.6e  max_rel=%.6e  -> %s\n",
                max_abs, rel, pass ? "PASS" : "FAIL");
    return pass;
}

// 单 kernel benchmark（含预热、events 计时、CSV 落盘）
void bench_one(int kernel_id, const sgemm::CliOptions& opt) {
    const std::string name = sgemm::kernel_name(kernel_id);
    const std::string state_before = sgemm::query_gpu_state();

    DeviceBuffers buf(opt.m, opt.n, opt.k);
    buf.fill(42u);

    SgemmFn fn = sgemm::kernel_fn(kernel_id);
    float *A = buf.A, *B = buf.B, *C = buf.C;
    int M = opt.m, N = opt.n, K = opt.k;

    auto launch = [&]() { fn(A, B, C, M, N, K); };
    sgemm::TimingStats st;
    if (opt.rounds > 1) {
        // 多轮门控（AR007）：跨轮 RSD>5% 自动追加（≤3 次），聚合取轮间 median
        sgemm::MultiRoundStats mr =
            sgemm::time_kernel_rounds(launch, opt.warmup, opt.iters, opt.rounds);
        st = sgemm::merge_round_stats(mr);
        std::printf("           rounds %d | per-round medians (ms):", mr.rounds);
        for (double t : mr.round_ms) std::printf(" %.4f", t);
        std::printf(" | cross-RSD %.2f%% (max within-round %.2f%%)\n",
                    mr.cross_rsd, mr.max_within_rsd);
    } else {
        st = sgemm::time_kernel(launch, opt.warmup, opt.iters);
    }
    CUDA_CHECK(cudaGetLastError());

    const double gf = sgemm::gflops_of(opt.m, opt.n, opt.k, st.ms_median);
    const double dram_gbs =
        sgemm::theoretical_min_dram_bytes(opt.m, opt.n, opt.k) /
        (st.ms_median * 1e6);   // 等效带宽下限（真实流量以 ncu 为准）
    const std::string state_after = sgemm::query_gpu_state();

    std::printf("%-9s  %4dx%4dx%4d  median %10.4f ms  min %10.4f  max %10.4f  "
                "RSD %5.2f%%  %9.2f GFLOPS  (>=min BW %7.1f GB/s)\n",
                name.c_str(), opt.m, opt.n, opt.k,
                st.ms_median, st.ms_min, st.ms_max, st.rsd, gf, dram_gbs);
    std::printf("           gpu_state before[%s] after[%s]\n",
                state_before.c_str(), state_after.c_str());

    if (opt.csv) {
        // regs/smem 注记来自 build.log（-Xptxas -v），此处标记获取方式；
        // SGEMM_CSV 环境变量可重定向落盘路径（run_matrix.ps1 消融分流用，默认不变）
        const char* env_csv = std::getenv("SGEMM_CSV");
        sgemm::append_csv_row(env_csv ? env_csv : "results/performance.csv", name,
                              opt.m, opt.n, opt.k, st, gf,
                              "see build.log", state_after);
    }
}

}  // namespace

int main(int argc, char** argv) {
    sgemm::CliOptions opt = sgemm::parse_cli(argc, argv);
    if (opt.help) {
        sgemm::print_usage(argv[0]);
        return 0;
    }

    // 全局消融旋钮注入
    sgemm::g_verbose = opt.verbose;
    if (opt.kernel == "smem1d") sgemm::g_smem1d_bk = opt.bk;

    if (opt.kernel == "tile2d") sgemm::g_tile2d_min_blocks = opt.lb;
    // swpipe/swsk：LB 消融已收窄（AR008 实测固化 128-reg/2-block 包络，见
    // sgemm_swpipe.cu 注释），--lb 不再影响 swpipe/swsk
    if (opt.kernel == "ws") {
        // ws：--lb 双语义（tile2d/ws 共用旋钮位，详设 §4.6）+ --stages 环深度
        //      + --wp producer warp 数（T008 消融三轴）
        sgemm::g_ws_min_blocks = opt.lb;
        sgemm::g_ws_stages     = opt.stages;
        sgemm::g_ws_prod_warps = opt.wp;
    } else if (opt.stages != 3) {
        std::fprintf(stderr,
                     "[note] --stages applies to kernel 'ws' only (ignored for '%s')\n",
                     opt.kernel.c_str());
    } else if (opt.wp != 2) {
        std::fprintf(stderr,
                     "[note] --wp applies to kernel 'ws' only (ignored for '%s')\n",
                     opt.kernel.c_str());
    }
    if (opt.kernel == "swsk" || opt.kernel == "wsk" || opt.kernel == "dsk") {
        sgemm::g_swsk_slices = opt.sk;
    } else if (opt.sk != 4) {
        std::fprintf(stderr,
                      "[note] --sk applies to kernel 'swsk'/'wsk'/'dsk' only (ignored for '%s')\n",
                      opt.kernel.c_str());
    }
    // AR009：wide/wsk 的 LB 消融旋钮（--wlb，默认 2 = 100% 占用目标；
    // 与 tile2d/ws 的 --lb 分钮，避免跨 kernel 默认语义污染）
    if (opt.kernel == "wide" || opt.kernel == "wsk") {
        sgemm::g_wide_min_blocks = opt.wlb;
    } else if (opt.wlb != 2) {
        std::fprintf(stderr,
                      "[note] --wlb applies to kernel 'wide'/'wsk' only (ignored for '%s')\n",
                      opt.kernel.c_str());
    }
    // AR010：deep/dsk 的 smem 双缓冲消融旋钮（--dbuf，默认 1 = 双缓冲单同步；
    // AR010 T004 实测 dbuf1 全尺寸 +3~14%，数据裁定翻转默认值）
    if (opt.kernel == "deep" || opt.kernel == "dsk") {
        sgemm::g_deep_dbuf = opt.dbuf;
    } else if (opt.dbuf != 1) {
        std::fprintf(stderr,
                      "[note] --dbuf applies to kernel 'deep'/'dsk' only (ignored for '%s')\n",
                      opt.kernel.c_str());
    }
    // AR010：split-K 归约 ILP2 消融旋钮（--rv2，全局生效：swsk/wsk/dsk 共享
    // detail::swsk_reduce 单一数值路径，v1/v2 逐位等价）
    sgemm::g_reduce_ilp2 = opt.rv2;
    // AR011：streamk 的波数旋钮（--waves，0 = auto 公式；T001 Red 阶段 kernel
    // 未接入，绑定先行——T002 Green 接入后即生效）
    if (opt.kernel == "streamk") {
        sgemm::g_streamk_waves = opt.waves;
    } else if (opt.waves != 0) {
        std::fprintf(stderr,
                      "[note] --waves applies to kernel 'streamk' only (ignored for '%s')\n",
                      opt.kernel.c_str());
    }
    // AR011：L2 persistence 钉 C 开关（--persist，dsk/streamk 有效；--hit 为其
    // hitRatio 子旋钮）。协议纪律（design §4.2.4）：pin 在计时区内、reset 在
    // 计时区后且先于任何 cuBLAS 锚定运行。
    // T003 能力墙收口（2026-10-07 runtime 实测）：TU104 sm_75 无 L2
    // persistence（persistingL2CacheMaxSize=0 / window=0 / setLimit 报
    // "not supported on this architecture"，Ampere+ 特性）→ FR2 判 N/A，
    // 优雅降级为 [note]（不写死代码路径；负结果证据见 environment.md §8）
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        const bool l2_persist_ok = (prop.persistingL2CacheMaxSize > 0);
        if (!l2_persist_ok && opt.persist != 0) {
            std::fprintf(stderr,
                          "[note] --persist 1 ignored: L2 persistence not "
                          "supported on this GPU (persistingL2CacheMaxSize=0, "
                          "%s sm_%d%d; Ampere+ feature)\n",
                          prop.name, prop.major, prop.minor);
        } else if (opt.kernel == "dsk" || opt.kernel == "streamk") {
            sgemm::g_l2_persist = opt.persist;
            sgemm::g_l2_hit     = opt.hit;
        } else if (opt.persist != 0) {
            std::fprintf(stderr,
                          "[note] --persist applies to kernel 'dsk'/'streamk' only "
                          "(ignored for '%s')\n",
                          opt.kernel.c_str());
        } else if (opt.hit != 0.8) {
            std::fprintf(stderr,
                          "[note] --hit is effective with --persist 1 only (ignored)\n");
        }
    }
    // 环境信息头（AGENTS.md §5：每次输出附版本信息）
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    sgemm::GpuInfo info = sgemm::query_gpu_info();
    std::printf("# cuda-sgemm bench | git=%s | driver=%d.%d | runtime=%d.%d\n",
                GIT_SHA, drv / 1000, drv % 1000, rt / 1000, rt % 1000);
    std::printf("# GPU: %s | %d SMs | boost %.0f MHz | %.1f GiB | state[%s]\n",
                info.name.c_str(), info.num_sms, info.clock_khz / 1000.0,
                info.total_mem / (1024.0 * 1024.0 * 1024.0),
                sgemm::query_gpu_state().c_str());

    if (opt.list_kernels) {
        std::printf("registered kernels:\n");
        for (int i = 0; i < sgemm::KERNEL_COUNT; ++i)
            std::printf("  %2d  %s\n", i, sgemm::kernel_name(i));
        return 0;
    }

    const int kid = sgemm::kernel_id(opt.kernel);
    if (kid < 0 && opt.kernel != "all") {
        std::fprintf(stderr, "[CLI_ERROR] unknown kernel: %s (try --list-kernels)\n",
                     opt.kernel.c_str());
        return EXIT_FAILURE;
    }

    if (opt.check) {
        std::printf("== correctness check vs cuBLAS FP32 (rel tolerance 1e-4) ==\n");
        bool all_pass = true;
        if (opt.kernel == "all") {
            for (int i = 0; i < sgemm::KERNEL_COUNT; ++i) {
                std::printf("  [%s]\n", sgemm::kernel_name(i));
                all_pass &= quick_check(i, opt.m, opt.n, opt.k);
            }
        } else {
            all_pass = quick_check(kid, opt.m, opt.n, opt.k);
        }
        if (!all_pass) {
            std::printf("== CHECK FAILED ==\n");
            return EXIT_FAILURE;
        }
        std::printf("== CHECK PASSED ==\n");
    }

    std::printf("== benchmark (warmup %d, iters %d, rounds %d) ==\n",
                opt.warmup, opt.iters, opt.rounds);
    if (opt.kernel == "all") {
        for (int i = 0; i < sgemm::KERNEL_COUNT; ++i) bench_one(i, opt);
    } else {
        bench_one(kid, opt);
    }
    return 0;
}
