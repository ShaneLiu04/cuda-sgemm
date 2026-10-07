// =====================================================================
// common.h — cuda-sgemm 公共基础设施
// 职责：CUDA_CHECK / CUBLAS_CHECK、CLI 解析、CUDA event 计时、
//       GFLOPS 计算、CSV 落盘、GPU 元数据与运行状态查询。
// 契约来源：specs/component-detail-design/cuda_sgemm_spec.md §4.3
// =====================================================================
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

// ---------------------------------------------------------------------
// 错误检查宏：所有 CUDA / cuBLAS API 必须包裹（AGENTS.md §4.1）
// ---------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _e = (expr);                                             \
        if (_e != cudaSuccess) {                                             \
            std::fprintf(stderr, "[CUDA_ERROR] %s:%d  %s  ->  %s\n",         \
                         __FILE__, __LINE__, #expr, cudaGetErrorString(_e)); \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(expr)                                                   \
    do {                                                                     \
        cublasStatus_t _s = (expr);                                          \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "[CUBLAS_ERROR] %s:%d  %s  ->  code %d\n",  \
                         __FILE__, __LINE__, #expr, (int)_s);                \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// Makefile / CMake 注入的 git 提交号（数据可复现性，AGENTS.md §5.5）
#ifndef GIT_SHA
#define GIT_SHA "unknown"
#endif

namespace sgemm {

// ---------------------------------------------------------------------
// CLI 选项（参数表冻结于 AR001，详设 §3）
// ---------------------------------------------------------------------
struct CliOptions {
    std::string kernel = "naive";   // naive|...|wide|wsk|deep|dsk|all
    int m = 4096;
    int n = 4096;
    int k = 4096;
    int warmup = 20;
    int iters = 100;
    bool check = false;             // 正确性检查（对照 cuBLAS FP32）
    bool csv = false;               // 结果追加写入 results/performance.csv
    bool verbose = false;           // 打印回退路径等细节
    int bk = 32;                    // smem1d 的 BK 消融旋钮（8/16/32；默认 32，AR007 实测固化）
    int lb = 1;                     // tile2d 的 __launch_bounds__ minBlocks 消融旋钮（1/2；
                                    //  swpipe/swsk 已固化 128-reg 包络，AR008 收窄语义）
    int sk = 4;                     // swsk/wsk/dsk 的 split-K 片数消融旋钮（1..16；默认 4，详设 §4.6）
    int stages = 3;                 // ws 的 smem 环深度消融旋钮（2/3；默认 3，AR008 详设 §4.6）
    int wp = 2;                     // ws 的 producer warp 数消融旋钮（1/2；默认 2，AR008 详设 §4.6）
    int wlb = 2;                    // wide/wsk 的 __launch_bounds__ minBlocks 消融旋钮（1/2；
                                    //  默认 2 = 64 regs/100% 占用目标，AR009 详设 §4.6）
    int dbuf = 1;                   // deep/dsk 的 smem 双缓冲消融旋钮（0/1；默认 1 = 双缓冲
                                    //  单同步，AR010 T004 实测全尺寸 +3~14% 翻转；0 = 单缓冲
                                    //  双同步，AR010 详设 §4.6）
    int rv2 = 0;                    // swsk/wsk/dsk 归约 ILP2 消融旋钮（0/1；默认 0 = 1 f4/线程
                                     //  v1，1 = 2 f4/线程 v2 逐位等价，AR010 详设 §4.6）
    int waves = 0;                  // streamk 的波数 W 消融旋钮（0..8；默认 0 = auto 公式
                                     //  clamp(floor(TOT/(48·16)),1,8)，AR011 详设 §4.6）
    int persist = 0;                // dsk/streamk 的 L2 persistence 钉 C 开关（0/1；默认 0，
                                     //  AR011 详设 §4.6；1 = 计时区内钉 C + 区后强制复位）
    double hit = 0.8;               // --persist 的 accessPolicyWindow hitRatio（0.5..1.0；
                                     //  默认 0.8，AR011 详设 §4.6）
    int rounds = 1;                 // 多轮统计轮数（AR007；1=单轮，与历史语义一致）
    bool list_kernels = false;
    bool help = false;
};

inline void print_usage(const char* prog) {
    std::printf(
        "Usage: %s --kernel <name> [options]\n"
        "  --kernel   naive|coalesced|smem1d|tile2d|vec4|cpasync|cpasync2|swpipe|swsk|ws|auto|cublas|wide|wsk|deep|dsk|streamk|all\n"
        "  --m/--n/--k          problem size (default 4096)\n"
        "  --warmup <n>         warmup iterations (default 20, spec: >=20)\n"
        "  --iters  <n>         timed iterations (default 100, spec: >=100)\n"
        "  --rounds <n>         independent timing rounds (default 1); n>1 aggregates\n"
        "                       per-round medians and reports cross-round RSD\n"
        "  --check              run correctness check vs cuBLAS FP32 before benchmark\n"
        "  --csv                append result row to results/performance.csv\n"
        "  --bk <8|16|32>       BK ablation knob for smem1d (default 32)\n"
        "  --lb <1|2>           __launch_bounds__ minBlocks ablation for tile2d/ws (default 1)\n"
        "  --sk <1..16>         split-K slices for swsk/wsk/dsk (default 4; 1 = bypass to swpipe/wide/deep)\n"
        "  --stages <2|3>       smem ring depth for ws (default 3)\n"
        "  --wp <1|2>           producer warp count for ws (default 2; 2+8=320 threads)\n"
        "  --wlb <1|2>          __launch_bounds__ minBlocks ablation for wide/wsk (default 2;\n"
        "                       2 = 64 regs -> 2 blocks/SM -> 100%% occupancy)\n"
        "  --dbuf <0|1>         smem double-buffer ablation for deep/dsk (default 1 = 2 buffers,\n"
        "                       1 sync/tile, 24832B; AR010 T004 measured +3~14%% at every size)\n"
        "  --rv2 <0|1|3>        split-K reduce variant ablation for swsk/wsk/dsk (default 0;\n"
        "                       0 = v1 1 f4/thread; 1 = v2 ILP2; 3 = v3 ILP4 + __ldcs/__stcs\n"
        "                       streaming hints, bit-identical chain order)\n"
        "  --waves <0..8>       Stream-K wave count W for streamk (default 0 = auto formula\n"
        "                       clamp(floor(TOT/(48*16)),1,8); AR011 design 4.2.6)\n"
        "  --persist <0|1>      L2 persistence window pinning C for dsk/streamk (default 0;\n"
        "                       1 = pin inside timed region + mandatory reset after,\n"
        "                       before any cuBLAS anchor run)\n"
        "  --hit <0.5..1.0>     accessPolicyWindow hitRatio (default 0.8; effective when\n"
        "                       --persist 1; C larger than persisting limit pins fractionally)\n"
        "  --verbose            print dispatch/fallback details\n"
        "  --list-kernels       list registered kernels\n"
        "  --help               this message\n",
        prog);
}

inline CliOptions parse_cli(int argc, char** argv) {
    CliOptions opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "[CLI_ERROR] missing value for %s\n", name);
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };
        if      (a == "--kernel")       opt.kernel = next(a.c_str());
        else if (a == "--m")            opt.m = std::atoi(next(a.c_str()));
        else if (a == "--n")            opt.n = std::atoi(next(a.c_str()));
        else if (a == "--k")            opt.k = std::atoi(next(a.c_str()));
        else if (a == "--warmup")       opt.warmup = std::atoi(next(a.c_str()));
        else if (a == "--iters")        opt.iters = std::atoi(next(a.c_str()));
        else if (a == "--rounds")       opt.rounds = std::atoi(next(a.c_str()));
        else if (a == "--bk")           opt.bk = std::atoi(next(a.c_str()));
        else if (a == "--lb")           opt.lb = std::atoi(next(a.c_str()));
        else if (a == "--sk")           opt.sk = std::atoi(next(a.c_str()));
        else if (a == "--stages")       opt.stages = std::atoi(next(a.c_str()));
        else if (a == "--wp")           opt.wp = std::atoi(next(a.c_str()));
        else if (a == "--wlb")          opt.wlb = std::atoi(next(a.c_str()));
        else if (a == "--dbuf")         opt.dbuf = std::atoi(next(a.c_str()));
        else if (a == "--rv2")          opt.rv2 = std::atoi(next(a.c_str()));
        else if (a == "--waves")        opt.waves = std::atoi(next(a.c_str()));
        else if (a == "--persist")      opt.persist = std::atoi(next(a.c_str()));
        else if (a == "--hit")          opt.hit = std::atof(next(a.c_str()));
        else if (a == "--check")        opt.check = true;
        else if (a == "--csv")          opt.csv = true;
        else if (a == "--verbose")      opt.verbose = true;
        else if (a == "--list-kernels") opt.list_kernels = true;
        else if (a == "--help" || a == "-h") opt.help = true;
        else {
            std::fprintf(stderr, "[CLI_ERROR] unknown argument: %s\n", a.c_str());
            std::exit(EXIT_FAILURE);
        }
    }
    if (opt.m <= 0 || opt.n <= 0 || opt.k <= 0) {
        std::fprintf(stderr, "[CLI_ERROR] M/N/K must be positive (got %d x %d x %d)\n",
                     opt.m, opt.n, opt.k);
        std::exit(EXIT_FAILURE);
    }
    if (opt.warmup < 0 || opt.iters < 1) {
        std::fprintf(stderr, "[CLI_ERROR] invalid warmup/iters\n");
        std::exit(EXIT_FAILURE);
    }
    if (opt.rounds < 1) {
        std::fprintf(stderr, "[CLI_ERROR] --rounds must be >= 1 (got %d)\n", opt.rounds);
        std::exit(EXIT_FAILURE);
    }
    if (opt.sk < 1 || opt.sk > 16) {
        std::fprintf(stderr, "[CLI_ERROR] --sk must be in 1..16 (got %d)\n", opt.sk);
        std::exit(EXIT_FAILURE);
    }
    if (opt.stages != 2 && opt.stages != 3) {
        std::fprintf(stderr, "[CLI_ERROR] --stages must be 2 or 3 (got %d)\n", opt.stages);
        std::exit(EXIT_FAILURE);
    }
    if (opt.wp != 1 && opt.wp != 2) {
        std::fprintf(stderr, "[CLI_ERROR] --wp must be 1 or 2 (got %d)\n", opt.wp);
        std::exit(EXIT_FAILURE);
    }
    if (opt.wlb != 1 && opt.wlb != 2) {
        std::fprintf(stderr, "[CLI_ERROR] --wlb must be 1 or 2 (got %d)\n", opt.wlb);
        std::exit(EXIT_FAILURE);
    }
    if (opt.dbuf != 0 && opt.dbuf != 1) {
        std::fprintf(stderr, "[CLI_ERROR] --dbuf must be 0 or 1 (got %d)\n", opt.dbuf);
        std::exit(EXIT_FAILURE);
    }
    if (opt.rv2 != 0 && opt.rv2 != 1 && opt.rv2 != 3) {
        std::fprintf(stderr, "[CLI_ERROR] --rv2 must be 0, 1 or 3 (got %d)\n", opt.rv2);
        std::exit(EXIT_FAILURE);
    }
    if (opt.waves < 0 || opt.waves > 8) {
        std::fprintf(stderr, "[CLI_ERROR] --waves must be in 0..8 (got %d)\n", opt.waves);
        std::exit(EXIT_FAILURE);
    }
    if (opt.persist != 0 && opt.persist != 1) {
        std::fprintf(stderr, "[CLI_ERROR] --persist must be 0 or 1 (got %d)\n", opt.persist);
        std::exit(EXIT_FAILURE);
    }
    if (opt.hit < 0.5 || opt.hit > 1.0) {
        std::fprintf(stderr, "[CLI_ERROR] --hit must be in 0.5..1.0 (got %.3f)\n", opt.hit);
        std::exit(EXIT_FAILURE);
    }
    return opt;
}

// ---------------------------------------------------------------------
// GPU 元数据
// ---------------------------------------------------------------------
struct GpuInfo {
    std::string name;
    int sm_major = 0, sm_minor = 0;
    size_t total_mem = 0;
    int clock_khz = 0;      // boost 时钟
    int num_sms = 0;
};

inline GpuInfo query_gpu_info() {
    GpuInfo info;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s (sm_%d%d)", prop.name, prop.major, prop.minor);
    info.name = buf;
    info.sm_major = prop.major;
    info.sm_minor = prop.minor;
    info.total_mem = prop.totalGlobalMem;
    info.clock_khz = prop.clockRate;
    info.num_sms = prop.multiProcessorCount;
    return info;
}

// 运行状态（时钟/温度/功耗）查询：best-effort，失败返回 "n/a"。
// 符合 AGENTS.md §5.3：每次正式测量前后记录。
inline std::string query_gpu_state() {
#if defined(_WIN32)
    FILE* f = _popen("nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw "
                     "--format=csv,noheader", "r");
#else
    FILE* f = popen("nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw "
                    "--format=csv,noheader", "r");
#endif
    if (!f) return "n/a";
    char line[256] = {0};
    const char* got = std::fgets(line, sizeof(line), f);
#if defined(_WIN32)
    _pclose(f);
#else
    pclose(f);
#endif
    if (!got) return "n/a";
    std::string s(line);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s.empty() ? "n/a" : s;
}

// ---------------------------------------------------------------------
// 测试矩阵初始化：均匀随机 [-1,1]，固定 seed 可复现（详设 §4.4）
// ---------------------------------------------------------------------
inline void init_matrix_host(float* h, size_t n, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < n; ++i) {
        h[i] = -1.0f + 2.0f * (float)rand() / (float)RAND_MAX;
    }
}

// ---------------------------------------------------------------------
// CUDA event 计时（详设 §4.3：只认 events；median/min/max；RSD）
// ---------------------------------------------------------------------
struct TimingStats {
    double ms_min = 0.0;
    double ms_median = 0.0;
    double ms_max = 0.0;
    double ms_mean = 0.0;
    double rsd = 0.0;             // 相对标准差（百分数），降频检测用
    int    iters = 0;
};

// Launch: 无参可调用对象（lambda 捕获参数后启动 kernel）
template <class Launch>
TimingStats time_kernel(Launch launch, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());

    std::vector<cudaEvent_t> start(iters), stop(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventCreate(&start[i]));
        CUDA_CHECK(cudaEventCreate(&stop[i]));
    }
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start[i]));
        launch();
        CUDA_CHECK(cudaEventRecord(stop[i]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> ms(iters);
    double sum = 0.0;
    for (int i = 0; i < iters; ++i) {
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, start[i], stop[i]));
        ms[i] = (double)t;
        sum += ms[i];
    }
    for (int i = 0; i < iters; ++i) {
        cudaEventDestroy(start[i]);
        cudaEventDestroy(stop[i]);
    }
    std::sort(ms.begin(), ms.end());
    TimingStats s;
    s.iters = iters;
    s.ms_min = ms.front();
    s.ms_max = ms.back();
    s.ms_median = (iters % 2 == 1) ? ms[iters / 2] : 0.5 * (ms[iters / 2 - 1] + ms[iters / 2]);
    s.ms_mean = sum / iters;
    double var = 0.0;
    for (double t : ms) var += (t - s.ms_mean) * (t - s.ms_mean);
    var /= iters;
    s.rsd = (s.ms_mean > 0.0) ? 100.0 * std::sqrt(var) / s.ms_mean : 0.0;
    return s;
}

inline double gflops_of(int m, int n, int k, double ms) {
    return 2.0 * (double)m * (double)n * (double)k / (ms * 1e6);
}

// ---------------------------------------------------------------------
// 多轮统计聚合层（AR007 详设 §4.3 补注）：time_kernel 单轮语义保持不变；
// rounds>1 时逐轮独立计时（每轮 warmup+iters），聚合取轮间 median，
// 跨轮 RSD 超门（5%）自动追加轮次（≤3 次重试）。
// ---------------------------------------------------------------------
struct MultiRoundStats {
    int rounds = 0;                    // 实际执行轮数（含自动重试）
    std::vector<double> round_ms;      // 逐轮 median
    double agg_ms = 0.0;               // 轮间 median（最终报告值）
    double cross_rsd = 0.0;            // 跨轮 RSD（百分数）
    double max_within_rsd = 0.0;       // 最大轮内 RSD
    TimingStats best_round;            // agg 对应轮的完整 min/max/median
};

// Launch: 无参可调用对象（同 time_kernel）
template <class Launch>
MultiRoundStats time_kernel_rounds(Launch launch, int warmup, int iters,
                                   int rounds, double rsd_gate_pct = 5.0,
                                   int max_retries = 3) {
    MultiRoundStats mr;
    std::vector<TimingStats> all_rounds;   // 逐轮完整统计（best_round 提取用）
    int retries = 0;
    auto cross_rsd_of = [](const std::vector<double>& v) {
        if (v.size() < 2) return 0.0;
        double sum = 0.0;
        for (double t : v) sum += t;
        const double mean = sum / v.size();
        double var = 0.0;
        for (double t : v) var += (t - mean) * (t - mean);
        var /= v.size();
        return (mean > 0.0) ? 100.0 * std::sqrt(var) / mean : 0.0;
    };

    for (;;) {
        TimingStats st = time_kernel(launch, warmup, iters);
        mr.round_ms.push_back(st.ms_median);
        all_rounds.push_back(st);
        mr.max_within_rsd = (st.rsd > mr.max_within_rsd) ? st.rsd : mr.max_within_rsd;
        // 轮数达标后按门控决定是否追加（每追加一次消耗一次重试额度）
        const bool planned_done = (int)mr.round_ms.size() >= rounds;
        if (planned_done) {
            mr.cross_rsd = cross_rsd_of(mr.round_ms);
            if (mr.cross_rsd > rsd_gate_pct && retries < max_retries) {
                ++retries;
                continue;   // 追加一轮（同 warmup+iters）
            }
            break;
        }
    }
    mr.rounds = (int)mr.round_ms.size();

    // 轮间 median（最终报告值）
    std::vector<double> sorted = mr.round_ms;
    std::sort(sorted.begin(), sorted.end());
    const int n = mr.rounds;
    mr.agg_ms = (n % 2 == 1) ? sorted[n / 2]
                             : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);

    // best_round：median 最接近 agg_ms 的一轮的完整统计（CSV min/max 来源）
    int best = 0;
    double best_diff = -1.0;
    for (int i = 0; i < n; ++i) {
        const double diff = std::fabs(mr.round_ms[i] - mr.agg_ms);
        if (best_diff < 0.0 || diff < best_diff) {
            best_diff = diff;
            best = i;
        }
    }
    mr.best_round = all_rounds[best];
    return mr;
}

// MultiRoundStats → TimingStats（CSV schema 冻结：rounds>1 时 rsd_pct = 跨轮
// RSD，min/max 取 best_round；rounds=1 与单轮完全一致）
inline TimingStats merge_round_stats(const MultiRoundStats& mr) {
    TimingStats st;
    st.ms_median = (mr.rounds > 1) ? mr.agg_ms : mr.best_round.ms_median;
    st.ms_min = mr.best_round.ms_min;
    st.ms_max = mr.best_round.ms_max;
    st.ms_mean = mr.best_round.ms_mean;
    st.rsd = (mr.rounds > 1) ? mr.cross_rsd : mr.best_round.rsd;
    st.iters = mr.best_round.iters;
    return st;
}

// 理论最小 DRAM 流量（字节）：各矩阵只读/写一次（ncu 对比基线，E4 实验）
inline double theoretical_min_dram_bytes(int m, int n, int k) {
    return ((double)m * k + (double)k * n + (double)m * n) * 4.0;
}

// ---------------------------------------------------------------------
// CSV 落盘（详设 §4.3 schema；append 模式，自动补表头与版本头）
// ---------------------------------------------------------------------
inline void append_csv_row(const std::string& path,
                           const std::string& kernel, int m, int n, int k,
                           const TimingStats& st, double gflops,
                           const std::string& regs /* "r128s0" 或 "n/a" */,
                           const std::string& gpu_state) {
    std::ifstream exist(path);
    bool need_header = !exist.good();
    exist.close();

    std::ofstream out(path, std::ios::app);
    if (!out) {
        std::fprintf(stderr, "[CSV_ERROR] cannot open %s\n", path.c_str());
        return;
    }
    if (need_header) {
        out << "# cuda-sgemm performance log | git=" << GIT_SHA << "\n";
        out << "kernel,m,n,k,ms_median,ms_min,ms_max,ms_mean,rsd_pct,gflops,"
               "regs_smem_note,gpu_state(sm_mhz,temp_c,power_w),git_sha,timestamp\n";
    }
    char ts[32];
    std::time_t now = std::time(nullptr);
    std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    out << kernel << ',' << m << ',' << n << ',' << k << ','
        << st.ms_median << ',' << st.ms_min << ',' << st.ms_max << ','
        << st.ms_mean << ',' << st.rsd << ',' << gflops << ','
        << regs << ",\"" << gpu_state << "\"," << GIT_SHA << ',' << ts << "\n";
}

}  // namespace sgemm
