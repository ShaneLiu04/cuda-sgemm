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
    std::string kernel = "naive";   // naive|coalesced|smem1d|tile2d|vec4|cpasync|cpasync2|cublas|all
    int m = 4096;
    int n = 4096;
    int k = 4096;
    int warmup = 20;
    int iters = 100;
    bool check = false;             // 正确性检查（对照 cuBLAS FP32）
    bool csv = false;               // 结果追加写入 results/performance.csv
    bool verbose = false;           // 打印回退路径等细节
    int bk = 16;                    // smem1d 的 BK 消融旋钮（8/16/32）
    int lb = 1;                     // tile2d 的 __launch_bounds__ minBlocks 消融旋钮（1/2）
    bool list_kernels = false;
    bool help = false;
};

inline void print_usage(const char* prog) {
    std::printf(
        "Usage: %s --kernel <name> [options]\n"
        "  --kernel   naive|coalesced|smem1d|tile2d|vec4|cpasync|cpasync2|cublas|all\n"
        "  --m/--n/--k          problem size (default 4096)\n"
        "  --warmup <n>         warmup iterations (default 20, spec: >=20)\n"
        "  --iters  <n>         timed iterations (default 100, spec: >=100)\n"
        "  --check              run correctness check vs cuBLAS FP32 before benchmark\n"
        "  --csv                append result row to results/performance.csv\n"
        "  --bk <8|16|32>       BK ablation knob for smem1d (default 16)\n"
        "  --lb <1|2>           __launch_bounds__ minBlocks ablation knob for tile2d (default 1)\n"
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
        else if (a == "--bk")           opt.bk = std::atoi(next(a.c_str()));
        else if (a == "--lb")           opt.lb = std::atoi(next(a.c_str()));
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
