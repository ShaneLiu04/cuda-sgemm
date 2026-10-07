// =====================================================================
// tools/cupti_trace.cu — AR012 T002（FR1 E-B）Level-1 采集器
// ---------------------------------------------------------------
// 目的：零权限获取 cuBLAS SGEMM 在各尺寸下实际启动的 kernel 名与
// launch 配置（grid/block/每线程寄存器/静态+动态 smem）+ 每次 launch
// 的 device 端时长（CUPTI activity 记录，ns / GPU 时钟域）。
//
// 为什么不用 nsys：本机 Nsight Systems CLI（2024.2.3，系统安装版与
// Nsight Compute 归档版两份）在无 admin 企业环境下 CLI→后端 protobuf
// IPC 静默失败（非 ASCII 路径报 "Message parsing failed"，ASCII 目录
// 下 exit 0 不拉起 app 不产 report；连 whoami.exe 都无法采集）——
// 负结果见 profile/cublas_disasm/report.md §1。进程内 CUPTI activity
// API 是同一数据源（nsys CUDA trace 的底层机制），不依赖注入与后端
// 服务，也不需要性能计数器权限（区别于 ncu 的 ERR_NVGPUCTRPERM）。
//
// 公平性契约（与 src/cublas_baseline.cu 逐条一致）：
//   * cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) 显式禁用 TF32；
//   * 行主序 C=A·B 换算：cublasSgemm(N, M, K, 1, B(ldb=N), A(lda=K),
//     0, C(ldc=N))；
//   * 输入种子 7/8（与 bench/tests 相同），保证与主 benchmark 同一
//     工作负载特征。
//
// 用法：cupti_trace.exe [M1 N1 K1] [M2 N2 K2] ...
//   无参数时跑默认尺寸集 512^3 / 1024^3 / 2048^3 / 4096^3。
//
// 输出（stdout，机器可解析）：
//   [cublas@MxNxK] launches=C name=<...> grid=(gx,gy,gz) block=(bx,by,bz)
//                  regs=<R> smem=<static+dyn>B dur_med=<us> dur_min=<us>
//   （同一 kernel+配置聚合计数与时长统计；warmup 阶段的 launch 一并
//    计数但时长单列——cuBLAS 首次调用可能触发 JIT/内部初始化 kernel）
//
// 构建（见 tools/build_cupti_trace.cmd）：
//   nvcc -O2 -std=c++17 -arch=sm_75 -I include ^
//        -I <cupti-arch>\include -L <cupti-arch>\lib -lcupti ^
//        -L <cuda-toolkit>\lib\x64 -lcublas -o build\cupti_trace.exe
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <array>
#include <vector>
#include <map>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cupti.h>
#include <cupti_activity.h>

// ---- 最小检查宏（独立于主工程 common.h，本文件为 E-B 工具不进套件）----
#define CUDA_CHECK_T(x)                                                        \
    do { cudaError_t e_ = (x);                                                 \
         if (e_ != cudaSuccess) {                                              \
             std::fprintf(stderr, "[cupti_trace] CUDA error %s at %s:%d\n",    \
                          cudaGetErrorString(e_), __FILE__, __LINE__);         \
             std::exit(EXIT_FAILURE); } } while (0)

#define CUBLAS_CHECK_T(x)                                                      \
    do { cublasStatus_t s_ = (x);                                              \
         if (s_ != CUBLAS_STATUS_SUCCESS) {                                    \
             std::fprintf(stderr, "[cupti_trace] cuBLAS error %d at %s:%d\n",  \
                          (int)s_, __FILE__, __LINE__);                        \
             std::exit(EXIT_FAILURE); } } while (0)

#define CUPTI_CHECK_T(x)                                                       \
    do { CUptiResult r_ = (x);                                                 \
         if (r_ != CUPTI_SUCCESS) {                                            \
             const char* msg_ = nullptr;                                       \
             cuptiGetResultString(r_, &msg_);                                  \
             std::fprintf(stderr, "[cupti_trace] CUPTI error %s at %s:%d\n",   \
                          msg_ ? msg_ : "?", __FILE__, __LINE__);              \
             std::exit(EXIT_FAILURE); } } while (0)

// ---------------------------------------------------------------------
// activity 记录收集：新版 CUPTI（2024.x SDK）缓冲回调对
//   cuptiActivityRegisterCallbacks(request, complete)：
//     request  —— CUPTI 拉一块空缓冲（本工具 malloc 固定 1MB 环）
//     complete —— CUPTI 推回已填缓冲（迭代记录 + 拷贝 name + free）；
//                 name 指针仅在回调内有效，必须就地深拷贝
// （旧 cuptiActivitySubscribe 已从本 SDK 移除，见 cupti_activity.h）
// ---------------------------------------------------------------------
namespace {

struct KernelRec {
    std::string name;
    uint32_t gridX = 0, gridY = 0, gridZ = 0;
    uint32_t blockX = 0, blockY = 0, blockZ = 0;
    uint32_t regs = 0;
    uint64_t smem_static = 0, smem_dyn = 0;
    uint64_t start_ns = 0, end_ns = 0;   // CUPTI GPU 时钟域（ns 标度）
};

constexpr size_t kBufBytes = 1 << 20;    // 1MB 活动缓冲（launch 数 <200）

std::vector<KernelRec> g_records;

void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size,
                               size_t* maxNumRecords) {
    *buffer = (uint8_t*)malloc(kBufBytes);
    if (*buffer == nullptr) {
        std::fprintf(stderr, "[cupti_trace] buffer malloc failed\n");
        std::exit(EXIT_FAILURE);
    }
    *size = kBufBytes;
    *maxNumRecords = 0;   // 0 = CUPTI 自行打包至缓冲满
}

void handle_buffer(uint8_t* buf, size_t valid) {
    CUpti_Activity* rec = nullptr;
    CUptiResult status = CUPTI_SUCCESS;
    do {
        status = cuptiActivityGetNextRecord(buf, valid, &rec);
        if (status == CUPTI_SUCCESS) {
            if (rec->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
                rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                const CUpti_ActivityKernel6* k =
                    reinterpret_cast<const CUpti_ActivityKernel6*>(rec);
                KernelRec r;
                r.name       = k->name ? k->name : "<null>";
                r.gridX      = k->gridX;  r.gridY = k->gridY;  r.gridZ = k->gridZ;
                r.blockX     = k->blockX; r.blockY = k->blockY; r.blockZ = k->blockZ;
                r.regs       = k->registersPerThread;
                r.smem_static= k->staticSharedMemory;
                r.smem_dyn   = k->dynamicSharedMemory;
                r.start_ns   = k->start;
                r.end_ns     = k->end;
                g_records.push_back(r);
            }
        } else if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
            const char* msg = nullptr;
            cuptiGetResultString(status, &msg);
            std::fprintf(stderr, "[cupti_trace] get record failed: %s\n",
                         msg ? msg : "?");
        }
    } while (status == CUPTI_SUCCESS);
}

void CUPTIAPI buffer_completed(CUcontext /*ctx*/, uint32_t /*streamId*/,
                               uint8_t* buffer, size_t /*size*/,
                               size_t validSize) {
    handle_buffer(buffer, validSize);
    free(buffer);
}

// 聚合键：kernel 名 + launch 配置（同一 kernel 理论上配置恒定；
// 聚合是为了稳健暴露 cuBLAS 可能的 per-size 多 kernel/多配置结构）
struct AggKey {
    std::string name;
    uint32_t gx, gy, gz, bx, by, bz, regs;
    uint64_t smem;
    bool operator<(const AggKey& o) const {
        if (name != o.name) return name < o.name;
        if (gx != o.gx) return gx < o.gx;
        if (gy != o.gy) return gy < o.gy;
        if (gz != o.gz) return gz < o.gz;
        if (bx != o.bx) return bx < o.bx;
        if (by != o.by) return by < o.by;
        if (bz != o.bz) return bz < o.bz;
        if (regs != o.regs) return regs < o.regs;
        return smem < o.smem;
    }
};

void dump_records(const char* tag, int M, int N, int K,
                  const std::vector<KernelRec>& recs) {
    std::map<AggKey, std::vector<double>> agg;   // key -> durations (us)
    for (const auto& r : recs) {
        AggKey k{r.name, r.gridX, r.gridY, r.gridZ,
                 r.blockX, r.blockY, r.blockZ, r.regs,
                 r.smem_static + r.smem_dyn};
        agg[k].push_back((double)(r.end_ns - r.start_ns) / 1000.0);
    }
    for (const auto& kv : agg) {
        const AggKey& k = kv.first;
        std::vector<double> d = kv.second;
        std::sort(d.begin(), d.end());
        const double med = d.empty() ? 0.0 : d[d.size() / 2];
        const double dmin = d.empty() ? 0.0 : d.front();
        const double dmax = d.empty() ? 0.0 : d.back();
        std::printf("[%s@%dx%dx%d] launches=%zu name=%s grid=(%u,%u,%u) "
                    "block=(%u,%u,%u) regs=%u smem=%lluB "
                    "dur_med=%.1fus dur_min=%.1fus dur_max=%.1fus\n",
                    tag, M, N, K, d.size(), k.name.c_str(),
                    k.gx, k.gy, k.gz, k.bx, k.by, k.bz, k.regs,
                    (unsigned long long)k.smem, med, dmin, dmax);
    }
    if (agg.empty())
        std::printf("[%s@%dx%dx%d] NO KERNEL RECORDS (CUPTI empty)\n",
                    tag, M, N, K);
}

// 与主工程 init_matrix_host 同族：确定性伪随机 [0,1)（种子 7/8）
void init_matrix(std::vector<float>& v, unsigned seed) {
    uint32_t s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = (float)(s >> 8) / 16777216.0f;
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::printf("# cupti_trace | CUPTI_API_VERSION=%u\n",
                (unsigned)CUPTI_API_VERSION);

    // CUPTI 缓冲回调注册须先于 CUDA context 创建（cublasCreate/cudaMalloc 之前）
    CUPTI_CHECK_T(cuptiActivityRegisterCallbacks(buffer_requested,
                                                 buffer_completed));
    // KERNEL 与 CONCURRENT_KERNEL 互斥（同 enable 报 NOT_COMPATIBLE），
    // 取 CONCURRENT_KERNEL（记录结构相同，语义更全）
    CUPTI_CHECK_T(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    // 尺寸集：默认 512/1024/2048/4096 立方（G1/G6/G7 判定尺寸），可覆写
    std::vector<std::array<int, 3>> sizes;
    for (int i = 1; i + 2 < argc; i += 3)
        sizes.push_back({std::atoi(argv[i]), std::atoi(argv[i + 1]),
                         std::atoi(argv[i + 2])});
    if (sizes.empty()) {
        sizes = {{512, 512, 512}, {1024, 1024, 1024},
                 {2048, 2048, 2048}, {4096, 4096, 4096}};
    }

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK_T(cublasCreate(&handle));
    CUBLAS_CHECK_T(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    const int WARMUP = 3, ITERS = 5;
    for (auto& s : sizes) {
        const int M = s[0], N = s[1], K = s[2];
        float *dA, *dB, *dC;
        CUDA_CHECK_T(cudaMalloc(&dA, (size_t)M * K * sizeof(float)));
        CUDA_CHECK_T(cudaMalloc(&dB, (size_t)K * N * sizeof(float)));
        CUDA_CHECK_T(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
        std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
        init_matrix(hA, 7u);
        init_matrix(hB, 8u);
        CUDA_CHECK_T(cudaMemcpy(dA, hA.data(), hA.size() * 4,
                                cudaMemcpyHostToDevice));
        CUDA_CHECK_T(cudaMemcpy(dB, hB.data(), hB.size() * 4,
                                cudaMemcpyHostToDevice));
        CUDA_CHECK_T(cudaMemset(dC, 0, (size_t)M * N * 4));

        const size_t mark = g_records.size();
        for (int it = 0; it < WARMUP + ITERS; ++it) {
            const float alpha = 1.0f, beta = 0.0f;
            // 行主序换算（cublas_baseline.cu 同式）
            CUBLAS_CHECK_T(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                       N, M, K,
                                       &alpha, dB, N, dA, K, &beta, dC, N));
            CUDA_CHECK_T(cudaDeviceSynchronize());
        }
        CUPTI_CHECK_T(cuptiActivityFlushAll(0));
        std::vector<KernelRec> slice(g_records.begin() + mark,
                                     g_records.end());
        // 首轮（warmup 第 1 次）可能混初始化 kernel：按 launch 序打印时
        // 聚合已合并；若 slice 非单一配置，读者可直接看 launches 计数
        dump_records("cublas", M, N, K, slice);
        std::fflush(stdout);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    CUPTI_CHECK_T(cuptiActivityDisable(
        CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUBLAS_CHECK_T(cublasDestroy(handle));
    CUDA_CHECK_T(cudaDeviceSynchronize());
    return 0;
}
