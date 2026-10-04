// =====================================================================
// cublas_baseline.cu — cuBLAS FP32 参考线（AR001，详设 §4.2）
// ---------------------------------------------------------------
// 公平性契约：
//   * cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) —— 显式禁用 TF32，
//     否则"97% of cuBLAS"对比失真（E2 实验提供 TF32 on/off 对照证据）。
//   * cuBLAS 为列主序。行主序 C=A·B 等价于列主序 C'=B'·A'
//     （A'=Aᵀ、B'=Bᵀ、C'=Cᵀ，同一 buffer 的另一种视角）：
//         cublasSgemm(N, M, K, alpha, B(ldb=N), A(lda=K), beta, C(ldc=N))
//   * 自校验义务：tests/test_correctness.cu 在小尺寸与 CPU double 交叉验证。
// 本文件同时定义全局消融旋钮 g_verbose（供回退路径打印，链接给所有目标）。
// =====================================================================
#include "sgemm_kernels.h"
#include "common.h"
#include <cstdio>

namespace sgemm {
bool g_verbose = false;   // 定义处（声明见 sgemm_kernels.h）
}

namespace {

cublasHandle_t g_handle = nullptr;

cublasHandle_t get_handle() {
    if (!g_handle) {
        CUBLAS_CHECK(cublasCreate(&g_handle));
        // 严格 FP32：禁用 TF32（Ada 上默认关闭，但必须显式声明并固定）
        CUBLAS_CHECK(cublasSetMathMode(g_handle, CUBLAS_DEFAULT_MATH));
    }
    return g_handle;
}

}  // namespace

void sgemm_cublas(const float* A, const float* B, float* C, int M, int N, int K) {
    cublasHandle_t h = get_handle();
    const float alpha = 1.0f, beta = 0.0f;
    // 行主序 -> 列主序换算（见文件头）：C'(N×M) = B'(N×K) · A'(K×M)
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha, B, N,
                             A, K,
                             &beta, C, N));
}
