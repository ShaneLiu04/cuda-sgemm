// tests/tmp_bitwise_ar010.cu — AR010 临时逐位验证工具（不入 CMake；验收后删除）
// 用途：deep vs swpipe 主路径逐位一致性 + deep DBUF{0,1} 实例逐位一致性 +
//       dsk vs swsk 逐位一致性（T003）+ --rv2{0,1} 归约逐位等价（T003）
#include "common.h"
#include "sgemm_kernels.h"

#include <cstdio>
#include <vector>

namespace sgemm { bool g_verbose = false; }   // 工具自备（正常构建中由 main.cu 定义）

static int failures = 0;

void run_case(const char* tag, int M, int N, int K) {
    float *dA, *dB, *C1, *C2;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C1, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C2, (size_t)M * N * sizeof(float)));
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    sgemm::init_matrix_host(hA.data(), hA.size(), 7u);
    sgemm::init_matrix_host(hB.data(), hB.size(), 8u);
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));
    const size_t cn = (size_t)M * N;
    std::vector<float> r1(cn), r2(cn);

    // ① deep(dbuf=0) vs swpipe
    sgemm::g_deep_dbuf = 0;
    sgemm_swpipe(dA, dB, C1, M, N, K);
    sgemm_deep(dA, dB, C2, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(r1.data(), C1, cn * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r2.data(), C2, cn * 4, cudaMemcpyDeviceToHost));
    bool eq = memcmp(r1.data(), r2.data(), cn * 4) == 0;
    std::printf("[%s] %dx%dx%d deep(dbuf0)==swpipe bitwise: %s\n",
                tag, M, N, K, eq ? "YES" : "NO");
    if (!eq) ++failures;

    // ② deep(dbuf=1) vs deep(dbuf=0)
    CUDA_CHECK(cudaMemset(C2, 0, cn * 4));
    sgemm::g_deep_dbuf = 1;
    sgemm_deep(dA, dB, C2, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(r2.data(), C2, cn * 4, cudaMemcpyDeviceToHost));
    eq = memcmp(r1.data(), r2.data(), cn * 4) == 0;
    std::printf("[%s] %dx%dx%d deep(dbuf1)==deep(dbuf0) bitwise: %s\n",
                tag, M, N, K, eq ? "YES" : "NO");
    if (!eq) ++failures;
    sgemm::g_deep_dbuf = 0;

    // ③ dsk(sk=4) vs swsk(sk=4)：同 BK=8 k-tile 切分 + 单一归约路径 → 逐位同源
    //    （仅主尺寸；非 4 倍数尺寸走回退不适用）
    if (N % 4 == 0 && K % 4 == 0) {
        sgemm::g_swsk_slices = 4;
        sgemm_swpipe_sk(dA, dB, C1, M, N, K);
        sgemm_dsk(dA, dB, C2, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(r1.data(), C1, cn * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(r2.data(), C2, cn * 4, cudaMemcpyDeviceToHost));
        eq = memcmp(r1.data(), r2.data(), cn * 4) == 0;
        std::printf("[%s] %dx%dx%d dsk(sk4)==swsk(sk4) bitwise: %s\n",
                    tag, M, N, K, eq ? "YES" : "NO");
        if (!eq) ++failures;

        // ④ --rv2 逐位等价：dsk sk4 归约 v1 vs v2
        CUDA_CHECK(cudaMemset(C2, 0, cn * 4));
        sgemm::g_reduce_ilp2 = 1;
        sgemm_dsk(dA, dB, C2, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(r2.data(), C2, cn * 4, cudaMemcpyDeviceToHost));
        eq = memcmp(r1.data(), r2.data(), cn * 4) == 0;
        std::printf("[%s] %dx%dx%d dsk(sk4,rv2)==dsk(sk4,rv1) bitwise: %s\n",
                    tag, M, N, K, eq ? "YES" : "NO");
        if (!eq) ++failures;
        sgemm::g_reduce_ilp2 = 0;

        // ⑤ dsk 确定性：两次独立调用逐位一致
        CUDA_CHECK(cudaMemset(C2, 0, cn * 4));
        sgemm_dsk(dA, dB, C2, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(r2.data(), C2, cn * 4, cudaMemcpyDeviceToHost));
        eq = memcmp(r1.data(), r2.data(), cn * 4) == 0;
        std::printf("[%s] %dx%dx%d dsk(sk4) run-to-run deterministic: %s\n",
                    tag, M, N, K, eq ? "YES" : "NO");
        if (!eq) ++failures;
        sgemm::g_swsk_slices = 4;
    }

    cudaFree(dA); cudaFree(dB); cudaFree(C1); cudaFree(C2);
}

int main() {
    run_case("main", 1024, 1024, 1024);
    run_case("nonsq", 1000, 1016, 1024);
    run_case("small", 256, 256, 256);
    run_case("large", 2048, 2048, 2048);
    std::printf(failures ? "BITWISE FAILED (%d)\n" : "ALL BITWISE YES\n", failures);
    return failures ? 1 : 0;
}
