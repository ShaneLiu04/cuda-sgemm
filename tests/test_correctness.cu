// =====================================================================
// test_correctness.cu — 全 kernel × 全测试矩阵 一键正确性回归（AR001 T003）
// ---------------------------------------------------------------
// 判据（详设 §4.4）：max|C_gpu - C_ref| / max(|C_ref|, eps) <= 1e-4
// 参考：
//   * CPU double 累加（小尺寸，且用于 cuBLAS 行主序换算的自校验）
//   * cuBLAS FP32（全尺寸，经自校验后才可信）
// 测试矩阵（冻结）：
//   4096^3（主场景）/ 1024^3 / 256^3（快速回归）/ 1000x1016x1024（非方阵）/
//   1023x1024x511（边界：非 tile 对齐且 K%4!=0 → 触发 vec4/cpasync 回退）/
//   130x257x66（边界：N%4!=0 → 回退路径专项）/
//   64x64x1（K 退化）/ 1x1x1 / 17x33x65（微小奇尺寸）
// 路径覆盖：--verbose 时 launcher 打印 "scalar fallback"，输出表格标注
//   每个尺寸实际走的路径（GPU 环境可 grep 断言）。
// 退出码：任一 FAIL -> 非零（CI 可用）。
// =====================================================================
#include "common.h"
#include "sgemm_kernels.h"

#include <cstdio>
#include <vector>

namespace {

// ---- CPU double 参考（i-k-j 序，cache 友好）----
void sgemm_cpu_double(const std::vector<float>& A, const std::vector<float>& B,
                      std::vector<double>& C, int M, int N, int K) {
    C.assign((size_t)M * N, 0.0);
    for (int i = 0; i < M; ++i)
        for (int kk = 0; kk < K; ++kk) {
            const double a = A[(size_t)i * K + kk];
            for (int j = 0; j < N; ++j)
                C[(size_t)i * N + j] += a * B[(size_t)kk * N + j];
        }
}

struct CaseResult {
    bool pass = false;
    double max_abs = 0.0, rel = 0.0;
    double max_abs_cpu = 0.0, rel_cpu = 0.0;
    bool has_cpu_ref = false;
};

// 对比辅助：返回 {max_abs, rel}
std::pair<double, double> compare(const std::vector<float>& got,
                                  const std::vector<double>& ref) {
    double max_abs = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = std::fabs((double)got[i] - ref[i]);
        if (d > max_abs) max_abs = d;
        double r = std::fabs(ref[i]);
        if (r > max_ref) max_ref = r;
    }
    return {max_abs, max_abs / (max_ref > 1e-30 ? max_ref : 1e-30)};
}

CaseResult run_case(int kernel_id, int M, int N, int K, bool with_cpu_ref) {
    CaseResult res;

    // 设备内存与输入
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    sgemm::init_matrix_host(hA.data(), hA.size(), 7u);
    sgemm::init_matrix_host(hB.data(), hB.size(), 8u);
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    // ---- 参考 1：cuBLAS FP32 ----
    std::vector<float> ref_f((size_t)M * N);
    sgemm_cublas(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(ref_f.data(), dC, ref_f.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 参考 2（小尺寸）：CPU double（同时自校验 cuBLAS 行主序换算）----
    std::vector<double> ref_d;
    if (with_cpu_ref) sgemm_cpu_double(hA, hB, ref_d, M, N, K);

    // ---- 待测 kernel ----
    CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));
    if (sgemm::g_verbose) {
        std::printf("    [%s] %dx%dx%d dispatch:\n", sgemm::kernel_name(kernel_id), M, N, K);
    }
    SgemmFn fn = sgemm::kernel_fn(kernel_id);
    if (!fn) {
        // Red 状态可观察：kernel 未接入注册表 → 该格 FAIL（非崩溃）
        std::printf("    [ERROR] kernel '%s' not registered (fn=nullptr)\n",
                    sgemm::kernel_name(kernel_id));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        return res;   // pass=false
    }
    fn(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    std::vector<float> got((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(got.data(), dC, got.size() * sizeof(float), cudaMemcpyDeviceToHost));

    if (err != cudaSuccess) {
        std::printf("    [ERROR] kernel launch: %s\n", cudaGetErrorString(err));
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        return res;   // pass=false
    }

    // 与 cuBLAS FP32 对比
    std::vector<double> ref_d_f(ref_f.begin(), ref_f.end());
    auto r1 = compare(got, ref_d_f);
    res.max_abs = r1.first;
    res.rel = r1.second;

    // 与 CPU double 对比（若启用）
    if (with_cpu_ref) {
        auto r2 = compare(got, ref_d);
        res.max_abs_cpu = r2.first;
        res.rel_cpu = r2.second;
        res.has_cpu_ref = true;
    }

    res.pass = (res.rel <= 1e-4) &&
               (!res.has_cpu_ref || res.rel_cpu <= 1e-4);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return res;
}

struct TestCase {
    int m, n, k;
    const char* note;
    bool cpu_ref;   // M*N*K <= ~2^28 时启用 CPU double 参考
};

// ---- AR011 bitwise 锚链载体（design §6.1；AR010 21/21 ad-hoc 专项永久化）----
// 同一输入（与 run_case 同种子 7/8）单 kernel 落盘 host，供跨 kernel memcmp。
void run_dump(int kernel_id, int M, int N, int K, std::vector<float>& out) {
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    sgemm::init_matrix_host(hA.data(), hA.size(), 7u);
    sgemm::init_matrix_host(hB.data(), hB.size(), 8u);
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));
    SgemmFn fn = sgemm::kernel_fn(kernel_id);
    fn(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    out.resize((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(out.data(), dC, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

}  // namespace

int main(int argc, char** argv) {
    // --verbose 透传给 launcher 的回退路径打印（路径覆盖断言用）
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--verbose") sgemm::g_verbose = true;

    std::printf("==================================================================\n");
    std::printf(" cuda-sgemm correctness suite | git=%s\n", GIT_SHA);
    std::printf(" tolerance: max|C-C_ref| / max|C_ref| <= 1e-4\n");
    std::printf("==================================================================\n");

    // 测试矩阵（详设 §4.4 + 微小退化用例；cpu_ref 阈值按算量取舍）
    const std::vector<TestCase> cases = {
        {4096, 4096, 4096, "main scene",            false},
        {1024, 1024, 1024, "fast regression",       false},
        {256,   256,   256, "fast regression",       true},
        {1000, 1016, 1024, "non-square",            false},
        {1023, 1024,  511, "edge: no tile align, K%4!=0 -> fallback", true},
        {130,   257,   66, "edge: N%4!=0 -> fallback",                 true},
        {64,     64,    1, "degenerate K=1",         true},
        {1,       1,    1, "degenerate 1x1x1",       true},
        {17,     33,   65, "tiny odd sizes",         true},
    };

    int total = 0, passed = 0;
    bool all_ok = true;

    for (int kid = 0; kid < sgemm::KERNEL_COUNT; ++kid) {
        const char* kname = sgemm::kernel_name(kid);
        std::printf("------------------------------------------------------------------\n");
        std::printf(" kernel: %s\n", kname);
        std::printf("------------------------------------------------------------------\n");
        for (const auto& tc : cases) {
            ++total;
            CaseResult r = run_case(kid, tc.m, tc.n, tc.k, tc.cpu_ref);
            if (r.pass) ++passed;
            else all_ok = false;
            std::printf("  %4dx%4dx%4d  %-42s  max_abs=%.3e rel=%.3e",
                        tc.m, tc.n, tc.k, tc.note, r.max_abs, r.rel);
            if (r.has_cpu_ref)
                std::printf("  (vs CPU: abs=%.3e rel=%.3e)", r.max_abs_cpu, r.rel_cpu);
            std::printf("  %s\n", r.pass ? "PASS" : "FAIL");
        }
    }

    // cuBLAS 行主序换算自校验（详设 §4.2 义务）：在两个小尺寸 vs CPU double
    std::printf("------------------------------------------------------------------\n");
    std::printf(" cublas row-major self-check (vs CPU double)\n");
    {
        const int sizes[][3] = {{256, 256, 256}, {100, 253, 61}};
        for (auto& s : sizes) {
            ++total;
            CaseResult r = run_case(sgemm::K_CUBLAS, s[0], s[1], s[2], true);
            // 只看 vs CPU 的判据（cublas vs cublas 恒等，无意义）
            r.pass = r.has_cpu_ref && r.rel_cpu <= 1e-4;
            if (r.pass) ++passed; else all_ok = false;
            std::printf("  %4dx%4dx%4d  vs CPU double rel=%.3e  %s\n",
                        s[0], s[1], s[2], r.rel_cpu, r.pass ? "PASS" : "FAIL");
        }
    }

    // ---- AR011 bitwise 锚链专项（design §6.1 三锚链 + 确定性双跑）----
    // 判据：跨 kernel / 双跑 memcmp 全位一致（0 容差）。旋钮就地设置并复原。
    std::printf("------------------------------------------------------------------\n");
    std::printf(" AR011 bitwise anchors (streamk, memcmp exact)\n");
    std::printf("------------------------------------------------------------------\n");
    auto anchor_case = [&](const char* name, const std::vector<float>& got,
                           const std::vector<float>& ref) {
        ++total;
        const bool eq = (got.size() == ref.size()) &&
                        std::memcmp(got.data(), ref.data(),
                                    got.size() * sizeof(float)) == 0;
        if (eq) ++passed; else all_ok = false;
        std::printf("  %-58s  %s\n", name, eq ? "BITWISE-PASS" : "BITWISE-FAIL");
    };
    {   // 锚 1：TOT<48 单波旁路 == deep（wrapper 直通；256x512x64: TOT=32）
        std::vector<float> g, r;
        run_dump(sgemm::K_STREAMK, 256, 512, 64, g);
        run_dump(sgemm::K_DEEP, 256, 512, 64, r);
        anchor_case("streamk(TOT<48 bypass) == deep @256x512x64", g, r);
    }
    {   // 锚 2：W=1（U=nt=64，cover=1 满波直写）== deep
        const int sw = sgemm::g_streamk_waves;
        sgemm::g_streamk_waves = 1;
        std::vector<float> g, r;
        run_dump(sgemm::K_STREAMK, 1536, 1024, 512, g);
        run_dump(sgemm::K_DEEP, 1536, 1024, 512, r);
        sgemm::g_streamk_waves = sw;
        anchor_case("streamk(W=1, U=nt=64, cover=1) == deep @1536x1024x512", g, r);
    }
    {   // 锚 3：W=2（U=32 切点 == dsk sk2 tps=32 逐点重合）== dsk(direct)
        const int sw = sgemm::g_streamk_waves, sk = sgemm::g_swsk_slices,
                  rv = sgemm::g_reduce_ilp2;
        sgemm::g_streamk_waves = 2;
        sgemm::g_swsk_slices = 2;
        sgemm::g_reduce_ilp2 = 0;
        std::vector<float> g, r;
        run_dump(sgemm::K_STREAMK, 1536, 1024, 512, g);
        run_dump(sgemm::K_DSK, 1536, 1024, 512, r);
        sgemm::g_streamk_waves = sw; sgemm::g_swsk_slices = sk;
        sgemm::g_reduce_ilp2 = rv;
        anchor_case("streamk(W=2, U=32) == dsk(sk2,direct) @1536x1024x512", g, r);
    }
    {   // 锚 4：1024^3 W=2 —— U=43 与 nt=128 非整除：切点 43b 相对 128c 逐 tile
        // 漂移，128c≡0 (mod 43) 仅 c=0 成立 → 只有 tile0（C[0:256,0:128]）
        // 与 dsk sk3 tps=43 逐点重合（design §4.2.1 例），故只对 tile0 区域
        // memcmp（cover=3 票据归并路径在 1024^3 的链序实证）
        const int sw = sgemm::g_streamk_waves, sk = sgemm::g_swsk_slices,
                  rv = sgemm::g_reduce_ilp2;
        sgemm::g_streamk_waves = 2;
        sgemm::g_swsk_slices = 3;
        sgemm::g_reduce_ilp2 = 0;
        std::vector<float> g, r;
        run_dump(sgemm::K_STREAMK, 1024, 1024, 1024, g);
        run_dump(sgemm::K_DSK, 1024, 1024, 1024, r);
        sgemm::g_streamk_waves = sw; sgemm::g_swsk_slices = sk;
        sgemm::g_reduce_ilp2 = rv;
        ++total;
        bool eq = true;
        for (int i = 0; i < 256 && eq; ++i)
            eq = std::memcmp(&g[(size_t)i * 1024], &r[(size_t)i * 1024],
                             128 * sizeof(float)) == 0;
        if (eq) ++passed; else all_ok = false;
        std::printf("  %-58s  %s\n",
                    "streamk(W=2) tile0 region == dsk(sk3,direct) @1024^3",
                    eq ? "BITWISE-PASS" : "BITWISE-FAIL");
    }
    {   // 锚 5：确定性双跑（auto W=5 票据归并路径，赢家不确定 → 结果须逐位确定）
        std::vector<float> g1, g2;
        run_dump(sgemm::K_STREAMK, 1024, 1024, 1024, g1);
        run_dump(sgemm::K_STREAMK, 1024, 1024, 1024, g2);
        anchor_case("streamk(auto W=5) double-run determinism @1024^3", g1, g2);
    }

    std::printf("==================================================================\n");
    std::printf(" SUMMARY: %d / %d PASS (incl. %d bitwise anchors)  ->  %s\n",
                passed, total, 5, all_ok ? "ALL PASS" : "FAILED");
    std::printf("==================================================================\n");
    return all_ok ? 0 : EXIT_FAILURE;
}
