// =====================================================================
// sgemm_swpipe_sk.cu — Kernel 6 变体 swsk：split-K + 确定性固定序归约
// ---------------------------------------------------------------------
// 动机（AR008）：swpipe 128x128 tile 在中小尺寸 wave 饥饿——512^3 仅
// 16 blocks / 48 SM（33% 占用面），1024^3 64 blocks = 1.33 波（尾波空转）。
// split-K 沿 K 维切 z 片：grid=(n_t, m_t, sk)，block 数 x sk，直接
// 抬波数；代价是 sk 倍 P 写出 + 一次归约读回（512^3 sk=4 时 P=4MB，
// 0.33ms@6GB/s 量级 << kernel 收益）。
//
// 主体复用：swpipe tile 主体经参数化共享（detail::swpipe_tile_grid，
// 见 sgemm_swpipe.cu）——tile 128x128、BK=8、TM=TN=8、A 转置 + B swizzle
// 布局、寄存器预取流水、LB 消融全部继承，本文件不重复实现。
//
// 归约（确定性）：独立 kernel，每线程 1x float4，固定 z 序 s=0..sk-1
// 串行累加 P[s*MN + i]——无 atomic、无锁、逐位可复现（--check 两次
// max_abs 相等即为证据）；非结合 FP32 加法顺序固定是其根因。
//
// workspace：冻结签名 sgemm_x(A,B,C,M,N,K) 下的必然取舍（AR008 design
// §4.2.1）——静态 RAII grow-only（只增不减，避免 free/malloc 抖动污染
// 计时；bench 事件窗内仅首次分配一次，warmup 后零分配）。
//
// 边界语义：sk > num_tiles 时空片（t0>=t1）自然写全零；sk=1 旁路直走
// sgemm_swpipe 单波路径（零归约开销）；对齐不满足（N%4/K%4/16B）回退
// sgemm_2d_tile 谓词化路径。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>

namespace sgemm {
int g_swsk_slices = 4;  // --sk 片数（默认 4；1..16，CLI 校验）
}

namespace {

// ---- grow-only workspace（RAII；程序退出统一释放）--------------------
struct Workspace {
    float*  p   = nullptr;
    size_t  cap = 0;        // 容量（floats）
    ~Workspace() {
        if (p) cudaFree(p);
    }
    void ensure(size_t floats) {
        if (cap >= floats) return;
        if (p) cudaFree(p);
        cudaError_t err = cudaMalloc(&p, floats * sizeof(float));
        if (err != cudaSuccess) {
            p = nullptr; cap = 0;
            std::fprintf(stderr, "[swsk] workspace alloc failed (%zu floats): %s\n",
                         floats, cudaGetErrorString(err));
            std::exit(EXIT_FAILURE);
        }
        cap = floats;
    }
};
Workspace g_ws;  // 静态实例：首次调用分配，之后零开销

// ---- 确定性归约：C[i] = sum_s P[s*MN + i]，固定 z 序，每线程 1x float4 --
__global__ void swsk_reduce_kernel(const float* __restrict__ P,
                                   float* __restrict__ C,
                                   long long mn4, int sk) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < mn4) {
        const float4* p4 = reinterpret_cast<const float4*>(P);
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int s = 0; s < sk; ++s) {
            const float4 v = p4[s * mn4 + idx];
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
        reinterpret_cast<float4*>(C)[idx] = acc;
    }
}

void launch_reduce(const float* P, float* C, long long mn, int sk) {
    const long long mn4 = mn / 4;   // 主路径 N%4==0 ⇒ M*N 整除 4
    const int block = 256;
    const long long grid = (mn4 + block - 1) / block;
    swsk_reduce_kernel<<<(unsigned)grid, block>>>(P, C, mn4, sk);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[swsk] reduce launch failed: %s\n",
                     cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

void sgemm_swpipe_sk(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    const int sk = sgemm::g_swsk_slices;

    // 对齐不满足 → 谓词化回退（与 swpipe 同条件；split-K 主路径依赖
    // float4 写 P 与归约 float4 读 P/C）
    const bool aligned =
        (N % 4 == 0) && (K % 4 == 0) &&
        (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0);
    if (!aligned) {
        if (sgemm::g_verbose) {
            std::printf("[swsk] size %dx%dx%d -> scalar fallback (tile2d)\n",
                        M, N, K);
        }
        sgemm_2d_tile(A, B, C, M, N, K);
        return;
    }

    // sk=1 旁路：直走 swpipe 单波路径（零 workspace / 零归约开销）
    if (sk < 2) {
        if (sgemm::g_verbose) {
            std::printf("[swsk] sk=%d -> bypass to swpipe single-wave\n", sk);
        }
        sgemm_swpipe(A, B, C, M, N, K);
        return;
    }

    constexpr int BM = 128, BN = 128, BK = 8;
    const int num_tiles = (K + BK - 1) / BK;
    const int grid_n = (N + BN - 1) / BN;
    const int grid_m = (M + BM - 1) / BM;
    const int tps = (num_tiles + sk - 1) / sk;   // 每片 k-tile 数（ceil）

    // workspace：sk 个 M*N 切片（grow-only；空片由主体写零，全量有效）
    g_ws.ensure((size_t)sk * M * N);

    // ① split-K 主体：grid=(grid_n, grid_m, sk)，Out=P
    sgemm::detail::swpipe_tile_grid(A, B, g_ws.p, M, N, K,
                                    tps, num_tiles, grid_n, grid_m, sk);
    // ② 确定性归约：P → C（固定 z 序）
    launch_reduce(g_ws.p, C, (long long)M * N, sk);
}
