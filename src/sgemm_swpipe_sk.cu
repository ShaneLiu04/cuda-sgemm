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
int g_reduce_ilp2 = 0;  // --rv2（AR010）：0 = v1（1 f4/线程），1 = v2（2 f4/线程，
                        // 逐位等价——仅线程-元素映射改变，逐元素 z 升序链不变）
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

// ---- 归约 v2（AR010 --rv2）：每线程 2 个相邻 float4，2x 独立 load 链 ----
// 逐位等价证明：逐元素的加法链仍为 s=0..sk-1 升序串行（v1 相同），
// 仅线程-元素映射改变（idx2 处理 {2*idx2, 2*idx2+1}）；mn4 奇数时尾线程
// 只写 i0（i1 越界护卫）。
__global__ void swsk_reduce_v2_kernel(const float* __restrict__ P,
                                      float* __restrict__ C,
                                      long long mn4, int sk) {
    const long long idx2 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long i0 = idx2 * 2;
    if (i0 < mn4) {
        const bool has1 = (i0 + 1) < mn4;
        const float4* p4 = reinterpret_cast<const float4*>(P);
        float4 acc0 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 acc1 = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int s = 0; s < sk; ++s) {
            const float4 v0 = p4[s * mn4 + i0];
            acc0.x += v0.x; acc0.y += v0.y; acc0.z += v0.z; acc0.w += v0.w;
            if (has1) {
                const float4 v1 = p4[s * mn4 + i0 + 1];
                acc1.x += v1.x; acc1.y += v1.y; acc1.z += v1.z; acc1.w += v1.w;
            }
        }
        reinterpret_cast<float4*>(C)[i0] = acc0;
        if (has1) reinterpret_cast<float4*>(C)[i0 + 1] = acc1;
    }
}

// ---- 归约 v3（AR010 T007 G1@1024³ 攻坚 --rv2=3）：每线程 4 个相邻 float4
// （ILP4，4x 独立 load 链）+ __ldcs/__stcs 流式提示（P 不复用、C 不再读，
// evict-first 减少 L2 污染）。逐位等价证明：每元素的加法链仍为 s=0..sk-1
// 升序串行（与 v1 完全相同），仅线程-元素映射改变（idx4 处理
// {4*idx4 .. 4*idx4+3}）；mn4 非整除时逐槽谓词护卫。
__global__ void swsk_reduce_v3_kernel(const float* __restrict__ P,
                                      float* __restrict__ C,
                                      long long mn4, int sk) {
    const long long idx4 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long i0 = idx4 * 4;
    if (i0 < mn4) {
        const bool has1 = (i0 + 1) < mn4;
        const bool has2 = (i0 + 2) < mn4;
        const bool has3 = (i0 + 3) < mn4;
        const float4* p4 = reinterpret_cast<const float4*>(P);
        float4 acc0 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 acc1 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 acc2 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 acc3 = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int s = 0; s < sk; ++s) {
            const float4 v0 = __ldcs(&p4[s * mn4 + i0]);
            acc0.x += v0.x; acc0.y += v0.y; acc0.z += v0.z; acc0.w += v0.w;
            if (has1) {
                const float4 v1 = __ldcs(&p4[s * mn4 + i0 + 1]);
                acc1.x += v1.x; acc1.y += v1.y; acc1.z += v1.z; acc1.w += v1.w;
            }
            if (has2) {
                const float4 v2 = __ldcs(&p4[s * mn4 + i0 + 2]);
                acc2.x += v2.x; acc2.y += v2.y; acc2.z += v2.z; acc2.w += v2.w;
            }
            if (has3) {
                const float4 v3 = __ldcs(&p4[s * mn4 + i0 + 3]);
                acc3.x += v3.x; acc3.y += v3.y; acc3.z += v3.z; acc3.w += v3.w;
            }
        }
        float4* c4 = reinterpret_cast<float4*>(C);
        __stcs(&c4[i0], acc0);
        if (has1) __stcs(&c4[i0 + 1], acc1);
        if (has2) __stcs(&c4[i0 + 2], acc2);
        if (has3) __stcs(&c4[i0 + 3], acc3);
    }
}

// ---- 归约 direct（AR010 T007 dsk last-slice-direct 主路径）------------------
// 语义：C[i] = ((0 + P_0[i]) + … + P_{sk-2}[i]) + C[i]，P 含 sk-1 个切片，
// 末源为 C 自身（由主体末片直写，值与旧全 P 路径的 P_{sk-1}[i] 逐位相同，
// 加法链完全同序 ⇒ 与 v1 全 P 归约逐位等价）。
// 实现：ILP4（4 相邻 float4/线程）+ P 读 __ldcs（一次性流读）+ C 写 __stcs；
// C 读用默认加载（末片刚写、L2 热）。sk>=2 由调用方保证。
__global__ void swsk_reduce_direct_kernel(const float* __restrict__ P,
                                          float* __restrict__ C,
                                          long long mn4, int sk) {
    const long long idx4 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long i0 = idx4 * 4;
    if (i0 < mn4) {
        const bool has1 = (i0 + 1) < mn4;
        const bool has2 = (i0 + 2) < mn4;
        const bool has3 = (i0 + 3) < mn4;
        const float4* p4 = reinterpret_cast<const float4*>(P);
        float4* c4 = reinterpret_cast<float4*>(C);
        float4 acc0 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 acc1 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 acc2 = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 acc3 = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int s = 0; s < sk - 1; ++s) {
            const float4 v0 = __ldcs(&p4[s * mn4 + i0]);
            acc0.x += v0.x; acc0.y += v0.y; acc0.z += v0.z; acc0.w += v0.w;
            if (has1) {
                const float4 v1 = __ldcs(&p4[s * mn4 + i0 + 1]);
                acc1.x += v1.x; acc1.y += v1.y; acc1.z += v1.z; acc1.w += v1.w;
            }
            if (has2) {
                const float4 v2 = __ldcs(&p4[s * mn4 + i0 + 2]);
                acc2.x += v2.x; acc2.y += v2.y; acc2.z += v2.z; acc2.w += v2.w;
            }
            if (has3) {
                const float4 v3 = __ldcs(&p4[s * mn4 + i0 + 3]);
                acc3.x += v3.x; acc3.y += v3.y; acc3.z += v3.z; acc3.w += v3.w;
            }
        }
        // 末源 = C（链序：全 P 路径的 P_{sk-1} 位置）
        const float4 u0 = c4[i0];
        acc0.x += u0.x; acc0.y += u0.y; acc0.z += u0.z; acc0.w += u0.w;
        if (has1) {
            const float4 u1 = c4[i0 + 1];
            acc1.x += u1.x; acc1.y += u1.y; acc1.z += u1.z; acc1.w += u1.w;
        }
        if (has2) {
            const float4 u2 = c4[i0 + 2];
            acc2.x += u2.x; acc2.y += u2.y; acc2.z += u2.z; acc2.w += u2.w;
        }
        if (has3) {
            const float4 u3 = c4[i0 + 3];
            acc3.x += u3.x; acc3.y += u3.y; acc3.z += u3.z; acc3.w += u3.w;
        }
        __stcs(&c4[i0], acc0);
        if (has1) __stcs(&c4[i0 + 1], acc1);
        if (has2) __stcs(&c4[i0 + 2], acc2);
        if (has3) __stcs(&c4[i0 + 3], acc3);
    }
}

}  // namespace

namespace sgemm {
namespace detail {

// ---- 确定性归约：C[i] = sum_s P[s*MN + i]，固定 z 序，每线程 1x float4 --
// AR009 提升：自匿名命名空间 launch_reduce 暴露为跨单元入口（wsk 共享，
// swsk/wsk 数值路径单一来源；实现零改动，仅命名空间与声明位置变化）
void swsk_reduce(const float* P, float* C, long long mn, int sk) {
    const long long mn4 = mn / 4;   // 主路径 N%4==0 ⇒ M*N 整除 4
    const int block = 256;
    if (sgemm::g_reduce_ilp2 == 3) {
        // v3（AR010 T007 --rv2=3）：ILP4 + __ldcs/__stcs（逐位等价，见 kernel 注释）
        const long long quads = (mn4 + 3) / 4;
        const long long grid = (quads + block - 1) / block;
        swsk_reduce_v3_kernel<<<(unsigned)grid, block>>>(P, C, mn4, sk);
    } else if (sgemm::g_reduce_ilp2) {
        // v2（AR010 --rv2=1）：每线程 2 相邻 f4（MLP x2，逐位等价）
        const long long pairs = (mn4 + 1) / 2;
        const long long grid = (pairs + block - 1) / block;
        swsk_reduce_v2_kernel<<<(unsigned)grid, block>>>(P, C, mn4, sk);
    } else {
        // v1（默认，AR008 以来语义/实现零改动）
        const long long grid = (mn4 + block - 1) / block;
        swsk_reduce_kernel<<<(unsigned)grid, block>>>(P, C, mn4, sk);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[swsk_reduce] launch failed: %s\n",
                     cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

// ---- 归约 direct（AR010 T007 dsk last-slice-direct 主路径）------------------
// 前置：P 含 sk-1 个切片（末片已由主体直写 C）；sk>=2；mn%4==0。
// 逐位等价：C = ((0+P_0)+…+P_{sk-2})+C，链序与全 P 归约 v1 完全一致。
void swsk_reduce_direct(const float* P, float* C, long long mn, int sk) {
    const long long mn4 = mn / 4;
    const int block = 256;
    const long long quads = (mn4 + 3) / 4;
    const long long grid = (quads + block - 1) / block;
    swsk_reduce_direct_kernel<<<(unsigned)grid, block>>>(P, C, mn4, sk);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[swsk_reduce_direct] launch failed: %s\n",
                     cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace detail
}  // namespace sgemm

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
    // ② 确定性归约：P → C（固定 z 序；AR009 经 detail::swsk_reduce 共享）
    sgemm::detail::swsk_reduce(g_ws.p, C, (long long)M * N, sk);
}
