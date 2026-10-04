// =====================================================================
// sgemm_cpasync.cu — Kernel 5：cp.async 双缓冲流水（AR006）
// ---------------------------------------------------------------
// 提供两个可消融的布局方案（AR006 T003 要求实测定稿）：
//
// 【方案甲 sgemm_cpasync】A、B 均用 cp.async 直拷（不转置）
//   * As[2][128][BK+4]：行距 12 floats=48B（保持 16B 对齐，PAD=4 强制——
//     PAD=5 会破坏 cp.async 的 16B dst 对齐）。代价：a-frag 标量读
//     As[ty*8+i][kk]，warp 内两行相差 8*12=96 ≡ 0 (mod 32) → 2-way
//     bank conflict【已知项，ncu 复核】；cp.async 不能转置，这是直拷的
//     固有取舍。
//   * Bs[2][8][128]：XOR swizzle（同 vec4），b-frag = 2x float4，0 冲突。
//   * 边界：cp.async 16B + zfill=16（主路径保证 N%4==0、K%4==0 ⇒ 每个
//     16B quad 全在界内或全在界外，一次谓词即可，无需部分拷贝）。
//
// 【方案乙 sgemm_cpasync_v2】B 用 cp.async+swizzle；A 保留 float4+转置写入
//   （vec4 路径，0 冲突），并以"寄存器预取下一 tile"隐藏 A 的全局延迟：
//   每线程每 tile 只搬 1 个 float4 → 可整块缓存在寄存器中，提前一拍发射。
//
// 双缓冲流水（2-stage，两方案共用骨架）：
//   issue(tile 0) -> commit
//   for t: [甲:issue(t+1)] [乙:store As(t); prefetch A(t+1); issue B(t+1)]
//          -> commit -> wait_prior(1)（末 tile wait_prior(0)）
//          -> __syncthreads -> compute(t) -> __syncthreads
//   两个 __syncthreads 保证：buf 交替写读无竞争（racecheck 抽查，AR006 T004）。
// 主路径条件：N%4==0 且 K%4==0 且指针 16B 对齐；否则回退 tile2d（任意尺寸）。
// 预期性能：~6.61 TFLOPS（≥97% cuBLAS FP32）；
//   预期 stall 主因从 long scoreboard 转为 barrier/依赖等待（E9）。
// =====================================================================
#include "sgemm_kernels.h"
#include <cuda_pipeline_primitives.h>
#include <cstdio>
#include <cstdint>

namespace {

constexpr int BM = 128, BN = 128, BK = 8;
constexpr int TM = 8,   TN = 8;

union Frag8 {
    float4 v4[2];
    float  f[8];
};

// cp.async 16B：valid=false 时 zfill=16（只写零，不读全局）
__device__ __forceinline__ void cp_async16(float* dst, const float* src, bool valid) {
    __pipeline_memcpy_async(dst, src, 16, valid ? 0 : 16);
}

// 64-FMA 外积核心（片段来源由调用方决定）
__device__ __forceinline__ void fma64(float c[TM][TN], const Frag8& a, const Frag8& b) {
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            c[i][j] = a.f[i] * b.f[j] + c[i][j];
}

// C 回写：每行 2x float4（N%4==0 ⇒ quad 全有或全无）
__device__ __forceinline__ void store_c(float* C, const float c[TM][TN],
                                        int row0, int col0, int M, int N) {
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = row0 + i;
        if (row < M) {
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int col = col0 + q * 4;
                if (col < N) {
                    float4 v;
                    v.x = c[i][q * 4 + 0];
                    v.y = c[i][q * 4 + 1];
                    v.z = c[i][q * 4 + 2];
                    v.w = c[i][q * 4 + 3];
                    *reinterpret_cast<float4*>(&C[(long long)row * N + col]) = v;
                }
            }
        }
    }
}

// =====================================================================
// 方案甲：A、B 全 cp.async 直拷
// =====================================================================
__global__ __launch_bounds__(256)
void sgemm_cpasync_kernel(const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C,
                     int M, int N, int K) {
    constexpr int PAD_A = 4;                       // 12 floats/行 = 48B，16B 对齐保持
    __shared__ __align__(16) float As[2][BM][BK + PAD_A];
    __shared__ __align__(16) float Bs[2][BK][BN];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;  // block(16,16)
    const int tid = ty * 16 + tx;

    float c[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) c[i][j] = 0.0f;

    // 每线程每 tile 的搬运任务（与 vec4 相同划分）
    const int ld_a_row  = tid >> 1;                // [0,128)
    const int ld_a_kq   = tid & 1;                 // k quad：0/4
    const int ld_b_krow = tid >> 5;                // [0,8)
    const int ld_b_unit = tid & 31;                // 16B 单位 [0,32)

    const int a_glb_row = by * BM + ld_a_row;
    const int b_glb_col = bx * BN + ld_b_unit * 4;
    const int num_tiles = (K + BK - 1) / BK;

    // 发射 tile t 的 A+B 到 buffer buf（全 cp.async，边界 zfill=16）
    auto issue_tile = [&](int t, int buf) {
        const int k0 = t * BK;
        // A：16B 沿 K（直拷，不转置）
        {
            const bool valid = (a_glb_row < M) && (k0 + ld_a_kq * 4 < K);
            // zfill=16 时不读 src；越界时传全局基地址保证指针始终合法
            const float* src = valid
                ? A + (long long)a_glb_row * K + k0 + ld_a_kq * 4 : A;
            cp_async16(&As[buf][ld_a_row][ld_a_kq * 4], src, valid);
        }
        // B：16B 沿 N，swizzle 写入（物理单位 = 逻辑单位 ^ (krow&7)）
        {
            const bool valid = (k0 + ld_b_krow < K) && (b_glb_col < N);
            float4* dst = &reinterpret_cast<float4*>(&Bs[buf][ld_b_krow][0])
                              [ld_b_unit ^ (ld_b_krow & 7)];
            const float* src = valid
                ? B + (long long)(k0 + ld_b_krow) * N + b_glb_col : B;
            cp_async16(reinterpret_cast<float*>(dst), src, valid);
        }
    };

    issue_tile(0, 0);
    __pipeline_commit();

    for (int t = 0; t < num_tiles; ++t) {
        const int buf = t & 1;
        if (t + 1 < num_tiles) {
            issue_tile(t + 1, buf ^ 1);            // 预取下一 tile
            __pipeline_commit();
            __pipeline_wait_prior(1);              // 等 tile t 就绪（保留最新一组）
        } else {
            __pipeline_wait_prior(0);              // 末 tile：全部等完
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            Frag8 a, b;
            // a-frag：标量 x8（直拷布局的固有 2-way 冲突，见文件头）
#pragma unroll
            for (int i = 0; i < TM; ++i) a.f[i] = As[buf][ty * TM + i][kk];
            // b-frag：2x float4 + swizzle（0 冲突）
            const float4* bs_row = reinterpret_cast<const float4*>(&Bs[buf][kk][0]);
            const int sw = kk & 7;
            b.v4[0] = bs_row[(tx * 2) ^ sw];
            b.v4[1] = bs_row[(tx * 2 + 1) ^ sw];
            fma64(c, a, b);
        }
        __syncthreads();                           // buf 在 t+2 复用前的写读隔离
    }

    store_c(C, c, by * BM + ty * TM, bx * BN + tx * TN, M, N);
}

// =====================================================================
// 方案乙：A 用 float4+转置写入（寄存器预取）；B 用 cp.async+swizzle
// =====================================================================
__global__ __launch_bounds__(256)
void sgemm_cpasync_v2_kernel(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int M, int N, int K) {
    constexpr int PAD_A = 4;                       // As 转置 [BK][BM+PAD]：132 floats/行
    __shared__ __align__(16) float As[2][BK][BM + PAD_A];
    __shared__ __align__(16) float Bs[2][BK][BN];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 16 + tx;

    float c[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) c[i][j] = 0.0f;

    const int ld_a_row  = tid >> 1;
    const int ld_a_kq   = tid & 1;
    const int ld_b_krow = tid >> 5;
    const int ld_b_unit = tid & 31;

    const int a_glb_row = by * BM + ld_a_row;
    const int b_glb_col = bx * BN + ld_b_unit * 4;
    const int num_tiles = (K + BK - 1) / BK;

    // A 的寄存器预取：每线程每 tile 恰好 1 个 float4
    auto load_a_tile = [&](int t) -> float4 {
        const int k0 = t * BK;
        const bool valid = (a_glb_row < M) && (k0 + ld_a_kq * 4 < K);
        if (!valid) return make_float4(0.f, 0.f, 0.f, 0.f);
        return *reinterpret_cast<const float4*>(
            A + (long long)a_glb_row * K + k0 + ld_a_kq * 4);
    };
    // A 转置散射写入 As[buf]（bank = (kq*4+kk)*4 + row → 0 冲突，见 vec4 注释）
    auto store_a_tile = [&](int buf, const float4& av) {
        const float* af = reinterpret_cast<const float*>(&av);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            As[buf][ld_a_kq * 4 + kk][ld_a_row] = af[kk];
        }
    };
    // B：cp.async + swizzle
    auto issue_b_tile = [&](int t, int buf) {
        const int k0 = t * BK;
        const bool valid = (k0 + ld_b_krow < K) && (b_glb_col < N);
        float4* dst = &reinterpret_cast<float4*>(&Bs[buf][ld_b_krow][0])
                           [ld_b_unit ^ (ld_b_krow & 7)];
        const float* src = valid
            ? B + (long long)(k0 + ld_b_krow) * N + b_glb_col : B;
        cp_async16(reinterpret_cast<float*>(dst), src, valid);
    };

    issue_b_tile(0, 0);
    __pipeline_commit();
    float4 a_pref = load_a_tile(0);                // 寄存器预取 A tile 0

    for (int t = 0; t < num_tiles; ++t) {
        const int buf = t & 1;
        store_a_tile(buf, a_pref);                 // A tile t 入 smem（上一轮已加载）
        if (t + 1 < num_tiles) {
            a_pref = load_a_tile(t + 1);           // 提前发射 A(t+1) 全局读（隐藏于 compute）
            issue_b_tile(t + 1, buf ^ 1);          // 预取 B(t+1)
            __pipeline_commit();
            __pipeline_wait_prior(1);              // 等 B(t) 就绪（A 为普通访存，sync 即可见）
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            Frag8 a, b;
            const float4* as_row = reinterpret_cast<const float4*>(&As[buf][kk][0]);
            a.v4[0] = as_row[ty * 2];
            a.v4[1] = as_row[ty * 2 + 1];
            const float4* bs_row = reinterpret_cast<const float4*>(&Bs[buf][kk][0]);
            const int sw = kk & 7;
            b.v4[0] = bs_row[(tx * 2) ^ sw];
            b.v4[1] = bs_row[(tx * 2 + 1) ^ sw];
            fma64(c, a, b);
        }
        __syncthreads();
    }

    store_c(C, c, by * BM + ty * TM, bx * BN + tx * TN, M, N);
}

// 主路径对齐条件（两个方案共用）
inline bool cpasync_main_path_ok(const float* A, const float* B, const float* C,
                                 int N, int K) {
    return (N % 4 == 0) && (K % 4 == 0) &&
           (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
           (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
           (reinterpret_cast<uintptr_t>(C) % 16 == 0);
}

}  // namespace

void sgemm_cpasync(const float* A, const float* B, float* C, int M, int N, int K) {
    if (!cpasync_main_path_ok(A, B, C, N, K)) {
        if (sgemm::g_verbose) {
            std::printf("[cpasync] size %dx%dx%d -> scalar fallback (tile2d)\n", M, N, K);
        }
        sgemm_2d_tile(A, B, C, M, N, K);
        return;
    }
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_cpasync_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[sgemm_cpasync] launch failed: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

void sgemm_cpasync_v2(const float* A, const float* B, float* C, int M, int N, int K) {
    if (!cpasync_main_path_ok(A, B, C, N, K)) {
        if (sgemm::g_verbose) {
            std::printf("[cpasync2] size %dx%dx%d -> scalar fallback (tile2d)\n", M, N, K);
        }
        sgemm_2d_tile(A, B, C, M, N, K);
        return;
    }
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_cpasync_v2_kernel<<<grid, block>>>(A, B, C, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[sgemm_cpasync_v2] launch failed: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}
