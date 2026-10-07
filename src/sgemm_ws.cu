// =====================================================================
// sgemm_ws.cu — Kernel 7：warp 专属化 producer/consumer 软件流水（AR008）
// ---------------------------------------------------------------------
// 动机：AR007 ncu 结论（无计数器环境下以消融间接检验）——大尺寸稳态循环
// 中 FFMA 发射槽占比 ~55%，LDG/STS 指令与地址计算挤占消费者发射槽。本版
// 把全局搬运整段剥离给专职 producer warp，consumer 只剩 LDS.128+FFMA——
// pre-Ampere（sm_75 无 cp.async、无 mbarrier）上的"软件版 warp
// specialization"（Hopper 库标配结构的软件复刻）。
//
// 线程拓扑（PW+8 warp，PW∈{1,2} 消融轴，默认 2）：
//   - producer warp 0..PW-1（32/64 线程）：LDG float4 → 寄存器 → STS 灌 smem 环
//   - consumer warp PW..PW+7（256 线程）：纯 LDS.128 + FFMA + 回写 C
//   - consumer 输出划分与 swpipe 完全一致：tid_c=(ty,tx) 16x16，每线程
//     TM=TN=8（64 acc）→ 128x128 tile（bank 冲突结论第三次继承）
//   - PW=1 时 producer 每线程搬 8+8 个 float4（64 寄存器预取缓冲），
//     单 warp 搬运吞吐是否够 8 consumer warp 吃——T008 消融问题
//
// tile 与 smem 布局（继承 swpipe/vec4，见 sgemm_swpipe.cu 头注释）：
//   - BM=BN=128, BK=8；A 转置 As[BK][BM+PAD_A=4]（PAD 消转置写/片段读
//     冲突且保持 16B 对齐）；B 保持 Bs[BK][BN]，16B 单位 XOR swizzle
//     （unit ^ (krow & 7)）消除 b-frag stride-8 4-way 冲突。
//   - 环：STAGES 个 Stage（每级 8.3KB），STAGES=3 → 24960B ≤ 32KB/block
//     （LB=2 双 block 驻留需 ≤ 65536/2，满足）。环深 3 覆盖 ~2 tile 的
//     DRAM 延迟抖动：consumer 算 slot k 时 producer 已灌 k+1/k+2。
//   - producer 搬运划分（每 tile 每线程 A_PER+B_PER 个 float4，A_PER=256/NPROD）：
//     PW=2：A quad 编号 a=4p+q → row=a>>1, kq=a&1（共 256 quad = 128 行 x 2）；
//            B unit 编号 b=4p+q → krow=b>>5, unit=b&31（共 256 unit = 8 x 32）。
//     PW=1：编号同式，p∈0..31，A_PER=B_PER=8。
//
// 屏障契约（PTX named barriers，sm_75 合法；id 0 保留不用）：
//   full[s]  = id 1+s : producer bar.arrive（非阻塞通知已灌好），
//                        consumer bar.sync（阻塞等待就绪）
//   empty[s] = id 1+STAGES+s : consumer bar.arrive（通知已读完），
//                        producer bar.sync（阻塞等环位空出，仅 t>=STAGES）
//   到达计数恒 = 块内线程数（PW=2: 320 = 64 producer + 256 consumer；
//   PW=1: 288 = 32 + 256）——两指令点合计补齐后屏障释放并自动复位。
//   环满/空由屏障到达语义天然保证，无计数变量、无 smem 竞态面。
//   首轮 t<STAGES 时 producer 跳过 empty 等待（环初态视为空）；
//   短 K（num_tiles<STAGES）同理不触达未用屏障 id，无死锁路径。
//
// 流水阶段与缓冲数：2 阶段（搬运/计算）x STAGES 级环形缓冲（默认 3）。
//   尾声：consumer 独占回写 C（每行 2x float4，行/列守卫）；producer
//   空转退出（不参与尾声屏障，无死锁面）。
//
// 资源契约：寄存器按 kernel 统一分配 = max(producer 路径,
// consumer 路径 64 acc+16 frag+寻址)；__launch_bounds__(32*PW+256, LB)
// 消融（AGENTS §4 军规）：LB=1（默认）无压力；LB=2 收紧寄存器上限
// （PW=2: 102 regs；PW=1: 113 regs）——若 ptxas 0 spill 可达则双 block
// 驻留（PW=2: 62.5% 占用；PW=1: 320→288 线程双驻留 56.3%）（本设计
// 核心实验变量，spill 与否以 build.log -Xptxas -v 为准，如实记录）。
//
// spill 豁免记录（AR011 T009 清偿 AGENTS §4 军规 4）：stages=2 消融实例
// ws_kernel<PW=2,STAGES=2,WP∈{1,2}> 存在 8B spill stores / 4B spill
// loads（96 regs 锁定下环索引边缘寄存器，bench/test 双目标共 4 处编译
// 事件，build.log 留档）；现役默认 stages=3 实例全 0 spill。stages=2 为
// 纯消融旋钮（AR008 实测全尺寸劣于 stages=3，非默认路径不做性能主张），
// 豁免依据 = 消融可比性优先（改动代码会改变消融对照物）。
// =====================================================================
#include "sgemm_kernels.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>

namespace sgemm {
int g_ws_stages      = 3;   // --stages 环深度消融（{2,3}，默认 3）
int g_ws_min_blocks  = 1;   // --lb 对 ws 的 __launch_bounds__ 消融（{1,2}）
int g_ws_prod_warps  = 2;   // --wp producer warp 数消融（{1,2}，默认 2；2+8=320 线程）
}

namespace {

constexpr int BM = 128, BN = 128, BK = 8;
constexpr int TM = 8,   TN = 8;
constexpr int PAD_A = 4;
constexpr int NCONS = 256;      // consumer 线程数恒定（8 warp x 32）

// named barrier id 布局（0 保留给 __syncthreads 语义，本 kernel 不用）：
// full[s] = 1+s；empty[s] = 1+STAGES+s（STAGES<=3 → 最大 id 6 <= 15）
template <int STAGES>
__device__ __forceinline__ int bar_full(int s) { return 1 + s; }
template <int STAGES>
__device__ __forceinline__ int bar_empty(int s) { return 1 + STAGES + s; }

__device__ __forceinline__ void named_bar_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(count));
}
__device__ __forceinline__ void named_bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(count));
}

__device__ __forceinline__ float4 load_zero_guard(const float* base, long long idx,
                                                  bool valid) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (valid) v = *reinterpret_cast<const float4*>(base + idx);
    return v;
}

struct Stage {
    float as[BK][BM + PAD_A];   // A 转置（行距 132 floats，16B 对齐保持）
    float bs[BK][BN];           // B XOR swizzle
};

// LB = __launch_bounds__ minBlocks 消融（1: 1 block/SM 低占用；
//     2: 寄存器上限收紧，0 spill 可达则双 block 驻留）
// STAGES = 环深度消融（2/3；stage 8.3KB → 16.6/25.0KB smem）
// PW = producer warp 数消融（1/2；线程数 = 32*PW + 256，屏障 count 随之）
template <int LB, int STAGES, int PW>
__global__ __launch_bounds__(32 * PW + NCONS, LB)
void ws_kernel(const float* __restrict__ A,
               const float* __restrict__ B,
               float* __restrict__ C,
               int M, int N, int K, int num_tiles) {
    constexpr int NPROD    = 32 * PW;         // producer 线程数
    constexpr int NTHREADS = NPROD + NCONS;   // 屏障到达计数
    constexpr int A_PER    = 256 / NPROD;     // 每线程每 tile 的 A quad / B unit 数

    __shared__ __align__(16) Stage ring[STAGES];

    const int bx = blockIdx.x, by = blockIdx.y;

    if (threadIdx.x < NPROD) {
        // ================= producer（warp 0..PW-1）=======================
        const int p = threadIdx.x;               // 0..NPROD-1
        int a_row[A_PER], a_kq[A_PER], b_krow[A_PER], b_unit[A_PER];
        const int a_row_base = by * BM;
        const int b_col_base = bx * BN;
#pragma unroll
        for (int q = 0; q < A_PER; ++q) {
            const int a = p * A_PER + q;         // A quad 编号 0..255
            a_row[q] = a >> 1;
            a_kq[q]  = a & 1;
            const int b = p * A_PER + q;         // B unit 编号 0..255
            b_krow[q] = b >> 5;
            b_unit[q] = b & 31;
        }

        for (int t = 0; t < num_tiles; ++t) {
            const int s = t % STAGES;
            if (t >= STAGES) {
                // 环位 s 上 consumers 已读完（empty[s] 到齐 NTHREADS）才可覆写
                named_bar_sync(bar_empty<STAGES>(s), NTHREADS);
            }
            const int k0 = t * BK;
            // ---- LDG → 寄存器（越界零守卫，语义同 swpipe）----
            float4 areg[A_PER], breg[A_PER];
#pragma unroll
            for (int q = 0; q < A_PER; ++q) {
                areg[q] = load_zero_guard(
                    A, (long long)(a_row_base + a_row[q]) * K + k0 + a_kq[q] * 4,
                    (a_row_base + a_row[q] < M) && (k0 + a_kq[q] * 4 < K));
                breg[q] = load_zero_guard(
                    B, (long long)(k0 + b_krow[q]) * N + b_col_base + b_unit[q] * 4,
                    (k0 + b_krow[q] < K) && (b_col_base + b_unit[q] * 4 < N));
            }
            // ---- 寄存器 → smem slot s（A 转置散射；B swizzle 直拷）----
            Stage& st = ring[s];
#pragma unroll
            for (int q = 0; q < A_PER; ++q) {
                const float* af = reinterpret_cast<const float*>(&areg[q]);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    st.as[a_kq[q] * 4 + kk][a_row[q]] = af[kk];
                }
            }
#pragma unroll
            for (int q = 0; q < A_PER; ++q) {
                reinterpret_cast<float4*>(&st.bs[b_krow[q]][0])
                    [b_unit[q] ^ (b_krow[q] & 7)] = breg[q];
            }
            named_bar_arrive(bar_full<STAGES>(s), NTHREADS);  // 通知就绪（非阻塞）
        }
        // producer 尾声：空转退出（不参与回写，不触碰任何屏障）
    } else {
        // ================= consumer（warp 2-9，256 线程）===============
        const int tid_c = threadIdx.x - NPROD;   // 0..255
        const int ty = tid_c >> 4;               // 0..15（同 swpipe 16x16 划分）
        const int tx = tid_c & 15;               // 0..15

        float c[TM][TN];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) c[i][j] = 0.0f;

        for (int t = 0; t < num_tiles; ++t) {
            const int s = t % STAGES;
            named_bar_sync(bar_full<STAGES>(s), NTHREADS);   // 等 slot s 灌好
            const Stage& st = ring[s];
            // ---- 纯计算：每 k 一步 2+2 LDS.128 → 64 FFMA（同 swpipe ③）----
#pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                float4 a0, a1, b0, b1;
                const float4* as_row = reinterpret_cast<const float4*>(&st.as[kk][0]);
                a0 = as_row[ty * 2];
                a1 = as_row[ty * 2 + 1];
                const float4* bs_row = reinterpret_cast<const float4*>(&st.bs[kk][0]);
                const int sw = kk & 7;
                b0 = bs_row[(tx * 2) ^ sw];
                b1 = bs_row[(tx * 2 + 1) ^ sw];
                const float af[8] = {a0.x, a0.y, a0.z, a0.w,
                                     a1.x, a1.y, a1.z, a1.w};
                const float bf[8] = {b0.x, b0.y, b0.z, b0.w,
                                     b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j)
                        c[i][j] = af[i] * bf[j] + c[i][j];
            }
            named_bar_arrive(bar_empty<STAGES>(s), NTHREADS); // 通知读完（非阻塞）
        }

        // ---- 回写：每行 2x float4（N%4==0 ⇒ quad 全有或全无），warp 合并 --
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int row = by * BM + ty * TM + i;
            if (row < M) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const int col = bx * BN + tx * TN + q * 4;
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
}

template <int LB, int STAGES, int PW>
void launch_ws(const float* A, const float* B, float* C,
               int M, int N, int K, int num_tiles, int grid_n, int grid_m) {
    const dim3 grid(grid_n, grid_m, 1);
    const dim3 block(32 * PW + NCONS, 1, 1);
    ws_kernel<LB, STAGES, PW><<<grid, block>>>(A, B, C, M, N, K, num_tiles);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[ws] launch failed (LB=%d, STAGES=%d, PW=%d): %s\n",
                     LB, STAGES, PW, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

void sgemm_ws(const float* A, const float* B, float* C, int M, int N, int K) {
    // 主路径条件：与 swpipe 相同（float4 全局读 + 16B smem 布局依赖）
    const bool aligned =
        (N % 4 == 0) && (K % 4 == 0) &&
        (reinterpret_cast<uintptr_t>(A) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(B) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(C) % 16 == 0);
    if (!aligned) {
        if (sgemm::g_verbose) {
            std::printf("[ws] size %dx%dx%d -> scalar fallback (tile2d)\n", M, N, K);
        }
        sgemm_2d_tile(A, B, C, M, N, K);   // 谓词化回退路径（任意尺寸正确）
        return;
    }

    const int num_tiles = (K + BK - 1) / BK;
    const int grid_n = (N + BN - 1) / BN;
    const int grid_m = (M + BM - 1) / BM;

    const int st = (sgemm::g_ws_stages == 2) ? 2 : 3;
    const int lb = (sgemm::g_ws_min_blocks == 2) ? 2 : 1;
    const int pw = (sgemm::g_ws_prod_warps == 1) ? 1 : 2;

    if (lb == 1) {
        if (pw == 2) {
            if (st == 3) launch_ws<1, 3, 2>(A, B, C, M, N, K, num_tiles, grid_n, grid_m);
            else         launch_ws<1, 2, 2>(A, B, C, M, N, K, num_tiles, grid_n, grid_m);
        } else {
            if (st == 3) launch_ws<1, 3, 1>(A, B, C, M, N, K, num_tiles, grid_n, grid_m);
            else         launch_ws<1, 2, 1>(A, B, C, M, N, K, num_tiles, grid_n, grid_m);
        }
    } else {
        if (pw == 2) {
            if (st == 3) launch_ws<2, 3, 2>(A, B, C, M, N, K, num_tiles, grid_n, grid_m);
            else         launch_ws<2, 2, 2>(A, B, C, M, N, K, num_tiles, grid_n, grid_m);
        } else {
            if (st == 3) launch_ws<2, 3, 1>(A, B, C, M, N, K, num_tiles, grid_n, grid_m);
            else         launch_ws<2, 2, 1>(A, B, C, M, N, K, num_tiles, grid_n, grid_m);
        }
    }
}
