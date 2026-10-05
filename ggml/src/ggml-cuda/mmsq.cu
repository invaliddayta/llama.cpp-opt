#include "mmsq.cuh"
#include "mmsq-kernels.cuh"

#include <cstdlib>

namespace {

struct mmsq_env {
    int n_min = 5;
    int n_max = 16;
    int split_target = 4;
    bool reuse_x = true;
    bool enabled = true;
    mmsq_env() {
        if (const char * e = getenv("GGML_CUDA_MMSQ")) {
            enabled = atoi(e) != 0;
        }
        if (const char * e = getenv("GGML_CUDA_MMSQ_NMIN")) {
            n_min = atoi(e);
        }
        if (const char * e = getenv("GGML_CUDA_MMSQ_REUSE_X")) {
            reuse_x = atoi(e) != 0;
        }
        if (const char * e = getenv("GGML_CUDA_MMSQ_SPLIT")) {
            split_target = atoi(e);
        }
        if (const char * e = getenv("GGML_CUDA_MMSQ_NMAX")) {
            n_max = atoi(e);
        }
    }
};

static const mmsq_env & get_env() {
    static mmsq_env env;
    return env;
}

static bool type_supported(ggml_type t) {
    return t == GGML_TYPE_IQ4_XS || t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K;
}

// kernel shape per (type, n8 tiles): warps along K, min CTAs per SM
template <mmsq::qtype T, int NT> struct launch_cfg          { static constexpr int KW = 2, MINB = NT == 1 ? 8 : 4; };
template <int NT> struct launch_cfg<mmsq::qtype::q4_K, NT>  { static constexpr int KW = 4, MINB = 3; };
template <int NT> struct launch_cfg<mmsq::qtype::q6_K, NT>  { static constexpr int KW = 4, MINB = 3; };

// split-K tile counters: zeroed once, each fixup resets its counter after use
constexpr int MMSQ_MAX_TILES = 1 << 16;

// each stream zeroes its own slice in stream order on first use (a captured memset only re-zeroes that slice)
static int * get_counters(ggml_backend_cuda_context & ctx) {
    if (ctx.mmsq_counters == nullptr) {
        CUDA_CHECK(cudaMalloc(&ctx.mmsq_counters, (size_t) GGML_CUDA_MAX_STREAMS * MMSQ_MAX_TILES * sizeof(int)));
    }
    int * slice = ctx.mmsq_counters + (size_t) ctx.curr_stream_no * MMSQ_MAX_TILES;
    if (!ctx.mmsq_counters_init[ctx.curr_stream_no]) {
        CUDA_CHECK(cudaMemsetAsync(slice, 0, MMSQ_MAX_TILES * sizeof(int), ctx.stream()));
        ctx.mmsq_counters_init[ctx.curr_stream_no] = true;
    }
    return slice;
}

template <mmsq::qtype T, int NT>
void launch(const uint8_t * W, int64_t row_bytes, const uint8_t * XF, int M, int K, int N, int64_t stride_dst_col,
            int nsm, int * counters, cudaStream_t stream, ggml_cuda_pool & pool, float * dst) {
    constexpr int KW   = launch_cfg<T, NT>::KW;
    constexpr int MINB = launch_cfg<T, NT>::MINB;
    constexpr int smem = mmsq::cfg<T, KW, 2>::SMEM;
    static_assert(smem <= 48 * 1024, "mmsq: dynamic smem above the default limit needs cudaFuncSetAttribute");

    const int nsb    = K / mmsq::QK;
    const int ctas_m = (M + 15) / 16;
    int splits = 1;
    while (ctas_m * splits < get_env().split_target * nsm && nsb / (splits * 2) >= KW) {
        splits *= 2;
    }
    if (ctas_m > MMSQ_MAX_TILES) {
        splits = 1;
    }
    const int sbps = ((nsb + splits - 1) / splits + KW - 1) / KW * KW;
    splits = (nsb + sbps - 1) / sbps;

    if (splits == 1) {
        mmsq::mul_mat<T, NT, KW, MINB, 2><<<dim3(ctas_m, 1), KW * 32, smem, stream>>>(W, row_bytes, XF, dst, M, K, N, stride_dst_col, nullptr, nullptr, sbps);
        return;
    }
    ggml_cuda_pool_alloc<float> part(pool, (size_t) splits * N * M);
    mmsq::mul_mat<T, NT, KW, MINB, 2><<<dim3(ctas_m, splits), KW * 32, smem, stream>>>(W, row_bytes, XF, dst, M, K, N, stride_dst_col, part.get(), counters, sbps);
}

template <mmsq::qtype T>
void launch_n(const uint8_t * W, int64_t row_bytes, const uint8_t * XF, int M, int K, int N, int64_t stride_dst_col,
              int nsm, int * counters, cudaStream_t stream, ggml_cuda_pool & pool, float * dst) {
    if (N <= 8) {
        launch<T, 1>(W, row_bytes, XF, M, K, N, stride_dst_col, nsm, counters, stream, pool, dst);
    } else {
        launch<T, 2>(W, row_bytes, XF, M, K, N, stride_dst_col, nsm, counters, stream, pool, dst);
    }
}

} // namespace

bool ggml_cuda_should_use_mmsq(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    const mmsq_env & env = get_env();
    if (!env.enabled || GGML_CUDA_CC_IS_AMD(cc) || GGML_CUDA_CC_IS_MTHREADS(cc) || cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    if (!type_supported(src0->type) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t N = src1->ne[1];
    if (N < env.n_min || N > env.n_max || N > 16) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] % mmsq::QK != 0 || src0->nb[0] != ggml_type_size(src0->type) || src0->nb[1] % 4 != 0) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || src1->nb[1] % 16 != 0 || ((uintptr_t) src1->data) % 16 != 0) {
        return false;
    }
    if (dst->nb[0] != sizeof(float)) {
        return false;
    }
    // runs are copied with 16B (8B/4B for q6_K) cp.async: rows and base must be aligned accordingly
    if (((uintptr_t) src0->data) % 16 != 0 || src0->nb[1] % (src0->type == GGML_TYPE_Q6_K ? 8 : 16) != 0) {
        return false;
    }
    return true;
}

void ggml_cuda_mul_mat_sq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int M = (int) src0->ne[1];
    const int K = (int) src0->ne[0];
    const int N = (int) src1->ne[1];
    const int nsb = K / mmsq::QK;
    const int ntiles = (N + 7) / 8;
    cudaStream_t stream = ctx.stream();
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;

    // persistent activation buffer, sized for the largest supported shape so CUDA graphs keep a fixed pointer
    constexpr int MAX_NSB = 128;
    const size_t xf_size = (size_t) 2 * MAX_NSB * mmsq::FSB;
    // the persistent buffer and reuse are restricted to stream 0; other streams use pool scratch
    const bool persistent = nsb <= MAX_NSB && ctx.curr_stream_no == 0;
    ggml_cuda_pool_alloc<uint8_t> xf_pool(ctx.pool());
    uint8_t * xf = nullptr;
    if (persistent) {
        if (ctx.mmsq_xf == nullptr) {
            CUDA_CHECK(cudaMalloc(&ctx.mmsq_xf, xf_size));
        }
        xf = (uint8_t *) ctx.mmsq_xf;
    } else {
        xf = xf_pool.alloc((size_t) ntiles * nsb * mmsq::FSB);
    }
    const bool reuse = get_env().reuse_x && persistent && ctx.mmsq_x_valid &&
        ctx.mmsq_x_src == src1->data && ctx.mmsq_x_ne[0] == K && ctx.mmsq_x_ne[1] == N && ctx.mmsq_x_nb1 == (int64_t) src1->nb[1];
    if (!reuse) {
        mmsq::quantize_x<<<dim3(nsb, ntiles), 32, 0, stream>>>((const float *) src1->data, xf, K, N, src1->nb[1] / sizeof(float));
    }
    if (persistent) {
        ctx.mmsq_x_valid = true;
        ctx.mmsq_x_src = src1->data;
        ctx.mmsq_x_end = (const char *) src1->data + ggml_nbytes(src1);
        ctx.mmsq_x_ne[0] = K;
        ctx.mmsq_x_ne[1] = N;
        ctx.mmsq_x_nb1 = src1->nb[1];
    }
    int * counters = get_counters(ctx);

    const uint8_t * W = (const uint8_t *) src0->data;
    const int64_t row_bytes = src0->nb[1];
    const int64_t stride_dst_col = dst->nb[1] / sizeof(float);
    float * d = (float *) dst->data;

    switch (src0->type) {
        case GGML_TYPE_IQ4_XS: launch_n<mmsq::qtype::iq4_xs>(W, row_bytes, xf, M, K, N, stride_dst_col, nsm, counters, stream, ctx.pool(), d); break;
        case GGML_TYPE_Q4_K:   launch_n<mmsq::qtype::q4_K>  (W, row_bytes, xf, M, K, N, stride_dst_col, nsm, counters, stream, ctx.pool(), d); break;
        case GGML_TYPE_Q5_K:   launch_n<mmsq::qtype::q5_K>  (W, row_bytes, xf, M, K, N, stride_dst_col, nsm, counters, stream, ctx.pool(), d); break;
        case GGML_TYPE_Q6_K:   launch_n<mmsq::qtype::q6_K>  (W, row_bytes, xf, M, K, N, stride_dst_col, nsm, counters, stream, ctx.pool(), d); break;
        default: GGML_ABORT("mmsq: unsupported type");
    }
    CUDA_CHECK(cudaGetLastError());
}

// F32 weights with few rows (e.g. ssm_alpha/beta): one warp per row, K split over CTAs, last CTA of a row group sums the splits
static constexpr int F32S_ROWS = 4;
static constexpr int F32S_KC   = 256;

template <int NC>
static __global__ void __launch_bounds__(F32S_ROWS * 32) mul_mat_f32_small(
        const float * __restrict__ W, const int64_t stride_w, const float * __restrict__ X, const int64_t stride_x,
        float * __restrict__ dst, const int64_t stride_dst, float * __restrict__ part, int * __restrict__ counters,
        const int M, const int K, const int N) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row  = blockIdx.x * F32S_ROWS + warp;
    const int k0   = blockIdx.y * F32S_KC;
    const int k1   = min(K, k0 + F32S_KC);

    float acc[NC];
#pragma unroll
    for (int n = 0; n < NC; ++n) {
        acc[n] = 0.0f;
    }
    if (row < M) {
        constexpr int IT = F32S_KC / 128;
        float4 w[IT];
#pragma unroll
        for (int it = 0; it < IT; ++it) {
            const int k = k0 + 128 * it + 4 * lane;
            w[it] = k < k1 ? __ldg((const float4 *) (W + (int64_t) row * stride_w + k)) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int it = 0; it < IT; ++it) {
            const int k = k0 + 128 * it + 4 * lane;
            if (k < k1) {
#pragma unroll
                for (int n = 0; n < NC; ++n) {
                    if (n < N) {
                        const float4 x = __ldg((const float4 *) (X + (int64_t) n * stride_x + k));
                        acc[n] += w[it].x * x.x + w[it].y * x.y + w[it].z * x.z + w[it].w * x.w;
                    }
                }
            }
        }
    }
#pragma unroll
    for (int n = 0; n < NC; ++n) {
        acc[n] = warp_reduce_sum(acc[n]);
    }

    if (gridDim.y == 1) {
        if (row < M && lane == 0) {
#pragma unroll
            for (int n = 0; n < NC; ++n) {
                if (n < N) {
                    dst[(int64_t) n * stride_dst + row] = acc[n];
                }
            }
        }
        return;
    }

    const int64_t psplit = (int64_t) N * M;
    if (row < M && lane == 0) {
#pragma unroll
        for (int n = 0; n < NC; ++n) {
            if (n < N) {
                part[blockIdx.y * psplit + (int64_t) n * M + row] = acc[n];
            }
        }
    }
    __threadfence();
    __syncthreads();
    __shared__ int prev;
    if (threadIdx.x == 0) {
        prev = atomicAdd(&counters[blockIdx.x], 1);
    }
    __syncthreads();
    if (prev != (int) gridDim.y - 1) {
        return;
    }
    __threadfence();
    for (int i = threadIdx.x; i < F32S_ROWS * N; i += blockDim.x) {
        const int r = blockIdx.x * F32S_ROWS + i % F32S_ROWS;
        const int n = i / F32S_ROWS;
        if (r < M) {
            float v = 0.0f;
            for (int s = 0; s < (int) gridDim.y; ++s) {
                v += __ldcg(part + s * psplit + (int64_t) n * M + r);
            }
            dst[(int64_t) n * stride_dst + r] = v;
        }
    }
    if (threadIdx.x == 0) {
        counters[blockIdx.x] = 0;
    }
}

bool ggml_cuda_should_use_mul_mat_f32_small(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    static const bool enabled = getenv("GGML_CUDA_F32_SMALL") == nullptr || atoi(getenv("GGML_CUDA_F32_SMALL")) != 0;
    if (!enabled || !GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // K splits go to gridDim.y
    if (src0->ne[1] > 512 || src1->ne[1] > 16 || src0->ne[0] % 4 != 0 || src0->ne[0] > (int64_t) 65535 * F32S_KC) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    if (src0->nb[1] % 16 != 0 || src1->nb[1] % 16 != 0 || ((uintptr_t) src0->data) % 16 != 0 || ((uintptr_t) src1->data) % 16 != 0) {
        return false;
    }
    return true;
}

void ggml_cuda_mul_mat_f32_small(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int M = (int) src0->ne[1];
    const int K = (int) src0->ne[0];
    const int N = (int) src1->ne[1];
    cudaStream_t stream = ctx.stream();

    const int row_groups = (M + F32S_ROWS - 1) / F32S_ROWS;
    const int splits     = (int) (((int64_t) K + F32S_KC - 1) / F32S_KC);
    GGML_ASSERT(row_groups <= MMSQ_MAX_TILES && splits <= 65535);

    ggml_cuda_pool_alloc<float> part(ctx.pool());
    int * counters = nullptr;
    if (splits > 1) {
        part.alloc((size_t) splits * N * M);
        counters = get_counters(ctx);
    }
    const dim3 grid(row_groups, splits);
    const float * W = (const float *) src0->data;
    const float * X = (const float *) src1->data;
    float * d = (float *) dst->data;
    const int64_t sw = src0->nb[1] / sizeof(float), sx = src1->nb[1] / sizeof(float), sd = dst->nb[1] / sizeof(float);
    if (N <= 1) {
        mul_mat_f32_small<1><<<grid, F32S_ROWS * 32, 0, stream>>>(W, sw, X, sx, d, sd, part.ptr, counters, M, K, N);
    } else if (N <= 4) {
        mul_mat_f32_small<4><<<grid, F32S_ROWS * 32, 0, stream>>>(W, sw, X, sx, d, sd, part.ptr, counters, M, K, N);
    } else if (N <= 8) {
        mul_mat_f32_small<8><<<grid, F32S_ROWS * 32, 0, stream>>>(W, sw, X, sx, d, sd, part.ptr, counters, M, K, N);
    } else {
        mul_mat_f32_small<16><<<grid, F32S_ROWS * 32, 0, stream>>>(W, sw, X, sx, d, sd, part.ptr, counters, M, K, N);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mmsq_note_writes(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int first, int last) {
    if (!ctx.mmsq_x_valid) {
        return;
    }
    if (ctx.curr_stream_no != 0) {
        ctx.mmsq_x_valid = false;
        return;
    }
    for (int i = first; i <= last; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        // optimizer steps also write into their sources
        if (t->op == GGML_OP_OPT_STEP_ADAMW || t->op == GGML_OP_OPT_STEP_SGD) {
            ctx.mmsq_x_valid = false;
            return;
        }
        const char * beg = (const char *) t->data;
        const char * end = beg + ggml_nbytes(t);
        if (beg < (const char *) ctx.mmsq_x_end && (const char *) ctx.mmsq_x_src < end) {
            ctx.mmsq_x_valid = false;
            return;
        }
    }
}
