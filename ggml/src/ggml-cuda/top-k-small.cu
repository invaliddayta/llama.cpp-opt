#include "top-k-small.cuh"

#include <cfloat>

namespace {

constexpr int TK_THREADS = 256;
constexpr int TK_CHUNKS  = 16;   // CTAs per row in pass 1

// ordering: larger value first, ties -> smaller index first; empty slots (index < 0) last.
// NaN is ranked as -inf (callers pass values through tk_key).
__device__ __forceinline__ bool tk_better(float va, int ia, float vb, int ib) {
    if (ib < 0) {
        return ia >= 0;
    }
    if (ia < 0) {
        return false;
    }
    return va > vb || (va == vb && ia < ib);
}

__device__ __forceinline__ float tk_key(float v) {
    return isnan(v) ? -INFINITY : v;
}

// bitonic sort of n (power of 2) (value, index) pairs in shared memory, best first
__device__ void tk_bitonic(float * v, int * idx, int n) {
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int t = threadIdx.x; t < n / 2; t += blockDim.x) {
                const int i = (t / stride) * stride * 2 + (t % stride);
                const int j = i + stride;
                const bool dir = ((i & size) == 0);   // true: best first in this sub-sequence
                const bool swap = dir ? tk_better(v[j], idx[j], v[i], idx[i]) : tk_better(v[i], idx[i], v[j], idx[j]);
                if (swap) {
                    const float tv = v[i]; v[i] = v[j]; v[j] = tv;
                    const int   ti = idx[i]; idx[i] = idx[j]; idx[j] = ti;
                }
            }
        }
    }
    __syncthreads();
}

template <int K>
__global__ void __launch_bounds__(TK_THREADS)
top_k_small_pass1(const float * __restrict__ src, float * __restrict__ cand_v, int * __restrict__ cand_i, const int64_t ncols) {
    const int row   = blockIdx.y;
    const int chunk = blockIdx.x;
    const int64_t per   = (ncols + TK_CHUNKS - 1) / TK_CHUNKS;
    const int64_t c_beg = chunk * per;
    const int64_t c_end = min(ncols, c_beg + per);
    const float * x = src + (int64_t) row * ncols;

    float tv[K];
    int   ti[K];
#pragma unroll
    for (int i = 0; i < K; ++i) { tv[i] = -INFINITY; ti[i] = -1; }
    // slot to replace next: an empty slot first, then the worst kept value
    float vmin = -INFINITY;
    int   imin = -1;
    int   pmin = 0;

    for (int64_t c = c_beg + threadIdx.x; c < c_end; c += TK_THREADS) {
        const float v = tk_key(x[c]);
        if (tk_better(v, (int) c, vmin, imin)) {
#pragma unroll
            for (int i = 0; i < K; ++i) {
                if (i == pmin) { tv[i] = v; ti[i] = (int) c; }
            }
            vmin = tv[0]; imin = ti[0]; pmin = 0;
#pragma unroll
            for (int i = 1; i < K; ++i) {
                if (tk_better(vmin, imin, tv[i], ti[i])) { vmin = tv[i]; imin = ti[i]; pmin = i; }
            }
        }
    }

    __shared__ float sv[TK_THREADS * K];
    __shared__ int   si[TK_THREADS * K];
#pragma unroll
    for (int i = 0; i < K; ++i) {
        sv[threadIdx.x * K + i] = tv[i];
        si[threadIdx.x * K + i] = ti[i];
    }
    tk_bitonic(sv, si, TK_THREADS * K);
    for (int i = threadIdx.x; i < K; i += TK_THREADS) {
        cand_v[((int64_t) row * TK_CHUNKS + chunk) * K + i] = sv[i];
        cand_i[((int64_t) row * TK_CHUNKS + chunk) * K + i] = si[i];
    }
}

template <int K>
__global__ void __launch_bounds__(TK_THREADS)
top_k_small_pass2(const float * __restrict__ cand_v, const int * __restrict__ cand_i, int * __restrict__ dst, const int k) {
    const int row = blockIdx.x;
    constexpr int N = TK_CHUNKS * K;
    __shared__ float sv[N];
    __shared__ int   si[N];
    for (int i = threadIdx.x; i < N; i += TK_THREADS) {
        sv[i] = cand_v[(int64_t) row * N + i];
        si[i] = cand_i[(int64_t) row * N + i];
    }
    tk_bitonic(sv, si, N);
    for (int i = threadIdx.x; i < k; i += TK_THREADS) {
        dst[(int64_t) row * k + i] = si[i];
    }
}

// ------------------------- new: histogram selection -------------------------
// pass A: S blocks per row histogram the top byte of the orderable key (warp-private bins, global flush)
// pass B: S blocks per row collect candidates above/at the boundary bucket into a global pool
// pass C: filter the second-byte bucket, then sort or use repeated max selection on overflow.
constexpr int HS_THREADS = 256;
#ifndef HS_SPLITS_DEF
#define HS_SPLITS_DEF 32
#endif
constexpr int HS_SPLITS  = HS_SPLITS_DEF;
constexpr int HS_HEAD    = 64;     // head slots per row (elements above the bucket, < k)
constexpr int HS_CAP     = 4032;   // boundary-bucket candidates per row; HS_HEAD + HS_CAP is a power of two

__device__ __forceinline__ uint32_t hs_key(float v) {
    const uint32_t u = __float_as_uint(isnan(v) ? -INFINITY : (v == 0.0f ? 0.0f : v));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// descending order: larger key first, then smaller index (0 encodes an empty slot)
__device__ __forceinline__ uint64_t hs_pack(uint32_t key, int idx) {
    return ((uint64_t) key << 32) | (uint32_t)(0xFFFFFFFFu - (uint32_t) idx);
}

__host__ __device__ __forceinline__ int hs_split_per(int ncols) {
    return ((ncols + HS_SPLITS - 1) / HS_SPLITS + 3) & ~3;
}

__global__ void __launch_bounds__(HS_THREADS)
topk_hist_a(const float * __restrict__ src, uint32_t * __restrict__ g_hist, const int ncols) {
    const int row   = blockIdx.y;
    const int split = blockIdx.x;
    const int per   = hs_split_per(ncols);
    const int c_beg = split * per;
    const int c_end = min(ncols, c_beg + per);
    const float * x = src + (int64_t) row * ncols;
    const int warp  = threadIdx.x / 32;

    __shared__ uint32_t wh[HS_THREADS / 32][256];
    for (int i = threadIdx.x; i < (HS_THREADS / 32) * 256; i += HS_THREADS) ((uint32_t *) wh)[i] = 0;
    __syncthreads();

    const int nc4 = c_end & ~3;
    for (int c4 = c_beg / 4 + threadIdx.x; c4 < nc4 / 4; c4 += HS_THREADS) {
        const float4 v = *((const float4 *) x + c4);
        atomicAdd(&wh[warp][hs_key(v.x) >> 24], 1);
        atomicAdd(&wh[warp][hs_key(v.y) >> 24], 1);
        atomicAdd(&wh[warp][hs_key(v.z) >> 24], 1);
        atomicAdd(&wh[warp][hs_key(v.w) >> 24], 1);
    }
    for (int c = max(nc4, c_beg) + threadIdx.x; c < c_end; c += HS_THREADS) {
        atomicAdd(&wh[warp][hs_key(x[c]) >> 24], 1);
    }
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += HS_THREADS) {
        uint32_t t = 0;
#pragma unroll
        for (int w = 0; w < HS_THREADS / 32; ++w) t += wh[w][b];
        if (t != 0) atomicAdd(&g_hist[(int64_t) row * 256 + b], t);
    }
}

// warp 0 only: given a 256-bin histogram in shared memory, find the highest bucket b where the count of
// elements in buckets >= b reaches `target`. returns b (or -1 if never reached).
__device__ __forceinline__ int hs_select_bucket(const uint32_t * h, uint32_t target) {
    const int lane = threadIdx.x % 32;
    uint32_t c[8];
    uint32_t s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) { c[j] = h[255 - 8 * lane - j]; s += c[j]; }
    uint32_t inc = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t t = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += t;
    }
    const unsigned hit = __ballot_sync(0xffffffffu, inc >= target);
    int b = -1;
    if (hit != 0) {
        const int first = __ffs(hit) - 1;
        if (lane == first) {
            uint32_t cum = inc - s;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (b < 0 && cum + c[j] >= target) b = 255 - 8 * lane - j;
                cum += c[j];
            }
        }
        b = __shfl_sync(0xffffffffu, b, first);
    }
    return b;
}

__global__ void __launch_bounds__(HS_THREADS)
topk_hist_b(const float * __restrict__ src, const uint32_t * __restrict__ g_hist, uint64_t * __restrict__ g_pool,
            int32_t * __restrict__ g_lo, int32_t * __restrict__ g_cnt, uint32_t * __restrict__ g_hist2, const int ncols, const int k, const int region_stride) {
    const int row   = blockIdx.y;
    const int split = blockIdx.x;
    const int per   = hs_split_per(ncols);
    const int c_beg = split * per;
    const int c_end = min(ncols, c_beg + per);
    const float * x = src + (int64_t) row * ncols;
    uint64_t * pool = g_pool + (int64_t) row * (HS_HEAD + HS_SPLITS * region_stride);
    uint64_t * region = pool + HS_HEAD + (int64_t) split * region_stride;

    __shared__ int b_star;
    __shared__ uint32_t hs[256];
    for (int i = threadIdx.x; i < 256; i += HS_THREADS) hs[i] = g_hist[(int64_t) row * 256 + i];
    __syncthreads();
    if (threadIdx.x < 32) {
        const int b = hs_select_bucket(hs, (uint32_t) k);
        if (threadIdx.x == 0) b_star = b;
    }
    __syncthreads();

    __shared__ uint32_t h2[256];
    __shared__ int      cnt;
    for (int i = threadIdx.x; i < 256; i += HS_THREADS) h2[i] = 0;
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();

    const int nc4 = c_end & ~3;
    for (int c4 = c_beg / 4 + threadIdx.x; c4 < nc4 / 4; c4 += HS_THREADS) {
        const float4 v = *((const float4 *) x + c4);
        const uint32_t key[4] = { hs_key(v.x), hs_key(v.y), hs_key(v.z), hs_key(v.w) };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t b = key[j] >> 24;
            if (b > b_star) {
                pool[atomicAdd(&g_lo[row], 1)] = hs_pack(key[j], 4 * c4 + j);
            } else if (b == b_star) {
                region[atomicAdd(&cnt, 1)] = hs_pack(key[j], 4 * c4 + j);
                atomicAdd(&h2[(key[j] >> 16) & 0xFF], 1);
            }
        }
    }
    for (int c = max(nc4, c_beg) + threadIdx.x; c < c_end; c += HS_THREADS) {
        const uint32_t key = hs_key(x[c]);
        const uint32_t b = key >> 24;
        if (b > b_star) {
            pool[atomicAdd(&g_lo[row], 1)] = hs_pack(key, c);
        } else if (b == b_star) {
            region[atomicAdd(&cnt, 1)] = hs_pack(key, c);
            atomicAdd(&h2[(key >> 16) & 0xFF], 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) g_cnt[(int64_t) row * HS_SPLITS + split] = cnt;
    for (int i = threadIdx.x; i < 256; i += HS_THREADS) {
        if (h2[i] != 0) atomicAdd(&g_hist2[(int64_t) row * 256 + i], h2[i]);
    }
}

__device__ void hs_bitonic_u64(uint64_t * a, int n) {
    for (int size = 2; size <= n; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            for (int t = threadIdx.x; t < n / 2; t += blockDim.x) {
                const int i = (t / stride) * stride * 2 + (t % stride);
                const int j = i + stride;
                const bool dir = ((i & size) == 0);
                const bool swap = dir ? (a[j] > a[i]) : (a[i] > a[j]);
                if (swap) { const uint64_t tmp = a[i]; a[i] = a[j]; a[j] = tmp; }
            }
        }
    }
    __syncthreads();
}

__global__ void __launch_bounds__(HS_THREADS)
topk_hist_c(const uint64_t * __restrict__ g_pool, const int32_t * __restrict__ g_lo, const int32_t * __restrict__ g_cnt,
            const uint32_t * __restrict__ g_hist2, int * __restrict__ dst, const int k, const int region_stride) {
    const int row = blockIdx.x;
    const uint64_t * pool = g_pool + (int64_t) row * (HS_HEAD + HS_SPLITS * region_stride);
    const int lo = g_lo[row];
    int * out = dst + (int64_t) row * k;
    const int need = k - lo;

    // sub-boundary from the second-byte histogram of the boundary bucket
    __shared__ int b2_star;
    __shared__ uint32_t hs[256];
    __shared__ int scnt[HS_SPLITS];
    for (int i = threadIdx.x; i < 256; i += HS_THREADS) hs[i] = g_hist2[(int64_t) row * 256 + i];
    for (int i = threadIdx.x; i < HS_SPLITS; i += HS_THREADS) scnt[i] = g_cnt[(int64_t) row * HS_SPLITS + i];
    __syncthreads();
    if (threadIdx.x < 32) {
        const int b = hs_select_bucket(hs, (uint32_t) need);
        if (threadIdx.x == 0) b2_star = b;
    }
    __syncthreads();

    __shared__ uint64_t buf[HS_CAP + HS_HEAD];
    for (int i = threadIdx.x; i < lo; i += HS_THREADS) buf[i] = pool[i];   // level-1 head
    __shared__ int head;
    __shared__ int cnt;
    if (threadIdx.x == 0) { head = lo; cnt = 0; }
    __syncthreads();

    // one warp per split region so the regions are read in parallel
    for (int split = threadIdx.x / 32; split < HS_SPLITS; split += HS_THREADS / 32) {
        const uint64_t * region = pool + HS_HEAD + (int64_t) split * region_stride;
        const int n = scnt[split];
        for (int i = threadIdx.x % 32; i < n; i += 32) {
            const uint64_t p = region[i];
            const uint32_t b2 = (uint32_t) (p >> 48) & 0xFF;   // second byte of the key
            if (b2 > b2_star) {
                buf[atomicAdd(&head, 1)] = p;
            } else if (b2 == b2_star) {
                const int slot = atomicAdd(&cnt, 1);
                if (slot < HS_CAP) buf[HS_HEAD + slot] = p;
            }
        }
    }
    __syncthreads();

    if (cnt <= HS_CAP) {
        int n = 2;
        while (n < HS_HEAD + cnt) n <<= 1;
        for (int i = head + threadIdx.x; i < HS_HEAD; i += HS_THREADS) buf[i] = 0;
        for (int i = HS_HEAD + cnt + threadIdx.x; i < n; i += HS_THREADS) buf[i] = 0;
        __syncthreads();
        hs_bitonic_u64(buf, n);
        for (int i = threadIdx.x; i < k; i += HS_THREADS) out[i] = (int)(0xFFFFFFFFu - (uint32_t) buf[i]);
        return;
    }

    // pathological sub-boundary: extract by repeated block max over the pool regions, then sort once
    for (int j = 0; j < need; ++j) {
        __shared__ uint64_t limit;
        __shared__ uint64_t red[HS_THREADS];
        if (threadIdx.x == 0) limit = j == 0 ? ~0ull : red[0];
        __syncthreads();
        uint64_t mine = 0;
        for (int split = 0; split < HS_SPLITS; ++split) {
            const uint64_t * region = pool + HS_HEAD + (int64_t) split * region_stride;
            const int n = scnt[split];
            for (int i = threadIdx.x; i < n; i += HS_THREADS) {
                const uint64_t p = region[i];
                if (p < limit && p > mine) mine = p;
            }
        }
        red[threadIdx.x] = mine;
        __syncthreads();
        for (int off = HS_THREADS / 2; off > 0; off >>= 1) {
            if (threadIdx.x < off && red[threadIdx.x + off] > red[threadIdx.x]) red[threadIdx.x] = red[threadIdx.x + off];
            __syncthreads();
        }
        if (threadIdx.x == 0) buf[lo + j] = red[0];
        __syncthreads();
    }
    {
        int n = 2;
        while (n < k) n <<= 1;
        for (int i = k + threadIdx.x; i < n; i += HS_THREADS) buf[i] = 0;
        __syncthreads();
        hs_bitonic_u64(buf, n);
        for (int i = threadIdx.x; i < k; i += HS_THREADS) out[i] = (int)(0xFFFFFFFFu - (uint32_t) buf[i]);
    }
}

template <int K>
void launch(ggml_cuda_pool & pool, const float * src, int * dst, int64_t ncols, int64_t nrows, int k, cudaStream_t stream) {
    GGML_ASSERT(k <= K);
    ggml_cuda_pool_alloc<float> cv(pool, nrows * TK_CHUNKS * K);
    ggml_cuda_pool_alloc<int>   ci(pool, nrows * TK_CHUNKS * K);
    top_k_small_pass1<K><<<dim3(TK_CHUNKS, nrows), TK_THREADS, 0, stream>>>(src, cv.get(), ci.get(), ncols);
    top_k_small_pass2<K><<<nrows, TK_THREADS, 0, stream>>>(cv.get(), ci.get(), dst, k);
}

// scratch layout: hist[nrows][256] u32 | hist2[nrows][256] u32 | lo[nrows] i32 | cnt[nrows][S] i32
// pool layout per row: [head HS_HEAD | S regions of `region_stride` packs]
void ggml_cuda_top_k_hist(ggml_cuda_pool & pool, const float * src, int * dst, int64_t ncols64, int64_t nrows, int64_t k, cudaStream_t stream) {
    const int ncols = (int) ncols64;
    const int region_stride = hs_split_per(ncols);
    ggml_cuda_pool_alloc<uint32_t> scratch(pool, (size_t) nrows * (512 + 1 + HS_SPLITS));
    ggml_cuda_pool_alloc<uint64_t> cand(pool, (size_t) nrows * (HS_HEAD + (int64_t) HS_SPLITS * region_stride));
    uint32_t * g_hist  = scratch.ptr;
    uint32_t * g_hist2 = g_hist + (size_t) nrows * 256;
    int32_t  * g_lo    = (int32_t *) (g_hist2 + (size_t) nrows * 256);
    int32_t  * g_cnt   = g_lo + nrows;
    CUDA_CHECK(cudaMemsetAsync(scratch.ptr, 0, (size_t) nrows * (512 + 1 + HS_SPLITS) * 4, stream));
    topk_hist_a<<<dim3(HS_SPLITS, nrows), HS_THREADS, 0, stream>>>(src, g_hist, ncols);
    topk_hist_b<<<dim3(HS_SPLITS, nrows), HS_THREADS, 0, stream>>>(src, g_hist, cand.ptr, g_lo, g_cnt, g_hist2, ncols, (int) k, region_stride);
    topk_hist_c<<<nrows, HS_THREADS, 0, stream>>>(cand.ptr, g_lo, g_cnt, g_hist2, dst, (int) k, region_stride);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void ggml_cuda_top_k_small(ggml_cuda_pool & pool, const float * src, int * dst, int64_t ncols, int64_t nrows, int k, cudaStream_t stream) {
    if (k > TOPK_SMALL_MAX_K) {
        ggml_cuda_top_k_hist(pool, src, dst, ncols, nrows, k, stream);
        return;
    }
    if (k <= 8) {
        launch<8>(pool, src, dst, ncols, nrows, k, stream);
    } else {
        GGML_ASSERT(k <= TOPK_SMALL_MAX_K);
        launch<16>(pool, src, dst, ncols, nrows, k, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
