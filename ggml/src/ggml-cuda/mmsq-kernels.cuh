// mmsq: small-batch (N <= 16) quantized matrix multiplication for Ampere+ (sm_80+), weight-bandwidth bound.
//
//   dst[n][m] = sum_k W[m][k] * x[n][k]
//
// W: rows of k-quant super-blocks (IQ4_XS, Q4_K, Q5_K, Q6_K), streamed once with cp.async in long per-row runs.
// x: activations quantized to int8 with one scale per 256 values (q8_K granularity), in mma fragment order.
// Each warp dequantizes a 16x256 weight tile once and feeds int8 tensor-core mma for all N columns.
//
// Header-only so it can be used by the standalone lab and by ggml-cuda.

#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace mmsq {

constexpr int QK = 256;

// ---------------------------------------------------------------------------------------------
// activation format: per n8 tile t and super-block sb, FSB bytes:
//   [0, 2048)      8 x 256 B int8 quants, B-fragment order: lane l -> col l/4, k = 32j + (l%4)*4 + {0..3, 16..19}
//   [2048, 2080)   8 float scales u (one per column)
//   [2080, 2208)   8 x 8 int16 sums of each 32-block in units of u (index [j][col]) for k-quant mins
//   [2208, 2224)   8 x uint16 per column: 2-bit shift e of each 32-block j at bits [2j]
// A 32-block is stored with step u << e (e in 0..3), so small blocks keep more precision than one scale per 256.
constexpr int FSB = 2048 + 32 + 128 + 16;

// Quantize one column's super-block. 4 consecutive lanes hold a column (this lane: 64 values in v, l4 = lane % 4).
// All 32 lanes must call this together; lanes with active == false only take part in the shuffles.
__device__ __forceinline__ void quantize_sb_col(const float (&v)[8][8], uint8_t * blk, const int col, const int l4, const bool active = true) {
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[j][i]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    const float d  = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    uint32_t ebits = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float aj = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) aj = fmaxf(aj, fabsf(v[j][i]));
        aj = fmaxf(aj, __shfl_xor_sync(0xffffffff, aj, 1));
        aj = fmaxf(aj, __shfl_xor_sync(0xffffffff, aj, 2));
        // finest step u << e with |x| / (u << e) <= 127, u = d / 8
        int e = 3;
        while (e > 0 && aj * id * (float) (8 >> (e - 1)) <= 127.0f) {
            --e;
        }
        const float idj = id * (float) (8 >> e);
        uint32_t w0 = 0, w1 = 0;
        int s = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q0 = max(-127, min(127, __float2int_rn(v[j][i] * idj)));
            const int q1 = max(-127, min(127, __float2int_rn(v[j][4 + i] * idj)));
            s += q0 + q1;
            w0 |= (uint32_t) (uint8_t) (int8_t) q0 << (8 * i);
            w1 |= (uint32_t) (uint8_t) (int8_t) q1 << (8 * i);
        }
        if (active) {
            ((uint2 *) (blk + 256 * j))[col * 4 + l4] = make_uint2(w0, w1);
        }
        s += __shfl_xor_sync(0xffffffff, s, 1);
        s += __shfl_xor_sync(0xffffffff, s, 2);
        if (active && l4 == 0) {
            ((int16_t *) (blk + 2080))[j * 8 + col] = (int16_t) (s << e);
        }
        ebits |= (uint32_t) e << (2 * j);
    }
    if (active && l4 == 0) {
        ((uint16_t *) (blk + 2208))[col] = (uint16_t) ebits;
        ((float *) (blk + 2048))[col] = d / 8.0f;
    }
}

// load this lane's 64 values of a column super-block (zeros for padded columns)
__device__ __forceinline__ void load_sb_col(float (&v)[8][8], const float * src, const bool valid, const int l4) {
    const int c4 = l4 * 4;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (valid) {
            const float4 a = *(const float4 *) (src + 32 * j + c4);
            const float4 b = *(const float4 *) (src + 32 * j + 16 + c4);
            v[j][0] = a.x; v[j][1] = a.y; v[j][2] = a.z; v[j][3] = a.w;
            v[j][4] = b.x; v[j][5] = b.y; v[j][6] = b.z; v[j][7] = b.w;
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) v[j][i] = 0.0f;
        }
    }
}

__global__ void quantize_x(const float * __restrict__ x, uint8_t * __restrict__ y, const int K, const int N, const int64_t stride_col) {
    const int sb   = blockIdx.x;
    const int t    = blockIdx.y;
    const int lane = threadIdx.x;
    const int col  = t * 8 + lane / 4;
    float v[8][8];
    load_sb_col(v, x + (int64_t) col * stride_col + sb * QK, col < N, lane % 4);
    quantize_sb_col(v, y + ((int64_t) t * (K / QK) + sb) * FSB, lane / 4, lane % 4);
}

// ---------------------------------------------------------------------------------------------
// device helpers

__device__ __forceinline__ void cp_async16(void * smem, const void * gmem) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async8(void * smem, const void * gmem) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async4(void * smem, const void * gmem) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int n> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }

__device__ __forceinline__ void mma_k32(int (&c)[4], const int (&a)[4], const int b0, const int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_k16(int (&c)[4], const int a0, const int a1, const int b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}

// 32-bit shared-memory load from a 2-byte aligned address
__device__ __forceinline__ uint32_t lds32_a2(const uint8_t * p) {
    const uintptr_t a = (uintptr_t) p;
    if ((a & 3) == 0) {
        return *(const uint32_t *) p;
    }
    const uint32_t * w = (const uint32_t *) (a & ~(uintptr_t) 3);
    return __funnelshift_r(w[0], w[1], 16);
}

// 8 nibbles -> 8 int8 table values (low nibbles -> .x, high nibbles -> .y)
__device__ __forceinline__ int2 lut16(const uint32_t q4, const uint32_t * t) {
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16 * i;
        const uint32_t lo = __byte_perm(t[0], t[1], q4 >> sh);
        const uint32_t hi = __byte_perm(t[2], t[3], q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

// ---------------------------------------------------------------------------------------------
// per-type super-block handling. Each type provides:
//   BLK, CP16 (row runs are 16B-aligned), ROW_ALIGN_BYTES
//   struct rowdata; load(rowdata &, const uint8_t * blk)            per-row super-block header
//   accumulate<NT>(isum, imin, blk0, blk1, rd0, rd1, bq, ssum)       all 8 k32 steps of the super-block
//   finish: acc += dy * (d * isum - dmin * imin)

enum class qtype { iq4_xs, q4_K, q5_K, q6_K };

template <qtype T> struct traits;

template <> struct traits<qtype::iq4_xs> {
    static constexpr int BLK = 136;
    static constexpr bool CP16 = true;
    static constexpr bool HAS_MIN = false;
};
template <> struct traits<qtype::q4_K> {
    static constexpr int BLK = 144;
    static constexpr bool CP16 = true;
    static constexpr bool HAS_MIN = true;
};
template <> struct traits<qtype::q5_K> {
    static constexpr int BLK = 176;
    static constexpr bool CP16 = true;
    static constexpr bool HAS_MIN = true;
};
template <> struct traits<qtype::q6_K> {
    static constexpr int BLK = 210;
    static constexpr bool CP16 = false;
    static constexpr bool HAS_MIN = false;
};

// k-quant 6-bit scale/min unpack for sub-block j from the 12 scale bytes (as 3 words)
__device__ __forceinline__ void scale_min_k4(const int j, const uint32_t s0, const uint32_t s1, const uint32_t s2, int & sc, int & m) {
    const uint8_t * q;
    uint32_t w[3] = { s0, s1, s2 };
    q = (const uint8_t *) w;
    if (j < 4) {
        sc = q[j] & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

template <qtype T, int NT>
struct sb_math;

// IQ4_XS: w = d * (ls_j - 32) * kvalues[q]
template <int NT> struct sb_math<qtype::iq4_xs, NT> {
    __device__ __forceinline__ static void run(const uint8_t * blk0, const uint8_t * blk1, const uint2 (&bq)[NT][8], const uint32_t * table,
                                               const int lane, float (&dscale)[2], float (&dmin)[2], int (&isum)[NT][4], int (&imin)[NT][4], const uint32_t (&ssw)[NT][8], const uint32_t (&shw)[NT]) {
        const int c4 = (lane % 4) * 4;
        dscale[0] = __half2float(*(const half *) blk0);
        dscale[1] = __half2float(*(const half *) blk1);
        dmin[0] = dmin[1] = 0.0f;
        // unpack the 8 six-bit sub-block scales of each row into signed bytes: [even j] and [odd j]
        uint32_t lse[2], lso[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint8_t * blk = r ? blk1 : blk0;
            const uint32_t sh = *(const uint16_t *) (blk + 2);
            const uint32_t sl = *(const uint32_t *) (blk + 4);
            // high 2 bits: even j at bits 0-1,4-5,8-9,12-13; odd j at 2-3,6-7,10-11,14-15 -> spread to bytes
            uint32_t he = sh & 0x3333, ho = (sh >> 2) & 0x3333;
            he = ((he & 0xFF00) << 8) | (he & 0xFF); he = (he | (he << 4)) & 0x0F0F0F0F;
            ho = ((ho & 0xFF00) << 8) | (ho & 0xFF); ho = (ho | (ho << 4)) & 0x0F0F0F0F;
            lse[r] = __vsub4((sl & 0x0F0F0F0F) | (he << 4), 0x20202020);
            lso[r] = __vsub4(((sl >> 4) & 0x0F0F0F0F) | (ho << 4), 0x20202020);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int2 v0 = lut16(*(const uint32_t *) (blk0 + 8 + 16 * j + c4), table);
            const int2 v1 = lut16(*(const uint32_t *) (blk1 + 8 + 16 * j + c4), table);
            const int a[4] = { v0.x, v1.x, v0.y, v1.y };
            const int ls0 = (int) (int8_t) ((((j & 1) ? lso[0] : lse[0]) >> (8 * (j / 2))) & 0xFF);
            const int ls1 = (int) (int8_t) ((((j & 1) ? lso[1] : lse[1]) >> (8 * (j / 2))) & 0xFF);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                int c[4] = { 0, 0, 0, 0 };
                mma_k32(c, a, (int) bq[t][j].x, (int) bq[t][j].y);
                const int e0 = (shw[t] >> (2 * j)) & 3, e1 = (shw[t] >> (16 + 2 * j)) & 3;
                isum[t][0] += (ls0 << e0) * c[0];
                isum[t][1] += (ls0 << e1) * c[1];
                isum[t][2] += (ls1 << e0) * c[2];
                isum[t][3] += (ls1 << e1) * c[3];
            }
        }
    }
};

// Q4_K / Q5_K: w = d * sc_j * q - dmin * m_j, q in [0, 15] / [0, 31]
template <qtype T, int NT> struct sb_math_k45 {
    __device__ __forceinline__ static void run(const uint8_t * blk0, const uint8_t * blk1, const uint2 (&bq)[NT][8], const uint32_t * /*table*/,
                                               const int lane, float (&dscale)[2], float (&dmin)[2], int (&isum)[NT][4], int (&imin)[NT][4], const uint32_t (&ssw)[NT][8], const uint32_t (&shw)[NT]) {
        const int c4 = (lane % 4) * 4;
        const half2 dm0 = *(const half2 *) blk0;
        const half2 dm1 = *(const half2 *) blk1;
        dscale[0] = __low2float(dm0);  dmin[0] = __high2float(dm0);
        dscale[1] = __low2float(dm1);  dmin[1] = __high2float(dm1);
        const uint32_t a0 = *(const uint32_t *) (blk0 + 4), a1 = *(const uint32_t *) (blk0 + 8), a2 = *(const uint32_t *) (blk0 + 12);
        const uint32_t b0 = *(const uint32_t *) (blk1 + 4), b1 = *(const uint32_t *) (blk1 + 8), b2 = *(const uint32_t *) (blk1 + 12);
        constexpr int QS = T == qtype::q5_K ? 48 : 16;   // offset of qs
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int p = j / 2;
            const int shift = 4 * (j & 1);
            uint32_t x0 = (*(const uint32_t *) (blk0 + QS + 32 * p + c4)      >> shift) & 0x0F0F0F0F;
            uint32_t x1 = (*(const uint32_t *) (blk1 + QS + 32 * p + c4)      >> shift) & 0x0F0F0F0F;
            uint32_t y0 = (*(const uint32_t *) (blk0 + QS + 32 * p + 16 + c4) >> shift) & 0x0F0F0F0F;
            uint32_t y1 = (*(const uint32_t *) (blk1 + QS + 32 * p + 16 + c4) >> shift) & 0x0F0F0F0F;
            if constexpr (T == qtype::q5_K) {
                x0 |= ((*(const uint32_t *) (blk0 + 16 + c4)      >> j) & 0x01010101) << 4;
                x1 |= ((*(const uint32_t *) (blk1 + 16 + c4)      >> j) & 0x01010101) << 4;
                y0 |= ((*(const uint32_t *) (blk0 + 16 + 16 + c4) >> j) & 0x01010101) << 4;
                y1 |= ((*(const uint32_t *) (blk1 + 16 + 16 + c4) >> j) & 0x01010101) << 4;
            }
            const int a[4] = { (int) x0, (int) x1, (int) y0, (int) y1 };
            int sc0, m0, sc1, m1;
            scale_min_k4(j, a0, a1, a2, sc0, m0);
            scale_min_k4(j, b0, b1, b2, sc1, m1);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                int c[4] = { 0, 0, 0, 0 };
                mma_k32(c, a, (int) bq[t][j].x, (int) bq[t][j].y);
                const int e0 = (shw[t] >> (2 * j)) & 3, e1 = (shw[t] >> (16 + 2 * j)) & 3;
                isum[t][0] += (sc0 << e0) * c[0];
                isum[t][1] += (sc0 << e1) * c[1];
                isum[t][2] += (sc1 << e0) * c[2];
                isum[t][3] += (sc1 << e1) * c[3];
                const int sy0 = (int) (int16_t) (ssw[t][j] & 0xFFFF);
                const int sy1 = (int) (int16_t) (ssw[t][j] >> 16);
                imin[t][0] += m0 * sy0;
                imin[t][1] += m0 * sy1;
                imin[t][2] += m1 * sy0;
                imin[t][3] += m1 * sy1;
            }
        }
    }
};
template <int NT> struct sb_math<qtype::q4_K, NT> : sb_math_k45<qtype::q4_K, NT> {};
template <int NT> struct sb_math<qtype::q5_K, NT> : sb_math_k45<qtype::q5_K, NT> {};

// Q6_K: w = d * sc16 * (q - 32), 16-element scales -> two k16 mma per k32 step
template <int NT> struct sb_math<qtype::q6_K, NT> {
    __device__ __forceinline__ static void run(const uint8_t * blk0, const uint8_t * blk1, const uint2 (&bq)[NT][8], const uint32_t * /*table*/,
                                               const int lane, float (&dscale)[2], float (&dmin)[2], int (&isum)[NT][4], int (&imin)[NT][4], const uint32_t (&ssw)[NT][8], const uint32_t (&shw)[NT]) {
        const int c4 = (lane % 4) * 4;
        dscale[0] = __half2float(__ushort_as_half((unsigned short) (lds32_a2(blk0 + 206) >> 16)));
        dscale[1] = __half2float(__ushort_as_half((unsigned short) (lds32_a2(blk1 + 206) >> 16)));
        dmin[0] = dmin[1] = 0.0f;
        // 16 int8 sub-block scales per row
        uint32_t scw0[4], scw1[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { scw0[i] = lds32_a2(blk0 + 192 + 4 * i); scw1[i] = lds32_a2(blk1 + 192 + 4 * i); }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t hx0 = lds32_a2(blk0 + 128 + 32 * h + c4), hy0 = lds32_a2(blk0 + 128 + 32 * h + 16 + c4);
            const uint32_t hx1 = lds32_a2(blk1 + 128 + 32 * h + c4), hy1 = lds32_a2(blk1 + 128 + 32 * h + 16 + c4);
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int qlo = 64 * h + 32 * half;
                const uint32_t lx0 = lds32_a2(blk0 + qlo + c4), ly0 = lds32_a2(blk0 + qlo + 16 + c4);
                const uint32_t lx1 = lds32_a2(blk1 + qlo + c4), ly1 = lds32_a2(blk1 + qlo + 16 + c4);
#pragma unroll
                for (int hi = 0; hi < 2; ++hi) {
                    const int jj = half + 2 * hi;
                    const int j = 4 * h + jj;
                    const int nsh = 4 * hi;
                    uint32_t x0 = ((lx0 >> nsh) & 0x0F0F0F0F) | (((hx0 >> (2 * jj)) & 0x03030303) << 4);
                    uint32_t x1 = ((lx1 >> nsh) & 0x0F0F0F0F) | (((hx1 >> (2 * jj)) & 0x03030303) << 4);
                    uint32_t y0 = ((ly0 >> nsh) & 0x0F0F0F0F) | (((hy0 >> (2 * jj)) & 0x03030303) << 4);
                    uint32_t y1 = ((ly1 >> nsh) & 0x0F0F0F0F) | (((hy1 >> (2 * jj)) & 0x03030303) << 4);
                    x0 = __vsub4(x0, 0x20202020); x1 = __vsub4(x1, 0x20202020);
                    y0 = __vsub4(y0, 0x20202020); y1 = __vsub4(y1, 0x20202020);
                    const int si = 8 * h + 2 * jj;   // scale index of the first 16 elements
                    const int sa0 = (int) (int8_t) (scw0[si / 4] >> (8 * (si % 4)));
                    const int sb0 = (int) (int8_t) (scw0[si / 4] >> (8 * (si % 4 + 1)));
                    const int sa1 = (int) (int8_t) (scw1[si / 4] >> (8 * (si % 4)));
                    const int sb1 = (int) (int8_t) (scw1[si / 4] >> (8 * (si % 4 + 1)));
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        int ca[4] = { 0, 0, 0, 0 };
                        int cb[4] = { 0, 0, 0, 0 };
                        mma_k16(ca, (int) x0, (int) x1, (int) bq[t][j].x);
                        mma_k16(cb, (int) y0, (int) y1, (int) bq[t][j].y);
                        const int e0 = (shw[t] >> (2 * j)) & 3, e1 = (shw[t] >> (16 + 2 * j)) & 3;
                        isum[t][0] += (sa0 * ca[0] + sb0 * cb[0]) << e0;
                        isum[t][1] += (sa0 * ca[1] + sb0 * cb[1]) << e1;
                        isum[t][2] += (sa1 * ca[2] + sb1 * cb[2]) << e0;
                        isum[t][3] += (sa1 * ca[3] + sb1 * cb[3]) << e1;
                    }
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------------------------
// kernel

template <qtype T, int KW, int STAGES = 2> struct cfg {
    static constexpr int BLK   = traits<T>::BLK;
    static constexpr int ROWS  = 16;
    static constexpr int RUN   = KW * BLK;                                   // bytes per row per stage
    static constexpr int RUNP  = (RUN + 15) / 16 * 16;
    static constexpr int ROWB  = RUNP + (((RUNP / 4) % 8) == 4 ? 0 : 16);   // row stride == 4 mod 8 words
    static constexpr int WST   = ROWS * ROWB;
    static constexpr int SMEM  = STAGES * WST;
};

// grid: (ceil(M/16), splits); block: KW warps. Each CTA: 16 rows x a K range of `sb_per_split` super-blocks
// (a multiple of KW). Warp kw handles super-block kw of every stage; partial sums are reduced in smem.
template <qtype T, int NT, int KW, int MINB, int STAGES = 2>
__global__ void __launch_bounds__(KW * 32, MINB)
mul_mat(const uint8_t * __restrict__ W, const int64_t row_bytes, const uint8_t * __restrict__ XF, float * __restrict__ dst,
        const int M, const int K, const int N, const int64_t stride_dst_col, float * __restrict__ part, int * __restrict__ counters,
        const int sb_per_split) {
    using C = cfg<T, KW, STAGES>;
    constexpr int NTH = KW * 32;
    constexpr int BLK = C::BLK;
    extern __shared__ __align__(16) uint8_t smem[];
    const int kw   = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int nsb    = K / QK;
    const int rowc0  = blockIdx.x * 16;
    const int sb_beg = blockIdx.y * sb_per_split;
    const int sb_end = min(nsb, sb_beg + sb_per_split);
    const int nsteps = (sb_end - sb_beg + KW - 1) / KW;

    uint32_t table[4] = { 0, 0, 0, 0 };
    if constexpr (T == qtype::iq4_xs) {
        table[0] = 0xBFAD9881u; // kvalues_iq4nl {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113} packed little-endian
        table[1] = 0xF6EADDCFu;
        table[2] = 0x26190D01u;
        table[3] = 0x71594535u;
    }

    // copy a stage: KW super-blocks (fewer in the tail) of 16 rows, one contiguous run per row
    auto load_stage = [&](int st, int sb) {
        uint8_t * ws = smem + st * C::WST;
        const int nvalid = min(KW, sb_end - sb);
        if (traits<T>::CP16 && C::RUN % 16 == 0 && nvalid == KW) {
            constexpr int CH = C::RUN / 16;
            constexpr int WCH = 16 * CH;
#pragma unroll
            for (int q0 = 0; q0 < WCH; q0 += NTH) {
                const int q = q0 + threadIdx.x;
                if (WCH % NTH == 0 || q < WCH) {
                    const int r = q / CH, o = q % CH;
                    const int row = min(rowc0 + r, M - 1);
                    cp_async16(ws + r * C::ROWB + o * 16, W + row * row_bytes + (int64_t) sb * BLK + o * 16);
                }
            }
        } else if ((KW * BLK) % 8 == 0 && (BLK % 8 == 0 || nvalid == KW)) {
            const int CH = nvalid * BLK / 8;
            const int WCH = 16 * CH;
            for (int q = threadIdx.x; q < WCH; q += NTH) {
                const int r = q / CH, o = q % CH;
                const int row = min(rowc0 + r, M - 1);
                cp_async8(ws + r * C::ROWB + o * 8, W + row * row_bytes + (int64_t) sb * BLK + o * 8);
            }
        } else {
            const int CH = (nvalid * BLK + 3) / 4;   // q6_K: runs are only 4B aligned (host requires row_bytes % 4 == 0)
            const int WCH = 16 * CH;
            for (int q = threadIdx.x; q < WCH; q += NTH) {
                const int r = q / CH, o = q % CH;
                const int row = min(rowc0 + r, M - 1);
                cp_async4(ws + r * C::ROWB + o * 4, W + row * row_bytes + (int64_t) sb * BLK + o * 4);
            }
        }
    };

    const int nlo_ = (lane % 4) * 2;
    auto load_x = [&](uint2 (&bq)[NT][8], float2 (&dy)[NT], uint32_t (&ssw)[NT][8], uint32_t (&shw)[NT], int sb) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const uint8_t * fb = XF + ((int64_t) t * nsb + sb) * FSB;
            const bool colv = t * 8 + lane / 4 < N;
#pragma unroll
            for (int j = 0; j < 8; ++j) bq[t][j] = colv ? __ldg((const uint2 *) (fb + 256 * j + lane * 8)) : make_uint2(0, 0);
            dy[t] = t * 8 + nlo_ < N ? __ldg((const float2 *) (fb + 2048 + nlo_ * 4)) : make_float2(0.0f, 0.0f);
#pragma unroll
            for (int j = 0; j < 8; ++j) ssw[t][j] = traits<T>::HAS_MIN ? __ldg((const uint32_t *) (fb + 2080 + 16 * j + nlo_ * 2)) : 0u;
            shw[t] = __ldg((const uint32_t *) (fb + 2208 + (lane % 4) * 4));
        }
    };

    float acc[NT][4];
#pragma unroll
    for (int t = 0; t < NT; ++t) { acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f; }

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nsteps) load_stage(s, sb_beg + s * KW);
        cp_async_commit();
    }

    const int r_lo = lane / 4;
    const int nlo  = (lane % 4) * 2;

#ifndef MMSQ_NO_XPREFETCH
    uint2  bq[NT][8];
    float2 dy[NT];
    uint32_t ssw[NT][8];
    uint32_t shw[NT];
    if (sb_beg + kw < sb_end) load_x(bq, dy, ssw, shw, sb_beg + kw);
#endif

    for (int step = 0; step < nsteps; ++step) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (step + STAGES - 1 < nsteps) load_stage((step + STAGES - 1) % STAGES, sb_beg + (step + STAGES - 1) * KW);
        cp_async_commit();

        const int sb = sb_beg + step * KW + kw;
        if (sb >= sb_end) continue;   // tail stage: warp-uniform skip, the loop-top barrier is still reached
        const uint8_t * st   = smem + (step % STAGES) * C::WST;
        const uint8_t * blk0 = st + r_lo * C::ROWB + kw * BLK;
        const uint8_t * blk1 = blk0 + 8 * C::ROWB;

#ifdef MMSQ_NO_XPREFETCH
        uint2  bq[NT][8];
        float2 dy[NT];
        uint32_t ssw[NT][8];
        uint32_t shw[NT];
        load_x(bq, dy, ssw, shw, sb);
#else
        uint2  bqn[NT][8];
        float2 dyn[NT];
        uint32_t sswn[NT][8];
        uint32_t shwn[NT];
        const bool has_next = sb + KW < sb_end;
        if (has_next) load_x(bqn, dyn, sswn, shwn, sb + KW);
#endif
        int isum[NT][4], imin[NT][4];
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) { isum[t][e] = 0; imin[t][e] = 0; }
        float dsc[2], dmn[2];
        sb_math<T, NT>::run(blk0, blk1, bq, table, lane, dsc, dmn, isum, imin, ssw, shw);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if constexpr (traits<T>::HAS_MIN) {
                acc[t][0] += dy[t].x * (dsc[0] * (float) isum[t][0] - dmn[0] * (float) imin[t][0]);
                acc[t][1] += dy[t].y * (dsc[0] * (float) isum[t][1] - dmn[0] * (float) imin[t][1]);
                acc[t][2] += dy[t].x * (dsc[1] * (float) isum[t][2] - dmn[1] * (float) imin[t][2]);
                acc[t][3] += dy[t].y * (dsc[1] * (float) isum[t][3] - dmn[1] * (float) imin[t][3]);
            } else {
                acc[t][0] += dy[t].x * dsc[0] * (float) isum[t][0];
                acc[t][1] += dy[t].y * dsc[0] * (float) isum[t][1];
                acc[t][2] += dy[t].x * dsc[1] * (float) isum[t][2];
                acc[t][3] += dy[t].y * dsc[1] * (float) isum[t][3];
            }
        }
#ifndef MMSQ_NO_XPREFETCH
        if (has_next) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                dy[t] = dyn[t];
                shw[t] = shwn[t];
#pragma unroll
                for (int j = 0; j < 8; ++j) { bq[t][j] = bqn[t][j]; ssw[t][j] = sswn[t][j]; }
            }
        }
#endif
    }
    cp_async_wait<0>();

    if (KW > 1) {
        __syncthreads();
        float * red = (float *) smem;   // [KW][NT*4][32]
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) red[(kw * NT * 4 + t * 4 + e) * 32 + lane] = acc[t][e];
        __syncthreads();
        if (kw != 0) return;
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float v = 0.0f;
#pragma unroll
                for (int k = 0; k < KW; ++k) v += red[(k * NT * 4 + t * 4 + e) * 32 + lane];
                acc[t][e] = v;
            }
    }

    if (gridDim.y == 1) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int n0 = t * 8 + nlo;
            const int r0 = rowc0 + r_lo, r1 = r0 + 8;
            if (n0 < N) {
                if (r0 < M) dst[(int64_t) n0 * stride_dst_col + r0] = acc[t][0];
                if (r1 < M) dst[(int64_t) n0 * stride_dst_col + r1] = acc[t][2];
            }
            if (n0 + 1 < N) {
                if (r0 < M) dst[(int64_t) (n0 + 1) * stride_dst_col + r0] = acc[t][1];
                if (r1 < M) dst[(int64_t) (n0 + 1) * stride_dst_col + r1] = acc[t][3];
            }
        }
        return;
    }

    // split-K: write partials, the last CTA of this row tile sums them in split order (deterministic)
    const int64_t psplit = (int64_t) N * M;
    float * my = part + (int64_t) blockIdx.y * psplit;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int n0 = t * 8 + nlo;
        const int r0 = rowc0 + r_lo, r1 = r0 + 8;
        if (n0 < N) {
            if (r0 < M) my[(int64_t) n0 * M + r0] = acc[t][0];
            if (r1 < M) my[(int64_t) n0 * M + r1] = acc[t][2];
        }
        if (n0 + 1 < N) {
            if (r0 < M) my[(int64_t) (n0 + 1) * M + r0] = acc[t][1];
            if (r1 < M) my[(int64_t) (n0 + 1) * M + r1] = acc[t][3];
        }
    }
    __threadfence();
    __syncwarp();
    int prev = 0;
    if (lane == 0) {
        prev = atomicAdd(&counters[blockIdx.x], 1);
    }
    prev = __shfl_sync(0xffffffff, prev, 0);
    if (prev != (int) gridDim.y - 1) {
        return;
    }
    __threadfence();
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int n = t * 8 + nlo + (e & 1);
            const int r = rowc0 + r_lo + 8 * (e >> 1);
            if (n < N && r < M) {
                float v = 0.0f;
                for (int k = 0; k < (int) gridDim.y; ++k) {
                    v += __ldcg(part + (int64_t) k * psplit + (int64_t) n * M + r);
                }
                dst[(int64_t) n * stride_dst_col + r] = v;
            }
        }
    }
    if (lane == 0) {
        counters[blockIdx.x] = 0;
    }
}

} // namespace mmsq
