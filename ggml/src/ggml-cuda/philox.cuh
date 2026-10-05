#pragma once
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cuda_runtime.h>
#endif
#include <cstdint>

namespace gpu_rng_lab {

__host__ __device__ inline uint32_t multiply_high(uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__)
    return __umulhi(a, b);
#else
    return (uint32_t) (((uint64_t) a * b) >> 32);
#endif
}

// Philox4x32-10, Random123 constants and round order. Each token has an independent counter.
__host__ __device__ inline uint4 philox(uint64_t counter, uint64_t seed) {
    uint4 value = make_uint4((uint32_t) counter, (uint32_t) (counter >> 32), 0, 0);
    uint32_t k0 = (uint32_t) seed, k1 = (uint32_t) (seed >> 32);
    for (int round = 0; round < 10; ++round) {
        const uint32_t hi0 = multiply_high(0xD2511F53u, value.x);
        const uint32_t hi1 = multiply_high(0xCD9E8D57u, value.z);
        value = make_uint4(hi1 ^ value.y ^ k0, 0xCD9E8D57u * value.z,
                          hi0 ^ value.w ^ k1, 0xD2511F53u * value.x);
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return value;
}

__host__ __device__ inline float uniform(uint64_t counter, uint64_t seed) {
    return (philox(counter, seed).x >> 8) * (1.0f / 16777216.0f);
}

static __global__ void generate(uint64_t seed, uint64_t begin, float * output, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        output[i] = uniform(begin + (uint64_t) i, seed);
    }
}

} // namespace gpu_rng_lab
