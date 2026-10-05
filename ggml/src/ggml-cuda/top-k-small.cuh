#pragma once

#include "common.cuh"

// Top-k for small k (<= 64) over wide rows (e.g. vocab logits), no full sort.
// k <= 16: two passes with per-thread register lists. k <= 64: top-byte histogram selection (see top-k-small.cu).
// pass 1: each CTA keeps per-thread top-k in registers over a column chunk, then sorts the CTA candidates
// pass 2: one CTA per row merges the chunk candidates. Output indices are sorted by descending value.

static constexpr int TOPK_SMALL_MAX_K = 16;   // two-pass register kernel
static constexpr int TOPK_HIST_MAX_K  = 64;   // histogram selection kernel (k > TOPK_SMALL_MAX_K)

static bool ggml_cuda_top_k_small_supported(int64_t ncols, int64_t nrows, int64_t k) {
    if (k < 1 || k > TOPK_HIST_MAX_K || k > ncols || ncols < 4096 || ncols > INT_MAX || nrows < 1 || nrows > 65535) {
        return false;
    }
    return k <= TOPK_SMALL_MAX_K || (ncols <= INT_MAX - 256 && ncols % 4 == 0);
}

void ggml_cuda_top_k_small(ggml_cuda_pool & pool, const float * src, int * dst, int64_t ncols, int64_t nrows, int k, cudaStream_t stream);
