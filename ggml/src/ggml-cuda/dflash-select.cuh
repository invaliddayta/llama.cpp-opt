#pragma once

#include <cstdint>

// Match std::max_element: keep the first maximum and ignore later NaNs.
static __global__ void dflash_select_lattice(const float * lattice, int32_t * output,
        int hidden, int top_k, int block_size, int sequences) {
    const int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq >= sequences) {
        return;
    }
    output[seq * block_size] = -1;
    int predecessor = 0;
    for (int pos = 1; pos < block_size; ++pos) {
        const float * row = lattice + ((int64_t) seq * block_size + pos) * hidden;
        const float * scores = row + top_k + (int64_t) predecessor * top_k;
        int best = 0;
        for (int k = 1; k < top_k; ++k) {
            if (scores[best] < scores[k]) {
                best = k;
            }
        }
        predecessor = best;
        output[seq * block_size + pos] = (int32_t) row[best];
    }
}
