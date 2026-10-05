#pragma once
#include "../ggml-gpu-grammar.h"
#include <cmath>

namespace gpu_grammar_lab {

__device__ inline int32_t advance_char(const dfa_device & dfa, int32_t node, uint32_t character) {
    if (node < 0 || node >= dfa.n_states) return -1;
    int low = 0, high = dfa.n_classes;
    while (low + 1 < high) {
        const int mid = (low + high) / 2;
        if (dfa.classes[mid] <= character) low = mid;
        else high = mid;
    }
    return dfa.next[(size_t) node * dfa.n_classes + low];
}

__device__ inline bool partial_allowed(const dfa_device & dfa, const state & current) {
    if (current.remaining == 0) return true;
    if (current.remaining < 0 || (current.remaining == 1 && current.partial_value < 2)) return false;
    uint32_t low = current.partial_value << (current.remaining * 6);
    const uint32_t high = low | ((1u << (current.remaining * 6)) - 1);
    if (low == 0) {
        if (current.remaining == 2) low = 1u << 11;
        if (current.remaining == 3) low = 1u << 16;
    }
    for (uint32_t t = dfa.terms_begin[current.node]; t < dfa.terms_begin[current.node + 1]; ++t) {
        const terminal term = dfa.terminals[t];
        if (term.kind == any_terminal) return true;
        bool overlap = false;
        for (uint32_t r = term.ranges_begin; r < term.ranges_end; ++r) {
            const range part = dfa.ranges[r];
            overlap |= part.lower <= high && low <= part.upper;
        }
        if (overlap == (term.kind == char_terminal)) return true;
    }
    return false;
}

// Match the existing decoder, including its treatment of invalid continuation bytes within one piece.
__device__ inline state consume(const dfa_device & dfa, state current, const char * piece) {
    const int widths[16] = {1,1,1,1,1,1,1,1,0,0,0,0,2,2,3,4};
    const unsigned char * pos = (const unsigned char *) piece;
    const int32_t initial_node = current.node;
    uint32_t value = current.partial_value;
    int remaining = current.remaining;
    while (*pos && remaining > 0) {
        if ((*pos >> 6) != 2) return {current.node, 0, -1, 0};
        value = (value << 6) + (*pos & 63);
        ++pos;
        --remaining;
    }
    if (current.remaining > 0 && remaining == 0) current.node = advance_char(dfa, current.node, value);
    // The reference clears all decoded codepoints on a later invalid lead byte.
    while (*pos) {
        remaining = widths[*pos >> 4] - 1;
        if (remaining < 0) {
            current.node = initial_node;
            current.remaining = -1;
            current.partial_value = 0;
            return current;
        }
        value = *pos & ((1u << (7 - remaining)) - 1);
        ++pos;
        while (*pos && remaining > 0) {
            value = (value << 6) + (*pos & 63);
            ++pos;
            --remaining;
        }
        if (remaining == 0) current.node = advance_char(dfa, current.node, value);
    }
    current.partial_value = value;
    current.remaining = remaining;
    return current;
}

__device__ inline bool allowed(const dfa_device & dfa, const state & initial, const char * piece, bool eog) {
    if (initial.error || initial.node < 0 || initial.node >= dfa.n_states) return false;
    if (eog) return dfa.accept_end[initial.node] != 0;
    if (*piece == 0) return false;
    const state next = consume(dfa, initial, piece);
    return next.node >= 0 && partial_allowed(dfa, next);
}

__device__ inline state consume_accepted(const dfa_device & dfa, state current, const char * piece) {
    if (current.node < 0 || current.node >= dfa.n_states) { current.error = 1; return current; }
    current.node = dfa.without_empty[current.node];
    return consume(dfa, current, piece);
}

static __global__ void mask(const dfa_device dfa, const state * current, const char * pieces,
        const uint32_t * offsets, const int32_t * eog, const float * logits, float * output, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        output[i] = allowed(dfa, *current, pieces + offsets[i], eog[i]) ? logits[i] : -INFINITY;
    }
}

static __global__ void accept(const dfa_device dfa, state * current, const char * pieces,
        const uint32_t * offsets, const int32_t * eog, int32_t token) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (!allowed(dfa, *current, pieces + offsets[token], eog[token])) {
        current->error = 1;
        return;
    }
    if (!eog[token]) *current = consume_accepted(dfa, *current, pieces + offsets[token]);
}

} // namespace gpu_grammar_lab
