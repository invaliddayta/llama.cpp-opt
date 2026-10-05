#pragma once
#include "common.cuh"
#include "grammar-session.cuh"
#include "philox.cuh"

namespace gpu_grammar_lab {

template<class T> __device__ inline const T * part(const char * bytes, uint32_t offset) {
    return reinterpret_cast<const T *>(bytes + offset);
}

__device__ inline dfa_device unpack_dfa(const char * bytes) {
    const auto & h = *reinterpret_cast<const packed_header *>(bytes);
    return {part<uint32_t>(bytes, h.classes), part<int32_t>(bytes, h.next), part<uint32_t>(bytes, h.terms_begin),
        part<int32_t>(bytes, h.accept_end), part<int32_t>(bytes, h.without_empty), part<terminal>(bytes, h.terminals),
        part<range>(bytes, h.ranges), h.n_classes, h.n_states};
}

__device__ inline session_config unpack_config(const char * bytes) {
    const auto & h = *reinterpret_cast<const packed_header *>(bytes);
    session_config config{};
    config.trigger = bytes + h.trigger;
    config.trigger_fallback = part<int32_t>(bytes, h.trigger_fallback);
    config.trigger_size = h.trigger_size;
    config.trigger_token = h.trigger_token;
    config.reasoning_start = h.reasoning_start;
    config.n_reasoning_end = h.n_reasoning_end;
    for (int i = 0; i < max_end_patterns; ++i) config.reasoning_end[i] = h.reasoning_end[i];
    return config;
}

static __global__ void prepare_packed(const char * bytes, persistent_state * current,
        const int32_t * pending, const int32_t * previous) {
    if (threadIdx.x || blockIdx.x) return;
    const auto & h = *reinterpret_cast<const packed_header *>(bytes);
    const uint64_t packet = (uint32_t) pending[0] | ((uint64_t) (uint32_t) pending[1] << 32);
    if (!previous && packet == current->packet) {
        current->working = current->base;
        return;
    }
    session_prepare_impl(unpack_dfa(bytes), unpack_config(bytes), &current->base, &current->working,
        pending + 2, previous, bytes + h.pieces, part<uint32_t>(bytes, h.offsets), part<int32_t>(bytes, h.eog), h.n_vocab);
    if (!previous) current->packet = packet;
}

static __global__ void mask_packed(const char * bytes, const persistent_state * current,
        const float * logits, float * output, int count) {
    const auto & h = *reinterpret_cast<const packed_header *>(bytes);
    const dfa_device dfa = unpack_dfa(bytes);
    const auto & s = current->working;
    const bool passthrough = s.thinking || s.awaiting_trigger;
    const auto * offsets = part<uint32_t>(bytes, h.offsets);
    const auto * eog = part<int32_t>(bytes, h.eog);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        output[i] = !s.grammar.error && (passthrough || allowed(dfa, s.grammar, bytes + h.pieces + offsets[i], eog[i])) ? logits[i] : -INFINITY;
    }
}

static __global__ void uniform_packed(const char * bytes, const persistent_state * current, float * output) {
    if (!threadIdx.x && !blockIdx.x) {
        const auto & h = *reinterpret_cast<const packed_header *>(bytes);
        *output = gpu_rng_lab::uniform(current->working.draws, h.seed);
    }
}

static __global__ void check_sample(const int32_t * sampled, const persistent_state * current,
        const float * final_logit, int32_t * output, int32_t n_vocab) {
    if (threadIdx.x || blockIdx.x) return;
    const int32_t token = *sampled;
    *output = !current->working.grammar.error && token >= 0 && token < n_vocab && isfinite(*final_logit) ? token : -1;
}

} // namespace gpu_grammar_lab

static void ggml_cuda_op_grammar_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * tables = dst->src[1];
    auto * state = dst->src[2];
    const auto * pending = dst->src[3];
    const auto * previous = dst->src[4];
    GGML_ASSERT(ggml_nbytes(state) == sizeof(gpu_grammar_lab::persistent_state));
    GGML_ASSERT(ggml_nbytes(pending) == (gpu_grammar_lab::max_pending_tokens + 4) * sizeof(int32_t));
    gpu_grammar_lab::prepare_packed<<<1, 1, 0, ctx.stream()>>>((const char *) tables->data,
        (gpu_grammar_lab::persistent_state *) state->data, (const int32_t *) pending->data,
        previous ? (const int32_t *) previous->data : nullptr);
    const int count = ggml_nelements(dst);
    gpu_grammar_lab::mask_packed<<<std::min(1024, (count + 127) / 128), 128, 0, ctx.stream()>>>(
        (const char *) tables->data, (const gpu_grammar_lab::persistent_state *) state->data,
        (const float *) dst->src[0]->data, (float *) dst->data, count);
}

static void ggml_cuda_op_gpu_uniform(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    gpu_grammar_lab::uniform_packed<<<1, 1, 0, ctx.stream()>>>((const char *) dst->src[1]->data,
        (const gpu_grammar_lab::persistent_state *) dst->src[2]->data, (float *) dst->data);
}

static void ggml_cuda_op_gpu_sample_check(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    gpu_grammar_lab::check_sample<<<1, 1, 0, ctx.stream()>>>((const int32_t *) dst->src[0]->data,
        (const gpu_grammar_lab::persistent_state *) dst->src[1]->data, (const float *) dst->src[2]->data,
        (int32_t *) dst->data, ggml_get_op_params_i32(dst, 0));
}
