#pragma once
#include <cstdint>

namespace gpu_grammar_lab {

constexpr int char_terminal = 3;
constexpr int any_terminal = 7;
constexpr int max_end_patterns = 4;
constexpr int max_pattern_tokens = 8;
constexpr int max_pending_tokens = 256;

struct range { uint32_t lower, upper; };
struct terminal { uint32_t ranges_begin, ranges_end; int32_t kind; };
struct dfa_device {
    const uint32_t * classes;
    const int32_t * next;
    const uint32_t * terms_begin;
    const int32_t * accept_end;
    const int32_t * without_empty;
    const terminal * terminals;
    const range * ranges;
    int32_t n_classes, n_states;
};
struct state { int32_t node; uint32_t partial_value; int32_t remaining, error; };
struct token_pattern { int32_t tokens[max_pattern_tokens], fallback[max_pattern_tokens], size; };
struct session_config {
    const char * trigger;
    const int32_t * trigger_fallback;
    int32_t trigger_size;
    int32_t trigger_token = -1;
    token_pattern reasoning_start, reasoning_end[max_end_patterns];
    int32_t n_reasoning_end;
};
struct session_state {
    state grammar;
    int32_t awaiting_trigger, trigger_position, thinking, start_position, end_position[max_end_patterns];
    uint64_t draws;
};
struct persistent_state {
    session_state base, working;
    uint64_t packet;
};
struct packed_header {
    uint64_t seed;
    int32_t n_vocab, n_classes, n_states;
    uint32_t classes, next, terms_begin, accept_end, without_empty, terminals, ranges;
    uint32_t pieces, offsets, eog, trigger, trigger_fallback;
    int32_t trigger_size;
    int32_t trigger_token = -1;
    token_pattern reasoning_start, reasoning_end[max_end_patterns];
    int32_t n_reasoning_end;
};

} // namespace gpu_grammar_lab
