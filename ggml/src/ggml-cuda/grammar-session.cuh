#pragma once
#include "grammar-dfa.cuh"

namespace ggml_gpu_grammar {

__device__ inline bool match_token(const token_pattern & pattern, int32_t token, int32_t & position) {
    if (pattern.size == 0) return false;
    while (position > 0 && pattern.tokens[position] != token) position = pattern.fallback[position - 1];
    if (pattern.tokens[position] == token) ++position;
    if (position != pattern.size) return false;
    position = 0;
    return true;
}

__device__ inline void grammar_accept(const dfa_device & dfa, const session_config & config, session_state & current,
        const char * piece, int32_t length, bool eog, int32_t token = -1) {
    if (current.grammar.error || config.trigger_size < 0) return;
    if (current.awaiting_trigger) {
        if (config.trigger_token >= 0) {
            if (token != config.trigger_token) return;
            current.awaiting_trigger = 0;
            current.grammar = consume_accepted(dfa, current.grammar, piece);
            if (current.grammar.node < 0 || current.grammar.remaining < 0) current.grammar.error = 1;
            return;
        }
        const char * pos = piece;
        while (pos < piece + length && current.awaiting_trigger) {
            while (current.trigger_position > 0 && config.trigger[current.trigger_position] != *pos) {
                current.trigger_position = config.trigger_fallback[current.trigger_position - 1];
            }
            if (config.trigger[current.trigger_position] == *pos) ++current.trigger_position;
            ++pos;
            if (current.trigger_position == config.trigger_size) {
                current.awaiting_trigger = 0;
                current.trigger_position = 0;
                current.grammar = consume_accepted(dfa, current.grammar, config.trigger);
                if (*pos) current.grammar = consume(dfa, current.grammar, pos);
            }
        }
    } else if (!eog) {
        current.grammar = consume_accepted(dfa, current.grammar, piece);
    } else if (current.grammar.node < 0 || current.grammar.node >= dfa.n_states || !dfa.accept_end[current.grammar.node]) {
        current.grammar.error = 1;
    }
    if (current.grammar.node < 0 || current.grammar.remaining < 0) current.grammar.error = 1;
}

__device__ inline void session_accept(const dfa_device & dfa, const session_config & config, session_state & current,
        int32_t token, const char * pieces, const uint32_t * offsets, const int32_t * eog, bool reasoning_only = false) {
    const bool apply_grammar = !reasoning_only && !current.thinking;
    if (!reasoning_only) ++current.draws;
    if (current.thinking) {
        int longest = -1;
        for (int i = 0; i < config.n_reasoning_end; ++i) {
            if (match_token(config.reasoning_end[i], token, current.end_position[i]) &&
                    (longest < 0 || config.reasoning_end[i].size > config.reasoning_end[longest].size)) longest = i;
        }
        if (longest >= 0) {
            current.thinking = 0;
            for (int i = 0; i < config.n_reasoning_end; ++i) current.end_position[i] = 0;
            const token_pattern & pattern = config.reasoning_end[longest];
            for (int i = 0; !reasoning_only && i < pattern.size; ++i) {
                const int32_t end_token = pattern.tokens[i];
                grammar_accept(dfa, config, current, pieces + offsets[end_token], offsets[end_token + 1] - offsets[end_token] - 1, eog[end_token], end_token);
            }
        }
    } else if (match_token(config.reasoning_start, token, current.start_position)) {
        current.thinking = 1;
        for (int i = 0; i < config.n_reasoning_end; ++i) current.end_position[i] = 0;
    }
    if (apply_grammar) grammar_accept(dfa, config, current, pieces + offsets[token], offsets[token + 1] - offsets[token] - 1, eog[token], token);
}

__device__ inline void session_prepare_impl(const dfa_device dfa, const session_config config,
        session_state * base, session_state * working, const int32_t * pending, const int32_t * previous,
        const char * pieces, const uint32_t * offsets, const int32_t * eog, int count) {
    if (previous) {
        if (*previous < 0 || *previous >= count) working->grammar.error = 1;
        else session_accept(dfa, config, *working, *previous, pieces, offsets, eog);
    } else {
        if (pending[0] < 0 || pending[0] > max_pending_tokens || pending[1] < 0 || pending[1] > pending[0]) {
            base->grammar.error = 1;
            *working = *base;
            return;
        }
        for (int i = 0; i < pending[0]; ++i) {
            const int32_t token = pending[i + 2];
            if (token < 0 || token >= count) base->grammar.error = 1;
            else session_accept(dfa, config, *base, token, pieces, offsets, eog, i < pending[1]);
        }
        *working = *base;
    }
}

} // namespace ggml_gpu_grammar
