#pragma once

#include "llama-grammar.h"
#include "../ggml/src/ggml-gpu-grammar.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace gpu_grammar_lab {

static_assert(char_terminal == LLAMA_GRETYPE_CHAR && any_terminal == LLAMA_GRETYPE_CHAR_ANY, "grammar terminal layout mismatch");

struct dfa_host {
    std::vector<uint32_t> classes;
    std::vector<int32_t> next;
    std::vector<uint32_t> terms_begin;
    std::vector<int32_t> accept_end;
    std::vector<int32_t> without_empty;
    std::vector<terminal> terminals;
    std::vector<range> ranges;
    std::vector<std::vector<std::vector<int32_t>>> stacks;
};

// Compile at setup only. Recursive languages hit the state cap and are rejected, never approximated.
inline dfa_host compile(llama_grammar & grammar, size_t max_states = 16384) {
    if (grammar.partial_utf8.n_remain != 0 || grammar.lazy) {
        throw std::runtime_error("DFA compilation needs an unconsumed, non-lazy grammar");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    dfa_host out;
    out.classes.push_back(0);
    std::vector<const llama_grammar_element *> elements;
    std::unordered_map<const llama_grammar_element *, int32_t> indices;
    std::unordered_map<const llama_grammar_element *, size_t> rule_of;
    if (grammar.rules.size() > 4096) throw std::runtime_error("GPU grammar rule limit exceeded");
    std::vector<std::vector<std::pair<size_t, bool>>> edges(grammar.rules.size());
    for (size_t rule_id = 0; rule_id < grammar.rules.size(); ++rule_id) {
        const auto & rule = grammar.rules[rule_id];
        for (size_t i = 0; i < rule.size(); ++i) {
        const auto & elem = rule[i];
        if (elements.size() >= 200000) throw std::runtime_error("GPU grammar element limit exceeded");
        rule_of[&elem] = rule_id;
        indices[&elem] = (int32_t) elements.size();
        elements.push_back(&elem);
        if (elem.type == LLAMA_GRETYPE_RULE_REF) {
            if (elem.value >= grammar.rules.size() || i + 1 >= rule.size()) throw std::runtime_error("invalid rule reference");
            const bool growing = rule[i + 1].type != LLAMA_GRETYPE_END && rule[i + 1].type != LLAMA_GRETYPE_ALT;
            edges[rule_id].push_back({elem.value, growing});
        }
        if (elem.type == LLAMA_GRETYPE_CHAR || elem.type == LLAMA_GRETYPE_CHAR_NOT ||
                elem.type == LLAMA_GRETYPE_CHAR_ALT || elem.type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            out.classes.push_back(elem.value);
            if (elem.value < UINT32_MAX) out.classes.push_back(elem.value + 1);
        }
        }
    }
    std::vector<bool> reachable(grammar.rules.size(), false);
    std::vector<size_t> todo;
    size_t preflight_work = 0;
    auto preflight_check = [&]() {
        if (++preflight_work > 10000000 || ((preflight_work & 1023) == 0 && std::chrono::steady_clock::now() >= deadline)) {
            throw std::runtime_error("GPU grammar preflight-work/deadline limit exceeded");
        }
    };
    for (const auto & stack : grammar.stacks) for (const auto * pos : stack) todo.push_back(rule_of.at(pos));
    while (!todo.empty()) {
        preflight_check();
        const auto id = todo.back(); todo.pop_back();
        if (reachable[id]) continue;
        reachable[id] = true;
        for (auto edge : edges[id]) { preflight_check(); todo.push_back(edge.first); }
    }
    // A recursive edge with a return suffix grows continuation stacks. Reject before expansion.
    for (size_t from = 0; from < edges.size(); ++from) if (reachable[from]) {
        for (auto edge : edges[from]) if (edge.second) {
            preflight_check();
            std::vector<bool> visited(edges.size(), false);
            todo = {edge.first};
            while (!todo.empty()) {
                preflight_check();
                const auto id = todo.back(); todo.pop_back();
                if (id == from) throw std::runtime_error("stack-growing recursive grammar is unsupported on GPU");
                if (visited[id]) continue;
                visited[id] = true;
                for (auto next : edges[id]) { preflight_check(); todo.push_back(next.first); }
            }
        }
    }
    std::sort(out.classes.begin(), out.classes.end());
    out.classes.erase(std::unique(out.classes.begin(), out.classes.end()), out.classes.end());
    using key = std::vector<std::vector<int32_t>>;
    auto canonical = [&](const llama_grammar_stacks & stacks) {
        key result;
        for (const auto & stack : stacks) {
            std::vector<int32_t> row;
            for (const auto * pos : stack) row.push_back(indices.at(pos));
            result.push_back(std::move(row));
        }
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    };
    std::map<key, int32_t> seen;
    const auto initial = grammar.stacks;
    constexpr size_t max_depth = 256, max_stacks = 1024, max_stack_entries = 32768, max_epsilon_work = 32768;
    size_t total_state_entries = 0;
    auto intern = [&](key candidate) {
        if (candidate.empty()) return int32_t(-1);
        const auto found = seen.find(candidate);
        if (found != seen.end()) return found->second;
        if (out.stacks.size() >= max_states || out.stacks.size() * out.classes.size() >= 8000000) {
            throw std::runtime_error("grammar is not a bounded DFA; refusing approximation/CPU fallback");
        }
        size_t entries = 0;
        for (const auto & row : candidate) {
            if (row.size() > max_depth) throw std::runtime_error("GPU grammar continuation-depth limit exceeded");
            entries += row.size();
        }
        if (candidate.size() > max_stacks || entries > max_stack_entries || total_state_entries + entries > 8000000) {
            throw std::runtime_error("GPU grammar stack-entry limit exceeded");
        }
        total_state_entries += entries;
        const int32_t index = (int32_t) out.stacks.size();
        seen.emplace(candidate, index);
        out.stacks.push_back(std::move(candidate));
        return index;
    };
    auto end_sequence = [](const llama_grammar_element * pos) { return pos->type == LLAMA_GRETYPE_END || pos->type == LLAMA_GRETYPE_ALT; };
    auto step = [&](const llama_grammar_stacks & stacks, uint32_t character) {
        llama_grammar_stacks todo_stacks, result;
        for (const auto & stack : stacks) {
            if (stack.empty()) continue;
            const auto * pos = stack.back();
            if (pos->type == LLAMA_GRETYPE_TOKEN || pos->type == LLAMA_GRETYPE_TOKEN_NOT) {
                throw std::runtime_error("token terminals need a separate GPU transition table");
            }
            const bool positive = pos->type == LLAMA_GRETYPE_CHAR || pos->type == LLAMA_GRETYPE_CHAR_ANY;
            bool match = false;
            do {
                if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
                    match |= pos->value <= character && character <= pos[1].value; pos += 2;
                } else {
                    match |= pos->type == LLAMA_GRETYPE_CHAR_ANY || pos->value == character; ++pos;
                }
            } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);
            if (match != positive) continue;
            llama_grammar_stack next(stack.begin(), stack.end() - 1);
            if (!end_sequence(pos)) next.push_back(pos);
            todo_stacks.push_back(std::move(next));
        }
        std::set<std::vector<int32_t>> expanded;
        size_t work = 0, queued_entries = 0;
        for (const auto & row : todo_stacks) queued_entries += row.size();
        while (!todo_stacks.empty()) {
            if (++work > max_epsilon_work || std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("GPU grammar epsilon-work/deadline limit exceeded");
            auto stack = std::move(todo_stacks.back()); todo_stacks.pop_back();
            queued_entries -= stack.size();
            if (stack.size() > max_depth) throw std::runtime_error("GPU grammar continuation-depth limit exceeded");
            std::vector<int32_t> row;
            for (const auto * pos : stack) row.push_back(indices.at(pos));
            if (!expanded.insert(row).second) continue;
            if (stack.empty() || stack.back()->type != LLAMA_GRETYPE_RULE_REF) {
                result.push_back(std::move(stack));
                if (result.size() > max_stacks) throw std::runtime_error("GPU grammar stack-count limit exceeded");
                continue;
            }
            const auto * pos = stack.back();
            const auto * alternative = grammar.rules[pos->value].data();
            while (true) {
                llama_grammar_stack next(stack.begin(), stack.end() - 1);
                if (!end_sequence(pos + 1)) next.push_back(pos + 1);
                if (!end_sequence(alternative)) next.push_back(alternative);
                queued_entries += next.size();
                if (todo_stacks.size() >= max_stacks || queued_entries > max_stack_entries) throw std::runtime_error("GPU grammar epsilon-queue limit exceeded");
                todo_stacks.push_back(std::move(next));
                while (!end_sequence(alternative)) ++alternative;
                if (alternative->type != LLAMA_GRETYPE_ALT) break;
                ++alternative;
            }
        }
        return result;
    };
    intern(canonical(initial));
    try {
        for (size_t state = 0; state < out.stacks.size(); ++state) {
            if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("grammar DFA setup deadline exceeded");
            llama_grammar_stacks stacks;
            std::vector<int32_t> tops;
            bool empty = false;
            key nonempty;
            for (const auto & row : out.stacks[state]) {
                llama_grammar_stack stack;
                for (int32_t pos : row) stack.push_back(elements.at(pos));
                stacks.push_back(std::move(stack));
                if (row.empty()) empty = true;
                else { tops.push_back(row.back()); nonempty.push_back(row); }
            }
            out.without_empty.push_back(intern(std::move(nonempty)));
            std::sort(tops.begin(), tops.end());
            tops.erase(std::unique(tops.begin(), tops.end()), tops.end());
            out.accept_end.push_back(empty);
            out.terms_begin.push_back((uint32_t) out.terminals.size());
            for (int32_t top : tops) {
                const auto * pos = elements.at(top);
                if (pos->type == LLAMA_GRETYPE_TOKEN || pos->type == LLAMA_GRETYPE_TOKEN_NOT) {
                    throw std::runtime_error("token terminals need a separate GPU transition table");
                }
                terminal term{(uint32_t) out.ranges.size(), 0, (int32_t) pos->type};
                do {
                    if (pos->type == LLAMA_GRETYPE_CHAR_ANY) {
                        ++pos;
                    } else if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
                        out.ranges.push_back({pos->value, pos[1].value});
                        pos += 2;
                    } else {
                        out.ranges.push_back({pos->value, pos->value});
                        ++pos;
                    }
                } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);
                term.ranges_end = (uint32_t) out.ranges.size();
                out.terminals.push_back(term);
            }
            for (uint32_t character : out.classes) {
                const int32_t next = intern(canonical(step(stacks, character)));
                out.next.push_back(next);
            }
        }
    } catch (...) {
        grammar.stacks = initial;
        throw;
    }
    grammar.stacks = initial;
    out.terms_begin.push_back((uint32_t) out.terminals.size());
    return out;
}

} // namespace gpu_grammar_lab
