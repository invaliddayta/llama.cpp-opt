#include "llama-gpu-sampling.h"
#include "llama-grammar-dfa.h"
#include "llama-vocab.h"
#include "ggml-cpp.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>

using namespace gpu_grammar_lab;

bool llama_gpu_sampling_enabled() {
    const char * value = std::getenv("LLAMA_GPU_SAMPLING");
    return value && std::strcmp(value, "1") == 0;
}

static std::vector<int32_t> gpu_failure(const std::vector<int32_t> & input) {
    std::vector<int32_t> result(input.size());
    int j = 0;
    for (size_t i = 1; i < input.size(); ++i) {
        while (j > 0 && input[j] != input[i]) j = result[j - 1];
        if (input[j] == input[i]) ++j;
        result[i] = j;
    }
    return result;
}

static token_pattern gpu_pattern(const std::vector<llama_token> & tokens) {
    if (tokens.size() > max_pattern_tokens) throw std::runtime_error("GPU sampling: reasoning tag is too long");
    token_pattern result{};
    const auto fallback = gpu_failure(tokens);
    result.size = tokens.size();
    for (int i = 0; i < result.size; ++i) { result.tokens[i] = tokens[i]; result.fallback[i] = fallback[i]; }
    return result;
}

static std::shared_ptr<const dfa_host> gpu_compile(const llama_vocab * vocab, const std::string & source) {
    static std::mutex mutex;
    static std::map<std::pair<const llama_vocab *, std::string>, std::shared_ptr<const dfa_host>> cache;
    std::lock_guard<std::mutex> guard(mutex);
    auto key = std::make_pair(vocab, source);
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;
    std::unique_ptr<llama_grammar, decltype(&llama_grammar_free_impl)> grammar(
        llama_grammar_init_impl(vocab, source.c_str(), "root", false, nullptr, 0, nullptr, 0), llama_grammar_free_impl);
    if (!grammar) throw std::runtime_error("GPU sampling: grammar parse failed");
    auto result = std::make_shared<dfa_host>(compile(*grammar));
    result->stacks.clear();
    if (cache.size() >= 8) cache.erase(cache.begin());
    cache.emplace(std::move(key), result);
    return result;
}

struct llama_gpu_grammar_tables {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * tensor = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
};

struct llama_gpu_grammar {
    std::shared_ptr<std::vector<char>> packed;
    std::shared_ptr<llama_gpu_grammar_tables> shared_tables;
    std::vector<int32_t> prefill_initial, prefill, pending;
    size_t transmitted = 0;
    bool acknowledged = false;
    uint64_t packet = 1;
    persistent_state initial{};
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * tables = nullptr;
    ggml_tensor * state = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * previous = nullptr;

    void allocate(ggml_backend_buffer_type_t type, bool initialize = true) {
        if (buft) {
            if (buft != type) throw std::runtime_error("GPU sampling: sampler cannot migrate between devices");
            return;
        }
        if (!shared_tables) {
            auto * device = ggml_backend_buft_get_device(type);
            if (!device || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                    std::strncmp(ggml_backend_dev_name(device), "CUDA", 4) != 0) {
                throw std::runtime_error("GPU sampling: CUDA device required; CPU fallback is forbidden");
            }
            auto immutable = std::make_shared<llama_gpu_grammar_tables>();
            immutable->context.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
            if (!immutable->context) throw std::runtime_error("GPU sampling: table context allocation failed");
            immutable->tensor = ggml_new_tensor_1d(immutable->context.get(), GGML_TYPE_I32, packed->size() / sizeof(int32_t));
            ggml_set_name(immutable->tensor, "gpu_grammar_tables");
            immutable->buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(immutable->context.get(), type));
            if (!immutable->buffer) throw std::runtime_error("GPU sampling: table buffer allocation failed");
            immutable->buft = type;
            ggml_backend_tensor_set(immutable->tensor, packed->data(), 0, packed->size());
            shared_tables = std::move(immutable);
        }
        if (shared_tables->buft != type) throw std::runtime_error("GPU sampling: shared tables cannot migrate between devices");
        tables = shared_tables->tensor;
        context.reset(ggml_init({2 * ggml_tensor_overhead(), nullptr, true}));
        if (!context) throw std::runtime_error("GPU sampling: context allocation failed");
        state = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, sizeof(persistent_state) / sizeof(int32_t));
        input = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, max_pending_tokens + 4);
        ggml_set_name(state, "gpu_grammar_state");
        ggml_set_name(input, "gpu_grammar_pending");
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(context.get(), type));
        if (!buffer) throw std::runtime_error("GPU sampling: buffer allocation failed");
        buft = type;
        if (initialize) {
            ggml_backend_tensor_set(state, &initial, 0, sizeof(initial));
            const std::vector<int32_t> zero(max_pending_tokens + 4);
            ggml_backend_tensor_set(input, zero.data(), 0, zero.size() * sizeof(int32_t));
        }
    }

    void copy(const llama_gpu_grammar & src) {
        if (packed != src.packed && *packed != *src.packed) throw std::runtime_error("GPU sampling: incompatible sampler copy");
        prefill = src.prefill;
        prefill_initial = src.prefill_initial;
        pending = src.pending;
        transmitted = src.transmitted;
        acknowledged = src.acknowledged;
        packet = src.packet;
        initial = src.initial;
        if (src.buft) {
            if (!buft) shared_tables = src.shared_tables;
            allocate(src.buft, false);
            ggml_backend_tensor_copy(src.state, state);
            ggml_backend_tensor_copy(src.input, input);
        } else if (buft) {
            ggml_backend_tensor_set(state, &initial, 0, sizeof(initial));
        }
    }
};

static const char * gpu_grammar_name(const llama_sampler *) { return "gpu-grammar"; }
static void gpu_grammar_apply(llama_sampler *, llama_token_data_array *) {
    throw std::runtime_error("GPU sampling: CPU sampler invocation is forbidden");
}
static void gpu_grammar_accept(llama_sampler * sampler, llama_token token) {
    auto & ctx = *static_cast<llama_gpu_grammar *>(sampler->ctx);
    if (!ctx.acknowledged) {
        if (ctx.transmitted > ctx.pending.size()) throw std::runtime_error("GPU sampling: invalid acceptance packet");
        ctx.pending.erase(ctx.pending.begin(), ctx.pending.begin() + ctx.transmitted);
        ctx.prefill.clear();
        ctx.transmitted = 0;
        ctx.acknowledged = true;
        ++ctx.packet;
    }
    if (ctx.pending.size() >= max_pending_tokens) throw std::runtime_error("GPU sampling: accepted-token packet overflow");
    ctx.pending.push_back(token);
}
static void gpu_grammar_reset(llama_sampler * sampler) {
    auto & ctx = *static_cast<llama_gpu_grammar *>(sampler->ctx);
    ctx.prefill = ctx.prefill_initial;
    ctx.pending = ctx.prefill;
    ctx.transmitted = 0;
    ctx.acknowledged = false;
    ctx.packet = 1;
    ctx.previous = nullptr;
    if (ctx.state) ggml_backend_tensor_set(ctx.state, &ctx.initial, 0, sizeof(ctx.initial));
}
static bool gpu_grammar_backend_init(llama_sampler * sampler, ggml_backend_buffer_type_t buft, uint32_t) {
    static_cast<llama_gpu_grammar *>(sampler->ctx)->allocate(buft);
    return true;
}
static void gpu_grammar_backend_reset(llama_sampler * sampler) {
    static_cast<llama_gpu_grammar *>(sampler->ctx)->previous = nullptr;
}
static void gpu_grammar_backend_accept(llama_sampler * sampler, ggml_context *, ggml_cgraph *, ggml_tensor * token) {
    static_cast<llama_gpu_grammar *>(sampler->ctx)->previous = token;
}
static void gpu_grammar_backend_apply(llama_sampler * sampler, ggml_context * ctx, ggml_cgraph *, llama_sampler_data * data) {
    auto & s = *static_cast<llama_gpu_grammar *>(sampler->ctx);
    data->logits = ggml_grammar_mask(ctx, data->logits, s.tables, s.state, s.input, s.previous);
}
static void gpu_grammar_set_input(llama_sampler * sampler) {
    auto & s = *static_cast<llama_gpu_grammar *>(sampler->ctx);
    std::vector<int32_t> packet(max_pending_tokens + 4);
    packet[0] = (uint32_t) s.packet;
    packet[1] = (uint32_t) (s.packet >> 32);
    packet[2] = s.pending.size();
    packet[3] = s.prefill.size();
    std::copy(s.pending.begin(), s.pending.end(), packet.begin() + 4);
    ggml_backend_tensor_set(s.input, packet.data(), 0, packet.size() * sizeof(int32_t));
    s.transmitted = s.pending.size();
    s.acknowledged = false;
}
static void gpu_grammar_copy(const llama_sampler * src, llama_sampler * dst) {
    static_cast<llama_gpu_grammar *>(dst->ctx)->copy(*static_cast<const llama_gpu_grammar *>(src->ctx));
}
static llama_sampler * gpu_grammar_clone(const llama_sampler * sampler);
static void gpu_grammar_free(llama_sampler * sampler) { delete static_cast<llama_gpu_grammar *>(sampler->ctx); }

static llama_sampler_i gpu_grammar_iface = {
    gpu_grammar_name, gpu_grammar_accept, gpu_grammar_apply, gpu_grammar_reset, gpu_grammar_clone, gpu_grammar_free,
    gpu_grammar_backend_init, gpu_grammar_backend_accept, gpu_grammar_backend_apply, gpu_grammar_set_input,
    gpu_grammar_backend_reset, gpu_grammar_copy,
};

static llama_sampler * gpu_grammar_clone(const llama_sampler * sampler) {
    const auto & src = *static_cast<const llama_gpu_grammar *>(sampler->ctx);
    auto dst = std::make_unique<llama_gpu_grammar>();
    dst->packed = src.packed;
    dst->copy(src);
    return llama_sampler_init(&gpu_grammar_iface, dst.release());
}

LLAMA_API llama_sampler * llama_sampler_init_gpu_grammar(
        const llama_vocab * vocab, const std::string & source, const std::string & trigger,
        const std::vector<llama_token> & start, const std::vector<std::vector<llama_token>> & end,
        const std::vector<llama_token> & prefill, uint32_t seed, bool lazy, llama_token trigger_token) {
    if (source.size() > 200000 || trigger.size() > 256 || (!source.empty() && lazy && trigger.empty() && trigger_token < 0) ||
            end.size() > max_end_patterns || prefill.size() > max_pending_tokens || trigger.find('\0') != std::string::npos) {
        throw std::runtime_error("GPU sampling: unsupported grammar/trigger/tag limits");
    }
    const auto dfa = gpu_compile(vocab, source.empty() ? "root ::= .*\n" : source);
    auto s = std::make_unique<llama_gpu_grammar>();
    s->packed = std::make_shared<std::vector<char>>(sizeof(packed_header));
    packed_header h{};
    h.seed = seed == LLAMA_DEFAULT_SEED ? std::random_device{}() : seed;
    h.n_vocab = vocab->n_tokens();
    h.n_classes = dfa->classes.size();
    h.n_states = dfa->accept_end.size();
    auto append = [&](const auto & values) {
        s->packed->resize((s->packed->size() + 7) & ~size_t(7));
        uint32_t offset = s->packed->size();
        const char * data = reinterpret_cast<const char *>(values.data());
        if (!values.empty()) s->packed->insert(s->packed->end(), data, data + values.size() * sizeof(values[0]));
        return offset;
    };
    h.classes = append(dfa->classes); h.next = append(dfa->next); h.terms_begin = append(dfa->terms_begin);
    h.accept_end = append(dfa->accept_end); h.without_empty = append(dfa->without_empty);
    h.terminals = append(dfa->terminals); h.ranges = append(dfa->ranges);
    std::vector<char> pieces; std::vector<uint32_t> offsets; std::vector<int32_t> eog;
    for (int i = 0; i < h.n_vocab; ++i) {
        offsets.push_back(pieces.size());
        const auto & piece = vocab->token_to_piece(i);
        pieces.insert(pieces.end(), piece.begin(), piece.end()); pieces.push_back(0);
        eog.push_back(vocab->is_eog(i));
    }
    offsets.push_back(pieces.size());
    h.pieces = append(pieces); h.offsets = append(offsets); h.eog = append(eog);
    std::vector<char> text(trigger.begin(), trigger.end()); text.push_back(0);
    h.trigger = append(text);
    h.trigger_fallback = append(gpu_failure(std::vector<int32_t>(trigger.begin(), trigger.end())));
    h.trigger_size = source.empty() ? -1 : (int32_t) trigger.size();
    if (trigger_token < LLAMA_TOKEN_NULL || trigger_token >= h.n_vocab) throw std::runtime_error("GPU sampling: invalid trigger token");
    h.trigger_token = trigger_token;
    h.reasoning_start = gpu_pattern(start);
    h.n_reasoning_end = end.size();
    for (size_t i = 0; i < end.size(); ++i) h.reasoning_end[i] = gpu_pattern(end[i]);
    std::memcpy(s->packed->data(), &h, sizeof(h));
    s->packed->resize((s->packed->size() + 7) & ~size_t(7));
    s->initial.base.grammar.node = 0;
    s->initial.base.awaiting_trigger = source.empty() || lazy;
    s->initial.working = s->initial.base;
    s->prefill = prefill;
    s->prefill_initial = prefill;
    s->pending = prefill;
    return llama_sampler_init(&gpu_grammar_iface, s.release());
}
