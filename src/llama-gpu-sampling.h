#pragma once
#include "llama.h"
#include <string>
#include <vector>

LLAMA_API bool llama_gpu_sampling_enabled();
LLAMA_API llama_sampler * llama_sampler_init_gpu_grammar(
        const llama_vocab * vocab, const std::string & grammar, const std::string & trigger,
        const std::vector<llama_token> & reasoning_start, const std::vector<std::vector<llama_token>> & reasoning_end,
        const std::vector<llama_token> & prefill, uint32_t seed, bool lazy, llama_token trigger_token = LLAMA_TOKEN_NULL);
