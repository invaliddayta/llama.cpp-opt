// Emit llama.cpp's DFlash drafts at chosen anchors of a fixed token sequence, for parity tests
// against an external (PyTorch) drafter implementation.
//
// input : --in seqs.bin (same format as llama-dflash-dump), --anchors "a1,a2,..." (per first sequence, ascending)
// output: one line per anchor: "<anchor> <tok1> <tok2> ... <tokn>"
//
// usage: llama-dflash-parity -m target.gguf -md draft.gguf --spec-type draft-dflash --spec-draft-n-max 7 --in seq.bin --anchors 120,130

#include "arg.h"
#include "common.h"
#include "speculative.h"
#include "llama.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string in_path, anchors_str;
    std::vector<char *> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--in" && i + 1 < argc)           { in_path = argv[++i]; }
        else if (a == "--anchors" && i + 1 < argc) { anchors_str = argv[++i]; }
        else { rest.push_back(argv[i]); }
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }
    const auto lim = common_speculative_get_output_limits(params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = lim.total;
    params.n_outputs_max_per_seq = lim.per_seq;

    llama_backend_init();
    auto init_tgt = common_init_from_params(params);
    llama_model   * model_tgt = init_tgt->model();
    llama_context * ctx_tgt   = init_tgt->context();

    common_speculative_init_result_ptr spec_init;
    {
        common_params params_dft = common_base_params_to_speculative(params);
        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
    }
    llama_context * ctx_dft = params.speculative.draft.ctx_dft;
    common_speculative * spec = common_speculative_init(params.speculative, 1);
    GGML_ASSERT(spec);

    std::ifstream fin(in_path, std::ios::binary);
    uint32_t n = 0;
    fin.read((char *) &n, sizeof(n));
    std::vector<llama_token> toks(n);
    fin.read((char *) toks.data(), (std::streamsize) n * sizeof(uint32_t));

    std::vector<int> anchors;
    {
        std::stringstream ss(anchors_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            anchors.push_back(std::stoi(item));
        }
    }

    common_speculative_begin(spec, 0, llama_tokens(toks.begin(), toks.begin() + anchors[0]));

    int cur = 0;
    llama_tokens result;
    for (int a : anchors) {
        GGML_ASSERT(a >= cur && a < (int) n);
        // drop the previous noise block from the draft cache
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, cur, -1);
        if (a > cur) {
            llama_batch batch = llama_batch_init(a - cur, 0, 1);
            for (int i = cur; i < a; ++i) {
                common_batch_add(batch, toks[i], i, { 0 }, false);
            }
            GGML_ASSERT(llama_decode(ctx_tgt, batch) == 0);
            GGML_ASSERT(common_speculative_process(spec, batch));
            llama_batch_free(batch);
        }
        cur = a;

        llama_tokens prompt(toks.begin(), toks.begin() + a);
        result.clear();
        auto & dp = common_speculative_get_draft_params(spec, 0);
        dp.drafting = true;
        dp.n_max    = -1;
        dp.pos0     = a;
        dp.id_last  = toks[a];
        dp.prompt   = &prompt;
        dp.result   = &result;
        common_speculative_draft(spec);

        printf("%d", a);
        for (auto t : result) {
            printf(" %d", t);
        }
        printf("\n");
        fflush(stdout);
    }

    common_speculative_free(spec);
    llama_backend_free();
    return 0;
}
