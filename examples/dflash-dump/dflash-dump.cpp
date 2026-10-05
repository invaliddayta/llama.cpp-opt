// Dump DFlash training features from a target model.
//
// input  : binary file of records { uint32 n; uint32 tok[n]; }
// output : <out>.feat  f16 [total_tokens][n_layers * n_embd]  (target layer inputs)
//          <out>.topk  { int32 id[K]; f16 logit[K]; } per token (target top-K logits)
//          <out>.idx   { uint64 offset_tokens; uint32 n; } per sequence
//
// usage: llama-dflash-dump -m target.gguf --in seqs.bin --out shard0 --layers 6,20,34,48,62 [--topk 32] [--chunk 1024]

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static uint16_t f32_to_f16(float f) {
    return ggml_fp32_to_fp16(f);
}

int main(int argc, char ** argv) {
    std::string in_path, out_path, layers_str = "6,20,34,48,62";
    int topk  = 32;
    int chunk = 1024;
    int max_seqs = -1;

    // strip our own args, pass the rest to common
    std::vector<char *> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--in" && i + 1 < argc)          { in_path = argv[++i]; }
        else if (a == "--out" && i + 1 < argc)    { out_path = argv[++i]; }
        else if (a == "--layers" && i + 1 < argc) { layers_str = argv[++i]; }
        else if (a == "--topk" && i + 1 < argc)   { topk = std::stoi(argv[++i]); }
        else if (a == "--chunk" && i + 1 < argc)  { chunk = std::stoi(argv[++i]); }
        else if (a == "--max-seqs" && i + 1 < argc) { max_seqs = std::stoi(argv[++i]); }
        else { rest.push_back(argv[i]); }
    }
    if (in_path.empty() || out_path.empty()) {
        fprintf(stderr, "need --in and --out\n");
        return 1;
    }

    std::vector<uint32_t> layers;
    {
        std::stringstream ss(layers_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            layers.push_back((uint32_t) std::stoul(item));
        }
    }

    common_params params;
    params.n_batch  = chunk;
    params.n_ubatch = chunk;
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    params.n_batch  = std::max(params.n_batch, chunk);
    params.n_ubatch = std::max(params.n_ubatch, chunk);

    llama_backend_init();

    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) {
        fprintf(stderr, "failed to init\n");
        return 1;
    }

    const int n_embd  = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_ctx   = llama_n_ctx(ctx);
    const int n_feat  = (int) layers.size() * n_embd;

    for (uint32_t l : layers) {
        llama_set_embeddings_layer_inp(ctx, l, true);
    }

    std::ifstream fin(in_path, std::ios::binary);
    FILE * ffeat = fopen((out_path + ".feat").c_str(), "wb");
    FILE * ftopk = fopen((out_path + ".topk").c_str(), "wb");
    FILE * fidx  = fopen((out_path + ".idx").c_str(),  "wb");
    if (!fin || !ffeat || !ftopk || !fidx) {
        fprintf(stderr, "failed to open files\n");
        return 1;
    }

    std::vector<uint16_t> feat_buf;
    std::vector<uint8_t>  topk_buf;
    const size_t topk_rec = (size_t) topk * (sizeof(int32_t) + sizeof(uint16_t));

    const int n_threads = std::max(1, (int) std::thread::hardware_concurrency() - 2);

    uint64_t total = 0;
    int n_seq = 0;
    const int64_t t_start = ggml_time_us();

    while (true) {
        uint32_t n = 0;
        if (!fin.read((char *) &n, sizeof(n))) {
            break;
        }
        std::vector<llama_token> toks(n);
        fin.read((char *) toks.data(), (std::streamsize) n * sizeof(uint32_t));
        if ((int) n > n_ctx) {
            fprintf(stderr, "seq %d too long (%u > %d)\n", n_seq, n, n_ctx);
            return 1;
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        feat_buf.resize((size_t) n * n_feat);
        topk_buf.resize((size_t) n * topk_rec);

        for (uint32_t off = 0; off < n; off += chunk) {
            const int m = (int) std::min<uint32_t>(chunk, n - off);
            llama_batch batch = llama_batch_init(m, 0, 1);
            for (int i = 0; i < m; ++i) {
                common_batch_add(batch, toks[off + i], off + i, { 0 }, topk > 0);
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "decode failed\n");
                return 1;
            }
            llama_batch_free(batch);

            for (size_t k = 0; k < layers.size(); ++k) {
                const float * src = llama_get_embeddings_layer_inp(ctx, layers[k]);
                for (int i = 0; i < m; ++i) {
                    uint16_t * dst = feat_buf.data() + (size_t) (off + i) * n_feat + k * n_embd;
                    const float * s = src + (size_t) i * n_embd;
                    for (int j = 0; j < n_embd; ++j) {
                        dst[j] = f32_to_f16(s[j]);
                    }
                }
            }

            if (topk > 0) {
                const float * logits = llama_get_logits(ctx);
                auto work = [&](int beg, int end) {
                    std::vector<int32_t> idx(n_vocab);
                    for (int i = beg; i < end; ++i) {
                        const float * row = logits + (size_t) i * n_vocab;
                        std::iota(idx.begin(), idx.end(), 0);
                        std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(),
                                [row](int32_t a, int32_t b) { return row[a] > row[b]; });
                        uint8_t * rec = topk_buf.data() + (size_t) (off + i) * topk_rec;
                        int32_t  * ids = (int32_t *) rec;
                        uint16_t * lg  = (uint16_t *) (rec + topk * sizeof(int32_t));
                        for (int k = 0; k < topk; ++k) {
                            ids[k] = idx[k];
                            lg[k]  = f32_to_f16(row[idx[k]]);
                        }
                    }
                };
                std::vector<std::thread> th;
                const int per = (m + n_threads - 1) / n_threads;
                for (int t = 0; t < n_threads; ++t) {
                    const int b = t * per, e = std::min(m, b + per);
                    if (b < e) {
                        th.emplace_back(work, b, e);
                    }
                }
                for (auto & x : th) {
                    x.join();
                }
            }
        }

        fwrite(feat_buf.data(), sizeof(uint16_t), feat_buf.size(), ffeat);
        if (topk > 0) {
            fwrite(topk_buf.data(), 1, topk_buf.size(), ftopk);
        }
        fwrite(&total, sizeof(total), 1, fidx);
        fwrite(&n, sizeof(n), 1, fidx);
        total += n;
        n_seq++;

        if (n_seq % 20 == 0) {
            const double s = (ggml_time_us() - t_start) / 1e6;
            fprintf(stderr, "seqs=%d tokens=%llu  %.1f tok/s\n", n_seq, (unsigned long long) total, total / s);
        }
        if (max_seqs > 0 && n_seq >= max_seqs) {
            break;
        }
    }

    fclose(ffeat);
    fclose(ftopk);
    fclose(fidx);
    fprintf(stderr, "done: %d seqs, %llu tokens\n", n_seq, (unsigned long long) total);

    llama_backend_free();
    return 0;
}
