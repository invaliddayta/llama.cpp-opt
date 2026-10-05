#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// fuse the conv-state concat with its rolling-window snapshot copies; returns nodes consumed, 0 if no match
int ggml_cuda_try_concat_snap_fusion(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);
