#pragma once

#include "common.cuh"

// small-batch quantized mul_mat (int8 tensor cores, weight-bandwidth bound), see mmsq-kernels.cuh
bool ggml_cuda_should_use_mmsq(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_sq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// drop the cached quantized activations if nodes [first, last] wrote over their source (or ran on a side stream)
void ggml_cuda_mmsq_note_writes(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int first, int last);

// F32 weights with few rows and a small batch (split over K)
bool ggml_cuda_should_use_mul_mat_f32_small(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_f32_small(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
