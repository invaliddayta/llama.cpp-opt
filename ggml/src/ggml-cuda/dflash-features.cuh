#pragma once

#include "common.cuh"

static cudaError_t dflash_stage_features(const float * src, float * dst, size_t hidden,
        size_t layers, size_t layer, size_t begin, size_t rows, cudaStream_t stream) {
    return cudaMemcpy2DAsync(dst + begin * hidden * layers + layer * hidden,
            hidden * layers * sizeof(float), src, hidden * sizeof(float),
            hidden * sizeof(float), rows, cudaMemcpyDeviceToDevice, stream);
}

static cudaError_t dflash_copy_features(const float * src, float * dst, size_t width,
        size_t begin, size_t rows, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src + begin * width, rows * width * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}
