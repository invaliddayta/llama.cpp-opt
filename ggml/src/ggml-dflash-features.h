#pragma once

#include "ggml-backend.h"

using ggml_backend_dflash_features_stage_t = bool (*)(ggml_backend_t, const ggml_tensor *, ggml_tensor *, size_t, size_t);
using ggml_backend_dflash_features_copy_t = bool (*)(ggml_backend_t, const ggml_tensor *, ggml_tensor *, size_t);
