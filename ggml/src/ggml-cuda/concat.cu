#include "concat.cuh"

#include <algorithm>
#include <array>
#include <climits>
#include <stdint.h>

static bool concat_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
        t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// contiguous kernels
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_cont(const T * x,
                                                                             const T * y,
                                                                             T *       dst,
                                                                             int64_t   ne00,
                                                                             int64_t   ne01,
                                                                             int64_t   ne02,
                                                                             int64_t   ne0,
                                                                             int64_t   ne1,
                                                                             int64_t   ne2) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

    const int64_t n = ne0 * ne1 * ne2;

    ggml_cuda_pdl_sync();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        if constexpr (dim == 0) {
            const int64_t row = i / ne0;
            const int64_t i0  = i - row * ne0;

            if (i0 < ne00) {
                dst[i] = x[row * ne00 + i0];
            } else {
                dst[i] = y[row * (ne0 - ne00) + (i0 - ne00)];
            }
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0 * ne1;
            const int64_t src0_plane = ne0 * ne01;
            const int64_t src1_plane = dst_plane - src0_plane;
            const int64_t i2         = i / dst_plane;
            const int64_t i01        = i - i2 * dst_plane;

            if (i01 < src0_plane) {
                dst[i] = x[i2 * src0_plane + i01];
            } else {
                dst[i] = y[i2 * src1_plane + (i01 - src0_plane)];
            }
        } else {
            const int64_t src0_size = ne0 * ne1 * ne02;

            if (i < src0_size) {
                dst[i] = x[i];
            } else {
                dst[i] = y[i - src0_size];
            }
        }
    }
}

template <typename T>
static void concat_cont_cuda(const T * x,
                             const T * y,
                             T *       dst,
                             int64_t   ne00,
                             int64_t   ne01,
                             int64_t   ne02,
                             int64_t   ne0,
                             int64_t   ne1,
                             int64_t   ne2,
                             int       dim,
                             cudaStream_t stream) {
    const int64_t n          = ne0 * ne1 * ne2;
    const int     num_blocks = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;

    if (dim == 0) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream);
        ggml_cuda_kernel_launch(concat_cont<T, 0>, launch_params, x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    if (dim == 1) {
        concat_cont<T, 1><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    concat_cont<T, 2><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
}

// non-contiguous kernel (slow)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
            } else if constexpr (dim == 1) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
            } else if constexpr (dim == 2) {
                x = (const T *)(src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
            } else if constexpr (dim == 3) {
                x = (const T *)(src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

// non-contiguous, small ne0: flat index with i1 fastest, so reads of a transposed src are coalesced
template <typename T, int dim>
static __global__ void concat_non_cont_small(
        const char * src0, const char * src1, char * dst,
        const int ne00, const int ne01, const int ne02, const int ne03,
        const uint64_t nb00, const uint64_t nb01, const uint64_t nb02, const uint64_t nb03,
        const uint64_t nb10, const uint64_t nb11, const uint64_t nb12, const uint64_t nb13,
        const int ne0, const int ne1, const int ne2, const int64_t n,
        const uint64_t nb0, const uint64_t nb1, const uint64_t nb2, const uint64_t nb3) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    const int i1 = t % ne1;
    int64_t r = t / ne1;
    const int i0 = r % ne0;
    r /= ne0;
    const int i2 = r % ne2;
    const int i3 = r / ne2;

    const T * x;
    if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
        x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
    } else if constexpr (dim == 0) {
        x = (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
    } else if constexpr (dim == 1) {
        x = (const T *)(src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
    } else if constexpr (dim == 2) {
        x = (const T *)(src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
    } else {
        x = (const T *)(src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
    }
    *(T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0) = *x;
}

template <typename T>
static void concat_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim, cudaStream_t stream) {
    if (dim != 3 && ggml_is_contiguous_to_3(src0) && ggml_is_contiguous_to_3(src1)) {
        const T * src0_d = (const T *) src0->data;
        const T * src1_d = (const T *) src1->data;
        T *       dst_d  = (T *) dst->data;

        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            concat_cont_cuda(
                    src0_d + i3*(src0->nb[3] / sizeof(T)),
                    src1_d + i3*(src1->nb[3] / sizeof(T)),
                    dst_d  + i3*( dst->nb[3] / sizeof(T)),
                    ggml_row_size(src0->type, src0->ne[0])/sizeof(T), src0->ne[1], src0->ne[2],
                    ggml_row_size(dst->type, dst->ne[0])/sizeof(T),  dst->ne[1],  dst->ne[2], dim, stream);
        }
    } else if (dim == 3 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const size_t size0 = ggml_nbytes(src0);
        const size_t size1 = ggml_nbytes(src1);

        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data,         src0->data, size0, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + size0, src1->data, size1, cudaMemcpyDeviceToDevice, stream));
    } else {
        GGML_ASSERT(!ggml_is_quantized(src0->type));

        const int64_t n = ggml_nelements(dst);
        if (dst->ne[0] <= 32 && n <= INT_MAX) {
            auto launch_small = [&](auto dim) {
                concat_non_cont_small<T, dim><<<(n + 255) / 256, 256, 0, stream>>>(
                    (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                    src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                    dst->ne[0], dst->ne[1], dst->ne[2], n,
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            };
            switch (dim) {
                case 0: launch_small(std::integral_constant<int, 0>{}); break;
                case 1: launch_small(std::integral_constant<int, 1>{}); break;
                case 2: launch_small(std::integral_constant<int, 2>{}); break;
                case 3: launch_small(std::integral_constant<int, 3>{}); break;
                default: GGML_ABORT("Invalid dim: %d", dim);
            }
            return;
        }

        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_non_cont<T, dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(dst->type  == src0->type);

    if (ggml_is_quantized(src0->type)) {
        if (dim == 3) {
            GGML_ASSERT(ggml_is_contiguous(src0));
            GGML_ASSERT(ggml_is_contiguous(src1));
        } else {
            GGML_ASSERT(ggml_is_contiguous_to_3(src0));
            GGML_ASSERT(ggml_is_contiguous_to_3(src1));
        }
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        // if first 3 dimensions are contiguous and ne[0] is multiple of the block size we can concat both tensors as byte tensors
        concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
    } else {
        GGML_ASSERT(ggml_blck_size(src0->type) == 1);

        switch (ggml_type_size(src0->type)) {
            case 1:
                concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
                break;
            case 2:
                concat_cuda<uint16_t>(src0, src1, dst, dim, stream);
                break;
            case 4:
                concat_cuda<uint32_t>(src0, src1, dst, dim, stream);
                break;
            case 8:
                concat_cuda<uint64_t>(src0, src1, dst, dim, stream);
                break;
            default:
                GGML_ABORT("Unsupported type size: %zu", ggml_type_size(src0->type));
                break;
        }
    }
}

// concat along dim 0 that also writes the rolling conv-state snapshots:
// each snapshot slot s holds window [s_idx, s_idx + w) of the concat output, with s_idx = base_s_idx + step * s.
// One kernel instead of a concat plus one copy per slot.
template <typename T>
static __global__ void concat_non_cont_small_snap(
        const char * src0, const char * src1, char * dst,
        const int ne00, const int ne01, const int ne02, const int ne03,
        const uint64_t nb00, const uint64_t nb01, const uint64_t nb02, const uint64_t nb03,
        const uint64_t nb10, const uint64_t nb11, const uint64_t nb12, const uint64_t nb13,
        const int ne0, const int ne1, const int ne2, const int64_t n,
        const uint64_t nb0, const uint64_t nb1, const uint64_t nb2, const uint64_t nb3,
        T * snap, const uint64_t snap_seq, const int64_t snap_slot, const int n_snap,
        const int w, const int base_s_idx, const int step) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    const int i1 = t % ne1;
    int64_t r = t / ne1;
    const int i0 = r % ne0;
    r /= ne0;
    const int i2 = r % ne2;
    const int i3 = r / ne2;

    const char * x;
    if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
        x = src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00;
    } else {
        x = src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10;
    }
    const T v = *(const T *) x;
    *(T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0) = v;

    if (snap != nullptr) {
        // slot s covers columns [base_s_idx + step*s, +w); each slot holds a [w, ne1] window
        T * base = snap + (int64_t) i2 * snap_seq;
        for (int s = 0; s < n_snap; ++s) {
            const int beg = base_s_idx + step * s;
            if (i0 >= beg && i0 < beg + w) {
                base[(int64_t) s * snap_slot + (int64_t) i1 * w + (i0 - beg)] = v;
            }
        }
    }
}

// match the conv-state concat followed by its rolling-window snapshot copies (one cpy per rollback slot)
// and run both in one kernel; returns the number of additional nodes consumed, 0 if no match
int ggml_cuda_try_concat_snap_fusion(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * node = cgraph->nodes[node_idx];
    const int32_t dim  = ((const int32_t *) node->op_params)[0];

    static const bool enabled = getenv("GGML_CUDA_CONCAT_SNAP") == nullptr || atoi(getenv("GGML_CUDA_CONCAT_SNAP")) != 0;
    if (!enabled || node->op != GGML_OP_CONCAT || dim != 0 || node->type != GGML_TYPE_F32 ||
        (node->flags & GGML_TENSOR_FLAG_OUTPUT) || node->ne[0] > 32 || node->ne[3] != 1 || ggml_nelements(node) > INT_MAX) {
        return 0;
    }
    const ggml_tensor * s0 = node->src[0];
    const ggml_tensor * s1 = node->src[1];
    if (s0->type != GGML_TYPE_F32 || s1->type != GGML_TYPE_F32) {
        return 0;
    }
    const int64_t ne1 = node->ne[1], ne2 = node->ne[2], w = s0->ne[0];

    const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t pa = reinterpret_cast<uintptr_t>(a->data);
        const uintptr_t pb = reinterpret_cast<uintptr_t>(b->data);
        return pa <= pb ? pb - pa < ggml_nbytes(a) : pa - pb < ggml_nbytes(b);
    };
    if (overlaps(node, s0) || overlaps(node, s1)) {
        return 0;
    }

    struct snap_cpy { int s_idx; float * dst; const ggml_tensor * tensor; };
    snap_cpy snaps[64];
    int n_snap = 0, last = node_idx;

    for (int j = node_idx + 1; j < cgraph->n_nodes && n_snap < 64; ++j) {
        ggml_tensor * t = cgraph->nodes[j];
        if (concat_is_view_or_noop(t)) {
            continue;
        }
        if (t->op != GGML_OP_CPY || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            break;
        }
        const ggml_tensor * src = t->src[0];
        const ggml_tensor * dst = t->src[1];
        if (src->op != GGML_OP_VIEW || src->view_src != node || dst->op != GGML_OP_VIEW ||
            dst->type != GGML_TYPE_F32 || src->view_offs % sizeof(float) != 0) {
            break;
        }
        const std::array<int64_t, GGML_MAX_DIMS> want_ne = { w, ne1, ne2, 1 };
        if (!std::equal(want_ne.begin(), want_ne.end(), src->ne) ||
            src->nb[0] != sizeof(float) || src->nb[1] != node->nb[1] || src->nb[2] != node->nb[2] ||
            dst->ne[0] != w * ne1 || dst->ne[1] != ne2 || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            dst->nb[0] != sizeof(float) || dst->nb[1] != (uint64_t) w * ne1 * sizeof(float)) {
            break;
        }
        if (src->view_offs / sizeof(float) > (uint64_t) node->ne[0] ||
            overlaps(dst, node) || overlaps(dst, s0) || overlaps(dst, s1)) {
            return 0;
        }
        for (int i = 0; i < n_snap; ++i) {
            if (overlaps(dst, snaps[i].tensor)) {
                return 0;
            }
        }
        snaps[n_snap++] = { (int) (src->view_offs / sizeof(float)), (float *) dst->data, dst };
        last = j;
    }
    if (n_snap < 2) {
        return 0;
    }
    // order the slots by destination address; require an arithmetic window sequence (step -1 after the sort)
    std::sort(snaps, snaps + n_snap, [](const snap_cpy & a, const snap_cpy & b) {
        return reinterpret_cast<uintptr_t>(a.dst) < reinterpret_cast<uintptr_t>(b.dst);
    });
    const uintptr_t first = reinterpret_cast<uintptr_t>(snaps[0].dst);
    const uintptr_t slot_bytes = reinterpret_cast<uintptr_t>(snaps[1].dst) - first;
    if (slot_bytes == 0 || slot_bytes % sizeof(float) != 0 || slot_bytes / sizeof(float) > INT_MAX) {
        return 0;
    }
    const int64_t slot = slot_bytes / sizeof(float);
    for (int i = 1; i < n_snap; ++i) {
        if (reinterpret_cast<uintptr_t>(snaps[i].dst) - first != slot_bytes * i || snaps[i].s_idx != snaps[0].s_idx - i) {
            return 0;
        }
    }
    const int w_max = snaps[0].s_idx + w;
    if (w_max > node->ne[0] || snaps[n_snap - 1].s_idx < 0) {
        return 0;
    }

    cudaStream_t stream = ctx.stream();
    const int64_t n = ggml_nelements(node);
    concat_non_cont_small_snap<unsigned int><<<(n + 255) / 256, 256, 0, stream>>>(
        (const char *) s0->data, (const char *) s1->data, (char *) node->data,
        s0->ne[0], s0->ne[1], s0->ne[2], s0->ne[3],
        s0->nb[0], s0->nb[1], s0->nb[2], s0->nb[3],
        s1->nb[0], s1->nb[1], s1->nb[2], s1->nb[3],
        node->ne[0], node->ne[1], node->ne[2], n,
        node->nb[0], node->nb[1], node->nb[2], node->nb[3],
        (unsigned int *) snaps[0].dst, (uint64_t) (w * ne1), slot, n_snap, w, snaps[0].s_idx, -1);
    CUDA_CHECK(cudaGetLastError());
    return last - node_idx;
}
