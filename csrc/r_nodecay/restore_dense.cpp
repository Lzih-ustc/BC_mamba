/******************************************************************************
 * restore_dense.cpp
 * PyTorch/pybind11 entry-point for restore_dense_fwd + restore_dense_bwd.
 *
 * v2: zero-order hold (no decay). log_gamma removed from both kernels.
 *
 * Exposed Python API:
 *   restore_dense_cuda.restore_dense_fwd(
 *       y_flat,        # (E,)            fp16/bf16/fp32
 *       time_steps,    # (E,)            int32  (sorted per-channel ascending)
 *       cu_seqlens,    # (n_channels+1,) int32
 *       dim,           # int             D
 *       seqlen,        # int             L
 *       out_dtype,     # optional torch.dtype (defaults to y_flat.dtype)
 *   ) -> Tensor  (B, L, D)
 *
 *   restore_dense_cuda.restore_dense_bwd(
 *       dout,          # (B, L, D)       same dtype as y_flat
 *       time_steps,    # (E,)            int32
 *       cu_seqlens,    # (n_channels+1,) int32
 *       dim,           # int             D
 *       seqlen,        # int             L
 *   ) -> dy_flat       # (E,) float32
 *****************************************************************************/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "restore_dense.h"
#include "restore_dense_bwd.h"

// Forward declarations of templated launchers (defined in .cu files)
template<typename input_t, typename output_t>
void restore_dense_fwd_cuda(RestoreDenseParams &p, cudaStream_t stream);

template<typename input_t>
void restore_dense_bwd_cuda(RestoreDenseBwdParams &p, cudaStream_t stream);

// --------------------------------------------------------------------------
// Forward
// --------------------------------------------------------------------------
at::Tensor restore_dense_fwd(
    const at::Tensor &y_flat,       // (E,)            fp16/bf16/fp32
    const at::Tensor &time_steps,   // (E,)            int32
    const at::Tensor &cu_seqlens,   // (n_channels+1,) int32
    int64_t dim,                    // D
    int64_t seqlen,                 // L
    const at::optional<at::ScalarType> out_dtype_opt
) {
    // ---- dtype checks ----
    auto input_type = y_flat.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float  ||
                input_type == at::ScalarType::Half    ||
                input_type == at::ScalarType::BFloat16,
                "restore_dense_fwd: y_flat must be fp32, fp16, or bf16");
    TORCH_CHECK(time_steps.scalar_type() == at::ScalarType::Int,
                "restore_dense_fwd: time_steps must be int32 (torch.int32)");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int,
                "restore_dense_fwd: cu_seqlens must be int32 (torch.int32)");

    // ---- device / contiguity checks ----
    TORCH_CHECK(y_flat.is_cuda() && time_steps.is_cuda() && cu_seqlens.is_cuda(),
                "restore_dense_fwd: all tensors must be on CUDA");
    TORCH_CHECK(y_flat.is_contiguous() && time_steps.is_contiguous() &&
                cu_seqlens.is_contiguous(),
                "restore_dense_fwd: all tensors must be contiguous");

    // ---- shape checks ----
    TORCH_CHECK(y_flat.dim()    == 1, "restore_dense_fwd: y_flat must be 1-D");
    TORCH_CHECK(time_steps.sizes() == y_flat.sizes(),
                "restore_dense_fwd: time_steps shape must equal y_flat shape");
    TORCH_CHECK(cu_seqlens.dim() == 1, "restore_dense_fwd: cu_seqlens must be 1-D");
    TORCH_CHECK(dim    >= 1, "restore_dense_fwd: dim must be >= 1");
    TORCH_CHECK(seqlen >= 1, "restore_dense_fwd: seqlen must be >= 1");

    const int total_events = (int)y_flat.size(0);
    const int n_channels   = (int)cu_seqlens.size(0) - 1;
    TORCH_CHECK(n_channels >= 1,
                "restore_dense_fwd: cu_seqlens must have at least 2 elements");
    TORCH_CHECK(n_channels % dim == 0,
                "restore_dense_fwd: n_channels (B*D) must be divisible by dim");

    // ---- output dtype ----
    auto out_type = out_dtype_opt.value_or(input_type);
    TORCH_CHECK(out_type == at::ScalarType::Float  ||
                out_type == at::ScalarType::Half    ||
                out_type == at::ScalarType::BFloat16,
                "restore_dense_fwd: out_dtype must be fp32, fp16, or bf16");

    const int64_t B_batch = (int64_t)(n_channels / dim);
    // Output shape: (B, L, D)
    auto out = torch::zeros({B_batch, seqlen, dim},
                            y_flat.options().dtype(out_type));

    // ---- fill params ----
    RestoreDenseParams p;
    p.n_channels   = n_channels;
    p.dim          = (int)dim;
    p.seqlen       = (int)seqlen;
    p.total_events = total_events;
    p.cu_seqlens   = cu_seqlens.data_ptr<int32_t>();
    p.time_steps   = time_steps.data_ptr<int32_t>();
    p.y_flat       = y_flat.data_ptr();
    p.out          = out.data_ptr();
    p.out_scalar_type = (int)out_type;

    // ---- launch ----
    at::cuda::CUDAGuard device_guard{y_flat.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (input_type == at::ScalarType::Float) {
        if      (out_type == at::ScalarType::Float)
            restore_dense_fwd_cuda<float, float>(p, stream);
        else if (out_type == at::ScalarType::Half)
            restore_dense_fwd_cuda<float, at::Half>(p, stream);
        else
            restore_dense_fwd_cuda<float, at::BFloat16>(p, stream);
    } else if (input_type == at::ScalarType::Half) {
        TORCH_CHECK(out_type == at::ScalarType::Half,
                    "restore_dense_fwd: fp16 input only supports fp16 output");
        restore_dense_fwd_cuda<at::Half, at::Half>(p, stream);
    } else {
        TORCH_CHECK(out_type == at::ScalarType::BFloat16,
                    "restore_dense_fwd: bf16 input only supports bf16 output");
        restore_dense_fwd_cuda<at::BFloat16, at::BFloat16>(p, stream);
    }

    return out;
}

// --------------------------------------------------------------------------
// Backward
//
// Zero-order-hold backward: dy[e] = sum of dout over all t owned by event e.
// Returns a single float32 tensor (no dlog_gamma — gamma is gone).
// --------------------------------------------------------------------------
at::Tensor restore_dense_bwd(
    const at::Tensor &dout,          // (B, L, D)       fp16/bf16/fp32
    const at::Tensor &time_steps,    // (E,)            int32
    const at::Tensor &cu_seqlens,    // (n_channels+1,) int32
    int64_t dim,                     // D
    int64_t seqlen                   // L
) {
    auto input_type = dout.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float  ||
                input_type == at::ScalarType::Half   ||
                input_type == at::ScalarType::BFloat16,
                "restore_dense_bwd: dout must be fp32/fp16/bf16");
    TORCH_CHECK(time_steps.scalar_type() == at::ScalarType::Int,
                "restore_dense_bwd: time_steps must be int32");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int,
                "restore_dense_bwd: cu_seqlens must be int32");
    TORCH_CHECK(dout.is_cuda() && time_steps.is_cuda() && cu_seqlens.is_cuda(),
                "restore_dense_bwd: all tensors must be on CUDA");
    TORCH_CHECK(dout.is_contiguous() && time_steps.is_contiguous() &&
                cu_seqlens.is_contiguous(),
                "restore_dense_bwd: all tensors must be contiguous");
    TORCH_CHECK(dim    >= 1, "restore_dense_bwd: dim must be >= 1");
    TORCH_CHECK(seqlen >= 1, "restore_dense_bwd: seqlen must be >= 1");

    const int n_channels   = (int)cu_seqlens.size(0) - 1;
    const int total_events = (int)time_steps.size(0);

    TORCH_CHECK(dout.dim() == 3,
                "restore_dense_bwd: dout must be 3-D (B, L, D)");
    TORCH_CHECK(dout.size(1) == seqlen,
                "restore_dense_bwd: dout.size(1) must equal seqlen");
    TORCH_CHECK(dout.size(2) == dim,
                "restore_dense_bwd: dout.size(2) must equal dim (D)");
    TORCH_CHECK(dout.size(0) * dim == (int64_t)n_channels,
                "restore_dense_bwd: dout.size(0)*dim must equal n_channels");

    // dy is always float32 (native lock-free atomicAdd on all CUDA GPUs).
    auto dy = torch::zeros({total_events},
                           dout.options().dtype(at::ScalarType::Float));

    RestoreDenseBwdParams p;
    p.n_channels   = n_channels;
    p.dim          = (int)dim;
    p.seqlen       = (int)seqlen;
    p.total_events = total_events;
    p.cu_seqlens   = cu_seqlens.data_ptr<int32_t>();
    p.time_steps   = time_steps.data_ptr<int32_t>();
    p.dout         = dout.data_ptr();
    p.dy           = dy.data_ptr<float>();

    at::cuda::CUDAGuard device_guard{dout.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (input_type == at::ScalarType::Float) {
        restore_dense_bwd_cuda<float>(p, stream);
    } else if (input_type == at::ScalarType::Half) {
        restore_dense_bwd_cuda<at::Half>(p, stream);
    } else if (input_type == at::ScalarType::BFloat16) {
        restore_dense_bwd_cuda<at::BFloat16>(p, stream);
    } else {
        TORCH_CHECK(false, "restore_dense_bwd: unsupported dtype");
    }

    return dy;
}

// --------------------------------------------------------------------------
// pybind11 module
// --------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("restore_dense_fwd", &restore_dense_fwd,
          py::arg("y_flat"),
          py::arg("time_steps"),
          py::arg("cu_seqlens"),
          py::arg("dim"),
          py::arg("seqlen"),
          py::arg("out_dtype") = py::none(),
          "Zero-order-hold restore (B, L, D) from flat events (CUDA, forward). "
          "Non-event positions carry the last event value forward (no decay). "
          "out_dtype: optional target dtype; defaults to y_flat.dtype.");
    m.def("restore_dense_bwd", &restore_dense_bwd,
          py::arg("dout"),
          py::arg("time_steps"),
          py::arg("cu_seqlens"),
          py::arg("dim"),
          py::arg("seqlen"),
          "Backward pass for zero-order-hold restore_dense. "
          "Returns dy_flat (float32). No dlog_gamma — gamma is removed.");
}
