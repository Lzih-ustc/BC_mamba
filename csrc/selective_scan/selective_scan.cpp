/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/python.h>
#include <vector>

#include "selective_scan.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

#define DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(WTYPE, NAME, ...)                     \
    if (WTYPE == at::ScalarType::Half) {                                             \
        using weight_t = at::Half;                                                   \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::BFloat16) {                                  \
        using weight_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::Float)  {                                    \
        using weight_t = float;                                                      \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }

#define DISPATCH_WTYPE_FLOAT_AND_COMPLEX(WTYPE, NAME, ...)                           \
    if (WTYPE == at::ScalarType::Float) {                                            \
       using weight_t = float;                                                       \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::ComplexFloat) {                              \
        using weight_t = c10::complex<float>;                                        \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }

template<typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream);

template <typename input_t, typename weight_t>
void selective_scan_bwd_cuda(SSMParamsBwd &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void selective_scan_fwd_flat_ch_cuda(SSMParamsFlatSeg &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void selective_scan_bwd_flat_ch_cuda(SSMParamsFlatSegBwd &params, cudaStream_t stream);

void set_ssm_params_fwd(SSMParamsBase &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t n_chunks,
                        const bool is_variable_B,
                        const bool is_variable_C,
                        // device pointers
                        const at::Tensor u,
                        const at::Tensor delta,
                        const at::Tensor A,
                        const at::Tensor B,
                        const at::Tensor C,
                        const at::Tensor out,
                        const at::Tensor z,
                        const at::Tensor out_z,
                        void* D_ptr,
                        void* delta_bias_ptr,
                        void* x_ptr,
                        bool has_z,
                        bool delta_softplus) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.dstate = dstate;
    params.n_groups = n_groups;
    params.n_chunks = n_chunks;
    params.dim_ngroups_ratio = dim / n_groups;

    params.delta_softplus = delta_softplus;

    params.is_variable_B = is_variable_B;
    params.is_variable_C = is_variable_C;

    // Set the pointers and strides.
    params.u_ptr = u.data_ptr();
    params.delta_ptr = delta.data_ptr();
    params.A_ptr = A.data_ptr();
    params.B_ptr = B.data_ptr();
    params.C_ptr = C.data_ptr();
    params.D_ptr = D_ptr;
    params.delta_bias_ptr = delta_bias_ptr;
    params.out_ptr = out.data_ptr();
    params.x_ptr = x_ptr;
    params.z_ptr = has_z ? z.data_ptr() : nullptr;
    params.out_z_ptr = has_z ? out_z.data_ptr() : nullptr;
    // All stride are in elements, not bytes.
    params.A_d_stride = A.stride(0);
    params.A_dstate_stride = A.stride(1);
    if (!is_variable_B) {
        params.B_d_stride = B.stride(0);
    } else {
        params.B_batch_stride = B.stride(0);
        params.B_group_stride = B.stride(1);
    }
    params.B_dstate_stride = !is_variable_B ? B.stride(1) : B.stride(2);
    if (!is_variable_C) {
        params.C_d_stride = C.stride(0);
    } else {
        params.C_batch_stride = C.stride(0);
        params.C_group_stride = C.stride(1);
    }
    params.C_dstate_stride = !is_variable_C ? C.stride(1) : C.stride(2);
    params.u_batch_stride = u.stride(0);
    params.u_d_stride = u.stride(1);
    params.delta_batch_stride = delta.stride(0);
    params.delta_d_stride = delta.stride(1);
    if (has_z) {
        params.z_batch_stride = z.stride(0);
        params.z_d_stride = z.stride(1);
        params.out_z_batch_stride = out_z.stride(0);
        params.out_z_d_stride = out_z.stride(1);
    }
    params.out_batch_stride = out.stride(0);
    params.out_d_stride = out.stride(1);
}

void set_ssm_params_bwd(SSMParamsBwd &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t n_chunks,
                        const bool is_variable_B,
                        const bool is_variable_C,
                        // device pointers
                        const at::Tensor u,
                        const at::Tensor delta,
                        const at::Tensor A,
                        const at::Tensor B,
                        const at::Tensor C,
                        const at::Tensor z,
                        const at::Tensor out,
                        const at::Tensor out_z,
                        void* D_ptr,
                        void* delta_bias_ptr,
                        void* x_ptr,
                        const at::Tensor dout,
                        const at::Tensor du,
                        const at::Tensor ddelta,
                        const at::Tensor dA,
                        const at::Tensor dB,
                        const at::Tensor dC,
                        const at::Tensor dz,
                        void* dD_ptr,
                        void* ddelta_bias_ptr,
                        bool has_z,
                        bool delta_softplus,
                        bool recompute_out_z) {
    // Pass in "dout" instead of "out", we're not gonna use "out" unless we have z
    set_ssm_params_fwd(params, batch, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                       u, delta, A, B, C, has_z ? out : dout,
                       has_z ? z : dout,
                       // If not recompute_out_z, pass dout instead of out_z.
                       // This won't be used by the bwd kernel
                       recompute_out_z ? out_z : dout,
                       D_ptr, delta_bias_ptr, x_ptr, has_z, delta_softplus);
    if (!recompute_out_z) { params.out_z_ptr = nullptr; }

    // Set the pointers and strides.
    params.dout_ptr = dout.data_ptr();
    params.du_ptr = du.data_ptr();
    params.dA_ptr = dA.data_ptr();
    params.dB_ptr = dB.data_ptr();
    params.dC_ptr = dC.data_ptr();
    params.dD_ptr = dD_ptr;
    params.ddelta_ptr = ddelta.data_ptr();
    params.ddelta_bias_ptr = ddelta_bias_ptr;
    params.dz_ptr = has_z ? dz.data_ptr() : nullptr;
    // All stride are in elements, not bytes.
    params.dout_batch_stride = dout.stride(0);
    params.dout_d_stride = dout.stride(1);
    params.dA_d_stride = dA.stride(0);
    params.dA_dstate_stride = dA.stride(1);
    if (!is_variable_B) {
        params.dB_d_stride = dB.stride(0);
    } else {
        params.dB_batch_stride = dB.stride(0);
        params.dB_group_stride = dB.stride(1);
    }
    params.dB_dstate_stride = !is_variable_B ? dB.stride(1) : dB.stride(2);
    if (!is_variable_C) {
        params.dC_d_stride = dC.stride(0);
    } else {
        params.dC_batch_stride = dC.stride(0);
        params.dC_group_stride = dC.stride(1);
    }
    params.dC_dstate_stride = !is_variable_C ? dC.stride(1) : dC.stride(2);
    params.du_batch_stride = du.stride(0);
    params.du_d_stride = du.stride(1);
    params.ddelta_batch_stride = ddelta.stride(0);
    params.ddelta_d_stride = ddelta.stride(1);
    if (has_z) {
        params.dz_batch_stride = dz.stride(0);
        params.dz_d_stride = dz.stride(1);
    }
}

void set_ssm_params_flat_seg_fwd(
    SSMParamsFlatSeg &params,
    const int dim,
    const int dstate,
    const int n_channels,
    const int n_events,
    const int n_blocks,
    const int n_chunks,
    const int max_seqlen,
    const at::Tensor &u_flat,
    const at::Tensor &k_flat,
    const at::Tensor &cu_seqlens,
    const at::Tensor &channel_ids,
    const at::Tensor &head_flags,
    const at::Tensor &dt_idle,
    const at::Tensor &dt_active,
    const at::Tensor &A,
    const at::Tensor &B,
    const at::Tensor &C,
    const at::Tensor &out_flat,
    const at::Tensor &z_flat,
    void *out_z_ptr,          // caller passes has_z ? out_z.data_ptr() : nullptr
    void *D_ptr,
    void *block_ab_ptr,
    void *block_prefix_ptr,
    void *prefix_a_ptr,
    bool has_channel_ids,
    bool has_head_flags,
    bool has_z,
    void *x_ptr
) {
    memset(&params, 0, sizeof(params));

    params.u_ptr          = u_flat.data_ptr();
    params.k_ptr          = reinterpret_cast<int32_t *>(k_flat.data_ptr());
    params.cu_seqlens_ptr = reinterpret_cast<int32_t *>(cu_seqlens.data_ptr());
    params.channel_ids_ptr = has_channel_ids
                            ? reinterpret_cast<int32_t *>(channel_ids.data_ptr())
                            : nullptr;
    params.head_flags_ptr = has_head_flags
                            ? reinterpret_cast<int32_t *>(head_flags.data_ptr())
                            : nullptr;

    params.A_ptr          = A.data_ptr();
    params.B_ptr          = B.data_ptr();
    params.C_ptr          = C.data_ptr();
    params.D_ptr          = D_ptr;
    params.z_ptr          = has_z ? z_flat.data_ptr() : nullptr;

    params.dt_idle_ptr    = reinterpret_cast<float *>(dt_idle.data_ptr());
    params.dt_active_ptr  = reinterpret_cast<float *>(dt_active.data_ptr());

    params.out_ptr        = out_flat.data_ptr();
    params.out_z_ptr      = out_z_ptr;   // already nullptr when !has_z (caller's responsibility)
    params.x_ptr          = x_ptr;
    params.block_ab_ptr      = block_ab_ptr;
    params.block_prefix_ptr  = block_prefix_ptr;
    params.prefix_a_ptr      = prefix_a_ptr;

    params.dim            = dim;
    params.dstate         = dstate;
    params.n_channels     = n_channels;
    params.n_events       = n_events;
    params.n_blocks       = n_blocks;
    params.n_chunks       = n_chunks;
    params.max_seqlen     = max_seqlen;

    params.A_d_stride      = A.stride(0);
    params.A_dstate_stride = A.stride(1);

    // FlatSeg: B/C are in transposed [dstate, E] layout.
    // Variable B/C: size(1) == n_events.  Non-variable: size(1) == dim.
    params.is_variable_B = B.size(1) == n_events;
    params.is_variable_C = C.size(1) == n_events;
    params.B_row_stride = static_cast<int>(B.stride(0));
    params.C_row_stride = static_cast<int>(C.stride(0));
}

// set_ssm_params_flat_seg_bwd — flat per-channel backward parameter setup.
// Calls set_ssm_params_flat_seg_fwd to fill the inherited SSMParamsFlatSeg
// fields, then fills the gradient-specific fields.
//
// B/C layout (transposed):  [dstate, E] for variable,  [dstate, dim] for constant.
// dB/dC layout must match B/C.
void set_ssm_params_flat_seg_bwd(
    SSMParamsFlatSegBwd &params,
    const int dim,
    const int dstate,
    const int n_channels,
    const int n_events,
    const int n_chunks,
    const int max_seqlen,
    // forward inputs
    const at::Tensor &u_flat,        // (E,) input_t
    const at::Tensor &k_flat,        // (E,) int32
    const at::Tensor &cu_seqlens,    // (n_channels+1,) int32
    const at::Tensor &dt_idle,       // (dim,) float
    const at::Tensor &dt_active,     // (dim,) float
    const at::Tensor &A,             // (dim, dstate) float
    const at::Tensor &B,             // (dstate, E) or (dstate, dim) float
    const at::Tensor &C,             // (dstate, E) or (dstate, dim) float
    // out from forward (needed for z-gate backward; pass u_flat when !has_z)
    const at::Tensor &out_flat,
    const at::Tensor &z_flat,        // (E,) input_t  — ignored when !has_z
    void *D_ptr,                     // (dim,) float or nullptr
    bool has_z,
    void *x_ptr,                     // (n_channels, n_chunks, dstate*2) float
    // upstream gradient
    const at::Tensor &dout_flat,     // (E,) input_t
    // gradient outputs (pre-allocated by caller)
    const at::Tensor &du_flat,       // (E,) input_t
    const at::Tensor &dB,            // same layout as B
    const at::Tensor &dC,            // same layout as C
    const at::Tensor &dA,            // (dim, dstate) float
    const at::Tensor &ddt_idle,      // (dim,) float
    const at::Tensor &ddt_active,    // (dim,) float
    const at::Tensor &dD,            // (dim,) float  — ignored when !has_dD
    const at::Tensor &dz_flat,       // (E,) input_t  — ignored when !has_z
    bool has_dD
) {
    // Fill inherited SSMParamsFlatSeg fields.
    // Pass out_flat as the fwd "out" so the kernel can read ungated SSM output
    // during z-gate backward.  scratch buffers not used by flat_ch → nullptr.
    at::Tensor dummy;
    set_ssm_params_flat_seg_fwd(
        params, dim, dstate, n_channels, n_events,
        /*n_blocks=*/1, n_chunks, max_seqlen,
        u_flat, k_flat, cu_seqlens,
        dummy, dummy,           // channel_ids / head_flags: unused
        dt_idle, dt_active, A, B, C,
        out_flat,
        has_z ? z_flat : dummy,
        /*out_z_ptr=*/nullptr,  // not recomputed in bwd
        D_ptr,
        /*block_ab=*/nullptr, /*block_prefix=*/nullptr, /*prefix_a=*/nullptr,
        /*has_channel_ids=*/false, /*has_head_flags=*/false, has_z,
        x_ptr
    );

    // Gradient pointers.
    params.dout_ptr       = dout_flat.data_ptr();
    params.du_ptr         = du_flat.data_ptr();
    params.dB_ptr         = dB.data_ptr();
    params.dC_ptr         = dC.data_ptr();
    params.dA_ptr         = dA.data_ptr();
    params.ddt_idle_ptr   = reinterpret_cast<float *>(ddt_idle.data_ptr());
    params.ddt_active_ptr = reinterpret_cast<float *>(ddt_active.data_ptr());
    params.dD_ptr         = has_dD ? reinterpret_cast<float *>(dD.data_ptr()) : nullptr;
    params.dz_ptr         = has_z  ? dz_flat.data_ptr() : nullptr;

    // Unused global-scan scratch buffers (flat_ch has no inter-block scan).
    params.dx_partial_ptr    = nullptr;
    params.prefix_a_rev_ptr  = nullptr;
    params.block_postfix_ptr = nullptr;
    params.dx_true_ptr       = nullptr;
    params.debug_flag_ptr    = nullptr;

    // dA strides.
    params.dA_d_stride      = static_cast<int>(dA.stride(0));
    params.dA_dstate_stride = static_cast<int>(dA.stride(1));

    // dB/dC strides — layout is [dstate, dim] (non-variable) or [dstate, E] (variable).
    // The kernel uses dB_d_stride only for the initial pointer offset (non-variable path),
    // and dB_dstate_stride for the atomicAdd index (non-variable path).
    // Variable paths use B_row_stride (already set by set_ssm_params_flat_seg_fwd).
    params.dB_d_stride       = static_cast<int>(dB.stride(1));   // stride along dim / E axis
    params.dB_dstate_stride  = static_cast<int>(dB.stride(0));   // stride along dstate axis
    params.dC_d_stride       = static_cast<int>(dC.stride(1));
    params.dC_dstate_stride  = static_cast<int>(dC.stride(0));
}

std::vector<at::Tensor>
selective_scan_fwd(const at::Tensor &u, const at::Tensor &delta,
                  const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                  const c10::optional<at::Tensor> &D_,
                  const c10::optional<at::Tensor> &z_,
                  const c10::optional<at::Tensor> &delta_bias_,
                  bool delta_softplus) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float || weight_type == at::ScalarType::ComplexFloat);

    const bool is_variable_B = B.dim() >= 3;
    const bool is_variable_C = C.dim() >= 3;
    const bool is_complex = weight_type == at::ScalarType::ComplexFloat;

    TORCH_CHECK(delta.scalar_type() == input_type);
    TORCH_CHECK(B.scalar_type() == (!is_variable_B ? weight_type : input_type));
    TORCH_CHECK(C.scalar_type() == (!is_variable_C ? weight_type : input_type));

    TORCH_CHECK(u.is_cuda());
    TORCH_CHECK(delta.is_cuda());
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(B.is_cuda());
    TORCH_CHECK(C.is_cuda());

    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = A.size(1);
    const int n_groups = is_variable_B ? B.size(1) : 1;

    TORCH_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);
    if (!is_variable_B) {
        CHECK_SHAPE(B, dim, dstate);
    } else {
        CHECK_SHAPE(B, batch_size, n_groups, dstate, !is_complex ? seqlen : seqlen * 2);
        TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
    }
    if (!is_variable_C) {
        CHECK_SHAPE(C, dim, dstate);
    } else {
        CHECK_SHAPE(C, batch_size, n_groups, dstate, !is_complex ? seqlen: seqlen * 2);
        TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);
    }

    if (D_.has_value()) {
        auto D = D_.value();
        TORCH_CHECK(D.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
    }

    if (delta_bias_.has_value()) {
        auto delta_bias = delta_bias_.value();
        TORCH_CHECK(delta_bias.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(delta_bias.is_cuda());
        TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, dim);
    }

    at::Tensor z, out_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = z_.value();
        TORCH_CHECK(z.scalar_type() == input_type);
        TORCH_CHECK(z.is_cuda());
        TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        CHECK_SHAPE(z, batch_size, dim, seqlen);
        out_z = torch::empty_like(z);
    }

    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    // const int n_chunks = (seqlen + 1024 - 1) / 1024;
    // at::Tensor out = torch::empty_like(u);
    // Right now u has BHL layout and delta has HBL layout, and we want out to have HBL layout
    at::Tensor out = torch::empty_like(delta);
    at::Tensor x;
    x = torch::empty({batch_size, dim, n_chunks, dstate * 2}, u.options().dtype(weight_type));

    SSMParamsBase params;
    set_ssm_params_fwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                       u, delta, A, B, C, out, z, out_z,
                       D_.has_value() ? D_.value().data_ptr() : nullptr,
                       delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr,
                       x.data_ptr(),
                       has_z,
                       delta_softplus);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{u.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), "selective_scan_fwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_COMPLEX(A.scalar_type(), "selective_scan_fwd", [&] {
            selective_scan_fwd_cuda<input_t, weight_t>(params, stream);
        });
    });
    std::vector<at::Tensor> result = {out, x};
    if (has_z) { result.push_back(out_z); }
    return result;
}

std::vector<at::Tensor>
selective_scan_bwd(const at::Tensor &u, const at::Tensor &delta,
                  const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                  const c10::optional<at::Tensor> &D_,
                  const c10::optional<at::Tensor> &z_,
                  const c10::optional<at::Tensor> &delta_bias_,
                  const at::Tensor &dout,
                  const c10::optional<at::Tensor> &x_,
                  const c10::optional<at::Tensor> &out_,
                  c10::optional<at::Tensor> &dz_,
                  bool delta_softplus,
                  bool recompute_out_z) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float || weight_type == at::ScalarType::ComplexFloat);

    const bool is_variable_B = B.dim() >= 3;
    const bool is_variable_C = C.dim() >= 3;
    const bool is_complex = weight_type == at::ScalarType::ComplexFloat;

    TORCH_CHECK(delta.scalar_type() == input_type);
    TORCH_CHECK(B.scalar_type() == (!is_variable_B ? weight_type : input_type));
    TORCH_CHECK(C.scalar_type() == (!is_variable_C ? weight_type : input_type));
    TORCH_CHECK(dout.scalar_type() == input_type);

    TORCH_CHECK(u.is_cuda());
    TORCH_CHECK(delta.is_cuda());
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(B.is_cuda());
    TORCH_CHECK(C.is_cuda());
    TORCH_CHECK(dout.is_cuda());

    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);
    TORCH_CHECK(dout.stride(-1) == 1 || dout.size(-1) == 1);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = A.size(1);
    const int n_groups = is_variable_B ? B.size(1) : 1;

    TORCH_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);
    if (!is_variable_B) {
        CHECK_SHAPE(B, dim, dstate);
    } else {
        CHECK_SHAPE(B, batch_size, n_groups, dstate, !is_complex ? seqlen : seqlen * 2);
        TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
    }
    if (!is_variable_C) {
        CHECK_SHAPE(C, dim, dstate);
    } else {
        CHECK_SHAPE(C, batch_size, n_groups, dstate, !is_complex ? seqlen: seqlen * 2);
        TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);
    }
    CHECK_SHAPE(dout, batch_size, dim, seqlen);

    if (D_.has_value()) {
        auto D = D_.value();
        TORCH_CHECK(D.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
    }

    if (delta_bias_.has_value()) {
        auto delta_bias = delta_bias_.value();
        TORCH_CHECK(delta_bias.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(delta_bias.is_cuda());
        TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, dim);
    }

    at::Tensor z, out, dz, out_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = z_.value();
        TORCH_CHECK(z.scalar_type() == input_type);
        TORCH_CHECK(z.is_cuda());
        TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        CHECK_SHAPE(z, batch_size, dim, seqlen);

        TORCH_CHECK(out_.has_value());
        out = out_.value();
        TORCH_CHECK(out.scalar_type() == input_type);
        TORCH_CHECK(out.is_cuda());
        TORCH_CHECK(out.stride(-1) == 1 || out.size(-1) == 1);
        CHECK_SHAPE(out, batch_size, dim, seqlen);

        if (dz_.has_value()) {
            dz = dz_.value();
            TORCH_CHECK(dz.scalar_type() == input_type);
            TORCH_CHECK(dz.is_cuda());
            TORCH_CHECK(dz.stride(-1) == 1 || dz.size(-1) == 1);
            CHECK_SHAPE(dz, batch_size, dim, seqlen);
        } else {
            dz = torch::empty_like(z);
        }
        if (recompute_out_z) {
            out_z = torch::empty_like(out);
        }
    }

    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    // const int n_chunks = (seqlen + 1024 - 1) / 1024;
    if (n_chunks > 1) { TORCH_CHECK(x_.has_value()); }
    if (x_.has_value()) {
        auto x = x_.value();
        TORCH_CHECK(x.scalar_type() == weight_type);
        TORCH_CHECK(x.is_cuda());
        TORCH_CHECK(x.is_contiguous());
        CHECK_SHAPE(x, batch_size, dim, n_chunks, 2 * dstate);
    }

    at::Tensor du = torch::empty_like(u);
    at::Tensor ddelta = torch::empty_like(delta);
    at::Tensor dA = torch::zeros_like(A);
    at::Tensor dB = !is_variable_B ? torch::zeros_like(B) : torch::zeros_like(B, B.options().dtype(torch::kFloat32));
    at::Tensor dC = !is_variable_C ? torch::zeros_like(C) : torch::zeros_like(C, C.options().dtype(torch::kFloat32));
    at::Tensor dD;
    if (D_.has_value()) { dD = torch::zeros_like(D_.value()); }
    at::Tensor ddelta_bias;
    if (delta_bias_.has_value()) { ddelta_bias = torch::zeros_like(delta_bias_.value()); }

    SSMParamsBwd params;
    set_ssm_params_bwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                       u, delta, A, B, C, z, out, out_z,
                       D_.has_value() ? D_.value().data_ptr() : nullptr,
                       delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr,
                       x_.has_value() ? x_.value().data_ptr() : nullptr,
                       dout, du, ddelta, dA, dB, dC, dz,
                       D_.has_value() ? dD.data_ptr() : nullptr,
                       delta_bias_.has_value() ? ddelta_bias.data_ptr() : nullptr,
                       has_z, delta_softplus, recompute_out_z);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{u.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), "selective_scan_bwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_COMPLEX(A.scalar_type(), "selective_scan_bwd", [&] {
            selective_scan_bwd_cuda<input_t, weight_t>(params, stream);
        });
    });
    std::vector<at::Tensor> result = {du, ddelta, dA, dB.to(B.dtype()), dC.to(C.dtype()), dD, ddelta_bias};
    if (has_z) { result.push_back(dz); }
    if (recompute_out_z) { result.push_back(out_z); }
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Flat per-channel forward — one CUDA block per logical channel, no global scan.
////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<at::Tensor>
selective_scan_fwd_flat_ch(
    const at::Tensor &u_flat,
    const at::Tensor &k_flat,
    const at::Tensor &cu_seqlens,
    const int64_t     max_seqlen_,
    const at::Tensor &dt_idle,
    const at::Tensor &dt_active,
    const at::Tensor &A,
    const at::Tensor &B,         // depthwise: [dim, dstate] float32; legacy: [dstate, E] or tiled
    const at::Tensor &C,         // depthwise: [dim, dstate] float32; legacy: [dstate, E] or tiled
    const c10::optional<at::Tensor> &D_,
    const c10::optional<at::Tensor> &z_,
    const c10::optional<at::Tensor> &tile_offsets_,
    const int64_t     tile_chunk_size_,
    const bool        is_depthwise_bc_,
    const c10::optional<at::Tensor> &pw_B_,   // [dim, dim] float32  (depthwise-separable pointwise)
    const c10::optional<at::Tensor> &pw_C_    // [dim, dim] float32
) {
    const auto input_type = u_flat.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float ||
                input_type == at::ScalarType::Half  ||
                input_type == at::ScalarType::BFloat16,
                "selective_scan_fwd_flat_ch: u_flat must be fp16/bf16/fp32");

    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float,
                "selective_scan_fwd_flat_ch: A must be float32");
    TORCH_CHECK(B.scalar_type() == at::ScalarType::Float,
                "selective_scan_fwd_flat_ch: B must be float32");
    TORCH_CHECK(C.scalar_type() == at::ScalarType::Float,
                "selective_scan_fwd_flat_ch: C must be float32");

    TORCH_CHECK(u_flat.is_cuda() && k_flat.is_cuda() && cu_seqlens.is_cuda() &&
                dt_idle.is_cuda() && dt_active.is_cuda() && A.is_cuda() &&
                B.is_cuda() && C.is_cuda(),
                "selective_scan_fwd_flat_ch: all tensors must be CUDA tensors");

    TORCH_CHECK(u_flat.is_contiguous() && k_flat.is_contiguous() && cu_seqlens.is_contiguous(),
                "selective_scan_fwd_flat_ch: u_flat/k_flat/cu_seqlens must be contiguous");

    TORCH_CHECK(u_flat.dim() == 1, "selective_scan_fwd_flat_ch: u_flat must be 1-D");
    TORCH_CHECK(k_flat.dim() == 1, "selective_scan_fwd_flat_ch: k_flat must be 1-D");
    TORCH_CHECK(cu_seqlens.dim() == 1, "selective_scan_fwd_flat_ch: cu_seqlens must be 1-D");
    TORCH_CHECK(k_flat.scalar_type() == at::ScalarType::Int,
                "selective_scan_fwd_flat_ch: k_flat must be int32");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int,
                "selective_scan_fwd_flat_ch: cu_seqlens must be int32");

    const int total_events = static_cast<int>(u_flat.size(0));
    TORCH_CHECK((int)k_flat.size(0) == total_events,
                "selective_scan_fwd_flat_ch: k_flat must match u_flat shape");

    TORCH_CHECK(cu_seqlens.size(0) >= 2,
                "selective_scan_fwd_flat_ch: cu_seqlens must have at least 2 elements");
    const int n_channels = static_cast<int>(cu_seqlens.size(0)) - 1;

    TORCH_CHECK(dt_idle.scalar_type() == at::ScalarType::Float &&
                dt_active.scalar_type() == at::ScalarType::Float,
                "selective_scan_fwd_flat_ch: dt_idle/dt_active must be float32");
    TORCH_CHECK(dt_idle.dim() == 1 && dt_active.dim() == 1,
                "selective_scan_fwd_flat_ch: dt_idle/dt_active must be 1-D");

    TORCH_CHECK(A.dim() == 2, "selective_scan_fwd_flat_ch: A must be 2-D (dim, dstate)");
    const int dim = static_cast<int>(A.size(0));
    const int dstate = static_cast<int>(A.size(1));
    TORCH_CHECK(dstate <= 256, "selective_scan_fwd_flat_ch: dstate must be <= 256");
    CHECK_SHAPE(dt_idle, dim);
    CHECK_SHAPE(dt_active, dim);

    const bool use_tiled_bc  = tile_offsets_.has_value() && tile_offsets_.value().defined();
    const bool use_depthwise = is_depthwise_bc_ && !use_tiled_bc;

    if (use_depthwise) {
        // Depthwise: W_B / W_C are [dim, dstate] float32
        TORCH_CHECK(B.dim() == 2 && C.dim() == 2,
                    "selective_scan_fwd_flat_ch (depthwise): W_B/W_C must be 2-D [dim, dstate]");
        TORCH_CHECK((int)B.size(0) == dim && (int)B.size(1) == dstate,
                    "selective_scan_fwd_flat_ch (depthwise): W_B must be [dim, dstate]");
        TORCH_CHECK((int)C.size(0) == dim && (int)C.size(1) == dstate,
                    "selective_scan_fwd_flat_ch (depthwise): W_C must be [dim, dstate]");
        TORCH_CHECK(B.is_contiguous() && C.is_contiguous(),
                    "selective_scan_fwd_flat_ch (depthwise): W_B/W_C must be contiguous");
    } else if (use_tiled_bc) {
        // Tiled layout: B/C are [N_total_tiles, dstate, tile_chunk_size]
        TORCH_CHECK(B.dim() == 3 && C.dim() == 3,
                    "selective_scan_fwd_flat_ch: tiled B/C must be 3-D [N_tiles, dstate, tile_chunk_size]");
        TORCH_CHECK(B.size(1) == dstate && C.size(1) == dstate,
                    "selective_scan_fwd_flat_ch: tiled B/C dim-1 must equal dstate");
        TORCH_CHECK(B.is_contiguous() && C.is_contiguous(),
                    "selective_scan_fwd_flat_ch: tiled B/C must be contiguous");
    } else {
        TORCH_CHECK(B.dim() == 2 && C.dim() == 2,
                    "selective_scan_fwd_flat_ch: B/C must be 2-D (dstate, E) [transposed layout]");
        CHECK_SHAPE(B, dstate, total_events);
        CHECK_SHAPE(C, dstate, total_events);
        TORCH_CHECK(B.stride(1) == 1 && C.stride(1) == 1,
                    "selective_scan_fwd_flat_ch: B/C must have contiguous last dimension (stride(1) == 1)");
    }

    if (D_.has_value()) {
        TORCH_CHECK(D_.value().scalar_type() == at::ScalarType::Float,
                    "selective_scan_fwd_flat_ch: D must be float32");
        TORCH_CHECK(D_.value().is_cuda() && D_.value().is_contiguous());
        CHECK_SHAPE(D_.value(), dim);
    }

    at::Tensor z_flat, out_z_flat;
    const bool has_z = z_.has_value();
    if (has_z) {
        z_flat = z_.value();
        TORCH_CHECK(z_flat.scalar_type() == input_type,
                    "selective_scan_fwd_flat_ch: z must have same dtype as u_flat");
        TORCH_CHECK(z_flat.is_cuda() && z_flat.is_contiguous());
        TORCH_CHECK(z_flat.dim() == 1 && (int)z_flat.size(0) == total_events,
                    "selective_scan_fwd_flat_ch: z must have shape (E,)");
    }

    TORCH_CHECK(max_seqlen_ >= 0, "selective_scan_fwd_flat_ch: max_seqlen must be non-negative");
    const int max_seqlen = static_cast<int>(max_seqlen_);
    // n_chunks is not used by the kernel (it computes ch_n_chunks dynamically),
    // but we still set it for the params struct.
    int n_chunks = (max_seqlen + 2048 - 1) / 2048;
    if (n_chunks == 0) n_chunks = 1;

    // No global scan needed — each channel is independent.
    at::Tensor out_flat = torch::empty_like(u_flat);

    // Allocate a separate state slot for each chunk. Backward relies on this
    // layout to reconstruct the initial state of every chunk.
    at::Tensor x = torch::zeros({n_channels, n_chunks, dstate * 2},
                                u_flat.options().dtype(torch::kFloat32));
                                
    if (has_z) {
        out_z_flat = torch::empty_like(u_flat);
    }



    SSMParamsFlatSeg params;
    at::Tensor z_dummy, head_flags_dummy, channel_ids_dummy;
    set_ssm_params_flat_seg_fwd(
        params,
        dim, dstate, n_channels, total_events,
        /*n_blocks=*/1, n_chunks, max_seqlen,
        u_flat, k_flat, cu_seqlens,
        channel_ids_dummy, head_flags_dummy,
        dt_idle, dt_active, A, B, C,
        out_flat,
        has_z ? z_flat : z_dummy,
        has_z ? out_z_flat.data_ptr() : nullptr,
        D_.has_value() ? D_.value().data_ptr() : nullptr,
        /*block_ab=*/nullptr, /*block_prefix=*/nullptr, /*prefix_a=*/nullptr,
        /*has_channel_ids=*/false, /*has_head_flags=*/false, has_z,
        x.data_ptr()
    );

    // ── B/C layout override ───────────────────────────────────────────────────
    if (use_depthwise) {
        // Depthwise: W_B / W_C are [dim, dstate]. B_ptr/C_ptr are set to the
        // weight tensors; W_B_ptr/W_C_ptr mirror them (kernel reads W_B/W_C ptr).
        // is_variable_B/C = true so the kernel instantiates the variable branch,
        // then the is_depthwise_bc runtime flag selects the on-the-fly sub-path.
        params.is_variable_B   = true;
        params.is_variable_C   = true;
        params.is_depthwise_bc = true;
        params.W_B_ptr         = B.data_ptr();
        params.W_C_ptr         = C.data_ptr();
        params.B_ptr           = nullptr;
        params.C_ptr           = nullptr;
        params.tile_offsets_ptr = nullptr;
        params.tile_chunk_size  = 0;

        // ── Optional pointwise weights ────────────────────────────────────────
        if (pw_B_.has_value() && pw_B_.value().defined()) {
            const at::Tensor &pw_B = pw_B_.value();
            const at::Tensor &pw_C = pw_C_.value();
            TORCH_CHECK(pw_B.scalar_type() == at::ScalarType::Float &&
                        pw_C.scalar_type() == at::ScalarType::Float,
                        "selective_scan_fwd_flat_ch: pw_B/pw_C must be float32");
            TORCH_CHECK(pw_B.is_cuda() && pw_C.is_cuda() &&
                        pw_B.is_contiguous() && pw_C.is_contiguous(),
                        "selective_scan_fwd_flat_ch: pw_B/pw_C must be CUDA and contiguous");
            TORCH_CHECK(pw_B.dim() == 2 && (int)pw_B.size(0) == dim && (int)pw_B.size(1) == dim,
                        "selective_scan_fwd_flat_ch: pw_B must be [dim, dim]");
            TORCH_CHECK(pw_C.dim() == 2 && (int)pw_C.size(0) == dim && (int)pw_C.size(1) == dim,
                        "selective_scan_fwd_flat_ch: pw_C must be [dim, dim]");
            params.pw_B_ptr = pw_B.data_ptr();
            params.pw_C_ptr = pw_C.data_ptr();
        }
    } else if (use_tiled_bc) {
        // ── Tiled B/C layout support ─────────────────────────────────────────
        params.tile_offsets_ptr = tile_offsets_.value().data_ptr<int32_t>();
        params.tile_chunk_size  = static_cast<int>(tile_chunk_size_);
        params.is_variable_B = true;
        params.is_variable_C = true;
    } else {
        params.tile_offsets_ptr = nullptr;
        params.tile_chunk_size  = 0;
    }

    at::cuda::CUDAGuard device_guard{u_flat.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(input_type, "selective_scan_fwd_flat_ch", [&] {
        using weight_t = float;
        selective_scan_fwd_flat_ch_cuda<input_t, weight_t>(params, stream);
    });

    std::vector<at::Tensor> result = {out_flat, x};
    if (has_z) result.push_back(out_z_flat);
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Flat per-channel backward — one CUDA block per logical channel, no global scan.
// Mirrors selective_scan_fwd_flat_ch; takes the saved x from forward.
////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<at::Tensor>
selective_scan_bwd_flat_ch(
    const at::Tensor &u_flat,        // (E,)             input_t  — forward input
    const at::Tensor &k_flat,        // (E,)             int32    — inter-event intervals
    const at::Tensor &cu_seqlens,    // (n_channels+1,)  int32
    const int64_t     max_seqlen_,
    const at::Tensor &dt_idle,       // (dim,)           float32
    const at::Tensor &dt_active,     // (dim,)           float32
    const at::Tensor &A,             // (dim, dstate)    float32
    const at::Tensor &B,             // depthwise:[dim,dstate]; legacy:(dstate,E)/(dstate,dim)/tiled
    const at::Tensor &C,             // depthwise:[dim,dstate]; legacy:(dstate,E)/(dstate,dim)/tiled
    const at::Tensor &dout_flat,     // (E,)             input_t  — upstream gradient
    const at::Tensor &x,             // (n_channels, n_chunks, dstate*2) float32 — saved fwd state
    const c10::optional<at::Tensor> &D_,     // (dim,) float32  — optional
    const c10::optional<at::Tensor> &z_,     // (E,)   input_t  — optional gate
    const c10::optional<at::Tensor> &out_,   // (E,)   input_t  — fwd ungated output (req. when has_z)
    const c10::optional<at::Tensor> &tile_offsets_,
    const int64_t     tile_chunk_size_,
    const bool        is_depthwise_bc_,
    const c10::optional<at::Tensor> &pw_B_,  // [dim, dim] float32  — optional pointwise
    const c10::optional<at::Tensor> &pw_C_   // [dim, dim] float32
) {
    const auto input_type = u_flat.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float ||
                input_type == at::ScalarType::Half  ||
                input_type == at::ScalarType::BFloat16,
                "selective_scan_bwd_flat_ch: u_flat must be fp16/bf16/fp32");

    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float,
                "selective_scan_bwd_flat_ch: A must be float32");
    TORCH_CHECK(B.scalar_type() == at::ScalarType::Float,
                "selective_scan_bwd_flat_ch: B must be float32");
    TORCH_CHECK(C.scalar_type() == at::ScalarType::Float,
                "selective_scan_bwd_flat_ch: C must be float32");
    TORCH_CHECK(dout_flat.scalar_type() == input_type,
                "selective_scan_bwd_flat_ch: dout_flat must match u_flat dtype");

    TORCH_CHECK(u_flat.is_cuda() && k_flat.is_cuda() && cu_seqlens.is_cuda() &&
                dt_idle.is_cuda() && dt_active.is_cuda() && A.is_cuda() &&
                B.is_cuda() && C.is_cuda() && dout_flat.is_cuda() && x.is_cuda(),
                "selective_scan_bwd_flat_ch: all tensors must be CUDA tensors");

    TORCH_CHECK(u_flat.is_contiguous() && k_flat.is_contiguous() &&
                cu_seqlens.is_contiguous() && dout_flat.is_contiguous(),
                "selective_scan_bwd_flat_ch: u_flat/k_flat/cu_seqlens/dout_flat must be contiguous");

    TORCH_CHECK(u_flat.dim() == 1 && k_flat.dim() == 1 &&
                cu_seqlens.dim() == 1 && dout_flat.dim() == 1,
                "selective_scan_bwd_flat_ch: u_flat/k_flat/cu_seqlens/dout_flat must be 1-D");
    TORCH_CHECK(k_flat.scalar_type() == at::ScalarType::Int,
                "selective_scan_bwd_flat_ch: k_flat must be int32");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::ScalarType::Int,
                "selective_scan_bwd_flat_ch: cu_seqlens must be int32");

    const int total_events = static_cast<int>(u_flat.size(0));
    TORCH_CHECK((int)k_flat.size(0)    == total_events,
                "selective_scan_bwd_flat_ch: k_flat must match u_flat shape");
    TORCH_CHECK((int)dout_flat.size(0) == total_events,
                "selective_scan_bwd_flat_ch: dout_flat must match u_flat shape");

    TORCH_CHECK(cu_seqlens.size(0) >= 2,
                "selective_scan_bwd_flat_ch: cu_seqlens must have at least 2 elements");
    const int n_channels = static_cast<int>(cu_seqlens.size(0)) - 1;

    TORCH_CHECK(dt_idle.scalar_type()   == at::ScalarType::Float &&
                dt_active.scalar_type() == at::ScalarType::Float,
                "selective_scan_bwd_flat_ch: dt_idle/dt_active must be float32");
    TORCH_CHECK(dt_idle.dim() == 1 && dt_active.dim() == 1,
                "selective_scan_bwd_flat_ch: dt_idle/dt_active must be 1-D");

    TORCH_CHECK(A.dim() == 2, "selective_scan_bwd_flat_ch: A must be 2-D (dim, dstate)");
    const int dim    = static_cast<int>(A.size(0));
    const int dstate = static_cast<int>(A.size(1));
    TORCH_CHECK(dstate <= 256, "selective_scan_bwd_flat_ch: dstate must be <= 256");
    CHECK_SHAPE(dt_idle,   dim);
    CHECK_SHAPE(dt_active, dim);

    const bool use_tiled_bc  = tile_offsets_.has_value() && tile_offsets_.value().defined();
    const bool use_depthwise = is_depthwise_bc_ && !use_tiled_bc;

    if (use_depthwise) {
        TORCH_CHECK(B.dim() == 2 && C.dim() == 2,
                    "selective_scan_bwd_flat_ch (depthwise): W_B/W_C must be 2-D [dim, dstate]");
        TORCH_CHECK((int)B.size(0) == dim && (int)B.size(1) == dstate,
                    "selective_scan_bwd_flat_ch (depthwise): W_B must be [dim, dstate]");
        TORCH_CHECK((int)C.size(0) == dim && (int)C.size(1) == dstate,
                    "selective_scan_bwd_flat_ch (depthwise): W_C must be [dim, dstate]");
        TORCH_CHECK(B.is_contiguous() && C.is_contiguous(),
                    "selective_scan_bwd_flat_ch (depthwise): W_B/W_C must be contiguous");
    } else if (use_tiled_bc) {
        TORCH_CHECK(B.dim() == 3 && C.dim() == 3,
                    "selective_scan_bwd_flat_ch: tiled B/C must be 3-D [N_tiles, dstate, tile_chunk_size]");
        TORCH_CHECK(B.size(1) == dstate && C.size(1) == dstate,
                    "selective_scan_bwd_flat_ch: tiled B/C dim-1 must equal dstate");
        TORCH_CHECK(B.is_contiguous() && C.is_contiguous(),
                    "selective_scan_bwd_flat_ch: tiled B/C must be contiguous");
    } else {
        TORCH_CHECK(B.dim() == 2 && C.dim() == 2,
                    "selective_scan_bwd_flat_ch: B/C must be 2-D [dstate, E] or [dstate, dim]");
        TORCH_CHECK(B.stride(1) == 1 && C.stride(1) == 1,
                    "selective_scan_bwd_flat_ch: B/C must have contiguous last dimension");
    }

    if (D_.has_value()) {
        TORCH_CHECK(D_.value().scalar_type() == at::ScalarType::Float,
                    "selective_scan_bwd_flat_ch: D must be float32");
        TORCH_CHECK(D_.value().is_cuda() && D_.value().is_contiguous());
        CHECK_SHAPE(D_.value(), dim);
    }

    // Validate and unpack x (saved forward SSM state).
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float,
                "selective_scan_bwd_flat_ch: x must be float32");
    TORCH_CHECK(x.is_contiguous(), "selective_scan_bwd_flat_ch: x must be contiguous");
    TORCH_CHECK(x.dim() == 3 && (int)x.size(0) == n_channels && (int)x.size(2) == dstate * 2,
                "selective_scan_bwd_flat_ch: x must have shape (n_channels, n_chunks, dstate*2)");

    TORCH_CHECK(max_seqlen_ >= 0, "selective_scan_bwd_flat_ch: max_seqlen must be non-negative");
    const int max_seqlen = static_cast<int>(max_seqlen_);
    int n_chunks = (max_seqlen + 2048 - 1) / 2048;
    if (n_chunks == 0) n_chunks = 1;

    TORCH_CHECK((int)x.size(1) == n_chunks,
                "selective_scan_bwd_flat_ch: x.size(1) must equal n_chunks = "
                "(max_seqlen + 2047) / 2048");

    // z / out handling.
    at::Tensor z_flat, out_flat_fwd, dz_flat;
    const bool has_z = z_.has_value();
    if (has_z) {
        TORCH_CHECK(out_.has_value(),
                    "selective_scan_bwd_flat_ch: out (forward ungated output) must be "
                    "provided when z is provided");
        z_flat      = z_.value();
        out_flat_fwd = out_.value();
        TORCH_CHECK(z_flat.scalar_type()       == input_type &&
                    out_flat_fwd.scalar_type()  == input_type,
                    "selective_scan_bwd_flat_ch: z/out must match u_flat dtype");
        TORCH_CHECK(z_flat.is_cuda()       && out_flat_fwd.is_cuda() &&
                    z_flat.is_contiguous() && out_flat_fwd.is_contiguous());
        TORCH_CHECK((int)z_flat.size(0)      == total_events &&
                    (int)out_flat_fwd.size(0) == total_events,
                    "selective_scan_bwd_flat_ch: z/out must have shape (E,)");
        dz_flat = torch::zeros_like(u_flat);
    }

    // Allocate gradient output tensors.
    at::Tensor du_flat    = torch::zeros_like(u_flat);
    at::Tensor dA         = torch::zeros_like(A);
    // Depthwise: dW_B / dW_C have the same shape as W_B / W_C ([dim, dstate]).
    // Legacy/tiled: dB / dC mirror the B / C shape.
    at::Tensor dB, dC, dW_B, dW_C;
    at::Tensor dpw_B, dpw_C;       // only allocated when pw_B_ is provided
    bool has_pw = false;
    if (use_depthwise) {
        // dB / dC are not written by the kernel in this mode — provide tiny
        // placeholders (1 element each) to satisfy set_ssm_params_flat_seg_bwd.
        at::Tensor tiny = torch::zeros({1, 1}, B.options());
        dB = tiny; dC = tiny;
        dW_B = torch::zeros_like(B);   // [dim, dstate]  — depthwise weight grad
        dW_C = torch::zeros_like(C);
    } else {
        dB = torch::zeros_like(B);
        dC = torch::zeros_like(C);
    }
    at::Tensor ddt_idle   = torch::zeros_like(dt_idle);
    at::Tensor ddt_active = torch::zeros_like(dt_active);
    at::Tensor dD;
    if (D_.has_value()) dD = torch::zeros_like(D_.value());

    // When z is absent the kernel never reads out_ptr — reuse u_flat as a dummy.
    const at::Tensor &out_arg = has_z ? out_flat_fwd : u_flat;

    at::Tensor z_dummy, dD_dummy, dz_dummy;

    SSMParamsFlatSegBwd params;
    set_ssm_params_flat_seg_bwd(
        params,
        dim, dstate, n_channels, total_events, n_chunks, max_seqlen,
        u_flat, k_flat, cu_seqlens,
        dt_idle, dt_active, A, B, C,
        out_arg,
        has_z ? z_flat : z_dummy,
        D_.has_value() ? D_.value().data_ptr() : nullptr,
        has_z,
        x.data_ptr(),
        dout_flat,
        du_flat, dB, dC, dA, ddt_idle, ddt_active,
        D_.has_value() ? dD : dD_dummy,
        has_z ? dz_flat : dz_dummy,
        D_.has_value()
    );

    // ── B/C / W_B / W_C layout override ─────────────────────────────────────
    if (use_depthwise) {
        params.is_variable_B   = true;
        params.is_variable_C   = true;
        params.is_depthwise_bc = true;
        params.W_B_ptr         = B.data_ptr();
        params.W_C_ptr         = C.data_ptr();
        params.B_ptr           = nullptr;
        params.C_ptr           = nullptr;
        params.dW_B_ptr        = dW_B.data_ptr();
        params.dW_C_ptr        = dW_C.data_ptr();
        params.tile_offsets_ptr = nullptr;
        params.tile_chunk_size  = 0;

        // ── Optional pointwise weights ────────────────────────────────────────
        if (pw_B_.has_value() && pw_B_.value().defined()) {
            has_pw = true;
            const at::Tensor &pw_B = pw_B_.value();
            const at::Tensor &pw_C = pw_C_.value();
            TORCH_CHECK(pw_B.scalar_type() == at::ScalarType::Float &&
                        pw_C.scalar_type() == at::ScalarType::Float,
                        "selective_scan_bwd_flat_ch: pw_B/pw_C must be float32");
            TORCH_CHECK(pw_B.is_cuda() && pw_C.is_cuda() &&
                        pw_B.is_contiguous() && pw_C.is_contiguous(),
                        "selective_scan_bwd_flat_ch: pw_B/pw_C must be CUDA and contiguous");
            TORCH_CHECK(pw_B.dim() == 2 && (int)pw_B.size(0) == dim && (int)pw_B.size(1) == dim,
                        "selective_scan_bwd_flat_ch: pw_B must be [dim, dim]");
            TORCH_CHECK(pw_C.dim() == 2 && (int)pw_C.size(0) == dim && (int)pw_C.size(1) == dim,
                        "selective_scan_bwd_flat_ch: pw_C must be [dim, dim]");
            // dpw_B / dpw_C: gradients for the pointwise weights [dim, dim].
            // dW_B / dW_C:   gradients for the depthwise weights [dim, dstate].
            // Both are computed by the kernel; they are kept in separate tensors.
            dpw_B = torch::zeros_like(pw_B);
            dpw_C = torch::zeros_like(pw_C);
            params.pw_B_ptr  = pw_B.data_ptr();
            params.pw_C_ptr  = pw_C.data_ptr();
            params.dpw_B_ptr = dpw_B.data_ptr<float>();
            params.dpw_C_ptr = dpw_C.data_ptr<float>();
        }
    } else if (use_tiled_bc) {
        // ── Tiled B/C layout support ─────────────────────────────────────────
        params.tile_offsets_ptr = tile_offsets_.value().data_ptr<int32_t>();
        params.tile_chunk_size  = static_cast<int>(tile_chunk_size_);
        params.is_variable_B = true;
        params.is_variable_C = true;
    } else {
        params.tile_offsets_ptr = nullptr;
        params.tile_chunk_size  = 0;
    }

    at::cuda::CUDAGuard device_guard{u_flat.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(input_type, "selective_scan_bwd_flat_ch", [&] {
        using weight_t = float;
        selective_scan_bwd_flat_ch_cuda<input_t, weight_t>(params, stream);
    });

    std::vector<at::Tensor> result = {du_flat, dA,
        use_depthwise ? dW_B : dB,   // B-side: dW_B (depthwise) or dB (legacy)
        use_depthwise ? dW_C : dC,   // C-side: dW_C (depthwise) or dC (legacy)
        ddt_idle, ddt_active};
    if (D_.has_value()) result.push_back(dD);
    if (has_z)          result.push_back(dz_flat);
    // Pointwise gradients — appended LAST when pw_B was provided.
    // Python side indexes these as result[base+0] and result[base+1].
    if (has_pw) {
        result.push_back(dpw_B);
        result.push_back(dpw_C);
    }
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &selective_scan_fwd, "Selective scan forward");
    m.def("bwd", &selective_scan_bwd, "Selective scan backward");
    m.def("fwd_flat_ch", &selective_scan_fwd_flat_ch,
        pybind11::arg("u_flat"),
        pybind11::arg("k_flat"),
        pybind11::arg("cu_seqlens"),
        pybind11::arg("max_seqlen"),
        pybind11::arg("dt_idle"),
        pybind11::arg("dt_active"),
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("D") = c10::nullopt,
        pybind11::arg("z") = c10::nullopt,
        pybind11::arg("tile_offsets") = c10::nullopt,
        pybind11::arg("tile_chunk_size") = 0,
        pybind11::arg("is_depthwise_bc") = false,
        pybind11::arg("pw_B") = c10::nullopt,
        pybind11::arg("pw_C") = c10::nullopt,
        "Selective scan forward (flat per-channel, jagged layout)");
    m.def("bwd_flat_ch", &selective_scan_bwd_flat_ch,
        pybind11::arg("u_flat"),
        pybind11::arg("k_flat"),
        pybind11::arg("cu_seqlens"),
        pybind11::arg("max_seqlen"),
        pybind11::arg("dt_idle"),
        pybind11::arg("dt_active"),
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("dout_flat"),
        pybind11::arg("x"),
        pybind11::arg("D") = c10::nullopt,
        pybind11::arg("z") = c10::nullopt,
        pybind11::arg("out") = c10::nullopt,
        pybind11::arg("tile_offsets") = c10::nullopt,
        pybind11::arg("tile_chunk_size") = 0,
        pybind11::arg("is_depthwise_bc") = false,
        pybind11::arg("pw_B") = c10::nullopt,
        pybind11::arg("pw_C") = c10::nullopt,
        "Selective scan backward (flat per-channel, jagged layout)");
}
