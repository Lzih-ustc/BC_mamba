/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstdint>

////////////////////////////////////////////////////////////////////////////////////////////////////
// Decoupled Lookback tile status constants — shared by fwd and bwd kernels.
////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef SSM_TILE_STATUS_DEFINED
#define SSM_TILE_STATUS_DEFINED
static constexpr int32_t TILE_INVALID  = 0;   // not yet written
static constexpr int32_t TILE_PARTIAL  = 1;   // local aggregate is available
static constexpr int32_t TILE_COMPLETE = 2;   // global inclusive prefix is available
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SSMScanParamsBase {
    using index_t = uint32_t;

    int batch, seqlen, n_chunks;
    index_t a_batch_stride;
    index_t b_batch_stride;
    index_t out_batch_stride;

    // Common data pointers.
    void *__restrict__ a_ptr;
    void *__restrict__ b_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ x_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SSMParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, dstate, n_groups, n_chunks;
    int dim_ngroups_ratio;
    bool is_variable_B;
    bool is_variable_C;

    bool delta_softplus;

    index_t A_d_stride;
    index_t A_dstate_stride;
    index_t B_batch_stride;
    index_t B_d_stride;
    index_t B_dstate_stride;
    index_t B_group_stride;
    index_t C_batch_stride;
    index_t C_d_stride;
    index_t C_dstate_stride;
    index_t C_group_stride;
    index_t u_batch_stride;
    index_t u_d_stride;
    index_t delta_batch_stride;
    index_t delta_d_stride;
    index_t z_batch_stride;
    index_t z_d_stride;
    index_t out_batch_stride;
    index_t out_d_stride;
    index_t out_z_batch_stride;
    index_t out_z_d_stride;

    // Common data pointers.
    void *__restrict__ A_ptr;
    void *__restrict__ B_ptr;
    void *__restrict__ C_ptr;
    void *__restrict__ D_ptr;
    void *__restrict__ u_ptr;
    void *__restrict__ delta_ptr;
    void *__restrict__ delta_bias_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ x_ptr;
    void *__restrict__ z_ptr;
    void *__restrict__ out_z_ptr;
    // Optional per-position dim ids for packed dense 2D path.
    // Shape: (batch, seqlen) int32, where batch == n_blocks for mamba2d packing.
    // When nullptr / has_dim_ids == 0, kernels fall back to original dense dim semantics.
    void *__restrict__ dim_ids_ptr;
    int has_dim_ids;
    index_t dim_ids_batch_stride;
    index_t dim_ids_d_stride;

    // ── per-position head_flag ──────────────────────────────────────────────
    void *head_flag_ptr          = nullptr;
    int   has_head_flag          = 0;
    int   head_flag_batch_stride = 0;
    int   head_flag_d_stride     = 0;
};

// Segmented forward-scan params — SSMParamsBase + a reset-mask pointer.
// The mask (uint8_t[seqlen]) marks positions where the SSM state should be
// reset to zero (i.e. the first event of a new logical channel in a packed
// flat sequence).  nullptr is allowed and means no resets.
struct SSMParamsSegmented : public SSMParamsBase {
    // uint8_t[seqlen]  —  1 = first event of a new channel (reset state here).
    // nullptr means no resets (behaves identically to the original kernel).
    void *new_channel_mask_ptr;
};

struct SSMParamsBwd: public SSMParamsBase {
    index_t dout_batch_stride;
    index_t dout_d_stride;
    index_t dA_d_stride;
    index_t dA_dstate_stride;
    index_t dB_batch_stride;
    index_t dB_group_stride;
    index_t dB_d_stride;
    index_t dB_dstate_stride;
    index_t dC_batch_stride;
    index_t dC_group_stride;
    index_t dC_d_stride;
    index_t dC_dstate_stride;
    index_t du_batch_stride;
    index_t du_d_stride;
    index_t dz_batch_stride;
    index_t dz_d_stride;
    index_t ddelta_batch_stride;
    index_t ddelta_d_stride;

    // Common data pointers.
    void *__restrict__ dout_ptr;
    void *__restrict__ dA_ptr;
    void *__restrict__ dB_ptr;
    void *__restrict__ dC_ptr;
    void *__restrict__ dD_ptr;
    void *__restrict__ du_ptr;
    void *__restrict__ dz_ptr;
    void *__restrict__ ddelta_ptr;
    void *__restrict__ ddelta_bias_ptr;
};

// Segmented backward-scan params — SSMParamsBwd + a reset-mask pointer.
// The mask (uint8_t[seqlen]) marks positions where the SSM state is reset
// (i.e. the first event of a new logical channel in a packed flat sequence).
// nullptr is allowed and means no resets.
struct SSMParamsSegmentedBwd : public SSMParamsBwd {
    void *new_channel_mask_ptr;
};

// Flat/jagged segmented selective-scan params (forward).
// Layout: events for all channels are packed into 1-D arrays; channel ch slice is
// [cu_seqlens[ch], cu_seqlens[ch+1]).
#ifndef SSM_PARAMS_FLAT_SEG_DEFINED
#define SSM_PARAMS_FLAT_SEG_DEFINED
struct SSMParamsFlatSeg {
    void     *u_ptr;            // [E]             input_t
    int32_t  *k_ptr;            // [E]             int32
    int32_t  *cu_seqlens_ptr;   // [n_channels+1]  int32
    int32_t  *head_flags_ptr;   // [E]             int32  (nullable)
    int32_t  *channel_ids_ptr;  // [E]             int32  (nullable; dim_id = channel_id % dim)

    void     *A_ptr;            // [dim, dstate]   weight_t
    void     *B_ptr;            // [E, dstate]     float32
    void     *C_ptr;            // [E, dstate]     float32
    void     *D_ptr;            // [dim]           float32 or nullptr
    void     *z_ptr;            // [E]             input_t or nullptr

    float    *dt_idle_ptr;      // [dim]           float32
    float    *dt_active_ptr;    // [dim]           float32

    void     *out_ptr;          // [E]             input_t
    void     *out_z_ptr;        // [E]             input_t or nullptr
    // Block-boundary SSM states saved by the forward pass for use by backward.
    // Layout: [n_blocks, dstate] float2  (scan_t = float2).
    //   x_ptr[b * dstate + n] = inclusive forward exit state of block b for state n.
    // Written by forward Phase 3 (fixup output pass) at the end of each block.
    // Consumed by backward Phase 3 as the exclusive forward prefix entering block b+1:
    //   block b reads x_ptr[(b-1)*dstate+n] to obtain h_{block_start_of_b}.
    void     *x_ptr;

    // Scratch buffers for global 1-D segmented scan pipeline:
    // block-scan -> block-prefix -> fixup.
    // block_ab / block_prefix store float2 tuples (A_end, B_end) per [block, state].
    void     *block_ab_ptr;      // [n_blocks, dstate, 2] float32 (float2 view)
    void     *block_prefix_ptr;  // [n_blocks, dstate, 2] float32 (float2 view)
    void     *prefix_a_ptr;      // [E, dstate] float32 (optional cached local prefix A)

    int       dim;
    int       dstate;
    int       n_channels;
    int       n_events;
    int       n_blocks;
    int       n_chunks;
    int       max_seqlen;

    int       A_d_stride;
    int       A_dstate_stride;

    // B/C layout controls (match flat/event semantics):
    //   variable: (E, dstate)
    //   non-variable: (dim, dstate)
    bool      is_variable_B;
    bool      is_variable_C;
    int       B_row_stride;
    int       C_row_stride;

    // ── Tiled B/C layout support ───────────────────────────────────────────
    // When non-null, B/C are stored as [N_total_tiles, dstate, tile_chunk_size]
    // instead of [dstate, E].  tile_offsets[ch] is the global tile start index
    // for channel ch (prefix sum of per-channel chunk counts).
    int32_t  *tile_offsets_ptr = nullptr;  // [n_channels+1] int32  — nullptr = old layout
    int       tile_chunk_size  = 0;        // == kNThreads * kNItems used during Python tiling

    // ── Depthwise B/C ─────────────────────────────────────────────────────────
    // When is_depthwise_bc = true, B/C values are NOT pre-computed and passed
    // in B_ptr/C_ptr.  Instead the kernel computes them on-the-fly per event:
    //   B[event_i, state_s] = u[event_i] * W_B[dim_id, state_s]
    //   C[event_i, state_s] = u[event_i] * W_C[dim_id, state_s]
    // W_B_ptr and W_C_ptr point to [dim, dstate] float32 weight tensors.
    // B_ptr / C_ptr are unused (may be nullptr) in this mode.
    // is_variable_B / is_variable_C must be set to true so that the variable
    // template branch is instantiated; the depthwise runtime flag selects the
    // on-the-fly sub-path instead of the tiled/legacy memory load.
    bool  is_depthwise_bc  = false;
    void *W_B_ptr          = nullptr;  // [dim, dstate] float32
    void *W_C_ptr          = nullptr;  // [dim, dstate] float32

    // ── Depthwise + Pointwise B/C ─────────────────────────────────────────────
    // Optional pointwise weights applied on top of the depthwise above.
    // When pw_B_ptr != nullptr and is_depthwise_bc == true, each channel block
    // (dim_id = d) precomputes an effective weight before the scan loop:
    //
    //   W_B_eff[d, s] = Σ_{d'=0}^{dim-1}  pw_B[d, d'] * W_B[d', s]
    //   W_C_eff[d, s] = Σ_{d'=0}^{dim-1}  pw_C[d, d'] * W_C[d', s]
    //
    // and then uses  B[i, s] = u[i] * W_B_eff[d, s]  (same as pure depthwise
    // but with channel-mixed weights).  W_B/W_C themselves are unchanged.
    // pw_B_ptr / pw_C_ptr: [dim, dim] float32, row-major contiguous.
    void *pw_B_ptr = nullptr;  // [dim, dim] float32
    void *pw_C_ptr = nullptr;  // [dim, dim] float32
};
#endif

// Flat/jagged segmented selective-scan params (backward).
#ifndef SSM_PARAMS_FLAT_SEG_BWD_DEFINED
#define SSM_PARAMS_FLAT_SEG_BWD_DEFINED
struct SSMParamsFlatSegBwd : SSMParamsFlatSeg {
    void  *dout_ptr;        // [E]           input_t
    void  *du_ptr;          // [E]           input_t
    void  *dB_ptr;          // [E, dstate]   float32  (variable) or [dstate, dim] (non-variable)
    void  *dC_ptr;          // [E, dstate]   float32  (variable) or [dstate, dim] (non-variable)
    void  *dA_ptr;          // [dim, dstate] weight_t
    float *ddt_idle_ptr;    // [dim]         float32
    float *ddt_active_ptr;  // [dim]         float32
    float *dD_ptr;          // [dim]         float32
    void  *dz_ptr;          // [E]           input_t or nullptr

    // Depthwise B/C weight gradients (used when is_depthwise_bc = true).
    // Both have shape [dim, dstate] float32; accumulated via atomicAdd.
    void  *dW_B_ptr = nullptr;  // [dim, dstate] float32
    void  *dW_C_ptr = nullptr;  // [dim, dstate] float32

    // Pointwise weight gradients (used when pw_B_ptr != nullptr).
    // Both have shape [dim, dim] float32; accumulated via atomicAdd.
    float *dpw_B_ptr = nullptr;  // [dim, dim] float32
    float *dpw_C_ptr = nullptr;  // [dim, dim] float32

    // Backward global reverse-scan scratch buffers.
    void  *dx_partial_ptr;    // [E, dstate] float32
    void  *prefix_a_rev_ptr;   // [E, dstate] float32
    void  *block_postfix_ptr;  // [n_blocks, dstate, 2] float32 (float2 view)
    void  *dx_true_ptr;        // [E, dstate] float32 (fixed-up dx)
    void  *debug_flag_ptr;     // [1] int32 debug flag

    int    dA_d_stride;
    int    dA_dstate_stride;

    // Strides for non-variable dB/dC (layout: [dstate, dim], same as B/C).
    // Used by the kernel at:
    //   dB_base = dB_ptr + dim_id * dB_d_stride         (init pointer offset)
    //   dB_base[state_idx * dB_dstate_stride]           (atomicAdd)
    // For variable B/C these are unused (kernel uses B_row_stride instead).
    int    dB_d_stride;       // stride along dim dimension  (=stride(1), =1 when contiguous)
    int    dB_dstate_stride;  // stride along dstate dimension (=stride(0), =dim when contiguous)
    int    dC_d_stride;
    int    dC_dstate_stride;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// Event-driven SSM parameters (forward pass)
struct SSMParamsEvent {
    using index_t = uint32_t;

    int batch, dim, seqlen, dstate, n_groups, n_chunks;
    int dim_ngroups_ratio;
    int Nmax;  // Maximum number of events per channel
    bool is_variable_B;
    bool is_variable_C;

    // Strides for event tensors
    index_t k_batch_stride;
    index_t k_d_stride;
    index_t k_nmax_stride;
    index_t valid_batch_stride;
    index_t valid_d_stride;
    index_t valid_nmax_stride;

    // Strides for B/C (same as dense)
    index_t B_batch_stride;
    index_t B_group_stride;
    index_t B_dstate_stride;
    index_t B_nmax_stride;
    index_t C_batch_stride;
    index_t C_group_stride;
    index_t C_dstate_stride;
    index_t C_nmax_stride;

    // Strides for u/z (compressed)
    index_t u_batch_stride;
    index_t u_d_stride;
    index_t u_nmax_stride;
    index_t z_batch_stride;
    index_t z_d_stride;
    index_t z_nmax_stride;
    index_t out_batch_stride;
    index_t out_d_stride;
    index_t out_nmax_stride;
    index_t out_z_batch_stride;
    index_t out_z_d_stride;
    index_t out_z_nmax_stride;

    // Data pointers
    void *__restrict__ u_ptr;           // (B, D, Nmax) compressed input
    void *__restrict__ k_ptr;           // (B, D, Nmax) int32 event intervals
    void *__restrict__ valid_ptr;       // (B, D, Nmax) bool valid mask
    void *__restrict__ dt_idle_ptr;     // (D,) float32 idle time constant
    void *__restrict__ dt_active_ptr;   // (D,) float32 active time constant
    void *__restrict__ A_idle_ptr;      // (D, dstate) float32 idle state matrix
    void *__restrict__ A_active_ptr;    // (D, dstate) float32 active state matrix
    void *__restrict__ B_ptr;           // (B, dstate, Nmax) or (D, dstate)
    void *__restrict__ C_ptr;           // (B, dstate, Nmax) or (D, dstate)
    void *__restrict__ D_ptr;           // (D,) float32 skip connection
    void *__restrict__ z_ptr;           // (B, D, Nmax) compressed gating
    void *__restrict__ out_ptr;         // (B, D, Nmax) compressed output
    void *__restrict__ out_z_ptr;       // (B, D, Nmax) compressed gated output
    void *__restrict__ x_ptr;           // (B, D, n_chunks, dstate*2) state cache

    // fp32 accumulation buffer for tile-parallel full-dstate scan
    // shape: (B, D, Nmax) float32 — multiple state-tile blocks atomicAdd here
    void *__restrict__ out_accum_ptr;   // (B, D, Nmax) float32
    index_t out_accum_batch_stride;
    index_t out_accum_d_stride;
    index_t out_accum_nmax_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Event-driven SSM backward parameters
struct SSMParamsEventBwd: public SSMParamsEvent {
    // Additional strides for gradients
    index_t dout_batch_stride;
    index_t dout_d_stride;
    index_t dout_nmax_stride;
    index_t du_batch_stride;
    index_t du_d_stride;
    index_t du_nmax_stride;
    index_t dz_batch_stride;
    index_t dz_d_stride;
    index_t dz_nmax_stride;
    
    // Gradient pointers
    void *__restrict__ dout_ptr;        // (B, D, Nmax) output gradient
    void *__restrict__ du_ptr;          // (B, D, Nmax) u gradient
    void *__restrict__ ddt_idle_ptr;    // (D,) dt_idle gradient  
    void *__restrict__ ddt_active_ptr;  // (D,) dt_active gradient
    void *__restrict__ dA_idle_ptr;     // (D, dstate) A_idle gradient
    void *__restrict__ dA_active_ptr;   // (D, dstate) A_active gradient
    void *__restrict__ dB_ptr;          // Variable B gradient
    void *__restrict__ dC_ptr;          // Variable C gradient
    void *__restrict__ dD_ptr;          // (D,) D gradient
    void *__restrict__ dz_ptr;          // (B, D, Nmax) z gradient
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Event-driven SSM parameters (forward pass) - FLAT/JAGGED (1-D jagged layout)
// Layout: events for all channels are packed into 1-D arrays; cu_seqlens[ch..ch+1] gives the slice.
struct SSMParamsEventFlat {
    using index_t = uint32_t;

    // Sizes
    int n_channels;         // = B * D  (logical channel count)
    int dim;                // = D  (used to derive dim_id = ch % dim for per-dim params)
    int dstate;
    int n_groups;
    int dim_ngroups_ratio;
    int total_events;       // = E  (total number of events across all channels)

    bool is_variable_B;     // true  → B is (E, dstate); false → B is (dim, dstate)
    bool is_variable_C;     // true  → C is (E, dstate); false → C is (dim, dstate)

    // Row strides for B / C (in elements):
    //   is_variable_*=true  → B_row_stride = B.stride(0)  (= dstate when contiguous)
    //   is_variable_*=false → B_row_stride = dstate        (same, for uniform kernel code)
    int B_row_stride;
    int C_row_stride;

    // 1-D jagged data pointers (length E)
    void *__restrict__ u_ptr;            // (E,)  input_t   — packed activations
    void *__restrict__ k_ptr;            // (E,)  int32     — inter-event intervals
    void *__restrict__ z_ptr;            // (E,)  input_t   — gating; nullptr if !has_z
    void *__restrict__ out_ptr;          // (E,)  input_t   — ungated SSM output (y before gate; same as out when !has_z)
    void *__restrict__ out_z_ptr;        // (E,)  input_t   — gated output y*silu(z); nullptr if !has_z

    // Prefix-sum offsets: cu_seqlens[ch] … cu_seqlens[ch+1] is the event slice for channel ch
    void *__restrict__ cu_seqlens_ptr;   // (n_channels+1,)  int32

    // Optional segmented/debug helpers (may be nullptr)
    void *__restrict__ channel_ids_ptr;  // (E,) int32 channel id per event
    void *__restrict__ head_flags_ptr;   // (E,) int32 segment head flag
    void *__restrict__ tail_flags_ptr;   // (E,) int32 segment tail flag

    // Per-dim parameters — indexed by dim_id = ch % dim
    void *__restrict__ dt_idle_ptr;      // (dim,)         float
    void *__restrict__ dt_active_ptr;    // (dim,)         float
    void *__restrict__ A_idle_ptr;       // (dim, dstate)  float
    void *__restrict__ A_active_ptr;     // (dim, dstate)  float
    void *__restrict__ B_ptr;            // (dim, dstate) float  (non-variable) or (E, dstate) float (variable)
    void *__restrict__ C_ptr;            // (dim, dstate) float  (non-variable) or (E, dstate) float (variable)
    void *__restrict__ D_ptr;            // (dim,)         float  or nullptr

    // Per-channel final SSM state (written at the end of each channel's event sequence)
    void *__restrict__ x_ptr;            // (n_channels, dstate)  float

    // Per-event state history (written after each event's state update in forward pass)
    void *__restrict__ x_hist_ptr;       // (E, dstate)  float  — nullptr if not needed

    // Optional fp32 accumulation buffer (may be nullptr)
    void *__restrict__ out_accum_ptr;    // (E,) float32 or implementation-defined flat accumulation layout
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Event-driven SSM backward parameters — FLAT/JAGGED layout
struct SSMParamsEventFlatBwd : public SSMParamsEventFlat {
    // Gradient of output (upstream)
    void *__restrict__ dout_ptr;         // (E,)  input_t

    // Gradient outputs for inputs/gating
    void *__restrict__ du_ptr;           // (E,)  input_t
    void *__restrict__ dz_ptr;           // (E,)  input_t  — nullptr if !has_z

    // Gradient outputs for per-dim learnable parameters (float, accumulated via atomicAdd)
    void *__restrict__ ddt_idle_ptr;     // (dim,)         float
    void *__restrict__ ddt_active_ptr;   // (dim,)         float
    void *__restrict__ dA_idle_ptr;      // (dim, dstate)  float
    void *__restrict__ dA_active_ptr;    // (dim, dstate)  float
    void *__restrict__ dB_ptr;           // (dim, dstate) float (non-variable) or (E, dstate) float (variable)
    void *__restrict__ dC_ptr;           // (dim, dstate) float (non-variable) or (E, dstate) float (variable)
    void *__restrict__ dD_ptr;           // (dim,)         float  — nullptr if no D
};