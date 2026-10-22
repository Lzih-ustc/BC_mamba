/*
 * extract_events_cuda.cu
 *
 * Fused CUDA extension – three entry points:
 *
 *  1. extract_events        (LEGACY – kept for backward compat)
 *     mask[B,D,L] bool → (ch_ids, t_steps, k_flat, cu_seqlens, max_seqlen, head_flags)
 *
 *  2. extract_base_events   (NEW Phase 1)
 *     base_input[B,d_model,L] float/half/bf16 + threshold
 *     → (base_ch_ids, base_t_steps, base_k_flat,
 *        base_cu_seqlens, base_counts, base_max_seqlen)
 *     Thresholds directly on float, no bool mask materialised.
 *     k_flat is computed inline inside fill_base_kernel (fused pass 3+4).
 *     head_flags scratch buffer eliminated.
 *
 *  3. expand_base_events    (NEW Phase 1)
 *     base event stream + base_counts + (B, d_model, expand)
 *     → (channel_ids, time_steps, k_flat, cu_seqlens, max_seqlen)
 *     Logically replicates each base channel into `expand` consecutive
 *     d_inner channels using the same ordering as repeat_interleave.
 *
 * Design invariant: CUDA kernels produce int32 index tensors only.
 * Float gather (u_flat / x_conv_flat) stays in Python so that
 * PyTorch autograd can track it without a hand-written backward.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// ============================================================================
// ── LEGACY kernels (unchanged) ───────────────────────────────────────────────
// ============================================================================

template <int COUNT_BLK>
__global__ void count_kernel(
    const bool*  __restrict__ mask,
    int32_t*     __restrict__ counts,
    int L
) {
    int ch = blockIdx.x;
    const bool* row = mask + (long)ch * L;
    int cnt = 0;
    for (int t = threadIdx.x; t < L; t += COUNT_BLK)
        cnt += (int)row[t];
    using BR = cub::BlockReduce<int, COUNT_BLK>;
    __shared__ typename BR::TempStorage ts;
    cnt = BR(ts).Sum(cnt);
    if (threadIdx.x == 0) counts[ch] = cnt;
}

template <int FILL_BLK>
__global__ void fill_kernel(
    const bool*    __restrict__ mask,
    const int32_t* __restrict__ cu_seqlens,
    int32_t*       __restrict__ ch_ids,
    int32_t*       __restrict__ t_steps,
    int L
) {
    constexpr int WARPS = FILL_BLK / 32;
    int ch   = blockIdx.x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    const bool* rm = mask + (long)ch * L;
    __shared__ int warp_prefix[WARPS + 1];
    __shared__ int blk_off;
    if (threadIdx.x == 0) blk_off = 0;
    __syncthreads();
    int base = cu_seqlens[ch];
    for (int t0 = 0; t0 < L; t0 += FILL_BLK) {
        int  t  = t0 + threadIdx.x;
        bool ev = (t < L) && rm[t];
        unsigned ballot = __ballot_sync(0xffffffffu, ev);
        int warp_cnt    = __popc(ballot);
        int lane_off    = __popc(ballot & ((1u << lane) - 1u));
        if (lane == 0) warp_prefix[warp] = warp_cnt;
        __syncthreads();
        if (threadIdx.x == 0) {
            int acc = 0;
            for (int w = 0; w < WARPS; w++) {
                int tmp = warp_prefix[w];
                warp_prefix[w] = acc;
                acc += tmp;
            }
            warp_prefix[WARPS] = acc;
        }
        __syncthreads();
        if (ev) {
            int pos      = base + blk_off + warp_prefix[warp] + lane_off;
            ch_ids[pos]  = ch;
            t_steps[pos] = t;
        }
        __syncthreads();
        if (threadIdx.x == 0) blk_off += warp_prefix[WARPS];
        __syncthreads();
    }
}

__global__ void kflat_kernel(
    const int32_t* __restrict__ ch_ids,
    const int32_t* __restrict__ t_steps,
    int32_t*       __restrict__ k_flat,
    int32_t*       __restrict__ hflags,
    int E
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    bool is_start = (i == 0) || (ch_ids[i] != ch_ids[i - 1]);
    hflags[i] = (int32_t)is_start;
    k_flat[i]  = is_start ? (t_steps[i] + 1) : (t_steps[i] - t_steps[i - 1]);
}

// ── Legacy host entry ────────────────────────────────────────────────────────
std::vector<torch::Tensor> extract_events_fwd(torch::Tensor mask) {
    TORCH_CHECK(mask.is_cuda(),       "mask must be a CUDA tensor");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
    TORCH_CHECK(mask.dtype() == torch::kBool, "mask must be bool");

    const int64_t B = mask.size(0), D = mask.size(1), L = mask.size(2);
    const int NC   = (int)(B * D);
    auto dev       = mask.device();
    auto stream    = at::cuda::getCurrentCUDAStream();
    auto mf        = mask.view({NC, L});

    auto counts = torch::empty({NC},
        torch::TensorOptions().dtype(torch::kInt32).device(dev));
    constexpr int COUNT_BLK = 256;
    count_kernel<COUNT_BLK><<<NC, COUNT_BLK, 0, stream>>>(
        mf.data_ptr<bool>(), counts.data_ptr<int32_t>(), (int)L);

    auto cu_seqlens = torch::empty({NC + 1},
        torch::TensorOptions().dtype(torch::kInt32).device(dev));
    {
        auto* c_ptr  = counts.data_ptr<int32_t>();
        auto* cs_ptr = cu_seqlens.data_ptr<int32_t>();
        void*  d_tmp  = nullptr; size_t tmp_sz = 0;
        cub::DeviceScan::InclusiveSum(d_tmp, tmp_sz, c_ptr, cs_ptr + 1, NC, stream);
        auto tmp_buf = torch::empty({(int64_t)tmp_sz},
            torch::TensorOptions().dtype(torch::kByte).device(dev));
        cub::DeviceScan::InclusiveSum(
            tmp_buf.data_ptr(), tmp_sz, c_ptr, cs_ptr + 1, NC, stream);
        cudaMemsetAsync(cs_ptr, 0, sizeof(int32_t), stream);
    }

    int32_t E_host = 0;
    cudaMemcpyAsync(&E_host, cu_seqlens.data_ptr<int32_t>() + NC,
                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (E_host == 0) {
        auto ei = torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
        auto ms = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
        return {ei, ei, ei, cu_seqlens, ms, ei};
    }

    auto opt_i   = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    auto ch_ids  = torch::empty({E_host}, opt_i);
    auto t_steps = torch::empty({E_host}, opt_i);
    auto k_flat  = torch::empty({E_host}, opt_i);
    auto hflags  = torch::empty({E_host}, opt_i);

    constexpr int FILL_BLK = 256;
    fill_kernel<FILL_BLK><<<NC, FILL_BLK, 0, stream>>>(
        mf.data_ptr<bool>(), cu_seqlens.data_ptr<int32_t>(),
        ch_ids.data_ptr<int32_t>(), t_steps.data_ptr<int32_t>(), (int)L);

    {
        int grid4 = (E_host + 255) / 256;
        kflat_kernel<<<grid4, 256, 0, stream>>>(
            ch_ids.data_ptr<int32_t>(), t_steps.data_ptr<int32_t>(),
            k_flat.data_ptr<int32_t>(), hflags.data_ptr<int32_t>(), E_host);
    }

    auto max_seqlen_t = counts.max();
    return {ch_ids, t_steps, k_flat, cu_seqlens, max_seqlen_t, hflags};
}


// ============================================================================
// ── NEW: extract_base_events ─────────────────────────────────────────────────
//
// Thresholds directly on the float base_input tensor.
// No bool mask is materialised – saves one B*d_model*L bool buffer.
//
// Grid : (NC_base,)   Block : (BLK,)
// ============================================================================

// ── Kernel A: count events per base channel ──────────────────────────────────
template <typename scalar_t, int COUNT_BLK>
__global__ void count_base_kernel(
    const scalar_t* __restrict__ x,      // [NC_base, L]
    int32_t*        __restrict__ counts, // [NC_base]
    int L,
    float threshold
) {
    int ch = blockIdx.x;
    const scalar_t* row = x + (long)ch * L;
    int cnt = 0;
    for (int t = threadIdx.x; t < L; t += COUNT_BLK)
        cnt += (int)((float)row[t] > threshold);
    using BR = cub::BlockReduce<int, COUNT_BLK>;
    __shared__ typename BR::TempStorage ts;
    cnt = BR(ts).Sum(cnt);
    if (threadIdx.x == 0) counts[ch] = cnt;
}

// ── Kernel B: scatter base event indices + k_flat (fused, no kflat_kernel pass) ──
//
// k_flat definition (inlined from the old kflat_kernel):
//   first event in channel → k_flat = t + 1
//   subsequent events      → k_flat = t - prev_t
//
// Implementation:
//   • __shared__ prev_t_shared  – t of the last event written before this tile
//                                  (-1 sentinel = no event yet in this channel)
//   • __shared__ tile_event_ts  – t values of events in the current tile,
//                                  indexed by local position within the tile
//     After ballot+prefix we know each event's local_pos; we write t there,
//     then read tile_event_ts[local_pos - 1] for the previous t.
//     The first event in a tile (local_pos == 0) falls back to prev_t_shared.
//
// head_flags scratch buffer is no longer needed.
template <typename scalar_t, int FILL_BLK>
__global__ void fill_base_kernel(
    const scalar_t* __restrict__ x,          // [NC_base, L]
    const int32_t*  __restrict__ cu_seqlens, // [NC_base + 1]
    int32_t*        __restrict__ ch_ids,     // [E_base]
    int32_t*        __restrict__ t_steps,    // [E_base]
    int32_t*        __restrict__ k_flat,     // [E_base]  ← NEW: written inline
    int L,
    float threshold
) {
    constexpr int WARPS = FILL_BLK / 32;
    int ch   = blockIdx.x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    const scalar_t* row = x + (long)ch * L;

    __shared__ int warp_prefix[WARPS + 1];
    __shared__ int blk_off;
    __shared__ int prev_t_shared;          // t of last event before this tile; -1 = none
    __shared__ int tile_event_ts[FILL_BLK]; // t values for events in current tile

    if (threadIdx.x == 0) {
        blk_off       = 0;
        prev_t_shared = -1;   // sentinel: no previous event in this channel yet
    }
    __syncthreads();

    int base = cu_seqlens[ch];

    for (int t0 = 0; t0 < L; t0 += FILL_BLK) {
        int  t  = t0 + threadIdx.x;
        bool ev = (t < L) && ((float)row[t] > threshold);

        unsigned ballot   = __ballot_sync(0xffffffffu, ev);
        int warp_cnt      = __popc(ballot);
        int lane_off      = __popc(ballot & ((1u << lane) - 1u));
        int local_pos     = -1;  // only valid when ev == true

        if (lane == 0) warp_prefix[warp] = warp_cnt;
        __syncthreads();

        if (threadIdx.x == 0) {
            int acc = 0;
            for (int w = 0; w < WARPS; w++) {
                int tmp = warp_prefix[w];
                warp_prefix[w] = acc;
                acc += tmp;
            }
            warp_prefix[WARPS] = acc;   // total events in this tile
        }
        __syncthreads();

        // ── Stage 1: record t for each event in this tile ──────────────────
        // local_pos is the event's index within this tile (0-based).
        if (ev) {
            local_pos = warp_prefix[warp] + lane_off;
            tile_event_ts[local_pos] = t;
        }
        __syncthreads();   // tile_event_ts fully written before any read below

        // ── Stage 2: write outputs (ch_ids, t_steps, k_flat) ───────────────
        if (ev) {
            int pos = base + blk_off + local_pos;

            // Determine previous event's t:
            //   local_pos == 0 → first in tile → use prev_t_shared
            //                    (== -1 means first in channel → k_flat = t+1)
            //   local_pos  > 0 → previous event is tile_event_ts[local_pos - 1]
            int prev_t = (local_pos == 0) ? prev_t_shared
                                          : tile_event_ts[local_pos - 1];

            ch_ids[pos]  = ch;
            t_steps[pos] = t;
            k_flat[pos]  = (prev_t < 0) ? (t + 1) : (t - prev_t);
        }
        __syncthreads();

        // ── Stage 3: advance blk_off and prev_t_shared for next tile ───────
        if (threadIdx.x == 0) {
            int tile_cnt = warp_prefix[WARPS];
            if (tile_cnt > 0)
                prev_t_shared = tile_event_ts[tile_cnt - 1];
            blk_off += tile_cnt;
        }
        __syncthreads();
    }
}

// ── Host entry: extract_base_events ─────────────────────────────────────────
//
// Returns:
//   [0] base_ch_ids      [E_base]         int32
//   [1] base_t_steps     [E_base]         int32
//   [2] base_k_flat      [E_base]         int32
//   [3] base_cu_seqlens  [NC_base + 1]    int32
//   [4] base_counts      [NC_base]        int32  ← needed by expand_base_events
//   [5] base_max_seqlen  scalar tensor    int32
//
std::vector<torch::Tensor> extract_base_events_fwd(
    torch::Tensor base_input,  // [B, d_model, L]  float/half/bf16, C-contiguous
    double        threshold
) {
    TORCH_CHECK(base_input.is_cuda(),       "base_input must be CUDA");
    TORCH_CHECK(base_input.is_contiguous(), "base_input must be contiguous");
    TORCH_CHECK(base_input.dim() == 3,      "base_input must be 3-D [B, d_model, L]");

    const int B       = (int)base_input.size(0);
    const int d_model = (int)base_input.size(1);
    const int L       = (int)base_input.size(2);
    const int NC_base = B * d_model;
    const float thr   = (float)threshold;

    auto dev    = base_input.device();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opt_i  = torch::TensorOptions().dtype(torch::kInt32).device(dev);

    // Reshape to [NC_base, L] for kernel convenience
    auto x_flat = base_input.view({NC_base, L});

    // ── Pass 1: count ──────────────────────────────────────────────────────
    auto counts = torch::empty({NC_base}, opt_i);
    constexpr int COUNT_BLK = 256;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        base_input.scalar_type(), "count_base_kernel", [&] {
            count_base_kernel<scalar_t, COUNT_BLK><<<NC_base, COUNT_BLK, 0, stream>>>(
                x_flat.data_ptr<scalar_t>(),
                counts.data_ptr<int32_t>(),
                L, thr
            );
        }
    );

    // ── Pass 2: prefix sum → cu_seqlens ────────────────────────────────────
    auto cu_seqlens = torch::empty({NC_base + 1}, opt_i);
    {
        auto* c_ptr  = counts.data_ptr<int32_t>();
        auto* cs_ptr = cu_seqlens.data_ptr<int32_t>();
        void*  d_tmp  = nullptr; size_t tmp_sz = 0;
        cub::DeviceScan::InclusiveSum(d_tmp, tmp_sz, c_ptr, cs_ptr + 1, NC_base, stream);
        auto tmp_buf = torch::empty({(int64_t)tmp_sz},
            torch::TensorOptions().dtype(torch::kByte).device(dev));
        cub::DeviceScan::InclusiveSum(
            tmp_buf.data_ptr(), tmp_sz, c_ptr, cs_ptr + 1, NC_base, stream);
        cudaMemsetAsync(cs_ptr, 0, sizeof(int32_t), stream);
    }

    // ── Read total E_base (D2H sync – needed to allocate output buffers) ────
    int32_t E_base = 0;
    cudaMemcpyAsync(&E_base, cu_seqlens.data_ptr<int32_t>() + NC_base,
                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Edge case: no events
    if (E_base == 0) {
        auto ei  = torch::zeros({0}, opt_i);
        auto ms  = torch::zeros({1}, opt_i);
        return {ei, ei, ei, cu_seqlens, counts, ms};
    }

    // ── Allocate output buffers ────────────────────────────────────────────
    // base_k_flat is allocated here (not after fill) so we can pass it in;
    // hflags_tmp scratch is gone – k_flat is now computed inside fill_base_kernel.
    auto base_ch_ids  = torch::empty({E_base}, opt_i);
    auto base_t_steps = torch::empty({E_base}, opt_i);
    auto base_k_flat  = torch::empty({E_base}, opt_i);

    // ── Pass 3: scatter base indices + k_flat (fused, single kernel launch) ─
    constexpr int FILL_BLK = 256;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        base_input.scalar_type(), "fill_base_kernel", [&] {
            fill_base_kernel<scalar_t, FILL_BLK><<<NC_base, FILL_BLK, 0, stream>>>(
                x_flat.data_ptr<scalar_t>(),
                cu_seqlens.data_ptr<int32_t>(),
                base_ch_ids.data_ptr<int32_t>(),
                base_t_steps.data_ptr<int32_t>(),
                base_k_flat.data_ptr<int32_t>(),   // ← written inline now
                L, thr
            );
        }
    );

    // Pass 4 (kflat_kernel) removed: k_flat is computed inside fill_base_kernel.

    auto base_max_seqlen = counts.max();

    // Returns: base_ch_ids, base_t_steps, base_k_flat,
    //          base_cu_seqlens, base_counts, base_max_seqlen
    return {base_ch_ids, base_t_steps, base_k_flat,
            cu_seqlens, counts, base_max_seqlen};
}


// ============================================================================
// ── NEW: expand_base_events ──────────────────────────────────────────────────
//
// Logically replicates each base channel c (= b*d_model + d_in) into
// `expand` consecutive d_inner channels using the same layout as:
//   repeat_interleave(dim=channel, repeats=expand)
//
// Mapping:  base ch c  →  logical chs  c*expand + r,  r = 0..expand-1
// This equals:  (b*d_model + d_in)*expand + r
//             = b*d_inner + d_in*expand + r   ✓
//
// ============================================================================

// ── Kernel C: build logical_cu_seqlens directly from base_cu_seqlens ─────────
//
// For base channel c with:
//   base_start = base_cu[c]
//   cnt        = base_cu[c+1] - base_cu[c]
//
// The r-th replica's start in the expanded stream is:
//   logical_cu[c * expand + r] = base_start * expand + r * cnt
//
// The final sentinel:
//   logical_cu[NC_inner] = base_cu[NC_base] * expand
//
// Launch with (NC_base + 1) threads so the last thread (c == NC_base)
// writes the sentinel without a separate cudaMemset.
//
__global__ void build_logical_cu_kernel(
    const int32_t* __restrict__ base_cu,     // [NC_base + 1]
    int32_t*       __restrict__ logical_cu,  // [NC_inner + 1]  = [NC_base*expand + 1]
    int NC_base,
    int expand
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c > NC_base) return;          // strictly > so c == NC_base is handled

    if (c < NC_base) {
        int base_start = base_cu[c];
        int cnt        = base_cu[c + 1] - base_start;
        for (int r = 0; r < expand; ++r)
            logical_cu[c * expand + r] = base_start * expand + r * cnt;
    } else {
        // c == NC_base: write the closing sentinel
        logical_cu[NC_base * expand] = base_cu[NC_base] * expand;
    }
}

// ── Kernel D: scatter base events → expanded slots (analytic positions) ───────
//
// For base event i in channel base_ch:
//   base_start = base_cu[base_ch]
//   cnt        = base_cu[base_ch + 1] - base_start
//   local_off  = i - base_start
//
// Output position for replica r:
//   out_pos = base_start * expand + r * cnt + local_off
//
// No logical_cu_seqlens lookup needed — position is derived purely from
// base_cu_seqlens, eliminating the expand_counts_kernel + DeviceScan step.
//
__global__ void expand_events_direct_kernel(
    const int32_t* __restrict__ base_ch_ids,   // [E_base]
    const int32_t* __restrict__ base_t_steps,  // [E_base]
    const int32_t* __restrict__ base_k_flat,   // [E_base]
    const int32_t* __restrict__ base_cu,       // [NC_base + 1]
    int32_t*       __restrict__ ch_ids,        // [E]
    int32_t*       __restrict__ t_steps,       // [E]
    int32_t*       __restrict__ k_flat,        // [E]
    int E_base,
    int expand
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E_base) return;

    int base_ch    = base_ch_ids[i];
    int t          = base_t_steps[i];
    int k          = base_k_flat[i];
    int base_start = base_cu[base_ch];
    int cnt        = base_cu[base_ch + 1] - base_start;
    int local_off  = i - base_start;

    for (int r = 0; r < expand; ++r) {
        int out_pos      = base_start * expand + r * cnt + local_off;
        ch_ids[out_pos]  = base_ch * expand + r;
        t_steps[out_pos] = t;
        k_flat[out_pos]  = k;
    }
}

// ── Host entry: expand_base_events ──────────────────────────────────────────
//
// Inputs:
//   base_ch_ids, base_t_steps, base_k_flat  [E_base]    int32
//   base_cu_seqlens                          [NC_base+1] int32
//   B, d_model, expand                       ints
//
// Returns:
//   [0] channel_ids  [E]            int32  (E = E_base * expand)
//   [1] time_steps   [E]            int32
//   [2] k_flat       [E]            int32
//   [3] cu_seqlens   [NC_inner + 1] int32
//   [4] max_seqlen   scalar tensor  int32
//
// Design: no base_counts parameter, no logical_counts buffer, no second scan.
//   Step 1 – build_logical_cu_kernel    : O(NC_base) analytic writes
//   Step 2 – expand_events_direct_kernel: O(E_base) analytic scatter
//
std::vector<torch::Tensor> expand_base_events_fwd(
    torch::Tensor base_ch_ids,     // [E_base]     int32
    torch::Tensor base_t_steps,    // [E_base]     int32
    torch::Tensor base_k_flat,     // [E_base]     int32
    torch::Tensor base_cu_seqlens, // [NC_base+1]  int32
    int64_t B,
    int64_t d_model,
    int64_t expand
) {
    TORCH_CHECK(base_ch_ids.is_cuda(),       "base_ch_ids must be CUDA");
    TORCH_CHECK(base_ch_ids.is_contiguous(), "base_ch_ids must be contiguous");

    const int NC_base  = (int)(B * d_model);
    const int NC_inner = (int)(B * d_model * expand);
    const int E_base   = (int)base_ch_ids.size(0);
    const int exp      = (int)expand;

    auto dev    = base_ch_ids.device();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opt_i  = torch::TensorOptions().dtype(torch::kInt32).device(dev);

    // Edge case: nothing to expand
    if (E_base == 0) {
        auto cu_seqlens_out = torch::zeros({NC_inner + 1}, opt_i);
        auto ms             = torch::zeros({1}, opt_i);
        auto ei             = torch::zeros({0}, opt_i);
        return {ei, ei, ei, cu_seqlens_out, ms};
    }

    // ── Step 1: build logical_cu_seqlens analytically (no scan needed) ────
    // Kernel launches NC_base + 1 threads:
    //   threads 0..NC_base-1 : write expand entries each
    //   thread  NC_base      : write the closing sentinel
    auto logical_cu_seqlens = torch::empty({NC_inner + 1}, opt_i);
    {
        int threads = 256;
        int grid    = (NC_base + 1 + threads - 1) / threads;
        build_logical_cu_kernel<<<grid, threads, 0, stream>>>(
            base_cu_seqlens.data_ptr<int32_t>(),
            logical_cu_seqlens.data_ptr<int32_t>(),
            NC_base, exp
        );
    }

    // Total expanded events = E_base * expand (exact, no D2H needed)
    const int E = E_base * exp;

    // ── Step 2: scatter expanded events with analytic positions ───────────
    auto ch_ids  = torch::empty({E}, opt_i);
    auto t_steps = torch::empty({E}, opt_i);
    auto k_flat  = torch::empty({E}, opt_i);

    {
        int threads = 256;
        int grid    = (E_base + threads - 1) / threads;
        expand_events_direct_kernel<<<grid, threads, 0, stream>>>(
            base_ch_ids.data_ptr<int32_t>(),
            base_t_steps.data_ptr<int32_t>(),
            base_k_flat.data_ptr<int32_t>(),
            base_cu_seqlens.data_ptr<int32_t>(),
            ch_ids.data_ptr<int32_t>(),
            t_steps.data_ptr<int32_t>(),
            k_flat.data_ptr<int32_t>(),
            E_base, exp
        );
    }

    // max_seqlen: max over per-channel counts = max(base_cu[c+1] - base_cu[c])
    // Computed via ATen slice-diff — no extra kernel, no D2H, no base_counts.
    auto diff       = base_cu_seqlens.slice(0, 1, NC_base + 1)
                    - base_cu_seqlens.slice(0, 0, NC_base);
    auto max_seqlen = diff.max();

    return {ch_ids, t_steps, k_flat, logical_cu_seqlens, max_seqlen};
}


// ============================================================================
// ── NEW: gather_events  (fused x_conv gather, scatter-store backward) ─────────
//
// Design notes
// ------------
//  * Forward:  one kernel launch loads (ch, t) once and writes to out_x,
//              avoiding PyTorch advanced-index overhead.
//  * Backward: scatter-store (not atomicAdd).  Safe because each (ch, t) pair
//              is unique by construction: fill_base_kernel writes at most one
//              event per (channel, time) and expand_events_kernel maps each
//              base event to a distinct logical channel.  See design comments
//              at the top of expand_base_events for the full argument.
//  * int32 indices are used directly inside the kernel; the (long) cast on the
//              product prevents int overflow for large NC*L.
//  * Both functions are registered in the same extract_events_cuda extension so
//              no separate .so is needed.
// ============================================================================

// ── Kernel E: fused forward gather ───────────────────────────────────────────
//
// Conceptually: x_flat is a [NC, L] view.
// Each thread reads (ch, t) once and gathers from the array.
//
template <typename scalar_t>
__global__ void gather_events_fwd2_kernel(
    const scalar_t* __restrict__ x_flat,  // [NC, L]  x_conv reshaped
    const int32_t*  __restrict__ ch_ids,  // [E]
    const int32_t*  __restrict__ t_steps, // [E]
    scalar_t*       __restrict__ out_x,   // [E]
    int L,
    int E
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    long idx  = (long)ch_ids[e] * L + t_steps[e];
    out_x[e]  = x_flat[idx];
}

// ── Kernel F: fused backward scatter-store ────────────────────────────────────
//
// Scatter grad_x → dx_flat in one pass.
// Uses plain store (no atomic) — valid because each (ch, t) is unique.
//
template <typename scalar_t>
__global__ void scatter_events_bwd2_kernel(
    const scalar_t* __restrict__ grad_x,  // [E]
    const int32_t*  __restrict__ ch_ids,  // [E]
    const int32_t*  __restrict__ t_steps, // [E]
    scalar_t*       __restrict__ dx_flat, // [NC, L]   zero-initialised by caller
    int L,
    int E
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    long idx      = (long)ch_ids[e] * L + t_steps[e];
    dx_flat[idx]  = grad_x[e];
}

// ── Host entry: gather_events_fwd ─────────────────────────────────────────────
//
// x      : [B, D, L]  contiguous, CUDA
// ch_ids, t_steps : [E]  int32, CUDA
// Returns: {out_x[E]}
//
std::vector<torch::Tensor> gather_events_fwd(
    torch::Tensor x,           // [B, D, L]
    torch::Tensor channel_ids, // [E] int32
    torch::Tensor time_steps   // [E] int32
) {
    TORCH_CHECK(x.is_cuda(), "gather_events_fwd: x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "gather_events_fwd: x must be contiguous");
    TORCH_CHECK(channel_ids.scalar_type() == torch::kInt32,
                "gather_events_fwd: channel_ids must be int32");
    TORCH_CHECK(time_steps.scalar_type()  == torch::kInt32,
                "gather_events_fwd: time_steps must be int32");
    TORCH_CHECK(x.dim() == 3, "gather_events_fwd: x must be 3-D [B, D, L]");

    const int L = (int)x.size(2);
    const int E = (int)channel_ids.size(0);

    auto out_x = torch::empty({E}, x.options());

    if (E == 0) return {out_x};

    auto stream  = at::cuda::getCurrentCUDAStream();
    int  threads = 256;
    int  blocks  = (E + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "gather_events_fwd", [&] {
            gather_events_fwd2_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                channel_ids.data_ptr<int32_t>(),
                time_steps.data_ptr<int32_t>(),
                out_x.data_ptr<scalar_t>(),
                L, E
            );
        }
    );

    return {out_x};
}

// ── Host entry: gather_events_bwd ─────────────────────────────────────────────
//
// grad_x         : [E]   contiguous, CUDA
// ch_ids, t_steps: [E]   int32, CUDA
// NC, L          : x_conv.shape = [NC, L] logically (= B*D, L)
// Returns: {dx[NC, L]}  (zeros except at event positions)
//
std::vector<torch::Tensor> gather_events_bwd(
    torch::Tensor grad_x,      // [E]
    torch::Tensor channel_ids, // [E] int32
    torch::Tensor time_steps,  // [E] int32
    int64_t NC,
    int64_t L
) {
    TORCH_CHECK(grad_x.is_cuda(), "gather_events_bwd: grad tensor must be CUDA");
    TORCH_CHECK(channel_ids.scalar_type() == torch::kInt32,
                "gather_events_bwd: channel_ids must be int32");
    TORCH_CHECK(time_steps.scalar_type()  == torch::kInt32,
                "gather_events_bwd: time_steps must be int32");

    // Zero-init: positions not touched by scatter retain 0 gradient.
    auto dx = torch::zeros({NC, L}, grad_x.options());

    const int E = (int)grad_x.size(0);
    if (E == 0) return {dx};

    auto stream  = at::cuda::getCurrentCUDAStream();
    int  threads = 256;
    int  blocks  = (E + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad_x.scalar_type(), "gather_events_bwd", [&] {
            scatter_events_bwd2_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                grad_x.data_ptr<scalar_t>(),
                channel_ids.data_ptr<int32_t>(),
                time_steps.data_ptr<int32_t>(),
                dx.data_ptr<scalar_t>(),
                (int)L, E
            );
        }
    );

    return {dx};
}

// ============================================================================
// ── Module bindings ──────────────────────────────────────────────────────────
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "extract_events",
        &extract_events_fwd,
        "LEGACY – bool mask → (ch_ids, t_steps, k_flat, cu_seqlens, max_seqlen, head_flags). "
        "Float gather done in Python for autograd."
    );
    m.def(
        "extract_base_events",
        &extract_base_events_fwd,
        "NEW – float base_input + threshold → base event stream (no bool mask). "
        "Returns (base_ch_ids, base_t_steps, base_k_flat, "
        "base_cu_seqlens, base_counts, base_max_seqlen)."
    );
    m.def(
        "expand_base_events",
        &expand_base_events_fwd,
        "NEW – logically expand base event stream by `expand` (repeat_interleave semantics). "
        "No base_counts argument; no logical_counts buffer; no second DeviceScan. "
        "logical_cu_seqlens built analytically via build_logical_cu_kernel; "
        "scatter positions computed analytically via expand_events_direct_kernel. "
        "Returns (channel_ids, time_steps, k_flat, cu_seqlens, max_seqlen)."
    );
    m.def(
        "gather_events_fwd",
        &gather_events_fwd,
        "Fused gather of x_conv at (channel_ids, time_steps). "
        "x: [B,D,L]; ch_ids, t_steps: [E] int32. "
        "Returns out_x[E]. "
        "One kernel launch."
    );
    m.def(
        "gather_events_bwd",
        &gather_events_bwd,
        "Fused scatter-store backward for gather_events_fwd. "
        "grad_x: [E]; ch_ids, t_steps: [E] int32; NC, L: ints. "
        "Returns dx[NC,L]. "
        "Plain store (no atomicAdd): safe when (ch,t) pairs are unique."
    );
}
