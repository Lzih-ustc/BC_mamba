/******************************************************************************
 * restore_dense_fwd_kernel.cuh
 * Restore a (B, L, D) dense output from flat event values + time_steps.
 *
 * Semantics (zero-order hold / step function):
 *   - event positions:      out[b, t, d] = y_flat[e]
 *   - non-event positions with a previous event at t0:
 *                           out[b, t, d] = y_flat[e_prev]   (carry-forward)
 *   - non-event positions before the first event: 0  (zero-init by caller)
 *
 * Compared with the decay version, log_gamma is removed entirely.
 * The per-thread work is now just a binary search + one store — no __expf.
 *
 * Grid  : (n_channels, ceil(seqlen / BLOCK))  — one CTA per (channel, tile)
 * Block : (BLOCK,)  = (256,)
 *
 * Optimisations retained from the decay version:
 *  1. Shared-memory preload of time_steps + y_flat (≤ RD_FWD_MAX_SMEM_EVENTS).
 *  2. Branchless binary search (fixed 12-iteration #pragma unroll).
 *  3. Fully coalesced writes within a tile.
 ******************************************************************************/
#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "restore_dense.h"

static constexpr int RD_FWD_BLOCK           = 256;
static constexpr int RD_FWD_MAX_SMEM_EVENTS = 2048;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
template<typename input_t, typename output_t = input_t>
__global__ __launch_bounds__(RD_FWD_BLOCK)
void restore_dense_fwd_kernel(RestoreDenseParams p) {
    const int ch      = blockIdx.x;
    const int tile_id = blockIdx.y;
    if (ch >= p.n_channels) return;

    const int tid       = threadIdx.x;
    const int tile_base = tile_id * RD_FWD_BLOCK;

    const int event_start = p.cu_seqlens[ch];
    const int event_end   = p.cu_seqlens[ch + 1];
    const int n_ev        = event_end - event_start;

    // ------------------------------------------------------------------
    // Shared-memory preload
    //
    // Layout (extern dynamic smem):
    //   [0 .. MAX*4)           int32_t s_ts[MAX]
    //   [MAX*4 .. MAX*(4+S))   input_t s_y [MAX]
    // ------------------------------------------------------------------
    extern __shared__ char smem_buf[];
    int32_t* __restrict__ s_ts =
        reinterpret_cast<int32_t*>(smem_buf);
    input_t* __restrict__ s_y  =
        reinterpret_cast<input_t*>(smem_buf + RD_FWD_MAX_SMEM_EVENTS * sizeof(int32_t));

    const input_t* __restrict__ y_src = reinterpret_cast<const input_t*>(p.y_flat);
    const bool use_smem = (n_ev > 0) && (n_ev <= RD_FWD_MAX_SMEM_EVENTS);

    if (use_smem) {
        for (int i = tid; i < n_ev; i += RD_FWD_BLOCK) {
            s_ts[i] = p.time_steps[event_start + i];
            s_y[i]  = y_src[event_start + i];
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Per-thread work
    // ------------------------------------------------------------------
    const int t = tile_base + tid;
    if (t >= p.seqlen || n_ev == 0) return;

    // Choose source arrays (0-based local event index).
    const int32_t* __restrict__ ts =
        use_smem ? s_ts : (p.time_steps + event_start);
    const input_t* __restrict__ ys =
        use_smem ? s_y  : (y_src + event_start);

    // ------------------------------------------------------------------
    // Branchless binary search — upper_bound(t), 12 fixed iterations.
    // lo == number of events with ts[i] <= t.
    // e_local = lo - 1 is the index of the most recent event at or before t.
    // ------------------------------------------------------------------
    int lo = 0;
#pragma unroll
    for (int s = RD_FWD_MAX_SMEM_EVENTS; s >= 1; s >>= 1) {
        const int probe = lo + s;
        lo = (probe <= n_ev && ts[probe - 1] <= t) ? probe : lo;
    }
    const int e_local = lo - 1;   // -1 → t is before first event

    if (e_local < 0) return;      // before first event → leave zero

    // Zero-order hold: just carry the event value forward (no decay).
    const float val = static_cast<float>(ys[e_local]);

    // Write out[b, t, d]  →  ch = b*D + d
    const int    dim_id = ch - (ch / p.dim) * p.dim;   // ch % dim
    const int    b_wr   = ch / p.dim;
    const size_t out_idx =
        (size_t)b_wr * ((size_t)p.seqlen * p.dim)
        + (size_t)t  *  p.dim
        +  dim_id;
    reinterpret_cast<output_t*>(p.out)[out_idx] = static_cast<output_t>(val);
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
template<typename input_t, typename output_t = input_t>
void restore_dense_fwd_cuda(RestoreDenseParams &p, cudaStream_t stream) {
    if (p.n_channels == 0) return;
    const int    n_tiles = (p.seqlen + RD_FWD_BLOCK - 1) / RD_FWD_BLOCK;
    const dim3   grid(p.n_channels, n_tiles);
    const dim3   block(RD_FWD_BLOCK);
    const size_t smem_bytes =
        static_cast<size_t>(RD_FWD_MAX_SMEM_EVENTS) *
        (sizeof(int32_t) + sizeof(input_t));
    restore_dense_fwd_kernel<input_t, output_t>
        <<<grid, block, smem_bytes, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
