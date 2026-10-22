/******************************************************************************
 * restore_dense_bwd_kernel.cuh
 *
 * Backward pass for the zero-order-hold restore_dense_fwd.
 *
 * Forward semantics (v2, no decay):
 *   out[b, t, d] = y_flat[e_prev]   for all t in [t0, t1)
 *   where e_prev is the most recent event at or before t.
 *
 * Backward (chain rule):
 *   dy[e] = Σ_{t=t0}^{t1-1}  dout[b, t, d]
 *
 *   t1 = next event's time_step, or seqlen if e is the last event in channel.
 *   Since decay = 1 everywhere, there is no dlog_gamma term.
 *
 * Optimisations retained from the decay version:
 *  1. Shared-memory preload of time_steps (coalesced).
 *  2. Branchless binary search (12-iteration unrolled loop).
 *  3. p.dy is always float32; native atomicAdd on all CUDA GPUs.
 *
 * The block-level dlog_gamma reduction is removed (no longer needed).
 *
 * Grid  : (n_channels, ceil(seqlen / BLOCK))
 * Block : (BLOCK,) = (256,)
 ******************************************************************************/
#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "restore_dense_bwd.h"

static constexpr int RD_BWD_BLOCK           = 256;
static constexpr int RD_BWD_MAX_SMEM_EVENTS = 2048;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
template<typename input_t>
__global__ __launch_bounds__(RD_BWD_BLOCK)
void restore_dense_bwd_kernel(RestoreDenseBwdParams p) {
    const int ch      = blockIdx.x;
    const int tile_id = blockIdx.y;
    if (ch >= p.n_channels) return;

    const int tid       = threadIdx.x;
    const int tile_base = tile_id * RD_BWD_BLOCK;

    const int event_start = p.cu_seqlens[ch];
    const int event_end   = p.cu_seqlens[ch + 1];
    const int n_ev        = event_end - event_start;

    // ------------------------------------------------------------------
    // Shared-memory preload — time_steps only (no y_flat needed).
    //
    // Layout (extern dynamic smem):
    //   [0 .. MAX*4)   int32_t s_ts[MAX]
    // ------------------------------------------------------------------
    extern __shared__ char smem_buf[];
    int32_t* __restrict__ s_ts = reinterpret_cast<int32_t*>(smem_buf);

    const bool use_smem = (n_ev > 0) && (n_ev <= RD_BWD_MAX_SMEM_EVENTS);
    if (use_smem) {
        for (int i = tid; i < n_ev; i += RD_BWD_BLOCK)
            s_ts[i] = p.time_steps[event_start + i];
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Per-thread gradient computation.
    // ------------------------------------------------------------------
    const int t = tile_base + tid;
    if (t >= p.seqlen || n_ev == 0) return;

    const int32_t* __restrict__ ts =
        use_smem ? s_ts : (p.time_steps + event_start);

    // ch = b*D + d
    const int dim_id = ch - (ch / p.dim) * p.dim;
    const int b_ch   = ch / p.dim;

    const input_t* __restrict__ dout_base =
        reinterpret_cast<const input_t*>(p.dout);

    // dout[b_ch, t, dim_id]
    const float g = static_cast<float>(
        dout_base[(size_t)b_ch * ((size_t)p.seqlen * p.dim)
                  + (size_t)t * p.dim
                  + dim_id]
    );

    // ------------------------------------------------------------------
    // Branchless binary search — upper_bound(t), same 12-iteration form.
    // ------------------------------------------------------------------
    int lo = 0;
#pragma unroll
    for (int s = RD_BWD_MAX_SMEM_EVENTS; s >= 1; s >>= 1) {
        const int probe = lo + s;
        lo = (probe <= n_ev && ts[probe - 1] <= t) ? probe : lo;
    }
    const int e_local = lo - 1;

    if (e_local < 0) return;   // t is before the first event; grad is 0

    const int e_global = event_start + e_local;

    // dy[e] += dout[b, t, d]   (decay == 1, so no multiplication)
    atomicAdd(p.dy + e_global, g);
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
template<typename input_t>
void restore_dense_bwd_cuda(RestoreDenseBwdParams &p, cudaStream_t stream) {
    if (p.n_channels == 0 || p.total_events == 0) return;
    const int    n_tiles = (p.seqlen + RD_BWD_BLOCK - 1) / RD_BWD_BLOCK;
    const dim3   grid(p.n_channels, n_tiles);
    const dim3   block(RD_BWD_BLOCK);
    // Smem for time_steps preload only (no y_flat array needed any more).
    const size_t smem_bytes =
        static_cast<size_t>(RD_BWD_MAX_SMEM_EVENTS) * sizeof(int32_t);
    restore_dense_bwd_kernel<input_t>
        <<<grid, block, smem_bytes, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
