// csrc/r/restore_dense_bwd.h
// Parameter struct for restore_dense backward CUDA kernel
//
// v2: no decay — backward simply sums dout over the region owned by each event.
//     log_gamma / y_flat / dlog_gamma are all removed.
#pragma once
#include <cstdint>

struct RestoreDenseBwdParams {
    int n_channels;   // = cu_seqlens.size(0)-1  (B*D)
    int dim;          // D
    int seqlen;       // L
    int total_events; // E

    const int32_t* cu_seqlens;  // (n_channels+1,) int32
    const int32_t* time_steps;  // (E,) int32

    const void* dout;  // (B, L, D) input_t  — upstream gradient

    // dy is ALWAYS float32 (accumulated via atomicAdd across tiles).
    float* dy;         // (E,) float32 — output gradient w.r.t. y_flat
};
