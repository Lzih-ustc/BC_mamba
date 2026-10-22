// csrc/a/restore_dense.h
// Parameter struct for restore_dense CUDA kernel
//
// v2: no decay — non-event positions carry the last event value forward
//     (zero-order hold).  log_gamma is removed entirely.
#pragma once
#include <cstdint>

struct RestoreDenseParams {
    int n_channels;   // = cu_seqlens.size(0)-1  (B*D)
    int dim;          // = D  (used only to compute out layout: b = ch/dim)
    int seqlen;       // L
    int total_events; // E

    const int32_t* cu_seqlens;  // (n_channels+1,) int32
    const int32_t* time_steps;  // (E,) int32

    const void* y_flat;  // (E,) input_t (half/bfloat16/float)
    void* out;           // (B, L, D) output_t (zero-initialised by caller)

    // output scalar type — may differ from y_flat's type.
    // Encoded as c10::ScalarType int value; 0 => same as y_flat.
    int out_scalar_type;
};
