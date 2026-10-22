/******************************************************************************
 * restore_dense_fwd_fp32.cu  —  float32 explicit instantiation
 ******************************************************************************/
#include "restore_dense_fwd_kernel.cuh"

// All output-dtype combinations for float32 input:
//   float32 → float32   (identity / legacy path)
//   float32 → fp16      (used when hidden_states is fp16)
//   float32 → bf16      (used when hidden_states is bf16)
template void restore_dense_fwd_cuda<float, float>       (RestoreDenseParams &p, cudaStream_t stream);
template void restore_dense_fwd_cuda<float, at::Half>    (RestoreDenseParams &p, cudaStream_t stream);
template void restore_dense_fwd_cuda<float, at::BFloat16>(RestoreDenseParams &p, cudaStream_t stream);
