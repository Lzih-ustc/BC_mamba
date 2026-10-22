/******************************************************************************
 * restore_dense_fwd_bf16.cu  —  bfloat16 explicit instantiation
 ******************************************************************************/
#include "restore_dense_fwd_kernel.cuh"

// bf16 → bf16 identity (input and output both bfloat16)
template void restore_dense_fwd_cuda<at::BFloat16, at::BFloat16>(RestoreDenseParams &p, cudaStream_t stream);
