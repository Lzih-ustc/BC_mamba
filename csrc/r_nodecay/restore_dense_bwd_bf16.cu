/******************************************************************************
 * restore_dense_bwd_bf16.cu  —  bfloat16 explicit instantiation
 ******************************************************************************/
#include "restore_dense_bwd_kernel.cuh"

template void restore_dense_bwd_cuda<at::BFloat16>(RestoreDenseBwdParams &p, cudaStream_t stream);
