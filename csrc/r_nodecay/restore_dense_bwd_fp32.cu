/******************************************************************************
 * restore_dense_bwd_fp32.cu  —  float32 explicit instantiation
 ******************************************************************************/
#include "restore_dense_bwd_kernel.cuh"

template void restore_dense_bwd_cuda<float>(RestoreDenseBwdParams &p, cudaStream_t stream);
