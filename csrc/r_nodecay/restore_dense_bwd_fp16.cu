/******************************************************************************
 * restore_dense_bwd_fp16.cu  —  half (fp16) explicit instantiation
 ******************************************************************************/
#include "restore_dense_bwd_kernel.cuh"

template void restore_dense_bwd_cuda<at::Half>(RestoreDenseBwdParams &p, cudaStream_t stream);
