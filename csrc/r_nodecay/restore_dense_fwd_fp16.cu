/******************************************************************************
 * restore_dense_fwd_fp16.cu  —  half (fp16) explicit instantiation
 ******************************************************************************/
#include "restore_dense_fwd_kernel.cuh"

// fp16 → fp16 identity (input and output both half)
template void restore_dense_fwd_cuda<at::Half, at::Half>(RestoreDenseParams &p, cudaStream_t stream);
