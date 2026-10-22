/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Split into multiple files to compile in paralell

#include "selective_scan_bwd_kernel.cuh"

template void selective_scan_bwd_cuda<float, float>(SSMParamsBwd &params, cudaStream_t stream);
template void selective_scan_bwd_flat_ch_cuda<float, float>(SSMParamsFlatSegBwd &params, cudaStream_t stream);
