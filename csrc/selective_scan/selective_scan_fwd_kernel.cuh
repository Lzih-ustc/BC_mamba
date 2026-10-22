/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#ifndef USE_ROCM
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_scan.cuh>
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#endif

#include "selective_scan.h"
#include "selective_scan_common.h"
#include "static_switch.h"

template<int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_,
         bool kIsVariableB_, bool kIsVariableC_,
         bool kHasZ_, typename input_t_, typename weight_t_>
struct Selective_Scan_fwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNRows = kNRows_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsComplex = std::is_same_v<weight_t, complex_t>;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kIsVariableB = kIsVariableB_;
    static constexpr bool kIsVariableC = kIsVariableC_;
    static constexpr bool kHasZ = kHasZ_;

    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = std::conditional_t<!kIsComplex, float2, float4>;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, !kIsComplex ? kNItems : kNItems * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, !kIsComplex ? kNLoads : kNLoads * 2,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE  : cub::BLOCK_LOAD_DIRECT>;
    // Dedicated float32 loader for variable B/C in the flat-ch kernel.
    // The flat-ch path fixes B/C as float32 ([E, dstate]) regardless of input_t, so
    // we cannot reuse BlockLoadWeightT (which is typed on input_t and breaks when
    // input_t = BFloat16 / Half while the actual pointer is float*).
    using BlockLoadWeightFloatT = cub::BlockLoad<float, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    // BlockLoadKT: dedicated int32 loader for k (inter-event intervals) in flat-ch kernel.
    using BlockLoadKT = cub::BlockLoad<int32_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    static constexpr int kSmemIOSize = custom_max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 sizeof(typename BlockLoadKT::TempStorage),
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 // Float loader for flat-ch path: B/C are always float32 regardless of input_t.
                                                 // Must appear in kSmemIOSize so that smem regions for both B and C fit
                                                 // (the flat-ch kernel overlays smem_load_weight / smem_load_weight1 here).
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightFloatT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_fwd_kernel(SSMParamsBase params) {
    constexpr bool kIsComplex = Ktraits::kIsComplex;
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows;
    constexpr bool kDirectIO = Ktraits::kDirectIO;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    // cast to lvalue reference of expected type
    // char *smem_loadstorescan = smem_ + 2 * MAX_DSTATE * sizeof(weight_t);
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_ + 2 * MAX_DSTATE * sizeof(weight_t));
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_loadstorescan);
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    // weight_t *smem_a = reinterpret_cast<weight_t *>(smem_ + smem_loadstorescan_size);
    // weight_t *smem_bc = reinterpret_cast<weight_t *>(smem_a + MAX_DSTATE);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * kNRows * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * kNRows * params.delta_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * kNRows * params.A_d_stride;
    weight_t *B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * kNRows * params.B_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    weight_t *C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * kNRows * params.C_d_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.n_chunks * params.dstate;

    float D_val[kNRows] = {0};
    if (params.D_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            D_val[r] = reinterpret_cast<float *>(params.D_ptr)[dim_id * kNRows + r];
        }
    }
    float delta_bias[kNRows] = {0};
    if (params.delta_bias_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            delta_bias[r] = reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id * kNRows + r];
        }
    }

    // for (int state_idx = threadIdx.x; state_idx < params.dstate; state_idx += blockDim.x) {
    //     smem_a[state_idx] = A[state_idx * params.A_dstate_stride];
    //     smem_bc[state_idx] = B[state_idx * params.B_dstate_stride] * C[state_idx * params.C_dstate_stride];
    // }

    constexpr int kChunkSize = kNThreads * kNItems;
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        input_t u_vals[kNRows][kNItems], delta_vals_load[kNRows][kNItems];
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            load_input<Ktraits>(u + r * params.u_d_stride, u_vals[r], smem_load, params.seqlen - chunk * kChunkSize);
            if constexpr (!kDirectIO) { __syncthreads(); }
            load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
        }
        u += kChunkSize;
        delta += kChunkSize;
    
        float delta_vals[kNRows][kNItems], delta_u_vals[kNRows][kNItems], out_vals[kNRows][kNItems];
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float u_val = float(u_vals[r][i]);
                delta_vals[r][i] = float(delta_vals_load[r][i]) + delta_bias[r];
                if (params.delta_softplus) {
                    delta_vals[r][i] = delta_vals[r][i] <= 20.f ? log1pf(expf(delta_vals[r][i])) : delta_vals[r][i];
                }
                delta_u_vals[r][i] = delta_vals[r][i] * u_val;
                out_vals[r][i] = D_val[r] * u_val;
            }
        }

        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            weight_t A_val[kNRows];
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                A_val[r] = A[state_idx * params.A_dstate_stride + r * params.A_d_stride];
                // Multiply the real part of A with LOG2E so we can use exp2f instead of expf.
                constexpr float kLog2e = M_LOG2E;
                if constexpr (!kIsComplex) {
                    A_val[r] *= kLog2e;
                } else {
                    A_val[r].real_ *= kLog2e;
                }
            }
            // This variable holds B * C if both B and C are constant across seqlen. If only B varies
            // across seqlen, this holds C. If only C varies across seqlen, this holds B.
            // If both B and C vary, this is unused.
            weight_t BC_val[kNRows];
            weight_t B_vals[kNItems], C_vals[kNItems];
            if constexpr (kIsVariableB) {
                load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
                if constexpr (!kIsVariableC) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                    }
                }
            }
            if constexpr (kIsVariableC) {
                auto &smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight_C, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
                if constexpr (!kIsVariableB) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
                    }
                }
            }
            if constexpr (!kIsVariableB && !kIsVariableC) {
                #pragma unroll
                for (int r = 0; r < kNRows; ++r) {
                    BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride] * C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                }
            }

            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                if (r > 0) { __syncthreads(); }  // Scan could be using the same smem
                scan_t thread_data[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    if constexpr (!kIsComplex) {
                        // 🔥 Clamp exponent to prevent underflow/overflow
                        float exponent = safe_clamp(delta_vals[r][i] * A_val[r], -SAFE_EXP2F_RANGE, SAFE_EXP2F_RANGE);
                        thread_data[i] = make_float2(exp2f(exponent),
                                                     !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i]);
                        if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                            if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                                thread_data[i] = make_float2(1.f, 0.f);
                            }
                        }
                    } else {
                        // Pytorch's implementation of complex exp (which calls thrust) is very slow
                        // 🔥 Clamp complex exponent for stability
                        complex_t delta_a = delta_vals[r][i] * A_val[r];
                        delta_a.real_ = safe_clamp(delta_a.real_, -SAFE_EXP2F_RANGE, SAFE_EXP2F_RANGE);
                        complex_t delta_a_exp = cexp2f(delta_a);
                        weight_t B_delta_u_val = !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i];
                        thread_data[i] = make_float4(delta_a_exp.real_, delta_a_exp.imag_, B_delta_u_val.real_, B_delta_u_val.imag_);
                        if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                            if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                                thread_data[i] = make_float4(1.f, 0.f, 0.f, 0.f);
                            }
                        }
                    }
                }
                // Initialize running total
                scan_t running_prefix;
                if constexpr (!kIsComplex) {
                    // If we use WARP_SCAN then all lane 0 of all warps (not just thread 0) needs to read
                    running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx + r * MAX_DSTATE] : make_float2(1.f, 0.f);
                    // running_prefix = chunk > 0 && threadIdx.x == 0 ? smem_running_prefix[state_idx] : make_float2(1.f, 0.f);
                } else {
                    running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx + r * MAX_DSTATE] : make_float4(1.f, 0.f, 0.f, 0.f);
                    // running_prefix = chunk > 0 && threadIdx.x == 0 ? smem_running_prefix[state_idx] : make_float4(1.f, 0.f, 0.f, 0.f);
                }
                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                );
                // There's a syncthreads in the scan op, so we don't need to sync here.
                // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
                if (threadIdx.x == 0) {
                    smem_running_prefix[state_idx] = prefix_op.running_prefix;
                    x[(r * params.n_chunks + chunk) * params.dstate + state_idx] = prefix_op.running_prefix;
                }
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const weight_t C_val = !kIsVariableC
                        ? BC_val[r]
                        : (!kIsVariableB ? BC_val[r] * C_vals[i] : C_vals[i]);
                    if constexpr (!kIsComplex) {
                        out_vals[r][i] += thread_data[i].y * C_val;
                    } else {
                        out_vals[r][i] += (complex_t(thread_data[i].z, thread_data[i].w) * C_val).real_ * 2;
                    }
                }
            }
        }
        
        input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
            + dim_id * kNRows * params.out_d_stride + chunk * kChunkSize;
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            store_output<Ktraits>(out + r * params.out_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
        }

        if constexpr (kHasZ) {
            input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride
                + dim_id * kNRows * params.z_d_stride + chunk * kChunkSize;
            input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride
                + dim_id * kNRows * params.out_z_d_stride + chunk * kChunkSize;
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                input_t z_vals[kNItems];
                __syncthreads();
                load_input<Ktraits>(z + r * params.z_d_stride, z_vals, smem_load, params.seqlen - chunk * kChunkSize);
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    float z_val = z_vals[i];
                    out_vals[r][i] *= z_val / (1 + expf(-z_val));
                }
                __syncthreads();
                store_output<Ktraits>(out_z + r * params.out_z_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
            }
        }

        Bvar += kChunkSize * (!kIsComplex ? 1 : 2);
        Cvar += kChunkSize * (!kIsComplex ? 1 : 2);
    }
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_fwd_launch(SSMParamsBase &params, cudaStream_t stream) {
    // Only kNRows == 1 is tested for now, which ofc doesn't differ from previously when we had each block
    // processing 1 row.
    constexpr int kNRows = 1;
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.is_variable_B, kIsVariableB, [&] {
            BOOL_SWITCH(params.is_variable_C, kIsVariableC, [&] {
                BOOL_SWITCH(params.z_ptr != nullptr , kHasZ, [&] {
                    using Ktraits = Selective_Scan_fwd_kernel_traits<kNThreads, kNItems, kNRows, kIsEvenLen, kIsVariableB, kIsVariableC, kHasZ, input_t, weight_t>;
                    
                    constexpr int kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
                    dim3 grid(params.batch, params.dim / kNRows);

                    // Had to change this substantially since potentially the hip 
                    // interface for setting kernel launch attributes is slightly different from 
                    // cuda's. In particualar, it seems to expect a plain const void * pointer.

                    auto kernel = &selective_scan_fwd_kernel<Ktraits>;

                    
                    if (kSmemSize >= 48 * 1024) {
                        #ifndef USE_ROCM
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                        #else
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            (void *) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                            std::cerr << "Warning (selective_scan_fwd_kernel): attempting to set maxDynamicSharedMemorySize on an AMD GPU which is currently a non-op (in ROCm versions <= 6.1). This might lead to undefined behavior. \n" << std::endl;
                        #endif
                    }

                    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}

template<typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream) {

    #ifndef USE_ROCM
        if (params.seqlen <= 128) {           
            selective_scan_fwd_launch<32, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 256) {
            selective_scan_fwd_launch<32, 8, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_fwd_launch<32, 16, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
        } else {
            selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
        }
    #else
        if (params.seqlen <= 256) {
            selective_scan_fwd_launch<64, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_fwd_launch<64, 8, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
        } else {
            selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
        }
    #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// ── Flat per-channel forward kernel ──────────────────────────────────────────
// One CUDA block per logical channel.  Each channel has its own jagged event
// slice [cu_seqlens[ch], cu_seqlens[ch+1]).  No global scan / block-boundary
// prefix propagation is needed because each channel is fully contained in a
// single block.
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_flat_ch_fwd_kernel(SSMParamsFlatSeg params) {
    // ── Type / constant declarations (same as original kernel) ───────────────
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kHasZ        = Ktraits::kHasZ;
    constexpr int  kNThreads    = Ktraits::kNThreads;
    constexpr int  kNItems      = Ktraits::kNItems;
    constexpr int  kNRows       = Ktraits::kNRows;   // always 1
    constexpr bool kDirectIO    = Ktraits::kDirectIO;
    using input_t  = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t   = typename Ktraits::scan_t;

    // ── Shared memory (same layout as original kernel) ───────────────────────
    extern __shared__ char smem_[];
    auto& smem_load         = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    // flat-ch kernel: variable B/C are always float32, so use the float-specific loader's
    // TempStorage here.  Using BlockLoadWeightT (which is typed on input_t) would compile-error
    // when input_t = BFloat16/Half because the pointer type (float*) would not match.
    auto& smem_load_weight  = reinterpret_cast<typename Ktraits::BlockLoadWeightFloatT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightFloatT::TempStorage*>(
                                smem_ + sizeof(typename Ktraits::BlockLoadWeightFloatT::TempStorage));
    auto& smem_store        = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_scan         = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(
                                smem_ + Ktraits::kSmemIOSize);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);
    // Effective depthwise+pointwise weights: W_B_eff[s] = Σ_d' pw_B[dim_id,d'] * W_B[d',s].
    // Computed cooperatively before the chunk loop when params.pw_B_ptr != nullptr.
    float  *smem_W_B_eff = reinterpret_cast<float *>(smem_running_prefix + kNRows * MAX_DSTATE);
    float  *smem_W_C_eff = smem_W_B_eff + MAX_DSTATE;

    // Dedicated int32 loader for k values (overlaps same smem region)
    auto& smem_load_k = reinterpret_cast<typename Ktraits::BlockLoadKT::TempStorage&>(smem_);

    // ── Block → data mapping ─────────────────────────────────────────────────
    const int ch      = blockIdx.x;                         // channel id
    const int dim_id  = ch % params.dim;

    const int ev_start = params.cu_seqlens_ptr[ch];
    const int ch_len   = params.cu_seqlens_ptr[ch + 1] - ev_start;
    if (ch_len == 0) return;                                // empty channel

    // ── Pointer initialisation ───────────────────────────────────────────────
    input_t  *u     = reinterpret_cast<input_t *>(params.u_ptr)   + ev_start;
    int32_t  *k     = params.k_ptr + ev_start;

    weight_t *A     = reinterpret_cast<weight_t *>(params.A_ptr)
                      + dim_id * params.A_d_stride;

    // Variable B/C: [dstate, E] transposed layout — offset by ev_start in E dim
    // Legacy pointers (used only when tile_offsets_ptr == nullptr)
    float    *Bvar_legacy  = reinterpret_cast<float *>(params.B_ptr) + ev_start;
    float    *Cvar_legacy  = reinterpret_cast<float *>(params.C_ptr) + ev_start;

    // Tiled B/C: tile_base_idx is the global tile start for this channel
    const int32_t tile_base_idx =
        (params.tile_offsets_ptr != nullptr) ? params.tile_offsets_ptr[ch] : 0;

    // x layout: [n_channels, n_chunks, dstate]
    // The n_chunks stride must be preserved; otherwise, states from multiple
    // chunks in the same channel overwrite one another. Backward reconstruction
    // would then read incorrect chunk-boundary states and produce invalid gradients.
    scan_t   *x     = reinterpret_cast<scan_t *>(params.x_ptr)
                      + ch * params.n_chunks * params.dstate;

    // ── D loading (same as original) ─────────────────────────────────────────
    float D_val[kNRows] = {0};
    if (params.D_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r)
            D_val[r] = reinterpret_cast<float *>(params.D_ptr)[dim_id * kNRows + r];
    }

    // ── dt_idle / dt_active (replaces delta_bias + softplus) ─────────────────
    const float dt_idle   = params.dt_idle_ptr[dim_id];
    const float dt_active = params.dt_active_ptr[dim_id];

    // ── Chunk loop ───────────────────────────────────────────────────────────
    constexpr int kChunkSize = kNThreads * kNItems;
    // [BugFix #1] Compute per-channel chunk count instead of using global n_chunks
    const int ch_n_chunks = (ch_len + kChunkSize - 1) / kChunkSize;

    // ── Depthwise + pointwise precompute ─────────────────────────────────────
    // When both is_depthwise_bc and pw_B_ptr are set, each channel block (dim_id=d)
    // computes its row of the effective weight matrix once before scanning:
    //   W_B_eff[s] = Σ_{d'} pw_B[d, d'] * W_B[d', s]
    // Threads split work across dstate: thread tid handles state s = tid, tid+kNThreads, ...
    const bool has_pw = params.is_depthwise_bc && (params.pw_B_ptr != nullptr);
    if (has_pw) {
        const float *pw_B_row  = reinterpret_cast<const float *>(params.pw_B_ptr)
                                  + (long)dim_id * params.dim;
        const float *pw_C_row  = reinterpret_cast<const float *>(params.pw_C_ptr)
                                  + (long)dim_id * params.dim;
        const float *W_B_base  = reinterpret_cast<const float *>(params.W_B_ptr);
        const float *W_C_base  = reinterpret_cast<const float *>(params.W_C_ptr);
        for (int s = threadIdx.x; s < params.dstate; s += kNThreads) {
            float acc_B = 0.f, acc_C = 0.f;
            for (int d2 = 0; d2 < params.dim; ++d2) {
                acc_B += pw_B_row[d2] * W_B_base[(long)d2 * params.dstate + s];
                acc_C += pw_C_row[d2] * W_C_base[(long)d2 * params.dstate + s];
            }
            smem_W_B_eff[s] = acc_B;
            smem_W_C_eff[s] = acc_C;
        }
        __syncthreads();
    }

    for (int chunk = 0; chunk < ch_n_chunks; ++chunk) {
        const int remaining = ch_len - chunk * kChunkSize;

        input_t u_vals[kNRows][kNItems];
        int32_t k_vals_load[kNRows][kNItems];
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            load_input<Ktraits>(u + r * ch_len, u_vals[r], smem_load, remaining);
            if constexpr (!kDirectIO) { __syncthreads(); }

            // Load k values with dedicated int32 loader
            Ktraits::BlockLoadKT(smem_load_k).Load(
                k + r * ch_len,
                k_vals_load[r],
                remaining,
                1                        // padding: k=1 → dt_eff = dt_active (safe default)
            );
        }
        u += kChunkSize;
        k += kChunkSize;

        // ── delta computation: dt formula instead of bias + softplus ─────────
        float delta_vals[kNRows][kNItems], delta_u_vals[kNRows][kNItems], out_vals[kNRows][kNItems];
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float u_val = float(u_vals[r][i]);
                // dt_eff = dt_idle * (k - 1) + dt_active
                delta_vals[r][i] = dt_idle * float(k_vals_load[r][i] - 1) + dt_active;
                delta_u_vals[r][i] = delta_vals[r][i] * u_val;
                out_vals[r][i] = D_val[r] * u_val;
            }
        }

        // ── State loop (identical structure to original, stride names changed) ─
        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            weight_t A_val[kNRows];
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                A_val[r] = A[state_idx * params.A_dstate_stride + r * params.A_d_stride];
                // Multiply by LOG2E so we can use exp2f instead of expf
                A_val[r] *= M_LOG2E;
            }

            // B/C in flat-ch are always float32 (layout [E, dstate]).  Declare the
            // accumulation arrays as float so they match the float* source pointers
            // and the BlockLoadWeightFloatT loader.  (weight_t == float in this path,
            // but the explicit type prevents any accidental input_t contamination.)
            weight_t BC_val[kNRows];
            float B_vals[kNItems], C_vals[kNItems];
            if constexpr (kIsVariableB) {
                // ── Depthwise path: on-the-fly from W_B / W_C ─────────────────
                if (params.is_depthwise_bc) {
                    // When pw_B_ptr is set, smem_W_B_eff already holds the mixed weight;
                    // otherwise read W_B directly — both are [dim, dstate] float32.
                    const float W_B_s = has_pw
                        ? smem_W_B_eff[state_idx]
                        : reinterpret_cast<const float *>(params.W_B_ptr)
                              [dim_id * params.dstate + state_idx];
                    #pragma unroll
                    for (int i = 0; i < kNItems; ++i)
                        B_vals[i] = float(u_vals[0][i]) * W_B_s;
                } else if (params.tile_offsets_ptr != nullptr) {
                    // Tiled layout: B_tiled[tile_id, state_idx, i]
                    const long tile_id  = (long)(tile_base_idx + chunk);
                    float *B_tile_row   = reinterpret_cast<float *>(params.B_ptr)
                                          + tile_id * (long)params.dstate * kChunkSize
                                          + (long)state_idx * kChunkSize;
                    Ktraits::BlockLoadWeightFloatT(smem_load_weight).Load(
                        B_tile_row, B_vals, remaining, 0.f);
                } else {
                    Ktraits::BlockLoadWeightFloatT(smem_load_weight).Load(
                        Bvar_legacy + state_idx * params.B_row_stride,
                        B_vals, remaining, 0.f);
                }
                if constexpr (!kIsVariableC) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = reinterpret_cast<float *>(params.C_ptr)[
                            state_idx * params.C_row_stride + dim_id];
                    }
                }
            }
            if constexpr (kIsVariableC) {
                auto &smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                // ── Depthwise path ─────────────────────────────────────────────
                if (params.is_depthwise_bc) {
                    const float W_C_s = has_pw
                        ? smem_W_C_eff[state_idx]
                        : reinterpret_cast<const float *>(params.W_C_ptr)
                              [dim_id * params.dstate + state_idx];
                    #pragma unroll
                    for (int i = 0; i < kNItems; ++i)
                        C_vals[i] = float(u_vals[0][i]) * W_C_s;
                } else if (params.tile_offsets_ptr != nullptr) {
                    const long tile_id  = (long)(tile_base_idx + chunk);
                    float *C_tile_row   = reinterpret_cast<float *>(params.C_ptr)
                                          + tile_id * (long)params.dstate * kChunkSize
                                          + (long)state_idx * kChunkSize;
                    Ktraits::BlockLoadWeightFloatT(smem_load_weight_C).Load(
                        C_tile_row, C_vals, remaining, 0.f);
                } else {
                    Ktraits::BlockLoadWeightFloatT(smem_load_weight_C).Load(
                        Cvar_legacy + state_idx * params.C_row_stride,
                        C_vals, remaining, 0.f);
                }
                if constexpr (!kIsVariableB) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = reinterpret_cast<float *>(params.B_ptr)[
                            state_idx * params.B_row_stride + dim_id];
                    }
                }
            }
            if constexpr (!kIsVariableB && !kIsVariableC) {
                // Both constant: B[state_idx, dim_id] * C[state_idx, dim_id].
                #pragma unroll
                for (int r = 0; r < kNRows; ++r) {
                    BC_val[r] = reinterpret_cast<float *>(params.B_ptr)[
                                    state_idx * params.B_row_stride + dim_id]
                              * reinterpret_cast<float *>(params.C_ptr)[
                                    state_idx * params.C_row_stride + dim_id];
                }
            }

            // ── Scan data construction & BlockScan (same as original) ────────
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                if (r > 0) { __syncthreads(); }
                scan_t thread_data[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    float exponent = safe_clamp(delta_vals[r][i] * A_val[r], -SAFE_EXP2F_RANGE, SAFE_EXP2F_RANGE);
                    thread_data[i] = make_float2(exp2f(exponent),
                                                 !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i]);
                    if constexpr (!Ktraits::kIsEvenLen) {
                        if (threadIdx.x * kNItems + i >= remaining) {
                            thread_data[i] = make_float2(1.f, 0.f);
                        }
                    }
                }
                // Running prefix from previous chunk
                scan_t running_prefix;
                running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx + r * MAX_DSTATE] : make_float2(1.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                );
                if (threadIdx.x == 0) {
                    smem_running_prefix[state_idx] = prefix_op.running_prefix;
                    // ── Fix C: write each chunk-boundary state to global memory immediately ──
                    // Layout: x[chunk * dstate + state_idx]
                    // When backward processes chunk[i], it reads the state saved
                    // by chunk[i-1] to reconstruct every forward state in chunk[i].
                    // Writing only once after the loop (the original BugFix #3)
                    // lets later chunks overwrite earlier states, so backward sees
                    // the final state instead of each chunk's boundary state.
                    if (params.x_ptr != nullptr) {
                        x[chunk * params.dstate + state_idx] = prefix_op.running_prefix;
                    }
                }
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const weight_t C_val = !kIsVariableC
                        ? BC_val[r]
                        : (!kIsVariableB ? BC_val[r] * C_vals[i] : C_vals[i]);
                    out_vals[r][i] += thread_data[i].y * C_val;
                }
            }
        }

        // ── Output store ─────────────────────────────────────────────────────
        input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + ev_start + chunk * kChunkSize;
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            store_output<Ktraits>(out + r * ch_len, out_vals[r], smem_store, remaining);
        }

        if constexpr (kHasZ) {
            input_t *z     = reinterpret_cast<input_t *>(params.z_ptr)     + ev_start + chunk * kChunkSize;
            input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + ev_start + chunk * kChunkSize;
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                input_t z_vals[kNItems];
                __syncthreads();
                load_input<Ktraits>(z + r * ch_len, z_vals, smem_load, remaining);
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    float z_val = z_vals[i];
                    out_vals[r][i] *= z_val / (1 + expf(-z_val));
                }
                __syncthreads();
                store_output<Ktraits>(out_z + r * ch_len, out_vals[r], smem_store, remaining);
            }
        }

        // Tiled path uses tile_base_idx + chunk for addressing; no pointer advance needed.
        // Legacy path still needs pointer advancement.
        if (params.tile_offsets_ptr == nullptr) {
            Bvar_legacy += kChunkSize;
            Cvar_legacy += kChunkSize;
        }
    }
    // The original BugFix #3 (writing x once after the loop) has been removed.
    // It preserved only the final state and overwrote earlier states in multi-chunk scans.
    // States are now written per chunk immediately after BlockScan (see Fix C above).
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// ── Flat per-channel forward launch helpers ──────────────────────────────────
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_flat_ch_fwd_launch(SSMParamsFlatSeg &params, cudaStream_t stream) {
    // kNRows is always 1 (one block = one channel)
    constexpr int kNRows = 1;
    // kIsEvenLen is always false (each channel has a different length)
    constexpr bool kIsEvenLen = false;

    BOOL_SWITCH(params.is_variable_B, kIsVariableB, [&] {
        BOOL_SWITCH(params.is_variable_C, kIsVariableC, [&] {
            BOOL_SWITCH(params.z_ptr != nullptr, kHasZ, [&] {
                using Ktraits = Selective_Scan_fwd_kernel_traits<
                    kNThreads, kNItems, kNRows, kIsEvenLen,
                    kIsVariableB, kIsVariableC, kHasZ,
                    input_t, weight_t>;

                constexpr int kSmemSize = Ktraits::kSmemSize
                                        + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t)
                                        + 2 * MAX_DSTATE * sizeof(float);
                                        // +2*MAX_DSTATE*float for smem_W_B_eff / smem_W_C_eff
                                        // (used only when params.pw_B_ptr != nullptr; always allocated
                                        //  to keep the smem layout uniform and avoid a second kernel).

                // grid: one block per channel
                dim3 grid(params.n_channels);

                auto kernel = &selective_scan_flat_ch_fwd_kernel<Ktraits>;
                if (kSmemSize >= 48 * 1024) {
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                }
                kernel<<<grid, kNThreads, kSmemSize, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

template<typename input_t, typename weight_t>
void selective_scan_fwd_flat_ch_cuda(SSMParamsFlatSeg &params, cudaStream_t stream) {
    // Branch on max_seqlen (same thresholds as original)
    if (params.max_seqlen <= 128) {
        selective_scan_flat_ch_fwd_launch<32,  4, input_t, weight_t>(params, stream);
    } else if (params.max_seqlen <= 256) {
        selective_scan_flat_ch_fwd_launch<32,  8, input_t, weight_t>(params, stream);
    } else if (params.max_seqlen <= 512) {
        selective_scan_flat_ch_fwd_launch<32, 16, input_t, weight_t>(params, stream);
    } else if (params.max_seqlen <= 1024) {
        selective_scan_flat_ch_fwd_launch<64, 16, input_t, weight_t>(params, stream);
    } else {
        selective_scan_flat_ch_fwd_launch<128,16, input_t, weight_t>(params, stream);
    }
}
