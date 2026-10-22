/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <ATen/cuda/Atomic.cuh>  // For atomicAdd on complex

#ifndef USE_ROCM
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_scan.cuh>
    #include <cub/block/block_reduce.cuh>
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#endif

#include "selective_scan.h"
#include "selective_scan_common.h"
#include "reverse_scan.cuh"
#include "static_switch.h"

template<typename scalar_t> __device__ __forceinline__ scalar_t conj(scalar_t x);
template<> __device__ __forceinline__ float conj<float>(float x) { return x; }
template<> __device__ __forceinline__ complex_t conj<complex_t>(complex_t x) { return std::conj(x); }

template<int kNThreads_, int kNItems_, bool kIsEvenLen_, bool kIsVariableB_, bool kIsVariableC_,
         bool kDeltaSoftplus_, bool kHasZ_, typename input_t_, typename weight_t_>
struct Selective_Scan_bwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsComplex = std::is_same_v<weight_t, complex_t>;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kIsVariableB = kIsVariableB_;
    static constexpr bool kIsVariableC = kIsVariableC_;
    static constexpr bool kDeltaSoftplus = kDeltaSoftplus_;
    static constexpr bool kHasZ = kHasZ_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads with float improves occupancy.
    // For complex this would lead to massive register spilling, so we keep it at 2.
    static constexpr int kMinBlocks = kNThreads == 128 && !kIsComplex ? 3 : 2;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = std::conditional_t<!kIsComplex, float2, float4>;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, !kIsComplex ? kNItems : kNItems * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, !kIsComplex ? kNLoads : kNLoads * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockReverseScanT = BlockReverseScan<scan_t, kNThreads>;
    using BlockReduceT = cub::BlockReduce<scan_t, kNThreads>;
    using BlockReduceFloatT = cub::BlockReduce<float, kNThreads>;
    using BlockReduceComplexT = cub::BlockReduce<complex_t, kNThreads>;
    using BlockExchangeT = cub::BlockExchange<float, kNThreads, !kIsComplex ? kNItems : kNItems * 2>;

    static constexpr int kSmemIOSize = custom_max({sizeof(typename BlockLoadT::TempStorage),
                                                    sizeof(typename BlockLoadVecT::TempStorage),
                                                    (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightT::TempStorage),
                                                    (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                    sizeof(typename BlockStoreT::TempStorage),
                                                    sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemExchangeSize = (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockExchangeT::TempStorage);
    static constexpr int kSmemReduceSize = sizeof(typename BlockReduceT::TempStorage);
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize + kSmemReduceSize + sizeof(typename BlockScanT::TempStorage) + sizeof(typename BlockReverseScanT::TempStorage);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_bwd_kernel(SSMParamsBwd params) {
    constexpr bool kIsComplex = Ktraits::kIsComplex;
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kDeltaSoftplus = Ktraits::kDeltaSoftplus;
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
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
    auto& smem_exchange = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    auto& smem_exchange1 = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize + sizeof(typename Ktraits::BlockExchangeT::TempStorage));
    auto& smem_reduce = *reinterpret_cast<typename Ktraits::BlockReduceT::TempStorage*>(reinterpret_cast<char *>(&smem_exchange) + Ktraits::kSmemExchangeSize);
    auto& smem_reduce_float = *reinterpret_cast<typename Ktraits::BlockReduceFloatT::TempStorage*>(&smem_reduce);
    auto& smem_reduce_complex = *reinterpret_cast<typename Ktraits::BlockReduceComplexT::TempStorage*>(&smem_reduce);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(reinterpret_cast<char *>(&smem_reduce) + Ktraits::kSmemReduceSize);
    auto& smem_reverse_scan = *reinterpret_cast<typename Ktraits::BlockReverseScanT::TempStorage*>(reinterpret_cast<char *>(&smem_scan) + sizeof(typename Ktraits::BlockScanT::TempStorage));
    weight_t *smem_delta_a = reinterpret_cast<weight_t *>(smem_ + Ktraits::kSmemSize);
    scan_t *smem_running_postfix = reinterpret_cast<scan_t *>(smem_delta_a + 2 * MAX_DSTATE + kNThreads);
    weight_t *smem_da = reinterpret_cast<weight_t *>(smem_running_postfix + MAX_DSTATE);
    weight_t *smem_dbc = reinterpret_cast<weight_t *>(smem_da + MAX_DSTATE);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * params.delta_d_stride;
    input_t *dout = reinterpret_cast<input_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride
        + dim_id * params.dout_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * params.A_d_stride;
    weight_t *B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * params.B_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    weight_t *C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * params.C_d_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    weight_t *dA = reinterpret_cast<weight_t *>(params.dA_ptr) + dim_id * params.dA_d_stride;
    weight_t *dB = reinterpret_cast<weight_t *>(params.dB_ptr)
        + (!kIsVariableB ? dim_id * params.dB_d_stride : batch_id * (!kIsComplex ? params.dB_batch_stride : params.dB_batch_stride / 2) + group_id * params.dB_group_stride);
    weight_t *dC = reinterpret_cast<weight_t *>(params.dC_ptr)
        + (!kIsVariableC ? dim_id * params.dC_d_stride : batch_id * (!kIsComplex ? params.dC_batch_stride : params.dC_batch_stride / 2) + group_id * params.dC_group_stride);
    float *dD = params.dD_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dD_ptr) + dim_id;
    float D_val = params.D_ptr == nullptr ? 0 : reinterpret_cast<float *>(params.D_ptr)[dim_id];
    float *ddelta_bias = params.ddelta_bias_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.ddelta_bias_ptr) + dim_id;
    float delta_bias = params.delta_bias_ptr == nullptr ? 0 : reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id];
    scan_t *x = params.x_ptr == nullptr
        ? nullptr
        : reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id) * (params.n_chunks) * params.dstate;
    float dD_val = 0;
    float ddelta_bias_val = 0;

    constexpr int kChunkSize = kNThreads * kNItems;
    u += (params.n_chunks - 1) * kChunkSize;
    delta += (params.n_chunks - 1) * kChunkSize;
    dout += (params.n_chunks - 1) * kChunkSize;
    Bvar += (params.n_chunks - 1) * kChunkSize * (!kIsComplex ? 1 : 2);
    Cvar += (params.n_chunks - 1) * kChunkSize * (!kIsComplex ? 1 : 2);
    for (int chunk = params.n_chunks - 1; chunk >= 0; --chunk) {
        input_t u_vals[kNItems];
        input_t delta_vals_load[kNItems];
        input_t dout_vals_load[kNItems];
        __syncthreads();
        load_input<Ktraits>(u, u_vals, smem_load, params.seqlen - chunk * kChunkSize);
        u -= kChunkSize;
        __syncthreads();
        load_input<Ktraits>(delta, delta_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
        // Will reload delta at the same location if kDeltaSoftplus
        if constexpr (!kDeltaSoftplus) { delta -= kChunkSize; }
        __syncthreads();
        load_input<Ktraits>(dout, dout_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
        dout -= kChunkSize;

        float dout_vals[kNItems], delta_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            dout_vals[i] = float(dout_vals_load[i]);
            delta_vals[i] = float(delta_vals_load[i]) + delta_bias;
            if constexpr (kDeltaSoftplus) {
                delta_vals[i] = delta_vals[i] <= 20.f ? log1pf(expf(delta_vals[i])) : delta_vals[i];
            }
        }

        if constexpr (kHasZ) {
            input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride
                + dim_id * params.z_d_stride + chunk * kChunkSize;
            input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
                + dim_id * params.out_d_stride + chunk * kChunkSize;
            input_t *dz = reinterpret_cast<input_t *>(params.dz_ptr) + batch_id * params.dz_batch_stride
                + dim_id * params.dz_d_stride + chunk * kChunkSize;
            input_t z_vals[kNItems], out_vals[kNItems];
            __syncthreads();
            load_input<Ktraits>(z, z_vals, smem_load, params.seqlen - chunk * kChunkSize);
            __syncthreads();
            load_input<Ktraits>(out, out_vals, smem_load, params.seqlen - chunk * kChunkSize);
            float dz_vals[kNItems], z_silu_vals[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float z_val = z_vals[i];
                float z_sigmoid_val = 1.0f / (1.0f + expf(-z_val));
                z_silu_vals[i] = z_val * z_sigmoid_val;
                dz_vals[i] = dout_vals[i] * float(out_vals[i]) * z_sigmoid_val
                             * (1.0f + z_val * (1.0f - z_sigmoid_val));
                dout_vals[i] *= z_silu_vals[i];
            }
            __syncthreads();
            store_output<Ktraits>(dz, dz_vals, smem_store, params.seqlen - chunk * kChunkSize);
            if (params.out_z_ptr != nullptr) {  // Recompute and store out_z
                float out_z_vals[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) { out_z_vals[i] = float(out_vals[i]) * z_silu_vals[i]; }
                // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
                    // printf("out_val=%f, z_silu_val = %f, out_z_val = %f\n", float(out_vals[0]), z_silu_vals[0], out_z_vals[0]);
                // }
                input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride
                    + dim_id * params.out_z_d_stride + chunk * kChunkSize;
                __syncthreads();
                store_output<Ktraits>(out_z, out_z_vals, smem_store, params.seqlen - chunk * kChunkSize);
            }
        }

        float du_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) { du_vals[i] = D_val * dout_vals[i]; }
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) { dD_val += dout_vals[i] * float(u_vals[i]); }

        float ddelta_vals[kNItems] = {0};
        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            const weight_t A_val = A[state_idx * params.A_dstate_stride];
            // Multiply the real part of A with LOG2E so we can use exp2f instead of expf.
            weight_t A_scaled;
            constexpr float kLog2e = M_LOG2E;
            if constexpr (!kIsComplex) {
                A_scaled = A_val * kLog2e;
            } else {
                A_scaled = complex_t(A_val.real_ * kLog2e, A_val.imag_);
            }
            weight_t B_val, C_val;
            weight_t B_vals[kNItems], C_vals[kNItems];
            if constexpr (!kIsVariableB) {
                B_val = B[state_idx * params.B_dstate_stride];
            } else {
                load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
            }
            if constexpr (!kIsVariableC) {
                C_val = C[state_idx * params.C_dstate_stride];
            } else {
                auto &smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight_C, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
            }
            // const weight_t A_val = smem_a[state_idx];
            scan_t thread_data[kNItems], thread_reverse_data[kNItems];
            if constexpr (!kIsComplex) {
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    // 🔥 Clamp exponent to prevent underflow/overflow
                    float exponent = safe_clamp(delta_vals[i] * A_scaled, -SAFE_EXP2F_RANGE, SAFE_EXP2F_RANGE);
                    const float delta_a_exp = exp2f(exponent);
                    thread_data[i] = make_float2(delta_a_exp, !kIsVariableB ? delta_vals[i] * float(u_vals[i]) : delta_vals[i] * float(u_vals[i]) * B_vals[i]);
                    if (i == 0) {
                        smem_delta_a[threadIdx.x == 0 ? state_idx + (chunk % 2) * MAX_DSTATE : threadIdx.x + 2 * MAX_DSTATE] = delta_a_exp;
                    } else {
                        thread_reverse_data[i - 1].x = delta_a_exp;
                    }
                    thread_reverse_data[i].y = dout_vals[i] *
                        (!kIsVariableC
                         ? (!kIsVariableB ? B_val * C_val : C_val)
                         : (!kIsVariableB ? B_val * C_vals[i] : C_vals[i]));
                }
                __syncthreads();
                thread_reverse_data[kNItems - 1].x = threadIdx.x == kNThreads - 1
                    ? (chunk == params.n_chunks - 1 ? 1.f : smem_delta_a[state_idx + ((chunk + 1) % 2) * MAX_DSTATE])
                    : smem_delta_a[threadIdx.x + 1 + 2 * MAX_DSTATE];
                // Initialize running total
                scan_t running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? x[(chunk - 1) * params.dstate + state_idx] : make_float2(1.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                );
                scan_t running_postfix = chunk < params.n_chunks - 1 && threadIdx.x % 32 == 0 ? smem_running_postfix[state_idx] : make_float2(1.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> postfix_op(running_postfix);
                typename Ktraits::BlockReverseScanT(smem_reverse_scan).InclusiveReverseScan(
                    thread_reverse_data, thread_reverse_data, SSMScanOp<weight_t>(), postfix_op
                );
                if (threadIdx.x == 0) { smem_running_postfix[state_idx] = postfix_op.running_prefix; }
                weight_t dA_val = 0, dBC_val = 0;
                weight_t dB_vals[kNItems], dC_vals[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const float dx = thread_reverse_data[i].y;
                    const float ddelta_u = !kIsVariableB ? dx : dx * B_vals[i];
                    du_vals[i] += ddelta_u * delta_vals[i];
                    const float a = thread_data[i].y - (!kIsVariableB ? delta_vals[i] * float(u_vals[i]) : delta_vals[i] * float(u_vals[i]) * B_vals[i]);
                    ddelta_vals[i] += ddelta_u * float(u_vals[i]) + dx * A_val * a;
                    dA_val += dx * delta_vals[i] * a;
                    if constexpr (!kIsVariableB || !kIsVariableC) {
                        if constexpr (!kIsVariableB) {  // dBC_val is dB_val
                            dBC_val += dout_vals[i] * (!kIsVariableC ? thread_data[i].y : thread_data[i].y * C_vals[i]);
                        } else {  // dBC_val is dC_val
                            dBC_val += dout_vals[i] * thread_data[i].y;
                        }
                    }
                    if constexpr (kIsVariableB) { dB_vals[i] = dx * delta_vals[i] * float(u_vals[i]); }
                    if constexpr (kIsVariableC) {
                        dC_vals[i] = dout_vals[i] * (!kIsVariableB ? thread_data[i].y * B_val : thread_data[i].y);
                    }
                }
                // Block-exchange to make the atomicAdd's coalesced, otherwise they're much slower
                if constexpr (kIsVariableB || kIsVariableC) {
                    if constexpr (kIsVariableB) {
                        typename Ktraits::BlockExchangeT(smem_exchange).BlockedToStriped(dB_vals, dB_vals);
                    }
                    if constexpr (kIsVariableC) {
                        auto &smem_exchange_C = !kIsVariableB ? smem_exchange : smem_exchange1;
                        typename Ktraits::BlockExchangeT(smem_exchange_C).BlockedToStriped(dC_vals, dC_vals);
                    }
                    const int seqlen_remaining = params.seqlen - chunk * kChunkSize - threadIdx.x;
                    weight_t *dB_cur = dB + state_idx * params.dB_dstate_stride + chunk * kChunkSize + threadIdx.x;
                    weight_t *dC_cur = dC + state_idx * params.dC_dstate_stride + chunk * kChunkSize + threadIdx.x;
                    #pragma unroll
                    for (int i = 0; i < kNItems; ++i) {
                        if (i * kNThreads < seqlen_remaining) {
                            if constexpr (kIsVariableB) { gpuAtomicAdd(dB_cur + i * kNThreads, dB_vals[i]); }
                            if constexpr (kIsVariableC) { gpuAtomicAdd(dC_cur + i * kNThreads, dC_vals[i]); }
                        }
                    }
                }
                if constexpr (!kIsVariableB || !kIsVariableC) {
                    float2 dA_dBC_val = make_float2(dA_val, dBC_val);
                    dA_dBC_val = typename Ktraits::BlockReduceT(smem_reduce).Sum(dA_dBC_val);
                    dA_val = dA_dBC_val.x;
                    if (threadIdx.x == 0) {
                        smem_dbc[state_idx] = chunk == params.n_chunks - 1 ? dA_dBC_val.y : dA_dBC_val.y + smem_dbc[state_idx];
                    }
                } else {
                    dA_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dA_val);
                }
                if (threadIdx.x == 0) {
                    smem_da[state_idx] = chunk == params.n_chunks - 1 ? dA_val : dA_val + smem_da[state_idx];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    // Pytorch's implementation of complex exp (which calls thrust) is very slow
                    // 🔥 Clamp complex exponent for stability
                    complex_t delta_a = delta_vals[i] * A_scaled;
                    delta_a.real_ = safe_clamp(delta_a.real_, -SAFE_EXP2F_RANGE, SAFE_EXP2F_RANGE);
                    complex_t delta_a_exp = cexp2f(delta_a);
                    weight_t B_delta_u_val = !kIsVariableB ? delta_vals[i] * float(u_vals[i]) : B_vals[i] * delta_vals[i] * float(u_vals[i]);
                    thread_data[i] = make_float4(delta_a_exp.real_, delta_a_exp.imag_, B_delta_u_val.real_, B_delta_u_val.imag_);
                    if (i == 0) {
                        smem_delta_a[threadIdx.x == 0 ? state_idx + (chunk % 2) * MAX_DSTATE : threadIdx.x + 2 * MAX_DSTATE] = delta_a_exp;
                    } else {
                        thread_reverse_data[i - 1].x = delta_a_exp.real_;
                        thread_reverse_data[i - 1].y = -delta_a_exp.imag_;
                    }
                    complex_t dout_BC = 2 * dout_vals[i]
                        * conj(!kIsVariableC
                                ? (!kIsVariableB ? B_val * C_val : C_val)
                                : (!kIsVariableB ? B_val * C_vals[i] : C_vals[i]));
                    thread_reverse_data[i].z = dout_BC.real_;
                    thread_reverse_data[i].w = dout_BC.imag_;
                }
                __syncthreads();
                complex_t delta_a_exp = threadIdx.x == kNThreads - 1
                    ? (chunk == params.n_chunks - 1 ? 1.f : smem_delta_a[state_idx + ((chunk + 1) % 2) * MAX_DSTATE])
                    : smem_delta_a[threadIdx.x + 1 + 2 * MAX_DSTATE];
                thread_reverse_data[kNItems - 1].x = delta_a_exp.real_;
                thread_reverse_data[kNItems - 1].y = -delta_a_exp.imag_;
                // Initialize running total
                scan_t running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? x[(chunk - 1) * params.dstate + state_idx] : make_float4(1.f, 0.f, 0.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                );
                scan_t running_postfix = chunk < params.n_chunks - 1 && threadIdx.x % 32 == 0 ? smem_running_postfix[state_idx] : make_float4(1.f, 0.f, 0.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> postfix_op(running_postfix);
                typename Ktraits::BlockReverseScanT(smem_reverse_scan).InclusiveReverseScan(
                    thread_reverse_data, thread_reverse_data, SSMScanOp<weight_t>(), postfix_op
                );
                if (threadIdx.x == 0) { smem_running_postfix[state_idx] = postfix_op.running_prefix; }
                weight_t dA_val = 0, dBC_val = 0;
                weight_t dB_vals[kNItems], dC_vals[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    complex_t x = complex_t(thread_data[i].z, thread_data[i].w);
                    complex_t dx = complex_t(thread_reverse_data[i].z, thread_reverse_data[i].w);
                    float ddelta_u = !kIsVariableB ? dx.real_ : (dx * conj(B_vals[i])).real_;
                    if constexpr (!kIsVariableB || !kIsVariableC) {
                        if constexpr (!kIsVariableB) {  // dBC_val is dB_val
                            dBC_val += (2 * dout_vals[i]) * conj(!kIsVariableC ? x : x * C_vals[i]);
                        } else {  // dBC_val is dC_val
                            dBC_val += (2 * dout_vals[i]) * conj(x);
                        }
                    }
                    const complex_t a_conj = conj(x - (!kIsVariableB ? delta_vals[i] * float(u_vals[i]) : delta_vals[i] * float(u_vals[i]) * B_vals[i]));
                    du_vals[i] += ddelta_u * delta_vals[i];
                    ddelta_vals[i] += ddelta_u * float(u_vals[i]) + (dx * conj(A_val) * a_conj).real_;
                    dA_val += delta_vals[i] * dx * a_conj;
                    if constexpr (kIsVariableB) { dB_vals[i] = dx * delta_vals[i] * float(u_vals[i]); }
                    if constexpr (kIsVariableC) {
                        dC_vals[i] = (2 * dout_vals[i]) * conj(!kIsVariableB ? x * B_val : x);
                    }
                }
                // Block-exchange to make the atomicAdd's coalesced, otherwise they're much slower
                if constexpr (kIsVariableB || kIsVariableC) {
                    float dB_vals_f[kNItems * 2], dC_vals_f[kNItems * 2];
                    if constexpr (kIsVariableB) {
                        #pragma unroll
                        for (int i = 0; i < kNItems; ++i) {
                            dB_vals_f[i * 2] = dB_vals[i].real_;
                            dB_vals_f[i * 2 + 1] = dB_vals[i].imag_;
                        }
                        typename Ktraits::BlockExchangeT(smem_exchange).BlockedToStriped(dB_vals_f, dB_vals_f);
                    }
                    if constexpr (kIsVariableC) {
                        #pragma unroll
                        for (int i = 0; i < kNItems; ++i) {
                            dC_vals_f[i * 2] = dC_vals[i].real_;
                            dC_vals_f[i * 2 + 1] = dC_vals[i].imag_;
                        }
                        auto &smem_exchange_C = !kIsVariableB ? smem_exchange : smem_exchange1;
                        typename Ktraits::BlockExchangeT(smem_exchange_C).BlockedToStriped(dC_vals_f, dC_vals_f);
                    }
                    const int seqlen_remaining = (params.seqlen - chunk * kChunkSize) * 2 - threadIdx.x;
                    float *dB_cur = reinterpret_cast<float *>(dB) + state_idx * params.dB_dstate_stride + chunk * kChunkSize * 2 + threadIdx.x;
                    float *dC_cur = reinterpret_cast<float *>(dC) + state_idx * params.dC_dstate_stride + chunk * kChunkSize * 2 + threadIdx.x;
                    #pragma unroll
                    for (int i = 0; i < kNItems * 2; ++i) {
                        if (i * kNThreads < seqlen_remaining) {
                            if constexpr (kIsVariableB) { gpuAtomicAdd(dB_cur + i * kNThreads, dB_vals_f[i]); }
                            if constexpr (kIsVariableC) { gpuAtomicAdd(dC_cur + i * kNThreads, dC_vals_f[i]); }
                        }
                    }
                }
                if constexpr (!kIsVariableB || !kIsVariableC) {
                    float4 dA_dBC_val = make_float4(dA_val.real_, dA_val.imag_, dBC_val.real_, dBC_val.imag_);
                    dA_dBC_val = typename Ktraits::BlockReduceT(smem_reduce).Sum(dA_dBC_val);
                    dA_val = complex_t(dA_dBC_val.x, dA_dBC_val.y);
                    dBC_val = complex_t(dA_dBC_val.z, dA_dBC_val.w);
                    if (threadIdx.x == 0) {
                        smem_dbc[state_idx] = chunk == params.n_chunks - 1 ? dBC_val : dBC_val + smem_dbc[state_idx];
                    }
                } else {
                    dA_val = typename Ktraits::BlockReduceComplexT(smem_reduce_complex).Sum(dA_val);
                }
                if (threadIdx.x == 0) {
                    smem_da[state_idx] = chunk == params.n_chunks - 1 ? dA_val : dA_val + smem_da[state_idx];
                }
            }
        }

        if constexpr (kDeltaSoftplus) {
            __syncthreads();
            input_t delta_vals_load[kNItems];
            load_input<Ktraits>(delta, delta_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
            delta -= kChunkSize;
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float delta_val = float(delta_vals_load[i]) + delta_bias;
                float delta_val_neg_exp = expf(-delta_val);
                ddelta_vals[i] = delta_val <= 20.f
                    ? ddelta_vals[i] / (1.f + delta_val_neg_exp)
                    : ddelta_vals[i];
            }
        }
        for (int i = 0; i < kNItems; ++i) { ddelta_bias_val += ddelta_vals[i]; }

        input_t *du = reinterpret_cast<input_t *>(params.du_ptr) + batch_id * params.du_batch_stride
            + dim_id * params.du_d_stride + chunk * kChunkSize;
        input_t *ddelta = reinterpret_cast<input_t *>(params.ddelta_ptr) + batch_id * params.ddelta_batch_stride
            + dim_id * params.ddelta_d_stride + chunk * kChunkSize;
        __syncthreads();
        store_output<Ktraits>(du, du_vals, smem_store, params.seqlen - chunk * kChunkSize);
        __syncthreads();
        store_output<Ktraits>(ddelta, ddelta_vals, smem_store, params.seqlen - chunk * kChunkSize);

        Bvar -= kChunkSize * (!kIsComplex ? 1 : 2);
        Cvar -= kChunkSize * (!kIsComplex ? 1 : 2);
    }
    if (params.dD_ptr != nullptr) {
        dD_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dD_val);
        if (threadIdx.x == 0) { gpuAtomicAdd(dD, dD_val); }
    }
    if (params.ddelta_bias_ptr != nullptr) {
        __syncthreads();
        ddelta_bias_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(ddelta_bias_val);
        if (threadIdx.x == 0) { gpuAtomicAdd(ddelta_bias, ddelta_bias_val); }
    }
    for (int state_idx = threadIdx.x; state_idx < params.dstate; state_idx += blockDim.x) {
        gpuAtomicAdd(&(dA[state_idx * params.dA_dstate_stride]), smem_da[state_idx]);
        weight_t dBC_val;
        if (!kIsVariableB || !kIsVariableC) { dBC_val = smem_dbc[state_idx]; }
        if constexpr (!kIsVariableB) {
            gpuAtomicAdd(&(dB[state_idx * params.dB_dstate_stride]),
                         !kIsVariableC ? dBC_val * conj(C[state_idx * params.C_dstate_stride]) : dBC_val);
        }
        if constexpr (!kIsVariableC) {
            gpuAtomicAdd(&(dC[state_idx * params.dC_dstate_stride]),
                        !kIsVariableB ? dBC_val * conj(B[state_idx * params.B_dstate_stride]) : dBC_val);
        }
    }
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_bwd_launch(SSMParamsBwd &params, cudaStream_t stream) {
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.is_variable_B, kIsVariableB, [&] {
            BOOL_SWITCH(params.is_variable_C, kIsVariableC, [&] {
                BOOL_SWITCH(params.delta_softplus, kDeltaSoftplus, [&] {
                    BOOL_SWITCH(params.z_ptr != nullptr , kHasZ, [&] {
                        using Ktraits = Selective_Scan_bwd_kernel_traits<kNThreads, kNItems, kIsEvenLen, kIsVariableB, kIsVariableC, kDeltaSoftplus, kHasZ, input_t, weight_t>;
                        // using Ktraits = Selective_Scan_bwd_kernel_traits<kNThreads, kNItems, true, kIsVariableB, kIsVariableC, kDeltaSoftplus, kHasZ, input_t, weight_t>;
                        // TODO: check this
                        constexpr int kSmemSize = Ktraits::kSmemSize + MAX_DSTATE * sizeof(typename Ktraits::scan_t) + (kNThreads + 4 * MAX_DSTATE) * sizeof(typename Ktraits::weight_t);

                        dim3 grid(params.batch, params.dim);
                        
                        auto kernel = &selective_scan_bwd_kernel<Ktraits>;

                        if (kSmemSize >= 48 * 1024) {

                            #ifndef USE_ROCM
                            C10_CUDA_CHECK(cudaFuncSetAttribute(
                                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                            #else
                            C10_CUDA_CHECK(cudaFuncSetAttribute(
                                (void *) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                            std::cerr << "Warning (selective_scan_bwd_kernel): attempting to set maxDynamicSharedMemorySize on an AMD GPU which is currently a non-op (in ROCm versions <= 6.1). This might lead to undefined behavior. \n" << std::endl;
                            #endif

                        }

                        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });
}

template<typename input_t, typename weight_t>
void selective_scan_bwd_cuda(SSMParamsBwd &params, cudaStream_t stream) {

    #ifndef USE_ROCM
        if (params.seqlen <= 128) {
            selective_scan_bwd_launch<32, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 256) {
            selective_scan_bwd_launch<32, 8, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_bwd_launch<32, 16, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_bwd_launch<64, 16, input_t, weight_t>(params, stream);
        } else {
            selective_scan_bwd_launch<128, 16, input_t, weight_t>(params, stream);
        }
    #else 
        if (params.seqlen <= 256) {
            selective_scan_bwd_launch<64, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_bwd_launch<64, 8, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_bwd_launch<64, 16, input_t, weight_t>(params, stream);
        } else {
            selective_scan_bwd_launch<128, 16, input_t, weight_t>(params, stream);
        }
    #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// ── Flat per-channel backward kernel traits ─────────────────────────────────
// Inherits all definitions from the original bwd traits.
// Fixed: kIsEvenLen=false, kDeltaSoftplus=false, kNRows=1, kIsComplex=false.
// Added: BlockLoadWeightFloatT (float32 B/C), BlockLoadKT (int32 k).
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kNThreads_, int kNItems_, bool kIsVariableB_, bool kIsVariableC_,
         bool kHasZ_, typename input_t_, typename weight_t_>
struct Selective_Scan_flat_ch_bwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsComplex = false;  // flat_ch: always real
    static constexpr bool kIsEvenLen = false;  // per-channel variable length
    static constexpr bool kIsVariableB = kIsVariableB_;
    static constexpr bool kIsVariableC = kIsVariableC_;
    static constexpr bool kDeltaSoftplus = false;  // uses dt_idle/dt_active formula
    static constexpr bool kHasZ = kHasZ_;
    static constexpr int kMinBlocks = kNThreads == 128 ? 3 : 2;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float2;  // always real

    // Loaders / stores — same as original bwd
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads, cub::BLOCK_STORE_WARP_TRANSPOSE>;

    // NEW: float32 loader for B/C (flat_ch path B/C are always float32)
    using BlockLoadWeightFloatT = cub::BlockLoad<float, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    // NEW: int32 loader for k (inter-event intervals)
    using BlockLoadKT = cub::BlockLoad<int32_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;

    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    using BlockReverseScanT = BlockReverseScan<scan_t, kNThreads>;
    using BlockReduceT = cub::BlockReduce<scan_t, kNThreads>;
    using BlockReduceFloatT = cub::BlockReduce<float, kNThreads>;
    using BlockReduceComplexT = cub::BlockReduce<complex_t, kNThreads>;
    using BlockExchangeT = cub::BlockExchange<float, kNThreads, kNItems>;

    static constexpr int kSmemIOSize = custom_max({sizeof(typename BlockLoadT::TempStorage),
                                                    sizeof(typename BlockLoadVecT::TempStorage),
                                                    sizeof(typename BlockLoadKT::TempStorage),
                                                    (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightT::TempStorage),
                                                    (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                    (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightFloatT::TempStorage),
                                                    sizeof(typename BlockStoreT::TempStorage),
                                                    sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemExchangeSize = (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockExchangeT::TempStorage);
    static constexpr int kSmemReduceSize = sizeof(typename BlockReduceT::TempStorage);
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize + kSmemReduceSize + sizeof(typename BlockScanT::TempStorage) + sizeof(typename BlockReverseScanT::TempStorage);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// ── Flat per-channel backward kernel ────────────────────────────────────────
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_flat_ch_bwd_kernel(SSMParamsFlatSegBwd params) {
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;

    // ── Shared memory (same layout as original bwd kernel) ──────────────────
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    // flat-ch: use float-specific loader for B/C
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightFloatT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightFloatT::TempStorage*>(
                                smem_ + sizeof(typename Ktraits::BlockLoadWeightFloatT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_exchange = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    auto& smem_exchange1 = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize + sizeof(typename Ktraits::BlockExchangeT::TempStorage));
    auto& smem_reduce = *reinterpret_cast<typename Ktraits::BlockReduceT::TempStorage*>(reinterpret_cast<char *>(&smem_exchange) + Ktraits::kSmemExchangeSize);
    auto& smem_reduce_float = *reinterpret_cast<typename Ktraits::BlockReduceFloatT::TempStorage*>(&smem_reduce);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(reinterpret_cast<char *>(&smem_reduce) + Ktraits::kSmemReduceSize);
    auto& smem_reverse_scan = *reinterpret_cast<typename Ktraits::BlockReverseScanT::TempStorage*>(reinterpret_cast<char *>(&smem_scan) + sizeof(typename Ktraits::BlockScanT::TempStorage));
    weight_t *smem_delta_a = reinterpret_cast<weight_t *>(smem_ + Ktraits::kSmemSize);
    scan_t *smem_running_postfix = reinterpret_cast<scan_t *>(smem_delta_a + 2 * MAX_DSTATE + kNThreads);
    weight_t *smem_da = reinterpret_cast<weight_t *>(smem_running_postfix + MAX_DSTATE);
    weight_t *smem_dbc = reinterpret_cast<weight_t *>(smem_da + MAX_DSTATE);
    // Depthwise gradient accumulators — always laid out (compile-time constant size).
    // smem_dw_b / smem_dw_c each hold MAX_DSTATE floats.
    weight_t *smem_dw_b = reinterpret_cast<weight_t *>(smem_dbc + MAX_DSTATE);
    weight_t *smem_dw_c = reinterpret_cast<weight_t *>(smem_dw_b  + MAX_DSTATE);
    // Pointwise: effective weight cache.  Allocated as part of the extra smem
    // (see kSmemSize comment in launch helper).  Sized MAX_DSTATE (= max dstate).
    // smem_W_B_eff[s] = Σ_{d'} pw_B[dim_id,d'] * W_B[d',s] — precomputed once.
    weight_t *smem_W_B_eff = reinterpret_cast<weight_t *>(smem_dw_c    + MAX_DSTATE);
    weight_t *smem_W_C_eff = reinterpret_cast<weight_t *>(smem_W_B_eff + MAX_DSTATE);
    // NOTE: dpw_B / dpw_C are accumulated directly into global memory via
    // gpuAtomicAdd — no smem accumulator needed, which also removes the
    // constraint that dim <= MAX_DSTATE.

    // Dedicated int32 loader for k (overlaps same smem region)
    auto& smem_load_k = reinterpret_cast<typename Ktraits::BlockLoadKT::TempStorage&>(smem_);

    // ── Block → data mapping ────────────────────────────────────────────────
    const int ch     = blockIdx.x;
    const int dim_id = ch % params.dim;

    const int ev_start = params.cu_seqlens_ptr[ch];
    const int ch_len   = params.cu_seqlens_ptr[ch + 1] - ev_start;
    if (ch_len == 0) return;

    // ── Pointer initialisation ──────────────────────────────────────────────
    input_t *u    = reinterpret_cast<input_t *>(params.u_ptr)    + ev_start;
    int32_t *k    = params.k_ptr + ev_start;
    input_t *dout = reinterpret_cast<input_t *>(params.dout_ptr) + ev_start;

    weight_t *A  = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * params.A_d_stride;
    weight_t *dA = reinterpret_cast<weight_t *>(params.dA_ptr) + dim_id * params.dA_d_stride;

    // Variable B/C: [dstate, E] layout, offset by ev_start (legacy path)
    float *Bvar_legacy = reinterpret_cast<float *>(params.B_ptr) + ev_start;
    float *Cvar_legacy = reinterpret_cast<float *>(params.C_ptr) + ev_start;

    // Tiled B/C: tile_base_idx is the global tile start for this channel
    const int32_t tile_base_idx =
        (params.tile_offsets_ptr != nullptr) ? params.tile_offsets_ptr[ch] : 0;

    // dB/dC for variable path: same layout as B/C
    weight_t *dB = reinterpret_cast<weight_t *>(params.dB_ptr)
        + (!kIsVariableB ? dim_id * params.dB_d_stride : 0);
    weight_t *dC = reinterpret_cast<weight_t *>(params.dC_ptr)
        + (!kIsVariableC ? dim_id * params.dC_d_stride : 0);

    float *dD = params.dD_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dD_ptr) + dim_id;
    float D_val = params.D_ptr == nullptr ? 0 : reinterpret_cast<float *>(params.D_ptr)[dim_id];

    // x layout: [n_channels, n_chunks, dstate], preserving the n_chunks stride
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr)
                + ch * params.n_chunks * params.dstate;

    // dt_idle / dt_active (replaces delta_bias)
    const float dt_idle   = params.dt_idle_ptr[dim_id];
    const float dt_active = params.dt_active_ptr[dim_id];

    float dD_val = 0;
    float ddt_idle_val = 0;
    float ddt_active_val = 0;

    constexpr int kChunkSize = kNThreads * kNItems;
    const int ch_n_chunks = (ch_len + kChunkSize - 1) / kChunkSize;

    // ── Depthwise + pointwise precompute (mirrors fwd kernel) ────────────────
    // W_B_eff[s] = Σ_{d'} pw_B[d, d'] * W_B[d', s]
    const bool has_pw = params.is_depthwise_bc && (params.pw_B_ptr != nullptr);
    if (has_pw) {
        const float *pw_B_row = reinterpret_cast<const float *>(params.pw_B_ptr)
                                 + (long)dim_id * params.dim;
        const float *pw_C_row = reinterpret_cast<const float *>(params.pw_C_ptr)
                                 + (long)dim_id * params.dim;
        const float *W_B_base = reinterpret_cast<const float *>(params.W_B_ptr);
        const float *W_C_base = reinterpret_cast<const float *>(params.W_C_ptr);
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

    // Jump to last chunk (reverse traversal)
    u    += (ch_n_chunks - 1) * kChunkSize;
    k    += (ch_n_chunks - 1) * kChunkSize;
    dout += (ch_n_chunks - 1) * kChunkSize;
    // Tiled path uses tile_base_idx + chunk, no pointer advance needed
    if (params.tile_offsets_ptr == nullptr) {
        Bvar_legacy += (ch_n_chunks - 1) * kChunkSize;
        Cvar_legacy += (ch_n_chunks - 1) * kChunkSize;
    }

    for (int chunk = ch_n_chunks - 1; chunk >= 0; --chunk) {
        const int remaining = ch_len - chunk * kChunkSize;

        // ── Step 1: Load u, k, dout ─────────────────────────────────────────
        input_t u_vals[kNItems];
        int32_t k_vals[kNItems];
        input_t dout_vals_load[kNItems];
        __syncthreads();
        load_input<Ktraits>(u, u_vals, smem_load, remaining);
        __syncthreads();
        Ktraits::BlockLoadKT(smem_load_k).Load(k, k_vals, remaining, 1);
        __syncthreads();
        load_input<Ktraits>(dout, dout_vals_load, smem_load, remaining);

        float dout_vals[kNItems], delta_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            dout_vals[i] = float(dout_vals_load[i]);
            // ── Step 2: Reconstruct delta (no softplus, no delta_bias) ──────
            delta_vals[i] = dt_idle * float(k_vals[i] - 1) + dt_active;
        }

        // ── Step 3: [kHasZ] Gate backward (1D pointers) ────────────────────
        if constexpr (kHasZ) {
            input_t *z   = reinterpret_cast<input_t *>(params.z_ptr)   + ev_start + chunk * kChunkSize;
            input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + ev_start + chunk * kChunkSize;
            input_t *dz  = reinterpret_cast<input_t *>(params.dz_ptr)  + ev_start + chunk * kChunkSize;
            input_t z_vals[kNItems], out_vals[kNItems];
            __syncthreads();
            load_input<Ktraits>(z, z_vals, smem_load, remaining);
            __syncthreads();
            load_input<Ktraits>(out, out_vals, smem_load, remaining);
            float dz_vals[kNItems], z_silu_vals[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float z_val = z_vals[i];
                float z_sigmoid_val = 1.0f / (1.0f + expf(-z_val));
                z_silu_vals[i] = z_val * z_sigmoid_val;
                dz_vals[i] = dout_vals[i] * float(out_vals[i]) * z_sigmoid_val
                             * (1.0f + z_val * (1.0f - z_sigmoid_val));
                dout_vals[i] *= z_silu_vals[i];
            }
            __syncthreads();
            store_output<Ktraits>(dz, dz_vals, smem_store, remaining);
            if (params.out_z_ptr != nullptr) {
                float out_z_vals[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) { out_z_vals[i] = float(out_vals[i]) * z_silu_vals[i]; }
                input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + ev_start + chunk * kChunkSize;
                __syncthreads();
                store_output<Ktraits>(out_z, out_z_vals, smem_store, remaining);
            }
        }

        // ── Step 4: dD / du initialisation (same as original bwd) ───────────
        float du_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) { du_vals[i] = D_val * dout_vals[i]; }
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) { dD_val += dout_vals[i] * float(u_vals[i]); }

        float ddelta_vals[kNItems] = {0};

        // ── Step 5: State loop (state_idx = 0..dstate-1) ────────────────────
        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            // 5a. Load A (same as original, real only)
            const weight_t A_val = A[state_idx * params.A_dstate_stride];
            constexpr float kLog2e = M_LOG2E;
            weight_t A_scaled = A_val * kLog2e;

            // 5b. Load B/C (float32 with BlockLoadWeightFloatT)
            weight_t B_val, C_val;
            float B_vals[kNItems], C_vals[kNItems];
            if constexpr (!kIsVariableB) {
                B_val = reinterpret_cast<float *>(params.B_ptr)[state_idx * params.B_row_stride + dim_id];
            } else {
                // ── Depthwise path: on-the-fly W_B / W_C ─────────────────────
                if (params.is_depthwise_bc) {
                    const float W_B_s = has_pw
                        ? smem_W_B_eff[state_idx]
                        : reinterpret_cast<const float *>(params.W_B_ptr)
                              [dim_id * params.dstate + state_idx];
                    #pragma unroll
                    for (int i = 0; i < kNItems; ++i)
                        B_vals[i] = float(u_vals[i]) * W_B_s;
                } else if (params.tile_offsets_ptr != nullptr) {
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
            }
            if constexpr (!kIsVariableC) {
                C_val = reinterpret_cast<float *>(params.C_ptr)[state_idx * params.C_row_stride + dim_id];
            } else {
                auto &smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                // ── Depthwise path ────────────────────────────────────────────
                if (params.is_depthwise_bc) {
                    const float W_C_s = has_pw
                        ? smem_W_C_eff[state_idx]
                        : reinterpret_cast<const float *>(params.W_C_ptr)
                              [dim_id * params.dstate + state_idx];
                    #pragma unroll
                    for (int i = 0; i < kNItems; ++i)
                        C_vals[i] = float(u_vals[i]) * W_C_s;
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
            }

            // 5c. Construct thread_data (forward scan data) — real only
            scan_t thread_data[kNItems], thread_reverse_data[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float exponent = safe_clamp(delta_vals[i] * A_scaled, -SAFE_EXP2F_RANGE, SAFE_EXP2F_RANGE);
                const float delta_a_exp = exp2f(exponent);
                thread_data[i] = make_float2(delta_a_exp,
                    !kIsVariableB ? delta_vals[i] * float(u_vals[i]) : delta_vals[i] * float(u_vals[i]) * B_vals[i]);
                // smem_delta_a double-buffer trick (same as original bwd)
                if (i == 0) {
                    smem_delta_a[threadIdx.x == 0 ? state_idx + (chunk % 2) * MAX_DSTATE : threadIdx.x + 2 * MAX_DSTATE] = delta_a_exp;
                } else {
                    thread_reverse_data[i - 1].x = delta_a_exp;
                }
                // 5d. Construct thread_reverse_data (dout*C part)
                thread_reverse_data[i].y = dout_vals[i] *
                    (!kIsVariableC
                     ? (!kIsVariableB ? B_val * C_val : C_val)
                     : (!kIsVariableB ? B_val * C_vals[i] : C_vals[i]));

                // Zero out-of-bounds threads
                if (threadIdx.x * kNItems + i >= remaining) {
                    thread_data[i] = make_float2(1.f, 0.f);
                }
            }
            __syncthreads();
            thread_reverse_data[kNItems - 1].x = threadIdx.x == kNThreads - 1
                ? (chunk == ch_n_chunks - 1 ? 1.f : smem_delta_a[state_idx + ((chunk + 1) % 2) * MAX_DSTATE])
                : smem_delta_a[threadIdx.x + 1 + 2 * MAX_DSTATE];

            // 5e. running_prefix — read x using the corrected layout
            scan_t running_prefix = chunk > 0 && threadIdx.x % 32 == 0
                ? x[(chunk - 1) * params.dstate + state_idx]
                : make_float2(1.f, 0.f);

            // 5f. BlockScan (reconstruct x state) — same as original bwd
            SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
            typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
            );

            // 5g. BlockReverseScan — same as original bwd
            scan_t running_postfix = chunk < ch_n_chunks - 1 && threadIdx.x % 32 == 0
                ? smem_running_postfix[state_idx]
                : make_float2(1.f, 0.f);
            SSMScanPrefixCallbackOp<weight_t> postfix_op(running_postfix);
            typename Ktraits::BlockReverseScanT(smem_reverse_scan).InclusiveReverseScan(
                thread_reverse_data, thread_reverse_data, SSMScanOp<weight_t>(), postfix_op
            );
            if (threadIdx.x == 0) { smem_running_postfix[state_idx] = postfix_op.running_prefix; }

            // 5h. Gradient computation (same as original bwd, real non-complex branch)
            weight_t dA_val = 0, dBC_val = 0;
            weight_t dB_vals[kNItems], dC_vals[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                const float dx = thread_reverse_data[i].y;
                const float ddelta_u = !kIsVariableB ? dx : dx * B_vals[i];
                du_vals[i] += ddelta_u * delta_vals[i];
                const float a = thread_data[i].y - (!kIsVariableB ? delta_vals[i] * float(u_vals[i]) : delta_vals[i] * float(u_vals[i]) * B_vals[i]);
                ddelta_vals[i] += ddelta_u * float(u_vals[i]) + dx * A_val * a;
                dA_val += dx * delta_vals[i] * a;
                if constexpr (!kIsVariableB || !kIsVariableC) {
                    if constexpr (!kIsVariableB) {
                        dBC_val += dout_vals[i] * (!kIsVariableC ? thread_data[i].y : thread_data[i].y * C_vals[i]);
                    } else {
                        dBC_val += dout_vals[i] * thread_data[i].y;
                    }
                }
                if constexpr (kIsVariableB) { dB_vals[i] = dx * delta_vals[i] * float(u_vals[i]); }
                if constexpr (kIsVariableC) {
                    dC_vals[i] = dout_vals[i] * (!kIsVariableB ? thread_data[i].y * B_val : thread_data[i].y);
                }
            }

            // 5i. Gradient write-out: depthwise path vs. tiled/legacy variable path.
            if constexpr (kIsVariableB || kIsVariableC) {
                if (params.is_depthwise_bc) {
                    // ── Depthwise gradient path ───────────────────────────────
                    // W_B_eff[s] (or W_B[d,s] when no pointwise) was used to compute
                    // B[i,s] = u[i] * W_B_eff[s].
                    //
                    // Chain-rule additions to du:
                    //   dL/du[i] += dB_vals[i] * W_B_eff[s]
                    //   dL/du[i] += dC_vals[i] * W_C_eff[s]
                    //
                    // Weight gradients (accumulated over all events in this channel):
                    //   dW_B_eff[s] += sum_i dB_vals[i] * u[i]
                    //   (maps back to dW_B / dpw_B depending on has_pw — see epilogue)
                    const float W_B_s = has_pw ? smem_W_B_eff[state_idx]
                        : reinterpret_cast<const float *>(params.W_B_ptr)
                              [dim_id * params.dstate + state_idx];
                    const float W_C_s = has_pw ? smem_W_C_eff[state_idx]
                        : reinterpret_cast<const float *>(params.W_C_ptr)
                              [dim_id * params.dstate + state_idx];

                    float dW_B_thread = 0.f, dW_C_thread = 0.f;
                    #pragma unroll
                    for (int i = 0; i < kNItems; ++i) {
                        const float u_i = float(u_vals[i]);
                        if constexpr (kIsVariableB) {
                            du_vals[i] += dB_vals[i] * W_B_s;
                            dW_B_thread += dB_vals[i] * u_i;
                        }
                        if constexpr (kIsVariableC) {
                            du_vals[i] += dC_vals[i] * W_C_s;
                            dW_C_thread += dC_vals[i] * u_i;
                        }
                    }
                    // Per-block reduce of (dW_B_eff[s], dW_C_eff[s]).
                    float2 dW_reduced = typename Ktraits::BlockReduceT(smem_reduce).Sum(
                        make_float2(dW_B_thread, dW_C_thread));
                    if (threadIdx.x == 0) {
                        smem_dw_b[state_idx] = chunk == ch_n_chunks - 1
                            ? dW_reduced.x : dW_reduced.x + smem_dw_b[state_idx];
                        smem_dw_c[state_idx] = chunk == ch_n_chunks - 1
                            ? dW_reduced.y : dW_reduced.y + smem_dw_c[state_idx];
                    }
                } else {
                    // ── Tiled / legacy variable path (unchanged) ──────────────
                    if constexpr (kIsVariableB) {
                        typename Ktraits::BlockExchangeT(smem_exchange).BlockedToStriped(dB_vals, dB_vals);
                    }
                    if constexpr (kIsVariableC) {
                        auto &smem_exchange_C = !kIsVariableB ? smem_exchange : smem_exchange1;
                        typename Ktraits::BlockExchangeT(smem_exchange_C).BlockedToStriped(dC_vals, dC_vals);
                    }
                    const int seqlen_remaining = remaining - threadIdx.x;

                    if (params.tile_offsets_ptr != nullptr) {
                        // Tiled dB/dC: write into [tile_id, state_idx, offset] layout
                        const long tile_id = (long)(tile_base_idx + chunk);
                        weight_t *dB_tile_row = reinterpret_cast<weight_t *>(params.dB_ptr)
                                                + tile_id * (long)params.dstate * kChunkSize
                                                + (long)state_idx * kChunkSize;
                        weight_t *dC_tile_row = reinterpret_cast<weight_t *>(params.dC_ptr)
                                                + tile_id * (long)params.dstate * kChunkSize
                                                + (long)state_idx * kChunkSize;
                        #pragma unroll
                        for (int i = 0; i < kNItems; ++i) {
                            if (i * kNThreads < seqlen_remaining) {
                                if constexpr (kIsVariableB) { gpuAtomicAdd(dB_tile_row + threadIdx.x + i * kNThreads, dB_vals[i]); }
                                if constexpr (kIsVariableC) { gpuAtomicAdd(dC_tile_row + threadIdx.x + i * kNThreads, dC_vals[i]); }
                            }
                        }
                    } else {
                        // Legacy dB/dC: flat 1D, offset by ev_start
                        weight_t *dB_cur = dB + state_idx * params.B_row_stride + ev_start + chunk * kChunkSize + threadIdx.x;
                        weight_t *dC_cur = dC + state_idx * params.C_row_stride + ev_start + chunk * kChunkSize + threadIdx.x;
                        #pragma unroll
                        for (int i = 0; i < kNItems; ++i) {
                            if (i * kNThreads < seqlen_remaining) {
                                if constexpr (kIsVariableB) { gpuAtomicAdd(dB_cur + i * kNThreads, dB_vals[i]); }
                                if constexpr (kIsVariableC) { gpuAtomicAdd(dC_cur + i * kNThreads, dC_vals[i]); }
                            }
                        }
                    }
                }
            }

            // 5j. BlockReduce + smem_da/smem_dbc accumulation — same as original bwd
            if constexpr (!kIsVariableB || !kIsVariableC) {
                float2 dA_dBC_val = make_float2(dA_val, dBC_val);
                dA_dBC_val = typename Ktraits::BlockReduceT(smem_reduce).Sum(dA_dBC_val);
                dA_val = dA_dBC_val.x;
                if (threadIdx.x == 0) {
                    smem_dbc[state_idx] = chunk == ch_n_chunks - 1 ? dA_dBC_val.y : dA_dBC_val.y + smem_dbc[state_idx];
                }
            } else {
                dA_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dA_val);
            }
            if (threadIdx.x == 0) {
                smem_da[state_idx] = chunk == ch_n_chunks - 1 ? dA_val : dA_val + smem_da[state_idx];
            }
        }

        // ── Step 6: ddelta → ddt_idle / ddt_active (replaces ddelta_bias) ───
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            ddt_idle_val   += ddelta_vals[i] * float(k_vals[i] - 1);
            ddt_active_val += ddelta_vals[i];
        }

        // ── Step 7: Write du (no ddelta output — delta is not external) ─────
        input_t *du = reinterpret_cast<input_t *>(params.du_ptr) + ev_start + chunk * kChunkSize;
        __syncthreads();
        store_output<Ktraits>(du, du_vals, smem_store, remaining);

        // ── Step 8: Update pointers (reverse traversal) ─────────────────────
        u    -= kChunkSize;
        k    -= kChunkSize;
        dout -= kChunkSize;
        // Tiled path uses tile_base_idx + chunk, no pointer decrement needed.
        if (params.tile_offsets_ptr == nullptr) {
            Bvar_legacy -= kChunkSize;
            Cvar_legacy -= kChunkSize;
        }
    }

    // ── Loop epilogue: write accumulated gradients ──────────────────────────
    if (params.dD_ptr != nullptr) {
        dD_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dD_val);
        if (threadIdx.x == 0) { gpuAtomicAdd(dD, dD_val); }
    }
    // ddt_idle / ddt_active (replaces ddelta_bias)
    __syncthreads();
    ddt_idle_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(ddt_idle_val);
    if (threadIdx.x == 0) { gpuAtomicAdd(reinterpret_cast<float *>(params.ddt_idle_ptr) + dim_id, ddt_idle_val); }
    __syncthreads();
    ddt_active_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(ddt_active_val);
    if (threadIdx.x == 0) { gpuAtomicAdd(reinterpret_cast<float *>(params.ddt_active_ptr) + dim_id, ddt_active_val); }

    // dA / non-variable dB/dC — same as original bwd
    for (int state_idx = threadIdx.x; state_idx < params.dstate; state_idx += blockDim.x) {
        gpuAtomicAdd(&(dA[state_idx * params.dA_dstate_stride]), smem_da[state_idx]);
        weight_t dBC_val;
        if (!kIsVariableB || !kIsVariableC) { dBC_val = smem_dbc[state_idx]; }
        if constexpr (!kIsVariableB) {
            float C_val = reinterpret_cast<float *>(params.C_ptr)[state_idx * params.C_row_stride + dim_id];
            gpuAtomicAdd(&(dB[state_idx * params.dB_dstate_stride]),
                         !kIsVariableC ? dBC_val * C_val : dBC_val);
        }
        if constexpr (!kIsVariableC) {
            float B_val = reinterpret_cast<float *>(params.B_ptr)[state_idx * params.B_row_stride + dim_id];
            gpuAtomicAdd(&(dC[state_idx * params.dC_dstate_stride]),
                         !kIsVariableB ? dBC_val * B_val : dBC_val);
        }
        // ── Depthwise: atomicAdd dW_B and dW_C into [dim, dstate] tensors ────
        if (params.is_depthwise_bc && params.dW_B_ptr != nullptr) {
            if (!has_pw) {
                // Pure depthwise: smem_dw_b[s] = dL/dW_B[dim_id, s] directly.
                gpuAtomicAdd(reinterpret_cast<float *>(params.dW_B_ptr) + dim_id * params.dstate + state_idx,
                             smem_dw_b[state_idx]);
                gpuAtomicAdd(reinterpret_cast<float *>(params.dW_C_ptr) + dim_id * params.dstate + state_idx,
                             smem_dw_c[state_idx]);
            } else {
                // Depthwise + pointwise:
                //   smem_dw_b[s]  =  dL/dW_B_eff[s]   (gradient wrt the effective weight)
                //
                // Chain rule back to W_B and pw_B:
                //   dL/dW_B[d', s] += pw_B[d, d'] * smem_dw_b[s]   (for ALL d, not just dim_id)
                //   dL/dpw_B[d, d'] = Σ_s  smem_dw_b[s] * W_B[d', s]
                //
                // We write dW_B[d', s] via atomicAdd (many blocks share d').
                // We write dpw_B[d, d'] = Σ_s only from this block (one block owns dim_id=d),
                //   so we accumulate across s in smem and atomicAdd once per d'.
                const float dW_B_eff_s = smem_dw_b[state_idx];
                const float dW_C_eff_s = smem_dw_c[state_idx];
                const float *W_B_base  = reinterpret_cast<const float *>(params.W_B_ptr);
                const float *W_C_base  = reinterpret_cast<const float *>(params.W_C_ptr);
                float       *dW_B_base = reinterpret_cast<float *>(params.dW_B_ptr);
                float       *dW_C_base = reinterpret_cast<float *>(params.dW_C_ptr);
                const float *pw_B_row  = reinterpret_cast<const float *>(params.pw_B_ptr)
                                          + (long)dim_id * params.dim;
                const float *pw_C_row  = reinterpret_cast<const float *>(params.pw_C_ptr)
                                          + (long)dim_id * params.dim;

                // dW_B[d', state_idx] += pw_B[dim_id, d'] * dW_B_eff[state_idx]
                // Each thread owns one state_idx; loop over d'.
                for (int d2 = 0; d2 < params.dim; ++d2) {
                    gpuAtomicAdd(dW_B_base + (long)d2 * params.dstate + state_idx,
                                 pw_B_row[d2] * dW_B_eff_s);
                    gpuAtomicAdd(dW_C_base + (long)d2 * params.dstate + state_idx,
                                 pw_C_row[d2] * dW_C_eff_s);
                }

                // dpw_B[dim_id, d'] = Σ_s  dW_B_eff[s] * W_B[d', s]
                // Each state_idx thread owns one s; write directly to global.
                // No smem accumulator needed — dim is unbounded by MAX_DSTATE.
                if (params.dpw_B_ptr != nullptr) {
                    for (int d2 = 0; d2 < params.dim; ++d2) {
                        const float w_b = W_B_base[(long)d2 * params.dstate + state_idx];
                        const float w_c = W_C_base[(long)d2 * params.dstate + state_idx];
                        gpuAtomicAdd(params.dpw_B_ptr + (long)dim_id * params.dim + d2,
                                     dW_B_eff_s * w_b);
                        gpuAtomicAdd(params.dpw_C_ptr + (long)dim_id * params.dim + d2,
                                     dW_C_eff_s * w_c);
                    }
                }
            }
        }
    }   // end: for state_idx
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// ── Flat per-channel backward launch helpers ────────────────────────────────
////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_flat_ch_bwd_launch(SSMParamsFlatSegBwd &params, cudaStream_t stream) {
    BOOL_SWITCH(params.is_variable_B, kIsVariableB, [&] {
        BOOL_SWITCH(params.is_variable_C, kIsVariableC, [&] {
            BOOL_SWITCH(params.z_ptr != nullptr, kHasZ, [&] {
                using Ktraits = Selective_Scan_flat_ch_bwd_kernel_traits<
                    kNThreads, kNItems, kIsVariableB, kIsVariableC, kHasZ,
                    input_t, weight_t>;

                constexpr int kSmemSize = Ktraits::kSmemSize
                    + MAX_DSTATE * sizeof(typename Ktraits::scan_t)
                    + (kNThreads + 8 * MAX_DSTATE) * sizeof(typename Ktraits::weight_t);
                    // Layout after Ktraits::kSmemSize (weight_t units):
                    //   smem_delta_a      : 2*MAX_DSTATE + kNThreads
                    //   smem_running_postfix: MAX_DSTATE scan_t = 2*MAX_DSTATE wt
                    //   smem_da / smem_dbc: 2 * MAX_DSTATE
                    //   smem_dw_b / dw_c  : 2 * MAX_DSTATE  (depthwise grads)
                    //   smem_W_B/C_eff    : 2 * MAX_DSTATE  (pw effective weight cache)
                    //   total extra wt    : 2+2+2+2 = 8 * MAX_DSTATE
                    // dpw_B/C go directly to global via gpuAtomicAdd — no smem slot.

                dim3 grid(params.n_channels);

                auto kernel = &selective_scan_flat_ch_bwd_kernel<Ktraits>;
                if (kSmemSize >= 48 * 1024) {
                    #ifndef USE_ROCM
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                    #else
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        (void *) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                    #endif
                }
                kernel<<<grid, kNThreads, kSmemSize, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

template<typename input_t, typename weight_t>
void selective_scan_bwd_flat_ch_cuda(SSMParamsFlatSegBwd &params, cudaStream_t stream) {
    // Must match the forward dispatch to keep kChunkSize == tile_chunk_size
    // when tiled B/C layout is used.
    if (params.max_seqlen <= 128) {
        selective_scan_flat_ch_bwd_launch<32,  4, input_t, weight_t>(params, stream);
    } else if (params.max_seqlen <= 256) {
        selective_scan_flat_ch_bwd_launch<32,  8, input_t, weight_t>(params, stream);
    } else if (params.max_seqlen <= 512) {
        selective_scan_flat_ch_bwd_launch<32, 16, input_t, weight_t>(params, stream);
    } else if (params.max_seqlen <= 1024) {
        selective_scan_flat_ch_bwd_launch<64, 16, input_t, weight_t>(params, stream);
    } else {
        selective_scan_flat_ch_bwd_launch<128,16, input_t, weight_t>(params, stream);
    }
}
