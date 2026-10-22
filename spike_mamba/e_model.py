# Copyright (c) 2023, Tri Dao, Albert Gu.
"""Flat event-driven Spike-Mamba core used by this submission.

Only the flat-channel Mamba implementation, its spike sampler, fused event
compaction, selective scan dispatch, and zero-order-hold restoration live in
this module.  Dense and alternative Mamba variants are intentionally absent.
"""

import math
import os
import time
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from .runtime_breakdown import cuda_stage
except ImportError:
    from runtime_breakdown import cuda_stage



class CudaTimer:
    _global_stats: dict = {}

    def __init__(self, name: str, enabled: bool = True, reset_peak: bool = True):
        self.name = name
        self.enabled = enabled
        self.reset_peak = reset_peak
        self.elapsed_ms: float = 0.0
        self.peak_mb: float = 0.0
        self._start = None
        self._end = None

    def __enter__(self):
        if self.enabled and torch.cuda.is_available():
            if self.reset_peak:
                torch.cuda.reset_peak_memory_stats()
            self._start = torch.cuda.Event(enable_timing=True)
            self._end   = torch.cuda.Event(enable_timing=True)
            self._start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and torch.cuda.is_available() and self._start is not None:
            self._end.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self._start.elapsed_time(self._end)
            self.peak_mb    = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  [PROFILE] [{self.name:35s}] {self.elapsed_ms:8.3f} ms | "
                  f"peak_alloc={self.peak_mb:7.1f} MB")
            stats = CudaTimer._global_stats.setdefault(self.name, [])
            stats.append((self.elapsed_ms, self.peak_mb))
        return False

    @classmethod
    def summary(cls):
        if not cls._global_stats:
            return
        print("\n" + "=" * 70)
        print(f"  {'Tag':<35} {'avg_ms':>9} {'avg_peak_MB':>12} {'calls':>6}")
        print("=" * 70)
        for tag, records in sorted(cls._global_stats.items()):
            avg_ms   = sum(r[0] for r in records) / len(records)
            avg_peak = sum(r[1] for r in records) / len(records)
            print(f"  {tag:<35} {avg_ms:9.3f} {avg_peak:12.1f} {len(records):6d}")
        print("=" * 70 + "\n")

    @classmethod
    def reset(cls):
        cls._global_stats.clear()


def _profiling_enabled() -> bool:
    return os.environ.get("MAMBA_PROFILE", "0") == "1"

from selective_scan_interface import selective_scan_flat_ch_fn, selective_scan_depthwise_bc_fn


# -- simple-scan backend (mamba_ssm standard kernel) ---------------------------
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn, mamba_inner_fn = None, None

# Fused CUDA event-extraction extension (extract_events_cuda/extract_events_cuda.cu)
# Falls back gracefully to pure-PyTorch if the extension is not built yet.
try:
    import extract_events_cuda as _ee_cuda
    _HAS_EE_CUDA = True
except ImportError:
    _ee_cuda = None
    _HAS_EE_CUDA = False

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

try:
    import restore_dense_cuda_nodecay
except ImportError:
    restore_dense_cuda_nodecay = None

import sys

class RestoreDenseFn(torch.autograd.Function):
    """
    Zero-order-hold restore: non-event positions carry the last event value
    forward unchanged (no exponential decay).  log_gamma is removed entirely.

    forward inputs : y_flat, time_steps_i32, cu_seqlens, dim, seqlen
    forward output : (B, L, D) tensor
    backward output: (dy_flat, None, None, None, None)
    """
    @staticmethod
    def forward(ctx, y_flat, time_steps_i32, cu_seqlens, dim: int, seqlen: int):
        if restore_dense_cuda_nodecay is not None:
            out = restore_dense_cuda_nodecay.restore_dense_fwd(
                y_flat, time_steps_i32, cu_seqlens, dim, seqlen
            )
        else:
            raise RuntimeError("restore_dense_cuda_nodecay not available")
        ctx.save_for_backward(time_steps_i32, cu_seqlens)
        ctx.dim    = dim
        ctx.seqlen = seqlen
        return out

    @staticmethod
    def backward(ctx, dout):
        time_steps_i32, cu_seqlens = ctx.saved_tensors
        dout_c = dout.contiguous()
        if restore_dense_cuda_nodecay is not None:
            with cuda_stage("restore_backward"):
                dy = restore_dense_cuda_nodecay.restore_dense_bwd(
                    dout_c, time_steps_i32, cu_seqlens, ctx.dim, ctx.seqlen
                )
        else:
            raise RuntimeError("restore_dense_cuda_nodecay not available")
        # gradients for: y_flat, time_steps_i32, cu_seqlens, dim, seqlen
        return dy, None, None, None, None


def restore_dense_fn(y_flat, time_steps_i32, cu_seqlens, dim: int, seqlen: int):
    return RestoreDenseFn.apply(y_flat, time_steps_i32, cu_seqlens, dim, seqlen)


# ---------------------------------------------------------------------------
# GatherEventsFn  - fused differentiable gather for x_conv
# ---------------------------------------------------------------------------
class GatherEventsFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_conv, channel_ids_i32, time_steps_i32):
        x_c = x_conv.contiguous()
        out_x = _ee_cuda.gather_events_fwd(
            x_c, channel_ids_i32, time_steps_i32
        )[0]
        ctx.save_for_backward(channel_ids_i32, time_steps_i32)
        ctx.shape = x_c.shape   # (B, D, L)
        return out_x

    @staticmethod
    def backward(ctx, grad_x):
        channel_ids_i32, time_steps_i32 = ctx.saved_tensors
        B, D, L = ctx.shape
        NC = B * D
        with cuda_stage("gather_scatter_backward"):
            dx_flat = _ee_cuda.gather_events_bwd(
                grad_x.contiguous(),
                channel_ids_i32, time_steps_i32,
                NC, L,
            )[0]
        return dx_flat.view(B, D, L), None, None


def _gather_events_cuda(x_conv, channel_ids_i32, time_steps_i32):
    return GatherEventsFn.apply(x_conv, channel_ids_i32, time_steps_i32)


def _gather_events_python(x_conv, channel_ids_i32, time_steps_i32):
    B, D, L = x_conv.shape
    idx_ch = channel_ids_i32.long()
    idx_t  = time_steps_i32.long()
    x_flat = x_conv.reshape(B * D, L)[idx_ch, idx_t]
    return x_flat


_HAS_GATHER_CUDA = (
    _HAS_EE_CUDA and hasattr(_ee_cuda, "gather_events_fwd")
)
_gather_events_fn = _gather_events_cuda if _HAS_GATHER_CUDA else _gather_events_python


class TwoLevelDelta(nn.Module):
    def __init__(self, d_inner: int, dt_min=0.001, dt_max=0.1,
                 dt_init_floor=1e-4, eps=1e-6, device=None):
        super().__init__()
        self.eps = eps
        scale = math.log(dt_max) - math.log(dt_min)
        rand0 = torch.rand(d_inner, device=device) * scale + math.log(dt_min)
        rand1 = torch.rand(d_inner, device=device) * scale + math.log(dt_min)
        dt0 = torch.exp(rand0).clamp(min=dt_init_floor)
        dt1 = torch.exp(rand1).clamp(min=dt_init_floor)

        def inv_sp(y, eps=1e-6):
            y = y.clamp_min(eps)
            return torch.log(torch.expm1(y))
        self.a0 = nn.Parameter(inv_sp(dt0))
        self.a1 = nn.Parameter(inv_sp(dt1))

    def levels(self):
        dt_idle = F.softplus(self.a0) + self.eps
        dt_active = F.softplus(self.a1) + self.eps
        return dt_idle, dt_active


class SpikeSamplerSurrogate(nn.Module):
    def __init__(self, mode: str = "ste", sample_inference: bool = False, threshold: float = 0.5):
        super().__init__()
        assert mode in ("prob", "ste")
        self.mode = mode
        self.sample_inference = sample_inference
        self.threshold = threshold

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        if self.mode == "prob":
            if self.training:
                s = torch.bernoulli(p)
                return s.detach() + p - p.detach()
            else:
                if self.sample_inference:
                    return torch.bernoulli(p)
                else:
                    return (p >= self.threshold).to(dtype=p.dtype)

        # STE mode
        s = (p >= self.threshold).to(dtype=p.dtype)
        return s.detach() + p - p.detach()



class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,
        spiking=True,
        spike_mode="ste",
        prob_activation="sigmoid",
        sample_inference=False,
        threshold=0.5,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self._profiling: bool = True

        self.spiking = spiking
        assert spike_mode in ("ste", "prob")
        assert prob_activation in ("sigmoid", "clamp")
        self.spike_mode = spike_mode
        self.prob_activation = prob_activation
        self.sample_inference = sample_inference
        self.threshold = threshold
        self.scan_impl = "flat_ch"

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        fp32_kwargs = {"device": device, "dtype": torch.float32}

        self.W_BC = nn.Parameter(
            torch.empty(self.d_inner, self.d_state * 2, **fp32_kwargs)
        )
        nn.init.normal_(self.W_BC, std=0.02)

        self.D = nn.Parameter(torch.ones(self.d_inner, **fp32_kwargs))
        self.D._no_weight_decay = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.prob_head = nn.Linear(self.d_model, self.d_model, bias=True, **factory_kwargs)
        self.spike_sampler = SpikeSamplerSurrogate(
            mode=self.spike_mode,
            sample_inference=self.sample_inference,
            threshold=self.threshold,
        )

        self.delta_gen = TwoLevelDelta(self.d_inner, dt_min=dt_min, dt_max=dt_max,
                                       dt_init_floor=dt_init_floor, device=device)

    def to_spike(self, out: torch.Tensor) -> torch.Tensor:
        logits = self.prob_head(out)
        if self.prob_activation == "sigmoid":
            p = torch.sigmoid(logits)
        else:
            p = torch.clamp(logits, 0.0, 1.0)
        return self.spike_sampler(p)

    def _to_spike(self, out: torch.Tensor) -> torch.Tensor:
        if not self.spiking:
            return out
        return self.to_spike(out)

    def extract_events_1d(
        self,
        hidden_states: torch.Tensor,
        x_conv:        torch.Tensor,
        threshold:     float,
    ):
        assert _HAS_EE_CUDA, (
            "extract_events_cuda extension not found. "
            "Build it with: python setup.py install"
        )

        B, L, d_model = hidden_states.shape
        assert d_model == self.d_model, (
            f"hidden_states last dim {d_model} != self.d_model {self.d_model}"
        )
        base_input = rearrange(hidden_states, "b l d -> b d l").contiguous()
        with cuda_stage("routing_index_extraction"):
            (base_ch_ids,
             base_t_steps,
             base_k_flat,
             base_cu_seqlens,
             _base_counts,
             base_max_seqlen) = _ee_cuda.extract_base_events(
                base_input, threshold
            )

        with cuda_stage("active_token_packing"):
            (channel_ids,
             time_steps,
             k_flat,
             cu_seqlens,
             max_seqlen_t) = _ee_cuda.expand_base_events(
                base_ch_ids,
                base_t_steps,
                base_k_flat,
                base_cu_seqlens,
                B,
                self.d_model,
                self.expand,
            )
            max_seqlen = int(max_seqlen_t.item())

        with cuda_stage("active_gather_forward"):
            x_conv_flat = _gather_events_fn(
                x_conv, channel_ids, time_steps
            )

        return (channel_ids, time_steps, k_flat, cu_seqlens, max_seqlen,
                 x_conv_flat)

    def enable_profiling(self, flag: bool = True):
        self._profiling = flag
        return self

    def _prof(self, name: str) -> CudaTimer:
        enabled = self._profiling or _profiling_enabled()
        return CudaTimer(name, enabled=enabled)

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        x = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            x = x + rearrange(self.in_proj.bias.to(dtype=x.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())

        if conv_state is not None:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))

        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        (channel_ids, time_steps,
            k_flat_i32, cu_seqlens_i32,
            max_seqlen,
            x_conv_flat) = self.extract_events_1d(
            hidden_states=hidden_states,
            x_conv=x,
            threshold=self.threshold,
        )
        channel_ids_i32 = channel_ids

        dt_idle_vec, dt_active_vec = self.delta_gen.levels()

        D_vec = self.D
        A     = -torch.exp(self.A_log)

        W_B = self.W_BC[:, :self.d_state].contiguous()
        W_C = self.W_BC[:, self.d_state:].contiguous()

        time_steps_i32 = time_steps.to(torch.int32)

        with cuda_stage("sparse_scan_forward"):
            y_flat, _ = selective_scan_flat_ch_fn(
                x_conv_flat,
                k_flat_i32,
                cu_seqlens_i32,
                int(max_seqlen),
                dt_idle_vec,
                dt_active_vec,
                A,
                W_B,
                W_C,
                D_vec,
                None,
                tile_offsets=None,
                tile_chunk_size=0,
                is_depthwise_bc=True,
                pw_B=None,
                pw_C=None,
            )

        if ssm_state is not None:
            pass

        time_steps_i32 = time_steps.to(torch.int32)

        with cuda_stage("dense_restoration_forward"):
            y_dense = restore_dense_fn(
                y_flat, time_steps_i32, cu_seqlens_i32, self.d_inner, seqlen
            )
        y_full = y_dense

        out = self.out_proj(y_full)

        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        x = self.in_proj(hidden_states.squeeze(1))

        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x, conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias, self.activation,
            )

        W_B = self.W_BC[:, :self.d_state]
        W_C = self.W_BC[:, self.d_state:]

        pw_B_w = self.pw_B.to(dtype=x.dtype)
        pw_C_w = self.pw_C.to(dtype=x.dtype)

        W_B_eff = pw_B_w @ W_B.to(dtype=x.dtype)
        W_C_eff = pw_C_w @ W_C.to(dtype=x.dtype)

        B_mat = x.unsqueeze(-1) * W_B_eff.unsqueeze(0)
        C_mat = x.unsqueeze(-1) * W_C_eff.unsqueeze(0)

        dt_idle, dt_active = self.delta_gen.levels()
        dt_idle   = dt_idle.to(dtype=x.dtype)
        dt_active = dt_active.to(dtype=x.dtype)

        hs_curr = hidden_states.squeeze(1)
        if hs_curr.shape[-1] != self.d_inner:
            hs_expanded = hs_curr.repeat_interleave(self.expand, dim=-1)
        else:
            hs_expanded = hs_curr

        has_spike = (hs_expanded.abs() > self.threshold)
        dt = torch.where(has_spike, dt_active.unsqueeze(0), dt_idle.unsqueeze(0))
        A  = -torch.exp(self.A_log.float())

        if selective_state_update is None:
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = dt.unsqueeze(-1) * B_mat
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = (ssm_state.to(dtype) * C_mat).sum(-1)
            y = y + self.D.to(dtype) * x
        else:
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = dt.unsqueeze(-1) * B_mat
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = (ssm_state.to(dtype) * C_mat).sum(-1)
            y = y + self.D.to(dtype) * x

        out = self.out_proj(y)
        out = self._to_spike(out)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv,
            device=device, dtype=conv_dtype
        )
        ssm_dtype = self.A_log.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state,
            device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size, self.d_model * self.expand, self.d_conv,
                device=self.conv1d.weight.device, dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size, self.d_model * self.expand, self.d_state,
                device=self.A_log.device, dtype=self.A_log.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

