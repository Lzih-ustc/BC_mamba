# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F
from mamba_ssm.utils.torch import custom_bwd, custom_fwd

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.cpp_functions import causal_conv1d_fwd_function, causal_conv1d_bwd_function, causal_conv1d_update_function
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_fwd_function = None
    causal_conv1d_bwd_function = None
    causal_conv1d_update_function = None

from mamba_ssm.ops.triton.layer_norm import _layer_norm_fwd

import selective_scan_cuda
try:
    from .runtime_breakdown import cuda_stage
except ImportError:
    from runtime_breakdown import cuda_stage

# The LRA submission only uses the flat-channel backend.  Keep the legacy
# dense/event extension optional so importing this module never requires it.
try:
    import selective_scan_cuda_raw
except ImportError:
    selective_scan_cuda_raw = None

class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False, event_mode=False, k=None, valid=None,
                dt_idle=None, dt_active=None, A_idle=None, A_active=None):
        """
        Standard dense selective scan forward.

        If event_mode=True, uses event-driven kernel with:
            - k: (B, D, Nmax) int32, event intervals
            - valid: (B, D, Nmax) bool, valid mask
            - dt_idle, dt_active: (D,) float32, two-level dt
            - A_idle, A_active: (D, dstate) float32, two-level A

        2D input support (Option A):
            u and delta may be 2D (n_blocks, K) instead of 3D (n_blocks, dim, K).
            They are internally unsqueeze'd to 3D before the CUDA call and the
            outputs are squeeze'd back.  This is applied to the standard dense
            path only; the event_mode path is unaffected.
        """
        if event_mode:
            # ── Event-driven path (unchanged) ─────────────────────────────────
            assert k is not None and valid is not None, "event_mode requires k and valid"
            assert dt_idle is not None and dt_active is not None, "event_mode requires dt_idle and dt_active"
            assert A_idle is not None and A_active is not None, "event_mode requires A_idle and A_active"

            # Ensure contiguous
            if u.stride(-1) != 1:
                u = u.contiguous()
            if k.stride(-1) != 1:
                k = k.contiguous()
            if valid.stride(-1) != 1:
                valid = valid.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if z is not None and z.stride(-1) != 1:
                z = z.contiguous()
            if D is not None:
                D = D.contiguous()

            # Reshape B/C if needed
            if B.dim() == 3:
                B = rearrange(B, "b dstate l -> b 1 dstate l")
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = rearrange(C, "b dstate l -> b 1 dstate l")
                ctx.squeeze_C = True

            # [DEBUG A] Check dt ranges and finiteness before forward
            print("[Selective Scan Event Mode Forward Debug]")
            print(f"  dt_idle: min={dt_idle.min():.6f}, max={dt_idle.max():.6f}")
            print(f"  dt_active: min={dt_active.min():.6f}, max={dt_active.max():.6f}")
            print(f"  isfinite(dt_idle): {torch.isfinite(dt_idle).all()}")
            print(f"  isfinite(dt_active): {torch.isfinite(dt_active).all()}")

            # Call event-driven kernel
            out, x, *rest = selective_scan_cuda_raw.fwd_event(
                u, k, valid, dt_idle, dt_active, A_idle, A_active, B, C, D, z
            )

            ctx.event_mode = True
            ctx.has_z = z is not None
            # For event mode: x shape is (B, D, 1, dstate) for real-only
            last_state = x[:, :, 0, :] if x is not None else None  # Extract (B, D, dstate)

            if not ctx.has_z:
                ctx.save_for_backward(u, k, valid, dt_idle, dt_active, A_idle, A_active, B, C, D, x)
                return out if not return_last_state else (out, last_state)
            else:
                ctx.save_for_backward(u, k, valid, dt_idle, dt_active, A_idle, A_active, B, C, D, z, x, out)
                out_z = rest[0]
                return out_z if not return_last_state else (out_z, last_state)
        else:
            # ── Standard dense path ───────────────────────────────────────────

            # ── Option A: normalize 2D inputs ─────────────────────────────────
            input_is_2d = (u.dim() == 2)
            if input_is_2d:
                u     = u.unsqueeze(1)      # (n_blocks, K) → (n_blocks, 1, K)
                delta = delta.unsqueeze(1)  # (n_blocks, K) → (n_blocks, 1, K)
                if B.dim() == 3:            # variable B: (n_blocks, dstate, K) → (n_blocks, 1, dstate, K)
                    B = B.unsqueeze(1)
                if C.dim() == 3:
                    C = C.unsqueeze(1)
                if z is not None and z.dim() == 2:
                    z = z.unsqueeze(1)
            ctx.input_is_2d = input_is_2d
            # ──────────────────────────────────────────────────────────────────

            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if z is not None and z.stride(-1) != 1:
                z = z.contiguous()
            if B.dim() == 3:
                B = rearrange(B, "b dstate l -> b 1 dstate l")
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = rearrange(C, "b dstate l -> b 1 dstate l")
                ctx.squeeze_C = True

            out, x, *rest = selective_scan_cuda_raw.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
            ctx.event_mode = False
            ctx.delta_softplus = delta_softplus
            ctx.has_z = z is not None
            last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
            if not ctx.has_z:
                ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            else:
                ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)

            # ── Option A: restore 2D output shape ─────────────────────────────
            if input_is_2d:
                out = out.squeeze(1)        # (n_blocks, 1, K) → (n_blocks, K)
                if ctx.has_z:
                    rest[0] = rest[0].squeeze(1)
            # ──────────────────────────────────────────────────────────────────

            if not ctx.has_z:
                return out if not return_last_state else (out, last_state)
            else:
                out_z = rest[0]
                return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if getattr(ctx, "event_mode", False):
            # ── Event-driven backward (unchanged) ─────────────────────────────
            if not ctx.has_z:
                u, k, valid, dt_idle, dt_active, A_idle, A_active, B, C, D, x = ctx.saved_tensors
                z = None
                out = None
            else:
                u, k, valid, dt_idle, dt_active, A_idle, A_active, B, C, D, z, x, out = ctx.saved_tensors
            
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            
            # ========= 🔥 Priority 1: Check dout at backward entry ==========
            print("\n[GRADIENT STATS] selective_scan INPUT (dout, from loss):")
            dout_absmax = dout.abs().max().item() if dout.numel() > 0 else 0.0
            dout_mean = dout.abs().mean().item() if dout.numel() > 0 else 0.0
            dout_is_zero = (dout.abs() < 1e-8).all().item()
            dout_has_nan = torch.isnan(dout).any().item()
            dout_nonzero = (dout.abs() > 1e-10).long().sum().item()
            print(f"  shape: {dout.shape}")
            print(f"  absmax: {dout_absmax:.8e}")
            print(f"  mean: {dout_mean:.8e}")
            print(f"  nonzero_ratio: {dout_nonzero/dout.numel() if dout.numel() > 0 else 0:.4f}")
            print(f"  has_nan: {dout_has_nan}")
            if dout_absmax < 1e-8:
                print("  ⚠️  WARNING: dout is nearly zero → check: spike_sampler, prob_head, or loss")
            
            # ========= 🔥 Priority 2: Check event structure ==========
            print("\n[EVENT STRUCTURE DIAGNOSTICS]")
            valid_count = valid.long().sum().item()
            u_absmax = u.abs().max().item() if u.numel() > 0 else 0.0
            u_is_zero = (u.abs() < 1e-8).all().item()
            k_min = k.float().min().item() if k.numel() > 0 else 0.0
            k_max = k.float().max().item() if k.numel() > 0 else 0.0
            print(f"  valid_bdn.sum(): {valid_count} (total slots: {valid.numel()})")
            print(f"  u.absmax: {u_absmax:.8f} | is_nearly_zero: {u_is_zero}")
            print(f"  k range: [{k_min:.2f}, {k_max:.2f}]")
            
            if valid_count == 0:
                print("  ⚠️  WARNING: NO valid events → entire compression is zero")
            if u_is_zero:
                print("  ⚠️  WARNING: u ≈ 0 → no actual input to scan")
            
            result = selective_scan_cuda_raw.bwd_event(
                u, k, valid, dt_idle, dt_active, A_idle, A_active, B, C, D, z, dout, x
            )
            
            # ========= 🔥 Priority 3: Check kernel output du ==========
            du_kernel = result[0] if len(result) > 0 else None
            print("\n[GRADIENT STATS] selective_scan OUTPUT (du, to u_comp):")
            if du_kernel is None:
                print("  ❌ ERROR: du_kernel is None!")
            else:
                du_absmax = du_kernel.abs().max().item() if du_kernel.numel() > 0 else 0.0
                du_mean = du_kernel.abs().mean().item() if du_kernel.numel() > 0 else 0.0
                du_nonzero = (du_kernel.abs() > 1e-10).long().sum().item()
                print(f"  shape: {du_kernel.shape}")
                print(f"  absmax: {du_absmax:.8e}")
                print(f"  mean: {du_mean:.8e}")
                print(f"  nonzero_ratio: {du_nonzero/du_kernel.numel() if du_kernel.numel() > 0 else 0:.4f}")
                
                if dout_absmax > 1e-8 and du_absmax < 1e-8:
                    print("  ⚠️  PROBLEM: dout is large but du is nearly zero!")
                    print("     → Issue in selective_scan kernel or event structure")
            
            # [DEBUG B - COMPREHENSIVE] Check ALL gradient outputs from bwd_event
            print("\n[GRADIENT STATS] All outputs from backward kernel:")
            grad_names_core = ["du", "ddt_idle", "ddt_active", "dA_idle", "dA_active", "dB", "dC"]
            for idx, name in enumerate(grad_names_core):
                if idx < len(result):
                    t = result[idx]
                    if t is None:
                        print(f"  {name:15s}: None")
                    else:
                        t_absmax = t.abs().max().item() if t.numel() > 0 else 0.0
                        t_mean = t.abs().mean().item() if t.numel() > 0 else 0.0
                        is_finite = torch.isfinite(t).all().item()
                        status = "✓" if is_finite else "✗ NON-FINITE"
                        print(f"  {name:15s}: {status} | absmax={t_absmax:.8e}, mean={t_mean:.8e}")
            
            result_idx = 7
            if D is not None and result_idx < len(result):
                t = result[result_idx]
                if t is None:
                    print(f"  {'dD':15s}: None")
                else:
                    t_absmax = t.abs().max().item() if t.numel() > 0 else 0.0
                    t_mean = t.abs().mean().item() if t.numel() > 0 else 0.0
                    is_finite = torch.isfinite(t).all().item()
                    status = "✓" if is_finite else "✗ NON-FINITE"
                    print(f"  {'dD':15s}: {status} | absmax={t_absmax:.8e}, mean={t_mean:.8e}")
                result_idx += 1
            
            if ctx.has_z and result_idx < len(result):
                t = result[result_idx]
                if t is None:
                    print(f"  {'dz':15s}: None")
                else:
                    t_absmax = t.abs().max().item() if t.numel() > 0 else 0.0
                    t_mean = t.abs().mean().item() if t.numel() > 0 else 0.0
                    is_finite = torch.isfinite(t).all().item()
                    status = "✓" if is_finite else "✗ NON-FINITE"
                    print(f"  {'dz':15s}: {status} | absmax={t_absmax:.8e}, mean={t_mean:.8e}")
            
            du = result[0]
            ddt_idle = result[1]
            ddt_active = result[2]
            dA_idle = result[3]
            dA_active = result[4]
            dB = result[5]
            dC = result[6]
            
            result_idx = 7
            dD = result[result_idx] if D is not None else None
            if D is not None:
                result_idx += 1
            
            dz = result[result_idx] if ctx.has_z else None
            
            dk = None
            dvalid = None
            
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            
            return (du, None, None, dB, dC,
                    dD if D is not None else None,
                    dz,
                    None, None,
                    None,
                    None,
                    dk, dvalid,
                    ddt_idle, ddt_active,
                    dA_idle, dA_active)
        else:
            # ── Standard dense backward ───────────────────────────────────────
            if not ctx.has_z:
                u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
                z = None
                out = None
            else:
                u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()

            # ── Option A: normalize 2D gradient inputs ────────────────────────
            if ctx.input_is_2d:
                dout = dout.unsqueeze(1)    # (n_blocks, K) → (n_blocks, 1, K)
            # ──────────────────────────────────────────────────────────────────

            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_raw.bwd(
                u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
                False  # option to recompute out_z, not used here
            )
            dz = rest[0] if ctx.has_z else None

            # ── Option A: restore 2D gradient output shapes ──────────────────
            if ctx.input_is_2d:
                du     = du.squeeze(1)      # (n_blocks, 1, K) → (n_blocks, K)
                ddelta = ddelta.squeeze(1)
                if dB.dim() == 4:           # variable B: (n_blocks, 1, dstate, K) → (n_blocks, dstate, K)
                    dB = dB.squeeze(1)
                if dC.dim() == 4:
                    dC = dC.squeeze(1)
                if dz is not None and dz.dim() == 3:
                    dz = dz.squeeze(1)
            # ──────────────────────────────────────────────────────────────────

            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC,
                    dD if D is not None else None,
                    dz,
                    ddelta_bias if delta_bias is not None else None,
                    None,   # delta_softplus
                    None,   # return_last_state
                    None,   # event_mode
                    None, None, None, None, None, None)  # all new event parameters


class SelectiveScanEventFlatFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u_flat, k_flat, cu_seqlens,
                dt_idle, dt_active, A_idle, A_active, B, C,
                D=None, z=None,
                return_last_state=False,
                channel_ids=None, head_flags=None, tail_flags=None,
                use_segmented=False):

        if return_last_state:
            raise RuntimeError("event_flat wrapper does not support return_last_state=True for now")

        # contiguous
        u_flat = u_flat.contiguous()
        k_flat = k_flat.contiguous()
        cu_seqlens = cu_seqlens.contiguous()
        dt_idle = dt_idle.contiguous()
        dt_active = dt_active.contiguous()
        A_idle = A_idle.contiguous()
        A_active = A_active.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        if z is not None:
            z = z.contiguous()
        if channel_ids is not None:
            channel_ids = channel_ids.contiguous()
        if head_flags is not None:
            head_flags = head_flags.contiguous()
        if tail_flags is not None:
            tail_flags = tail_flags.contiguous()

        # C++ returns: out_flat, x, [out_z_flat], x_hist
        result = selective_scan_cuda.fwd_event_flat(
            u_flat, k_flat, cu_seqlens,
            dt_idle, dt_active, A_idle, A_active, B, C,
            D, z, channel_ids, head_flags, tail_flags, use_segmented
        )

        out = result[0]
        x_last = result[1]      # (n_channels, dstate)
        if z is not None:
            out_z = result[2]
            x_hist = result[3]
        else:
            out_z = None
            x_hist = result[2]

        ctx.has_z = z is not None
        ctx.save_for_backward(
            u_flat, k_flat, cu_seqlens,
            dt_idle, dt_active, A_idle, A_active, B, C,
            x_hist,
            D if D is not None else torch.empty(0, device=u_flat.device, dtype=torch.float32),
            z if z is not None else torch.empty(0, device=u_flat.device, dtype=u_flat.dtype),
        )

        return out_z if z is not None else out

    @staticmethod
    def backward(ctx, dout_flat):
        if dout_flat.stride(-1) != 1:
            dout_flat = dout_flat.contiguous()

        (u_flat, k_flat, cu_seqlens,
         dt_idle, dt_active, A_idle, A_active, B, C,
         x_hist, D_saved, z_saved) = ctx.saved_tensors

        D = None if D_saved.numel() == 0 else D_saved
        z = None if z_saved.numel() == 0 else z_saved

        grads = selective_scan_cuda.bwd_event_flat(
            u_flat, k_flat, cu_seqlens,
            dt_idle, dt_active, A_idle, A_active, B, C,
            x_hist,
            D, z, dout_flat,
            None,   # head_flags must be None for old event_flat bwd
        )

        du = grads[0]
        ddt_idle = grads[1]
        ddt_active = grads[2]
        dA_idle = grads[3]
        dA_active = grads[4]
        dB = grads[5]
        dC = grads[6]

        idx = 7
        dD = grads[idx] if D is not None else None
        if D is not None:
            idx += 1
        dz = grads[idx] if ctx.has_z else None

        return (
            du,            # u_flat
            None,          # k_flat
            None,          # cu_seqlens
            ddt_idle,      # dt_idle
            ddt_active,    # dt_active
            dA_idle,       # A_idle
            dA_active,     # A_active
            dB,            # B
            dC,            # C
            dD,            # D
            dz,            # z
            None,          # return_last_state
            None,          # channel_ids
            None,          # head_flags
            None,          # tail_flags
            None,          # use_segmented
        )


def selective_scan_event_flat_fn(u_flat, k_flat, cu_seqlens,
                                 dt_idle, dt_active, A_idle, A_active, B, C,
                                 D=None, z=None,
                                 return_last_state=False,
                                 channel_ids=None, head_flags=None, tail_flags=None,
                                 use_segmented=False):
    return SelectiveScanEventFlatFn.apply(
        u_flat, k_flat, cu_seqlens,
        dt_idle, dt_active, A_idle, A_active, B, C,
        D, z, return_last_state, channel_ids, head_flags, tail_flags, use_segmented
    )


class SelectiveScanFlatSegFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u_flat, k_flat, cu_seqlens, max_seqlen,
                dt_idle, dt_active, A, B, C,
                D=None, z=None, head_flags=None, channel_ids=None):
        if head_flags is None or channel_ids is None:
            raise RuntimeError(
                "selective_scan_flat_seg_fn requires both head_flags and channel_ids."
            )
        if not u_flat.is_contiguous():
            u_flat = u_flat.contiguous()
        if not k_flat.is_contiguous():
            k_flat = k_flat.contiguous()
        if not cu_seqlens.is_contiguous():
            cu_seqlens = cu_seqlens.contiguous()
        if not dt_idle.is_contiguous():
            dt_idle = dt_idle.contiguous()
        if not dt_active.is_contiguous():
            dt_active = dt_active.contiguous()
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
        if not C.is_contiguous():
            C = C.contiguous()
        if D is not None and not D.is_contiguous():
            D = D.contiguous()
        if z is not None and not z.is_contiguous():
            z = z.contiguous()
        if head_flags is not None and not head_flags.is_contiguous():
            head_flags = head_flags.contiguous()
        if channel_ids is not None and not channel_ids.is_contiguous():
            channel_ids = channel_ids.contiguous()

        out, x, *rest = selective_scan_cuda.fwd_flat_seg(
            u_flat, k_flat, cu_seqlens, max_seqlen,
            dt_idle, dt_active, A, B, C,
            D, z, head_flags, channel_ids
        )

        ctx.has_z = z is not None
        ctx.max_seqlen = int(max_seqlen)
        ctx.save_for_backward(
            u_flat, k_flat, cu_seqlens,
            dt_idle, dt_active, A, B, C,
            D if D is not None else torch.empty(0, device=u_flat.device, dtype=torch.float32),
            z if z is not None else torch.empty(0, device=u_flat.device, dtype=u_flat.dtype),
            head_flags if head_flags is not None else torch.empty(0, device=u_flat.device, dtype=torch.int32),
            channel_ids if channel_ids is not None else torch.empty(0, device=u_flat.device, dtype=torch.int32),
            x,
            out,
        )

        if ctx.has_z:
            out_z = rest[0]
            return out_z
        return out

    @staticmethod
    def backward(ctx, dout_flat):
        if not dout_flat.is_contiguous():
            dout_flat = dout_flat.contiguous()

        (u_flat, k_flat, cu_seqlens,
         dt_idle, dt_active, A, B, C,
         D_saved, z_saved, head_saved, channel_saved,
         x, out) = ctx.saved_tensors

        D = None if D_saved.numel() == 0 else D_saved
        z = None if z_saved.numel() == 0 else z_saved
        head_flags = None if head_saved.numel() == 0 else head_saved
        channel_ids = None if channel_saved.numel() == 0 else channel_saved
        out_arg = out if ctx.has_z else None

        grads = selective_scan_cuda.bwd_flat_seg(
            u_flat, k_flat, cu_seqlens, ctx.max_seqlen,
            dt_idle, dt_active, A, B, C,
            dout_flat, x,
            D, z, out_arg, head_flags, channel_ids
        )

        du = grads[0]
        dA = grads[1]
        dB = grads[2]
        dC = grads[3]
        ddt_idle = grads[4]
        ddt_active = grads[5]

        idx = 6
        dD = grads[idx] if D is not None else None
        if D is not None:
            idx += 1
        dz = grads[idx] if ctx.has_z else None

        return (
            du,
            None,
            None,
            None,  # max_seqlen
            ddt_idle,
            ddt_active,
            dA,
            dB,
            dC,
            dD,
            dz,
            None,
            None,
        )


def selective_scan_flat_seg_fn(u_flat, k_flat, cu_seqlens, max_seqlen,
                               dt_idle, dt_active, A, B, C,
                               D=None, z=None, head_flags=None, channel_ids=None):
    return SelectiveScanFlatSegFn.apply(
        u_flat, k_flat, cu_seqlens, max_seqlen,
        dt_idle, dt_active, A, B, C,
        D, z, head_flags, channel_ids
    )


def rms_norm_forward(
    x,
    weight,
    bias,
    eps=1e-6,
    is_rms_norm=True,
):
    # x (b l) d
    if x.stride(-1) != 1:
        x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y = _layer_norm_fwd(
        x, weight, bias, eps, None, residual_dtype=None, is_rms_norm=is_rms_norm
    )[0]
    # y (b l) d
    return y



class SelectiveScanDepthwiseBCFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, W_B, W_C, D=None, z=None, delta_bias=None,
                delta_softplus=False, return_last_state=False):
        """
        Dense selective_scan with flat-style depthwise B/C parameterization.

        Semantics without materializing dense B/C:
            B_t[b, d, n, t] = u[b, d, t] * W_B[d, n]
            C_t[b, d, n, t] = u[b, d, t] * W_C[d, n]

        This calls selective_scan_cuda_raw.fwd_depthwise_bc / bwd_depthwise_bc.
        """
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if W_B.stride(-1) != 1:
            W_B = W_B.contiguous()
        if W_C.stride(-1) != 1:
            W_C = W_C.contiguous()
        if D is not None:
            D = D.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if delta_bias is not None:
            delta_bias = delta_bias.contiguous()

        out, x, *rest = selective_scan_cuda_raw.fwd_depthwise_bc(
            u, delta, A, W_B, W_C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        ctx.save_for_backward(u, delta, A, W_B, W_C, D, z, delta_bias, x, out)

        # Same convention as the original wrapper. In the depthwise path x stores
        # the scan prefix before the final W_B factor; this is sufficient for bwd.
        last_state = x[:, :, -1, 1::2]
        if not ctx.has_z:
            return out if not return_last_state else (out, last_state)
        else:
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, W_B, W_C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        dz = None
        du, ddelta, dA, dW_B, dW_C, dD, ddelta_bias, *rest = selective_scan_cuda_raw.bwd_depthwise_bc(
            u, delta, A, W_B, W_C, D, z, delta_bias, dout, x, out, dz,
            ctx.delta_softplus, False
        )
        dz = rest[0] if ctx.has_z else None
        return (du, ddelta, dA, dW_B, dW_C,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,  # delta_softplus
                None)  # return_last_state


def selective_scan_depthwise_bc_fn(u, delta, A, W_B, W_C, D=None, z=None,
                                   delta_bias=None, delta_softplus=False,
                                   return_last_state=False):
    """Dense selective_scan using flat-style depthwise B/C weights without
    materializing B/C tensors of shape (B, D, N, L)."""
    return SelectiveScanDepthwiseBCFn.apply(
        u, delta, A, W_B, W_C, D, z, delta_bias, delta_softplus, return_last_state
    )

def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False, event_mode=False, k=None, valid=None,
                     dt_idle=None, dt_active=None, A_idle=None, A_active=None):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.

    If event_mode=True, uses event-driven kernel with k, valid, and two-level dt/A parameters.

    2D input support (Option A):
        u/delta may be 2D (n_blocks, K); they are internally unsqueeze'd/squeeze'd.
        Applies to the standard dense path only.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, 
                                 return_last_state, event_mode, k, valid,
                                 dt_idle, dt_active, A_idle, A_active)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


class MambaInnerFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight=None, c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6, return_dt_win: bool = False):
        """
             xz: (batch, dim, seqlen)
        """
        assert causal_conv1d_fwd_function is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_fwd_function(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)

        dt_win = None
        if return_dt_win:
            delta_eff = delta
            if delta_bias is not None:
                delta_eff = delta_eff + delta_bias.to(dtype=delta_eff.dtype).view(1, -1, 1)
            if delta_softplus:
                delta_eff = F.softplus(delta_eff)
            dt_win = delta_eff.mean(dim=1).unsqueeze(-1)  # ✅ (B, L, 1)

        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
            
        if b_rms_weight is not None:
            B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            B = rms_norm_forward(B, b_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if c_rms_weight is not None:
            C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            C = rms_norm_forward(C, c_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if dt_rms_weight is not None:
            delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
            delta = rms_norm_forward(delta, dt_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()

        out, scan_intermediates, out_z = selective_scan_cuda_raw.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.b_rms_weight = b_rms_weight
        ctx.c_rms_weight = c_rms_weight
        ctx.dt_rms_weight = dt_rms_weight
        ctx.b_c_dt_rms_eps = b_c_dt_rms_eps
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out)
        out_final = F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
        if return_dt_win:
            ctx.mark_non_differentiable(dt_win)
            return out_final, dt_win
        return out_final

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, d_dt_win=None):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_fwd_function is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_fwd_function(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
            if dt_rms_weight is not None:
                delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
                delta = rms_norm_forward(delta, ctx.dt_rms_weight, None, ctx.b_c_dt_rms_eps)
                delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
            if b_rms_weight is not None:
                B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                B = rms_norm_forward(
                    B, ctx.b_rms_weight, None, ctx.b_c_dt_rms_eps
                )
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            if c_rms_weight is not None:
                C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                C = rms_norm_forward(
                    C, ctx.c_rms_weight, None, ctx.b_c_dt_rms_eps
                )
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda_raw.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_bwd_function(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias, None, None, None, None, None, None, None)


def mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight= None, c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6, return_dt_win: bool = False
):
    return MambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps, return_dt_win)


def mamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)

# ── Replacement for the original selective_scan_flat_ch_fn ──────────────────

class SelectiveScanFlatChFn(torch.autograd.Function):
    """
    Wraps selective_scan_cuda_raw.fwd_flat_ch / bwd_flat_ch into an autograd-aware
    Function so that training backward passes use the custom ch CUDA kernel.

    Forward input order (positional, for .apply()):
        u_flat, k_flat, cu_seqlens, dt_idle, dt_active, A, B, C, D, z, max_seqlen

    D and z may be None.  max_seqlen is a Python int (non-differentiable).
    k_flat / cu_seqlens are int32 tensors (non-differentiable).
    """

    @staticmethod
    def forward(ctx, u_flat, k_flat, cu_seqlens,
                dt_idle, dt_active, A, B, C,
                D, z, max_seqlen, tile_offsets, tile_chunk_size,
                is_depthwise_bc,
                pw_B, pw_C):
        # ── Contiguous guards ─────────────────────────────────────────────────
        u_flat     = u_flat.contiguous()
        k_flat     = k_flat.contiguous()
        cu_seqlens = cu_seqlens.contiguous()
        dt_idle    = dt_idle.contiguous()
        dt_active  = dt_active.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        if D is not None: D = D.contiguous()
        if z is not None: z = z.contiguous()
        if tile_offsets is not None: tile_offsets = tile_offsets.contiguous()
        if pw_B is not None: pw_B = pw_B.contiguous()
        if pw_C is not None: pw_C = pw_C.contiguous()

        result = selective_scan_cuda.fwd_flat_ch(
            u_flat, k_flat, cu_seqlens, max_seqlen,
            dt_idle, dt_active, A, B, C, D, z,
            tile_offsets, tile_chunk_size,
            is_depthwise_bc,
            pw_B, pw_C
        )
        out_flat = result[0]   # (E,)  — pre-gate output (always present)
        x        = result[1]   # (n_channels, n_chunks, dstate*2)  saved SSM state

        has_z  = z is not None
        has_D  = D is not None
        has_pw = pw_B is not None
        ctx.has_z  = has_z
        ctx.has_D  = has_D
        ctx.has_pw = has_pw
        ctx.max_seqlen      = max_seqlen
        ctx.tile_chunk_size = tile_chunk_size
        ctx.is_depthwise_bc = is_depthwise_bc

        # save_for_backward does not accept None; pack conditionally.
        # Fixed layout:
        #   [u_flat, k_flat, cu_seqlens, dt_idle, dt_active, A, B, C, x,
        #    D?          (if has_D)
        #    z, out_flat (if has_z)
        #    tile_offsets (if not None)
        #    pw_B, pw_C  (if has_pw)]
        tensors_to_save = [u_flat, k_flat, cu_seqlens,
                           dt_idle, dt_active, A, B, C, x]
        if has_D:
            tensors_to_save.append(D)
        if has_z:
            out_z_flat = result[2]          # (E,) — gated output
            tensors_to_save.extend([z, out_flat])
        ctx.has_tile_offsets = tile_offsets is not None
        if tile_offsets is not None:
            tensors_to_save.append(tile_offsets)
        if has_pw:
            tensors_to_save.extend([pw_B, pw_C])

        ctx.save_for_backward(*tensors_to_save)

        # x is the SSM state — not differentiable w.r.t. the loss
        ctx.mark_non_differentiable(x)

        primary_out = out_z_flat if has_z else out_flat
        return primary_out, x

    @staticmethod
    def backward(ctx, dout, _dx):
        """
        dout : gradient w.r.t. primary output (E,)
        _dx  : gradient w.r.t. x — ignored (marked non-differentiable)
        """
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        # ── Unpack saved tensors (mirroring the packing order above) ─────────
        saved = list(ctx.saved_tensors)
        u_flat, k_flat, cu_seqlens, dt_idle, dt_active, A, B, C, x = saved[:9]
        idx = 9
        D = saved[idx] if ctx.has_D else None
        if ctx.has_D: idx += 1
        z        = saved[idx]     if ctx.has_z else None
        if ctx.has_z: idx += 1
        out_flat = saved[idx]     if ctx.has_z else None
        if ctx.has_z: idx += 1
        tile_offsets = saved[idx] if ctx.has_tile_offsets else None
        if ctx.has_tile_offsets: idx += 1
        pw_B = saved[idx]         if ctx.has_pw else None
        if ctx.has_pw: idx += 1
        pw_C = saved[idx]         if ctx.has_pw else None

        # ── Call ch bwd kernel ────────────────────────────────────────────────
        with cuda_stage("sparse_scan_backward"):
            result = selective_scan_cuda.bwd_flat_ch(
                u_flat, k_flat, cu_seqlens, ctx.max_seqlen,
                dt_idle, dt_active, A, B, C,
                dout, x,
                D, z, out_flat,
                tile_offsets, ctx.tile_chunk_size,
                ctx.is_depthwise_bc,
                pw_B, pw_C
            )

        # Result layout from bwd_flat_ch:
        #   [0] du_flat
        #   [1] dA
        #   [2] dW_B  (depthwise mode) or dB  (legacy)
        #   [3] dW_C  (depthwise mode) or dC  (legacy)
        #   [4] ddt_idle
        #   [5] ddt_active
        #   [6] dD          (if has_D)
        #   [+] dz          (if has_z)
        #   [+] dpw_B       (if has_pw)  ← gradient for pw_B
        #   [+] dpw_C       (if has_pw)  ← gradient for pw_C
        du_flat    = result[0]
        dA         = result[1]
        dW_B       = result[2]   # dW_B (or dB in legacy mode)
        dW_C       = result[3]
        ddt_idle   = result[4]
        ddt_active = result[5]

        ridx = 6
        dD = result[ridx] if ctx.has_D else None
        if ctx.has_D: ridx += 1
        dz = result[ridx] if ctx.has_z else None
        if ctx.has_z: ridx += 1
        dpw_B = result[ridx] if ctx.has_pw else None
        if ctx.has_pw: ridx += 1
        dpw_C = result[ridx] if ctx.has_pw else None

        # Must match forward positional arg order:
        # u_flat, k_flat, cu_seqlens, dt_idle, dt_active, A, B, C,
        # D, z, max_seqlen, tile_offsets, tile_chunk_size, is_depthwise_bc,
        # pw_B, pw_C
        return (du_flat,       # u_flat
                None,          # k_flat      (int32, no grad)
                None,          # cu_seqlens  (int32, no grad)
                ddt_idle,      # dt_idle
                ddt_active,    # dt_active
                dA,            # A
                dW_B,          # B  (= dW_B in depthwise mode)
                dW_C,          # C  (= dW_C in depthwise mode)
                dD,            # D  (None if not has_D)
                dz,            # z  (None if not has_z)
                None,          # max_seqlen  (Python int, no grad)
                None,          # tile_offsets (int32, no grad)
                None,          # tile_chunk_size (Python int, no grad)
                None,          # is_depthwise_bc (bool, no grad)
                dpw_B,         # pw_B  (None if not has_pw)
                dpw_C)         # pw_C  (None if not has_pw)


def selective_scan_flat_ch_fn(u_flat, k_flat, cu_seqlens, max_seqlen,
                               dt_idle, dt_active, A, B, C,
                               D=None, z=None,
                               tile_offsets=None, tile_chunk_size=0,
                               is_depthwise_bc=False,
                               pw_B=None, pw_C=None):
    """
    Autograd-enabled flat per-channel selective scan.

    Forward  → selective_scan_cuda.fwd_flat_ch
    Backward → selective_scan_cuda.bwd_flat_ch  (custom per-channel CUDA kernel)

    Depthwise mode (is_depthwise_bc=True):
      B and C must be [dim, dstate] float32 weight tensors (W_B, W_C).
      The kernel computes B[i,s] = u[i]*W_B[d,s] and C[i,s] = u[i]*W_C[d,s]
      on-the-fly — no segment_scatter, no tiling, no extra memory allocation.
      Backward returns dW_B and dW_C with shape [dim, dstate] in the B/C slots.

    Depthwise + pointwise mode (is_depthwise_bc=True, pw_B/pw_C provided):
      pw_B and pw_C must be [dim, dim] float32. The kernel precomputes:
        W_B_eff[d, s] = Σ_{d'} pw_B[d, d'] * W_B[d', s]
      and uses W_B_eff for B[i,s] = u[i]*W_B_eff[d,s] entirely in-kernel.
      Backward returns:
        B/C slots → dW_B / dW_C  [dim, dstate]  (grad for depthwise weights)
        pw_B slot  → dpw_B       [dim, dim]      (grad for pointwise weights)
        pw_C slot  → dpw_C       [dim, dim]

    Legacy mode (is_depthwise_bc=False):
      When tile_offsets is provided, B/C must be in tiled layout:
          [N_total_tiles, dstate, tile_chunk_size]
      tile_offsets: (n_channels+1,) int32 — prefix sum of per-channel chunk counts.
      tile_chunk_size: int — must equal kNThreads * kNItems used during tiling.

    Returns:
        out  : (E,)                             primary output (gated if z provided)
        x    : (n_channels, n_chunks, dstate*2) final SSM state (non-differentiable)
    """
    u_flat     = u_flat.contiguous()
    k_flat     = k_flat.contiguous()
    cu_seqlens = cu_seqlens.contiguous()
    dt_idle    = dt_idle.contiguous()
    dt_active  = dt_active.contiguous()
    A = A.contiguous(); B = B.contiguous(); C = C.contiguous()
    if D is not None: D = D.contiguous()
    if z is not None: z = z.contiguous()
    if tile_offsets is not None: tile_offsets = tile_offsets.contiguous()
    if pw_B is not None: pw_B = pw_B.contiguous()
    if pw_C is not None: pw_C = pw_C.contiguous()

    return SelectiveScanFlatChFn.apply(
        u_flat, k_flat, cu_seqlens,
        dt_idle, dt_active, A, B, C,
        D, z, max_seqlen, tile_offsets, tile_chunk_size,
        is_depthwise_bc,
        pw_B, pw_C
    )
