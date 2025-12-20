"""
Custom Triton kernels for Titans memory operations.

These kernels fuse multiple operations to reduce memory bandwidth
and kernel launch overhead.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _momentum_update_kernel(
    # Pointers
    surprises_ptr,  # [B, T, D] - input surprises (negative gradients)
    prev_momentum_ptr,  # [B, D] - previous momentum state
    output_ptr,  # [B, D] - final momentum output
    # Scalars
    momentum: tl.constexpr,  # momentum coefficient (e.g., 0.9)
    one_minus_momentum: tl.constexpr,  # 1 - momentum
    # Dimensions
    T: tl.constexpr,
    D: tl.constexpr,
    # Strides
    stride_sb,  # surprises batch stride
    stride_st,  # surprises time stride
    stride_sd,  # surprises D stride
    stride_pb,  # prev_momentum batch stride
    stride_pd,  # prev_momentum D stride
    stride_ob,  # output batch stride
    stride_od,  # output D stride
    # Block size
    BLOCK_D: tl.constexpr,
):
    """
    Fused momentum update kernel.

    Computes: m[t] = momentum * m[t-1] + (1 - momentum) * surprise[t]

    Parallelizes over batch and D dimensions, sequential over T.
    For T=64 (typical segment length), sequential is fine.
    """
    # Program ID
    pid_b = tl.program_id(0)  # batch index
    pid_d = tl.program_id(1)  # D block index

    # D offsets for this block
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Load previous momentum: [BLOCK_D]
    prev_ptr = prev_momentum_ptr + pid_b * stride_pb + d_offs * stride_pd
    m = tl.load(prev_ptr, mask=d_mask, other=0.0)

    # Sequential loop over T (small, typically 64)
    for t in range(T):
        # Load surprise at time t: [BLOCK_D]
        s_ptr = surprises_ptr + pid_b * stride_sb + t * stride_st + d_offs * stride_sd
        surprise = tl.load(s_ptr, mask=d_mask, other=0.0)

        # Update momentum: m = momentum * m + (1 - momentum) * surprise
        m = momentum * m + one_minus_momentum * surprise

    # Store final momentum: [BLOCK_D]
    out_ptr = output_ptr + pid_b * stride_ob + d_offs * stride_od
    tl.store(out_ptr, m, mask=d_mask)


def triton_momentum_update(
    surprises: torch.Tensor,
    momentum: float,
    prev_momentum: torch.Tensor,
) -> torch.Tensor:
    """
    Compute momentum update using fused Triton kernel.

    Args:
        surprises: [B, T, *shape] surprise values
        momentum: momentum coefficient (e.g., 0.9)
        prev_momentum: [B, *shape] previous momentum state

    Returns:
        [B, *shape] final momentum after processing all T timesteps
    """
    # Flatten to 3D: [B, T, D]
    B, T = surprises.shape[:2]
    orig_shape = surprises.shape[2:]
    D = surprises[0, 0].numel()

    surprises_flat = surprises.view(B, T, D).contiguous()
    prev_flat = prev_momentum.view(B, D).contiguous()
    output = torch.empty(B, D, device=surprises.device, dtype=surprises.dtype)

    # Launch kernel
    BLOCK_D = min(1024, triton.next_power_of_2(D))
    grid = (B, triton.cdiv(D, BLOCK_D))

    _momentum_update_kernel[grid](
        surprises_flat,
        prev_flat,
        output,
        momentum,
        1.0 - momentum,
        T,
        D,
        surprises_flat.stride(0),
        surprises_flat.stride(1),
        surprises_flat.stride(2),
        prev_flat.stride(0),
        prev_flat.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_D=BLOCK_D,
    )

    # Reshape back to original shape
    return output.view(B, *orig_shape)


@triton.jit
def _fused_weight_update_kernel(
    # Pointers - separate input/output to avoid cloning
    weights_in_ptr,  # [B, D] - current weights (read-only)
    weights_out_ptr,  # [B, D] - updated weights (write-only)
    grads_ptr,  # [B, T, D] - input gradients
    prev_momentum_ptr,  # [B, D] - previous momentum (read-only)
    new_momentum_ptr,  # [B, D] - updated momentum (write-only)
    # Scalars
    lr: tl.constexpr,  # learning rate
    momentum: tl.constexpr,  # momentum coefficient
    one_minus_momentum: tl.constexpr,
    decay_factor: tl.constexpr,  # 1 - decay
    # Dimensions
    T: tl.constexpr,
    D: tl.constexpr,
    # Strides
    stride_wb,
    stride_wd,
    stride_gb,
    stride_gt,
    stride_gd,
    stride_pb,
    stride_pd,
    # Block size
    BLOCK_D: tl.constexpr,
):
    """
    Fully fused weight update kernel with separate input/output.

    Computes in one kernel:
    1. Momentum accumulation over T timesteps
    2. Weight update with decay: w_new = decay_factor * w_old - lr * final_momentum
    3. Outputs updated weights and momentum (no cloning needed)
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Load previous momentum
    prev_ptr = prev_momentum_ptr + pid_b * stride_pb + d_offs * stride_pd
    m = tl.load(prev_ptr, mask=d_mask, other=0.0)

    # Sequential momentum accumulation
    for t in range(T):
        g_ptr = grads_ptr + pid_b * stride_gb + t * stride_gt + d_offs * stride_gd
        grad = tl.load(g_ptr, mask=d_mask, other=0.0)
        m = momentum * m + one_minus_momentum * grad

    # Store updated momentum to new location
    new_m_ptr = new_momentum_ptr + pid_b * stride_pb + d_offs * stride_pd
    tl.store(new_m_ptr, m, mask=d_mask)

    # Load current weights
    w_in_ptr = weights_in_ptr + pid_b * stride_wb + d_offs * stride_wd
    w = tl.load(w_in_ptr, mask=d_mask, other=0.0)

    # Update weights: w = decay_factor * w - lr * m
    w = decay_factor * w - lr * m

    # Store updated weights to new location
    w_out_ptr = weights_out_ptr + pid_b * stride_wb + d_offs * stride_wd
    tl.store(w_out_ptr, w, mask=d_mask)


def triton_fused_weight_update(
    weights_in: torch.Tensor,
    grads: torch.Tensor,
    prev_momentum: torch.Tensor,
    lr: float,
    momentum: float,
    decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fully fused weight update - returns new tensors (no cloning needed).

    Args:
        weights_in: [B, *shape] - current weights (read-only)
        grads: [B, T, *shape] - gradient values
        prev_momentum: [B, *shape] - previous momentum (read-only)
        lr: learning rate
        momentum: momentum coefficient
        decay: weight decay

    Returns:
        (new_weights, new_momentum) - updated tensors
    """
    B, T = grads.shape[:2]
    orig_shape = grads.shape[2:]
    D = grads[0, 0].numel()

    weights_flat = weights_in.view(B, D)
    grads_flat = grads.view(B, T, D).contiguous()
    prev_flat = prev_momentum.view(B, D)

    # Allocate output tensors
    weights_out = torch.empty_like(weights_flat)
    momentum_out = torch.empty_like(prev_flat)

    BLOCK_D = min(1024, triton.next_power_of_2(D))
    grid = (B, triton.cdiv(D, BLOCK_D))

    _fused_weight_update_kernel[grid](
        weights_flat,
        weights_out,
        grads_flat,
        prev_flat,
        momentum_out,
        lr,
        momentum,
        1.0 - momentum,
        1.0 - decay,
        T,
        D,
        weights_flat.stride(0),
        weights_flat.stride(1),
        grads_flat.stride(0),
        grads_flat.stride(1),
        grads_flat.stride(2),
        prev_flat.stride(0),
        prev_flat.stride(1),
        BLOCK_D=BLOCK_D,
    )

    # Reshape back to original shape
    return weights_out.view(B, *orig_shape), momentum_out.view(B, *orig_shape)

