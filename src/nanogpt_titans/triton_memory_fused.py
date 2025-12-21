"""
Optimized memory update for Titans.

This module provides memory-efficient gradient computation that avoids
materializing the [B, T, H, C] gradient tensors.

Key optimization:
- Old path: einsum('btc,bth->btch') produces [B, T, C, H] = 1.2B elements
- New path: einsum('btc,bth->bch') sums over T, produces [B, C, H] = 2.4M elements
- 500x memory reduction!
"""

from __future__ import annotations

import torch


def chunked_gradient_memory_update(
    keys: torch.Tensor,           # [B, T, C]
    values: torch.Tensor,         # [B, T, C]
    weights: dict[str, torch.Tensor],     # {name: [B, *shape]}
    momentum: dict[str, torch.Tensor],    # {name: [B, *shape]}
    lr: float,
    mom_coef: float,
    decay: float,
    chunk_size: int = 64,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Compute memory update with chunked gradient computation.

    Instead of computing all [B, T, H, C] gradients at once, we:
    1. Process T tokens in chunks of chunk_size
    2. For each chunk, compute gradients and immediately accumulate momentum
    3. This reduces peak memory from O(B*T*H*C) to O(B*chunk*H*C)

    Args:
        keys: [B, T, C] input keys
        values: [B, T, C] target values
        weights: Current MLP weights
        momentum: Current momentum state
        lr: Learning rate
        mom_coef: Momentum coefficient
        decay: Weight decay
        chunk_size: Number of tokens to process per chunk

    Returns:
        (new_weights, new_momentum)
    """
    _B, T, C = keys.shape

    W0 = weights['layers.0.weight']  # [B, H, C]
    W1 = weights['layers.1.weight']  # [B, C, H]

    has_bias = 'layers.0.bias' in weights
    b0 = weights.get('layers.0.bias')  # [B, H] or None
    b1 = weights.get('layers.1.bias')  # [B, C] or None

    # Initialize output momentum (will accumulate across chunks)
    new_mom_W0 = momentum['layers.0.weight'].clone()
    new_mom_W1 = momentum['layers.1.weight'].clone()
    if has_bias:
        new_mom_b0 = momentum['layers.0.bias'].clone()
        new_mom_b1 = momentum['layers.1.bias'].clone()

    one_minus_mom = 1.0 - mom_coef
    decay_factor = 1.0 - decay
    mse_scale = 2.0 / C

    num_chunks = (T + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, T)

        # Extract chunk
        keys_chunk = keys[:, start:end, :]      # [B, chunk_len, C]
        values_chunk = values[:, start:end, :]  # [B, chunk_len, C]

        # Forward pass for this chunk
        h1_pre = torch.bmm(keys_chunk, W0.transpose(-1, -2))  # [B, chunk_len, H]
        if b0 is not None:
            h1_pre = h1_pre + b0.unsqueeze(1)

        sig_h1 = torch.sigmoid(h1_pre)
        h1 = h1_pre * sig_h1

        pred = torch.bmm(h1, W1.transpose(-1, -2))  # [B, chunk_len, C]
        if b1 is not None:
            pred = pred + b1.unsqueeze(1)

        # Backward pass for this chunk
        d_pred = mse_scale * (pred - values_chunk)  # [B, chunk_len, C]

        # Gradient for W1: dW1[b,c,h] = sum_t(d_pred[b,t,c] * h1[b,t,h])
        # We accumulate across tokens in the chunk
        dW1_chunk = torch.einsum('btc,bth->bch', d_pred, h1)  # [B, C, H]

        dh1 = torch.bmm(d_pred, W1)  # [B, chunk_len, H]
        silu_grad = sig_h1 * (1 + h1_pre * (1 - sig_h1))
        dh1_pre = dh1 * silu_grad

        # Gradient for W0: dW0[b,h,c] = sum_t(dh1_pre[b,t,h] * keys[b,t,c])
        dW0_chunk = torch.einsum('bth,btc->bhc', dh1_pre, keys_chunk)  # [B, H, C]

        if has_bias:
            db0_chunk = dh1_pre.sum(dim=1)  # [B, H]
            db1_chunk = d_pred.sum(dim=1)   # [B, C]

        # Accumulate momentum (treating chunk as one "super-step")
        new_mom_W0 = mom_coef * new_mom_W0 + one_minus_mom * dW0_chunk
        new_mom_W1 = mom_coef * new_mom_W1 + one_minus_mom * dW1_chunk
        if has_bias:
            new_mom_b0 = mom_coef * new_mom_b0 + one_minus_mom * db0_chunk
            new_mom_b1 = mom_coef * new_mom_b1 + one_minus_mom * db1_chunk

    # Apply weight update
    new_W0 = decay_factor * W0 - lr * new_mom_W0
    new_W1 = decay_factor * W1 - lr * new_mom_W1

    # Build dicts in named_parameters() order: weight, bias, weight, bias
    if has_bias:
        new_b0 = decay_factor * b0 - lr * new_mom_b0
        new_b1 = decay_factor * b1 - lr * new_mom_b1
        new_weights = {
            'layers.0.weight': new_W0,
            'layers.0.bias': new_b0,
            'layers.1.weight': new_W1,
            'layers.1.bias': new_b1,
        }
        new_momentum_dict = {
            'layers.0.weight': new_mom_W0,
            'layers.0.bias': new_mom_b0,
            'layers.1.weight': new_mom_W1,
            'layers.1.bias': new_mom_b1,
        }
    else:
        new_weights = {
            'layers.0.weight': new_W0,
            'layers.1.weight': new_W1,
        }
        new_momentum_dict = {
            'layers.0.weight': new_mom_W0,
            'layers.1.weight': new_mom_W1,
        }

    return new_weights, new_momentum_dict


def aggregated_gradient_memory_update(
    keys: torch.Tensor,           # [B, T, C]
    values: torch.Tensor,         # [B, T, C]
    weights: dict[str, torch.Tensor],
    momentum: dict[str, torch.Tensor],
    lr: float | torch.Tensor,
    mom_coef: float | torch.Tensor,
    decay: float | torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Memory update with aggregated gradients (no per-token storage).

    Key insight: Instead of computing per-token gradients [B, T, H, C] and then
    processing with momentum, we aggregate gradients directly during computation:

    Old: einsum('btc,bth->btch') produces [B, T, C, H] = 1.2B elements
    New: einsum('btc,bth->bch') sums over T, produces [B, C, H] = 2.4M elements

    This is a 500x memory reduction!

    The momentum update uses the mean gradient:
        m_new = mom * m_old + (1-mom) * mean(grad_t)

    This is an approximation of the exact per-token momentum, but:
    - Exact when momentum=0
    - Very close for typical momentum values (0.9)
    - Much faster and more memory efficient

    Args:
        keys: [B, T, C] input keys
        values: [B, T, C] target values
        weights: Current MLP weights {name: [B, *shape]}
        momentum: Current momentum state {name: [B, *shape]}
        lr: Learning rate - float or [B, T, 1] adaptive tensor
        mom_coef: Momentum coefficient - float or [B, T, 1] adaptive tensor
        decay: Weight decay - float or [B, T, 1] adaptive tensor

    Returns:
        (new_weights, new_momentum)
    """
    _B, T, C = keys.shape

    W0 = weights['layers.0.weight']  # [B, H, C]
    W1 = weights['layers.1.weight']  # [B, C, H]

    has_bias = 'layers.0.bias' in weights
    b0 = weights.get('layers.0.bias')
    b1 = weights.get('layers.1.bias')

    # Forward pass (full batch)
    h1_pre = torch.bmm(keys, W0.transpose(-1, -2))  # [B, T, H]
    if b0 is not None:
        h1_pre = h1_pre + b0.unsqueeze(1)

    sig_h1 = torch.sigmoid(h1_pre)
    h1 = h1_pre * sig_h1

    pred = torch.bmm(h1, W1.transpose(-1, -2))  # [B, T, C]
    if b1 is not None:
        pred = pred + b1.unsqueeze(1)

    # Backward pass - AGGREGATE gradients directly (no per-token storage!)
    mse_scale = 2.0 / C
    d_pred = mse_scale * (pred - values)  # [B, T, C]

    # Aggregated gradient for W1: sum over T
    # dW1_agg[b,c,h] = sum_t(d_pred[b,t,c] * h1[b,t,h])
    dW1_agg = torch.einsum('btc,bth->bch', d_pred, h1)  # [B, C, H] - NOT [B,T,C,H]!

    dh1 = torch.bmm(d_pred, W1)  # [B, T, H]
    silu_grad = sig_h1 * (1 + h1_pre * (1 - sig_h1))
    dh1_pre = dh1 * silu_grad

    # Aggregated gradient for W0
    dW0_agg = torch.einsum('bth,btc->bhc', dh1_pre, keys)  # [B, H, C]

    if has_bias:
        db0_agg = dh1_pre.sum(dim=1)  # [B, H]
        db1_agg = d_pred.sum(dim=1)   # [B, C]

    # Handle adaptive parameters (tensors) vs fixed (floats)
    # For adaptive: use mean across T for the aggregated path
    if isinstance(lr, torch.Tensor):
        # lr is [B, T, 1], take mean -> [B, 1, 1] for broadcasting
        lr = lr.mean(dim=1, keepdim=True)  # [B, 1, 1]
    if isinstance(mom_coef, torch.Tensor):
        mom_coef = mom_coef.mean(dim=1, keepdim=True)  # [B, 1, 1]
    if isinstance(decay, torch.Tensor):
        decay = decay.mean(dim=1, keepdim=True)  # [B, 1, 1]

    # Momentum update with aggregated gradient
    # Scale by 1/T to get mean gradient
    one_minus_mom = 1.0 - mom_coef
    decay_factor = 1.0 - decay
    scale = 1.0 / T

    # For tensor params, need to reshape for broadcasting with [B, H, C] weights
    if isinstance(mom_coef, torch.Tensor):
        # mom_coef is [B, 1, 1], works with [B, H, C]
        new_mom_W0 = mom_coef * momentum['layers.0.weight'] + one_minus_mom * dW0_agg * scale
        new_mom_W1 = mom_coef * momentum['layers.1.weight'] + one_minus_mom * dW1_agg * scale
    else:
        new_mom_W0 = mom_coef * momentum['layers.0.weight'] + one_minus_mom * dW0_agg * scale
        new_mom_W1 = mom_coef * momentum['layers.1.weight'] + one_minus_mom * dW1_agg * scale

    if has_bias:
        if isinstance(mom_coef, torch.Tensor):
            mom_coef_bias = mom_coef.squeeze(-1)  # [B, 1] for [B, H] bias
            one_minus_mom_bias = 1.0 - mom_coef_bias
            new_mom_b0 = mom_coef_bias * momentum['layers.0.bias'] + one_minus_mom_bias * db0_agg * scale
            new_mom_b1 = mom_coef_bias * momentum['layers.1.bias'] + one_minus_mom_bias * db1_agg * scale
        else:
            new_mom_b0 = mom_coef * momentum['layers.0.bias'] + one_minus_mom * db0_agg * scale
            new_mom_b1 = mom_coef * momentum['layers.1.bias'] + one_minus_mom * db1_agg * scale

    # Weight update
    if isinstance(decay_factor, torch.Tensor) or isinstance(lr, torch.Tensor):
        new_W0 = decay_factor * W0 - lr * new_mom_W0
        new_W1 = decay_factor * W1 - lr * new_mom_W1
    else:
        new_W0 = decay_factor * W0 - lr * new_mom_W0
        new_W1 = decay_factor * W1 - lr * new_mom_W1

    # Build dicts in named_parameters() order: weight, bias, weight, bias
    if has_bias:
        if isinstance(decay_factor, torch.Tensor) or isinstance(lr, torch.Tensor):
            decay_factor_bias = decay_factor.squeeze(-1) if isinstance(decay_factor, torch.Tensor) else decay_factor
            lr_bias = lr.squeeze(-1) if isinstance(lr, torch.Tensor) else lr
            new_b0 = decay_factor_bias * b0 - lr_bias * new_mom_b0
            new_b1 = decay_factor_bias * b1 - lr_bias * new_mom_b1
        else:
            new_b0 = decay_factor * b0 - lr * new_mom_b0
            new_b1 = decay_factor * b1 - lr * new_mom_b1
        new_weights = {
            'layers.0.weight': new_W0.detach(),
            'layers.0.bias': new_b0.detach(),
            'layers.1.weight': new_W1.detach(),
            'layers.1.bias': new_b1.detach(),
        }
        new_momentum_dict = {
            'layers.0.weight': new_mom_W0.detach(),
            'layers.0.bias': new_mom_b0.detach(),
            'layers.1.weight': new_mom_W1.detach(),
            'layers.1.bias': new_mom_b1.detach(),
        }
    else:
        new_weights = {
            'layers.0.weight': new_W0.detach(),
            'layers.1.weight': new_W1.detach(),
        }
        new_momentum_dict = {
            'layers.0.weight': new_mom_W0.detach(),
            'layers.1.weight': new_mom_W1.detach(),
        }

    return new_weights, new_momentum_dict
