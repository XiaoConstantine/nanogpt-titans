"""
Custom Triton kernels for Titans memory operations.

These kernels fuse multiple operations to reduce memory bandwidth
and kernel launch overhead.

Kernels:
- triton_momentum_update: Fused momentum accumulation over timesteps
- triton_fused_weight_update: Fused momentum + weight update
- triton_cross_entropy: Fused log_softmax + nll_loss
- triton_layer_norm: Fused layer normalization
- triton_fused_linear_silu: Fused linear + SiLU activation
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# =============================================================================
# Cross-Entropy Kernels (fused log_softmax + nll_loss)
# =============================================================================


@triton.jit
def _cross_entropy_fwd_kernel(
    logits_ptr,
    targets_ptr,
    losses_ptr,
    n_cols: tl.constexpr,
    logits_stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused cross-entropy forward kernel.

    For each row:
    1. Find max for numerical stability
    2. Compute log(sum(exp(x - max)))
    3. Compute loss = -logits[target] + log_sum_exp
    """
    row_idx = tl.program_id(0)

    # Pointer to this row's logits
    row_start = logits_ptr + row_idx * logits_stride_row

    # Load target for this row
    target = tl.load(targets_ptr + row_idx)

    # First pass: find max for numerical stability
    max_val = float("-inf")
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        vals = tl.load(row_start + col_offs, mask=mask, other=float("-inf"))
        max_val = tl.maximum(max_val, tl.max(vals, axis=0))

    # Second pass: compute sum(exp(x - max))
    sum_exp = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        vals = tl.load(row_start + col_offs, mask=mask, other=float("-inf"))
        sum_exp += tl.sum(tl.exp(vals - max_val), axis=0)

    # log_sum_exp = max + log(sum_exp)
    log_sum_exp = max_val + tl.log(sum_exp)

    # Load target logit and compute loss
    # Handle ignore_index=-1 (padding)
    if target >= 0 and target < n_cols:
        target_logit = tl.load(row_start + target)
        loss = log_sum_exp - target_logit
    else:
        loss = 0.0  # Ignore this sample

    tl.store(losses_ptr + row_idx, loss)


@triton.jit
def _cross_entropy_bwd_kernel(
    logits_ptr,
    targets_ptr,
    grad_output_ptr,
    grad_logits_ptr,
    n_cols: tl.constexpr,
    logits_stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused cross-entropy backward kernel.

    Gradient: softmax(logits) - one_hot(target)
    Scaled by grad_output (usually 1/n for mean reduction)
    """
    row_idx = tl.program_id(0)

    row_start = logits_ptr + row_idx * logits_stride_row
    grad_row_start = grad_logits_ptr + row_idx * logits_stride_row

    target = tl.load(targets_ptr + row_idx)
    grad_out = tl.load(grad_output_ptr + row_idx)

    # Handle ignore_index
    if target < 0 or target >= n_cols:
        # Zero gradients for ignored samples
        for block_start in range(0, n_cols, BLOCK_SIZE):
            col_offs = block_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offs < n_cols
            tl.store(grad_row_start + col_offs, tl.zeros([BLOCK_SIZE], dtype=tl.float32), mask=mask)
        return

    # First pass: find max
    max_val = float("-inf")
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        vals = tl.load(row_start + col_offs, mask=mask, other=float("-inf"))
        max_val = tl.maximum(max_val, tl.max(vals, axis=0))

    # Second pass: compute sum(exp(x - max))
    sum_exp = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        vals = tl.load(row_start + col_offs, mask=mask, other=float("-inf"))
        sum_exp += tl.sum(tl.exp(vals - max_val), axis=0)

    # Third pass: compute gradients = grad_out * (softmax - one_hot)
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        vals = tl.load(row_start + col_offs, mask=mask, other=float("-inf"))

        # softmax = exp(x - max) / sum_exp
        softmax = tl.exp(vals - max_val) / sum_exp

        # one_hot: 1 at target position, 0 elsewhere
        one_hot = (col_offs == target).to(tl.float32)

        grad = grad_out * (softmax - one_hot)
        tl.store(grad_row_start + col_offs, grad, mask=mask)


class _TritonCrossEntropy(torch.autograd.Function):
    """Autograd wrapper for Triton cross-entropy."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)

        losses = torch.empty(n_rows, device=logits.device, dtype=logits.dtype)

        _cross_entropy_fwd_kernel[(n_rows,)](
            logits,
            targets,
            losses,
            n_cols,
            logits.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(logits, targets)
        ctx.n_cols = n_cols

        # Count non-ignored samples for mean
        valid_mask = (targets >= 0) & (targets < n_cols)
        n_valid = valid_mask.sum()

        if n_valid > 0:
            return losses.sum() / n_valid
        return losses.sum()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits, targets = ctx.saved_tensors
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)

        grad_logits = torch.empty_like(logits)

        # Compute per-sample grad_output (1/n for mean reduction)
        valid_mask = (targets >= 0) & (targets < n_cols)
        n_valid = valid_mask.sum()
        if n_valid > 0:
            grad_per_sample = grad_output / n_valid
        else:
            grad_per_sample = grad_output

        grad_output_expanded = torch.full(
            (n_rows,), grad_per_sample.item(), device=logits.device, dtype=logits.dtype
        )

        _cross_entropy_bwd_kernel[(n_rows,)](
            logits,
            targets,
            grad_output_expanded,
            grad_logits,
            n_cols,
            logits.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return grad_logits, None


def triton_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Fused cross-entropy loss using Triton.

    Combines log_softmax + nll_loss in a single kernel pass.

    Args:
        logits: [N, V] raw logits (not softmaxed)
        targets: [N] target class indices (use -1 for ignore)

    Returns:
        Scalar loss (mean over valid samples)
    """
    # Flatten for 2D processing
    if logits.ndim > 2:
        orig_shape = logits.shape
        logits = logits.view(-1, orig_shape[-1])
        targets = targets.view(-1)

    return _TritonCrossEntropy.apply(logits, targets)


# =============================================================================
# LayerNorm Kernel
# =============================================================================


@triton.jit
def _layer_norm_fwd_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused layer normalization forward kernel.

    Computes: y = (x - mean) / sqrt(var + eps) * weight + bias
    """
    row_idx = tl.program_id(0)
    row_start = x_ptr + row_idx * n_cols
    out_start = y_ptr + row_idx * n_cols

    # Compute mean
    mean = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        vals = tl.load(row_start + col_offs, mask=mask, other=0.0)
        mean += tl.sum(vals, axis=0)
    mean = mean / n_cols

    # Compute variance
    var = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols
        vals = tl.load(row_start + col_offs, mask=mask, other=0.0)
        diff = vals - mean
        var += tl.sum(diff * diff, axis=0)
    var = var / n_cols

    rstd = 1.0 / tl.sqrt(var + eps)

    # Store mean and rstd for backward
    if mean_ptr is not None:
        tl.store(mean_ptr + row_idx, mean)
    if rstd_ptr is not None:
        tl.store(rstd_ptr + row_idx, rstd)

    # Normalize and apply weight/bias
    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offs < n_cols

        x = tl.load(row_start + col_offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + col_offs, mask=mask, other=1.0)

        y = (x - mean) * rstd * w

        if HAS_BIAS:
            b = tl.load(bias_ptr + col_offs, mask=mask, other=0.0)
            y = y + b

        tl.store(out_start + col_offs, y, mask=mask)


def triton_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Fused layer normalization using Triton.

    Args:
        x: [..., D] input tensor
        weight: [D] scale parameter
        bias: [D] shift parameter (optional)
        eps: epsilon for numerical stability

    Returns:
        [..., D] normalized tensor
    """
    orig_shape = x.shape
    x = x.view(-1, orig_shape[-1])
    n_rows, n_cols = x.shape

    y = torch.empty_like(x)
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 1024)

    _layer_norm_fwd_kernel[(n_rows,)](
        x,
        y,
        weight,
        bias if bias is not None else x,  # Dummy pointer if no bias
        None,  # mean_ptr - not needed for inference
        None,  # rstd_ptr - not needed for inference
        n_cols,
        eps,
        HAS_BIAS=bias is not None,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y.view(orig_shape)


# =============================================================================
# Fused Linear + SiLU Kernel
# =============================================================================


@triton.jit
def _linear_silu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused linear + SiLU activation kernel.

    Computes: y = silu(x @ w.T + bias)
    Where silu(x) = x * sigmoid(x)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute block indices
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Main matmul loop
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # Load x block [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load w block [BLOCK_N, BLOCK_K] - w is [N, K], we want w.T
        w_ptrs = w_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk
        w_mask = (n_offs[:, None] < N) & (k_offs[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate: x @ w.T
        acc += tl.dot(x, tl.trans(w))

    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(b_ptr + n_offs, mask=n_offs < N, other=0.0)
        acc = acc + bias[None, :]

    # Apply SiLU: x * sigmoid(x)
    sigmoid = 1.0 / (1.0 + tl.exp(-acc))
    result = acc * sigmoid

    # Store result
    y_ptrs = y_ptr + m_offs[:, None] * stride_ym + n_offs[None, :] * stride_yn
    y_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(y_ptrs, result.to(tl.float16), mask=y_mask)


def triton_linear_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused linear + SiLU activation using Triton.

    Computes: silu(x @ weight.T + bias)

    Args:
        x: [..., K] input tensor
        weight: [N, K] weight matrix
        bias: [N] bias vector (optional)

    Returns:
        [..., N] output tensor
    """
    orig_shape = x.shape[:-1]
    K = x.shape[-1]
    N = weight.shape[0]

    x = x.view(-1, K)
    M = x.shape[0]

    y = torch.empty(M, N, device=x.device, dtype=x.dtype)

    # Block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _linear_silu_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return y.view(*orig_shape, N)


# =============================================================================
# Fused MSE Gradient Kernel (for memory update)
# =============================================================================


@triton.jit
def _mse_grad_kernel(
    pred_ptr,
    target_ptr,
    grad_ptr,
    n_elements,
    scale,  # 2.0 / n_elements for MSE gradient
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute gradient of MSE loss: grad = 2 * (pred - target) / n

    This is the gradient w.r.t. the prediction.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    pred = tl.load(pred_ptr + offs, mask=mask, other=0.0)
    target = tl.load(target_ptr + offs, mask=mask, other=0.0)

    grad = scale * (pred - target)

    tl.store(grad_ptr + offs, grad, mask=mask)


def triton_mse_grad(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE gradient using Triton.

    Args:
        pred: Predictions tensor (any shape)
        target: Targets tensor (same shape as pred)

    Returns:
        Gradient tensor: 2 * (pred - target) / n_elements
    """
    pred_flat = pred.view(-1).contiguous()
    target_flat = target.view(-1).contiguous()
    n_elements = pred_flat.numel()

    grad = torch.empty_like(pred_flat)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    _mse_grad_kernel[grid](
        pred_flat,
        target_flat,
        grad,
        n_elements,
        2.0 / n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return grad.view_as(pred)


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


def triton_batched_weight_update(
    weights_dict: dict[str, torch.Tensor],
    grads_dict: dict[str, torch.Tensor],
    momentum_dict: dict[str, torch.Tensor],
    lr: float,
    momentum: float,
    decay: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Batched weight update - processes ALL parameters in a single kernel launch.

    Instead of launching N kernels (one per parameter), this:
    1. Flattens and concatenates all params into single tensors
    2. Launches ONE kernel
    3. Splits results back to per-parameter tensors

    Args:
        weights_dict: {name: [B, *shape]} current weights
        grads_dict: {name: [B, T, *shape]} gradients
        momentum_dict: {name: [B, *shape]} previous momentum
        lr: learning rate
        momentum: momentum coefficient
        decay: weight decay

    Returns:
        (new_weights_dict, new_momentum_dict)
    """
    # Get batch size and time steps from first grad
    first_name = next(iter(grads_dict))
    B, T = grads_dict[first_name].shape[:2]
    device = grads_dict[first_name].device
    dtype = grads_dict[first_name].dtype

    # Track shapes for splitting later
    param_names = list(weights_dict.keys())
    param_sizes = []
    param_shapes = []

    # Flatten and concatenate all parameters
    weights_list = []
    grads_list = []
    momentum_list = []

    for name in param_names:
        w = weights_dict[name]
        g = grads_dict[name]
        m = momentum_dict[name]

        # Get flattened size (excluding batch dim for weights/momentum, batch+time for grads)
        flat_size = w[0].numel()
        param_sizes.append(flat_size)
        param_shapes.append(w.shape[1:])  # Shape without batch dim

        # Flatten to [B, D] and [B, T, D]
        weights_list.append(w.view(B, flat_size))
        grads_list.append(g.view(B, T, flat_size))
        momentum_list.append(m.view(B, flat_size))

    # Concatenate along D dimension: [B, total_D] and [B, T, total_D]
    all_weights = torch.cat(weights_list, dim=1)
    all_grads = torch.cat(grads_list, dim=2)
    all_momentum = torch.cat(momentum_list, dim=1)

    total_D = all_weights.shape[1]

    # Allocate output tensors
    weights_out = torch.empty(B, total_D, device=device, dtype=dtype)
    momentum_out = torch.empty(B, total_D, device=device, dtype=dtype)

    # Launch ONE kernel for all parameters
    BLOCK_D = min(1024, triton.next_power_of_2(total_D))
    grid = (B, triton.cdiv(total_D, BLOCK_D))

    _fused_weight_update_kernel[grid](
        all_weights,
        weights_out,
        all_grads,
        all_momentum,
        momentum_out,
        lr,
        momentum,
        1.0 - momentum,
        1.0 - decay,
        T,
        total_D,
        all_weights.stride(0),
        all_weights.stride(1),
        all_grads.stride(0),
        all_grads.stride(1),
        all_grads.stride(2),
        all_momentum.stride(0),
        all_momentum.stride(1),
        BLOCK_D=BLOCK_D,
    )

    # Split results back to per-parameter tensors
    new_weights = {}
    new_momentum = {}
    offset = 0

    for i, name in enumerate(param_names):
        size = param_sizes[i]
        shape = param_shapes[i]

        # Extract slice and reshape
        new_weights[name] = weights_out[:, offset : offset + size].view(B, *shape)
        new_momentum[name] = momentum_out[:, offset : offset + size].view(B, *shape)
        offset += size

    return new_weights, new_momentum

