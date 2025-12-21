"""
Titans: Learning to Memorize at Test Time
Implementation of Memory as Context (MAC) architecture on top of nanoGPT.

Based on: https://arxiv.org/abs/2501.00663

Fixed version addressing:
- Proper prefix-LM attention masking
- Per-sequence memory state (not global)
- Memory updates during training
- Correct state passing between segments
- FIXED: Causal memory retrieval (no future token leakage)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

# FlexAttention imports (available in PyTorch 2.5+)
# Works on L4, A100, H100. Fails on T4 (Triton register limit).
_FLEX_ATTENTION_AVAILABLE = False
try:
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention,
    )

    if torch.cuda.is_available():
        _FLEX_ATTENTION_AVAILABLE = True
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

# Triton kernels for fused memory operations
_TRITON_AVAILABLE = False
triton_momentum_update = None  # type: ignore[assignment]
triton_fused_weight_update = None  # type: ignore[assignment]
triton_cross_entropy = None  # type: ignore[assignment]
try:
    from nanogpt_titans.triton_kernels import (
        triton_cross_entropy,
        triton_fused_weight_update,
        triton_momentum_update,
    )

    if torch.cuda.is_available():
        _TRITON_AVAILABLE = True
except ImportError:
    pass


# --- FlexAttention mask functions ---


def causal_mask(_b, _h, q_idx, kv_idx):
    """Standard causal mask: each position can only attend to previous positions."""
    return q_idx >= kv_idx


def prefix_lm_mask(prefix_len: int):
    """
    Create prefix-LM mask function.

    Prefix tokens (0:prefix_len) can attend to all prefix tokens.
    Suffix tokens (prefix_len:) can attend to prefix + causally to suffix.
    """

    def mask_fn(_b, _h, q_idx, kv_idx):
        # If query is in prefix, it can attend to all prefix tokens
        in_prefix = q_idx < prefix_len
        # If key is in prefix, it can always be attended to
        key_in_prefix = kv_idx < prefix_len
        # Causal mask for suffix
        causal = q_idx >= kv_idx
        # Prefix can attend to prefix; suffix can attend to prefix + causal suffix
        return (
            (in_prefix & key_in_prefix)
            | (key_in_prefix & ~in_prefix)
            | (~in_prefix & ~key_in_prefix & causal)
        )

    return mask_fn


def document_causal_mask(document_ids: torch.Tensor):
    """
    Create document-aware causal mask for packed sequences.

    Each position can only attend to positions in the same document,
    with causal masking within each document.

    Args:
        document_ids: [seq_len] tensor where document_ids[i] = doc index for token i
    """

    def mask_fn(_b, _h, q_idx, kv_idx):
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        causal = q_idx >= kv_idx
        return same_doc & causal

    return mask_fn


@lru_cache(maxsize=32)
def get_causal_block_mask(seq_len: int, device: torch.device):
    """Cache causal block masks for common sequence lengths."""
    if not _FLEX_ATTENTION_AVAILABLE:
        return None
    return create_block_mask(
        causal_mask,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )


def parallel_scan_log(gates: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    Parallel associative scan via cumsum trick.

    Solves the recurrence: x[t] = gate[t] * x[t-1] + token[t]

    This is O(n) and fully parallelizable on CPU/GPU/TPU using cumsum.
    Based on: https://github.com/PeaBrane/mamba-tiny
    See also: https://srush.github.io/annotated-mamba/hard.html

    Args:
        gates: Decay/gate values [*, T] in range (0, 1), e.g., momentum coefficient
        tokens: Input values [*, T, ...] to accumulate, e.g., surprises

    Returns:
        Accumulated values [*, T, ...] where each position contains
        the momentum-weighted sum of all previous tokens
    """
    # Handle the case where tokens has more dims than gates
    # gates: [*, T], tokens: [*, T, D1, D2, ...]
    token_dims = tokens.dim() - gates.dim()
    for _ in range(token_dims):
        gates = gates.unsqueeze(-1)

    # Clamp gates to avoid log(0)
    gates = gates.clamp(min=1e-7, max=1.0 - 1e-7)

    # Log-space computation for numerical stability
    log_gates = torch.log(gates)
    log_cumsum_gates = torch.cumsum(log_gates, dim=-2 - token_dims + 1)  # cumsum along T dim

    # Scale tokens by inverse cumulative gate product
    # This "normalizes" each token to the same reference point
    scaling = torch.exp(-log_cumsum_gates)
    scaled_tokens = tokens * scaling

    # Cumsum of scaled tokens
    cumsum_scaled = torch.cumsum(scaled_tokens, dim=-2 - token_dims + 1)

    # Rescale back by cumulative gate product
    result = cumsum_scaled * torch.exp(log_cumsum_gates)

    return result


def parallel_momentum(
    surprises: torch.Tensor,
    momentum: float,
    prev_momentum: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute momentum updates in parallel using cumsum trick.

    Computes: m[t] = momentum * m[t-1] + (1 - momentum) * surprise[t]

    This is equivalent to exponential moving average of surprises.

    Args:
        surprises: Surprise values [B, T, ...] to accumulate
        momentum: Momentum coefficient (e.g., 0.9)
        prev_momentum: Previous momentum state [B, ...] from last segment

    Returns:
        Momentum values [B, T, ...] for each timestep
    """
    B, T = surprises.shape[:2]
    device = surprises.device

    # Gates are constant momentum value
    gates = torch.full((B, T), momentum, device=device, dtype=surprises.dtype)

    # Scale surprises by (1 - momentum)
    tokens = (1 - momentum) * surprises

    # If we have previous momentum, prepend it and adjust
    if prev_momentum is not None:
        # Prepend previous momentum as first "token" with gate 1.0
        # This propagates the previous state through the scan
        prev_expanded = prev_momentum.unsqueeze(1)  # [B, 1, ...]
        tokens = torch.cat([prev_expanded, tokens], dim=1)  # [B, T+1, ...]
        gates = torch.cat([torch.ones(B, 1, device=device), gates], dim=1)

        result = parallel_scan_log(gates, tokens)
        return result[:, 1:]  # Remove the prepended position

    return parallel_scan_log(gates, tokens)


@dataclass
class TitansConfig:
    """Configuration for TitansGPT model."""

    # Base GPT config (nanoGPT defaults)
    block_size: int = 2048
    vocab_size: int = 50304  # GPT-2 vocab size (50257 rounded for efficiency)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True

    # Titans-specific config
    segment_len: int = 128  # tokens per segment
    memory_depth: int = 2  # MLP layers in neural memory (paper uses 1-4, default 2)
    memory_expansion: int = 2  # hidden dim multiplier (2 = n_embd*2, reduced from 4 for memory)
    num_persist_mem: int = 4  # persistent memory tokens
    num_longterm_mem: int = 16  # long-term memory tokens retrieved
    memory_lr: float = 0.01  # learning rate for surprise-based updates
    memory_momentum: float = 0.9  # momentum for past surprise
    memory_decay: float = 0.001  # forgetting/decay factor
    memory_layer: int = -1  # which layer has memory (-1 = middle layer, -2 = all layers)


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch doesn't support bias=False directly)."""

    def __init__(self, ndim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


@dataclass
class MemoryState:
    """Per-sequence memory state - stores MLP weights as the memory."""

    # MLP weights per batch item: dict mapping param_name -> [B, *param_shape]
    weights: dict[str, torch.Tensor]
    # Last momentum value for each weight (for continuity across segments): [B, *param_shape]
    last_momentum: dict[str, torch.Tensor]
    # Last segment's output representation for causal retrieval: [B, T, C] or None
    last_segment_output: torch.Tensor | None = None
    step: int = 0


class MemoryMLP(nn.Module):
    """
    Small MLP whose weights serve as the memory.

    This is a stateless module - actual weights are passed in via functional_call.
    """

    def __init__(self, dim: int, hidden_dim: int, depth: int = 2, bias: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = dim
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            in_dim = hidden_dim
        self.layers.append(nn.Linear(in_dim, dim, bias=bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:  # type: ignore[union-attr]
            x = F.silu(layer(x))
        return self.layers[-1](x)


class NeuralMemory(nn.Module):
    """
    Neural Memory Module that learns at test time.

    Correct implementation based on Titans paper:
    - Memory IS the MLP weights (stored per-sequence in MemoryState)
    - Surprise = gradient of MSE loss w.r.t. MLP weights
    - Memory update = weight update with momentum and decay
    - Retrieval = forward pass through MLP with per-sequence weights

    FIXED: Retrieval now uses previous segment's output (stored in state)
    to avoid leaking future token information.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config  # type: ignore[assignment]
        self.n_embd = config.n_embd  # type: ignore[assignment]
        self.num_longterm_mem = config.num_longterm_mem  # type: ignore[assignment]

        # Template MLP - its weights are cloned per-sequence as initial memory
        # Use smaller hidden_dim for memory efficiency
        hidden_dim = config.n_embd * config.memory_expansion
        self.hidden_dim = hidden_dim  # Cache for batched gradient computation
        self.memory_mlp = MemoryMLP(
            dim=config.n_embd,
            hidden_dim=hidden_dim,
            depth=config.memory_depth,
            bias=config.bias,
        )

        # Query projection for retrieval
        self.query_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Key/Value projections for storage
        self.key_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Learned initial query for first segment (when no previous output exists)
        self.init_query = nn.Parameter(torch.randn(1, 1, config.n_embd) * 0.02)

        # Hyperparameters for test-time learning
        self.lr = config.memory_lr  # type: ignore[assignment]
        self.momentum = config.memory_momentum  # type: ignore[assignment]
        self.decay = config.memory_decay  # type: ignore[assignment]

        # Cache vmapped functions for performance (fallback only)
        self._batched_mlp_forward: callable | None = None  # type: ignore[valid-type]
        self._batched_grad_fn: callable | None = None  # type: ignore[valid-type]

        # Use batched matmul instead of vmap (much faster)
        self._use_batched_matmul = True

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        """Initialize memory state - clone MLP weights for each batch item."""
        weights: dict[str, torch.Tensor] = {}
        last_momentum: dict[str, torch.Tensor] = {}

        for name, param in self.memory_mlp.named_parameters():
            # Expand weights to [B, *param_shape] and clone - ensure contiguous
            expanded = param.detach().unsqueeze(0).expand(batch_size, *param.shape).clone().contiguous()
            weights[name] = expanded.to(device)
            # Last momentum starts at zero - already contiguous
            last_momentum[name] = torch.zeros_like(expanded)

        return MemoryState(
            weights=weights, last_momentum=last_momentum, last_segment_output=None, step=0
        )

    def reset_state(self, state: MemoryState) -> None:
        """Reset memory state in-place - avoids reallocation."""
        for name, param in self.memory_mlp.named_parameters():
            # Copy initial weights in-place
            state.weights[name].copy_(param.detach().unsqueeze(0).expand_as(state.weights[name]))
            state.last_momentum[name].zero_()
        state.last_segment_output = None
        state.step = 0

    def _compute_gradients_batched(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute per-token MSE gradients using batched matrix operations.

        This replaces vmap with efficient einsum operations, reducing
        O(B*T) small kernel launches to O(1) large matrix operations.

        For 2-layer MLP with SiLU activation:
            h1 = silu(keys @ W0.T + b0)
            pred = h1 @ W1.T + b1
            loss = MSE(pred, values)

        Args:
            keys: Input keys [B, T, C]
            values: Target values [B, T, C]
            weights: Per-batch MLP weights {name: [B, *param_shape]}

        Returns:
            Gradients {name: [B, T, *param_shape]}
        """
        B, T, C = keys.shape
        H = self.hidden_dim

        # Extract weights (PyTorch Linear: weight is [out, in])
        W0 = weights['layers.0.weight']  # [B, H, C]
        b0 = weights['layers.0.bias']    # [B, H]
        W1 = weights['layers.1.weight']  # [B, C, H]
        b1 = weights['layers.1.bias']    # [B, C]

        # Forward pass (batched over all tokens)
        # h1_pre = keys @ W0.T + b0 = einsum('btc,bhc->bth', keys, W0) + b0
        h1_pre = torch.einsum('btc,bhc->bth', keys, W0) + b0.unsqueeze(1)  # [B, T, H]

        # SiLU activation: silu(x) = x * sigmoid(x)
        sig_h1 = torch.sigmoid(h1_pre)
        h1 = h1_pre * sig_h1  # [B, T, H]

        # Output layer: pred = h1 @ W1.T + b1
        pred = torch.einsum('bth,bch->btc', h1, W1) + b1.unsqueeze(1)  # [B, T, C]

        # MSE gradient: d_pred = 2 * (pred - values) / C
        # We absorb the 2/C factor into the learning rate for efficiency
        d_pred = pred - values  # [B, T, C]

        # Backprop through output layer
        # dW1[b,t,c,h] = d_pred[b,t,c] * h1[b,t,h] (outer product per token)
        dW1 = torch.einsum('btc,bth->btch', d_pred, h1)  # [B, T, C, H]
        db1 = d_pred  # [B, T, C]

        # dh1 = d_pred @ W1
        dh1 = torch.einsum('btc,bch->bth', d_pred, W1)  # [B, T, H]

        # Backprop through SiLU
        # silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        #          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        silu_grad = sig_h1 * (1 + h1_pre * (1 - sig_h1))
        dh1_pre = dh1 * silu_grad  # [B, T, H]

        # Backprop through input layer
        # dW0[b,t,h,c] = dh1_pre[b,t,h] * keys[b,t,c] (outer product per token)
        dW0 = torch.einsum('bth,btc->bthc', dh1_pre, keys)  # [B, T, H, C]
        db0 = dh1_pre  # [B, T, H]

        return {
            'layers.0.weight': dW0,
            'layers.0.bias': db0,
            'layers.1.weight': dW1,
            'layers.1.bias': db1,
        }

    def _forward_with_weights(
        self,
        x: torch.Tensor,
        weights: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Forward pass through MLP using specific weights for one batch item."""
        # Extract weights for this batch item
        params = {name: w[batch_idx] for name, w in weights.items()}

        # Use functional_call to apply MLP with these weights
        return torch.func.functional_call(self.memory_mlp, params, (x,))

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState,
    ) -> torch.Tensor:
        """
        Retrieve from memory using PREVIOUS segment's output (causal).

        Args:
            x: Input tensor [B, T, C] - current segment (NOT used for retrieval to maintain causality)
            state: Current memory state (contains previous segment's output)

        Returns:
            retrieved: Memory context [B, num_longterm_mem, C]
        """
        b, _t, _c = x.shape

        # FIXED: Use previous segment's output for retrieval, NOT current segment
        # This ensures causality - we only use information from the past
        if state.last_segment_output is not None:
            query_source = state.last_segment_output  # [B, T_prev, C]
        else:
            # First segment: use learned initial query expanded for batch
            query_source = self.init_query.expand(b, -1, -1)  # [B, 1, C]

        # Project to queries - ensure contiguous for vmap
        queries = self.query_proj(query_source).contiguous()  # [B, T_prev, C] or [B, 1, C]

        # Retrieve by passing queries through per-sequence MLP (vectorized with vmap)
        def single_batch_forward(params: dict[str, torch.Tensor], q: torch.Tensor) -> torch.Tensor:
            # q: [T, C], params: {name: [*param_shape]}
            q = q.unsqueeze(0)  # [1, T, C]
            return torch.func.functional_call(self.memory_mlp, params, (q,)).squeeze(0)  # [T, C]

        # vmap over batch dimension
        if self._batched_mlp_forward is None:
            params_in_dims = dict.fromkeys(state.weights, 0)
            self._batched_mlp_forward = torch.func.vmap(
                single_batch_forward, in_dims=(params_in_dims, 0)
            )

        # Ensure weights are contiguous for vmap
        weights_contig = {k: v.contiguous() for k, v in state.weights.items()}
        retrieved = self._batched_mlp_forward(weights_contig, queries)  # [B, T_prev, C]

        # Pool to num_longterm_mem tokens
        B, T_prev, C = retrieved.shape
        if T_prev != self.num_longterm_mem:
            # Use chunk-based pooling instead of adaptive_avg_pool1d (MPS compatible)
            # Divide T_prev into num_longterm_mem chunks and average each
            if T_prev > self.num_longterm_mem:
                # Truncate to divisible size, then reshape and mean
                chunk_size = T_prev // self.num_longterm_mem
                usable_len = chunk_size * self.num_longterm_mem
                retrieved = retrieved[:, :usable_len, :]  # [B, usable_len, C]
                retrieved = retrieved.view(B, self.num_longterm_mem, chunk_size, C).mean(dim=2)
            else:
                # Pad and reshape (rare case: T_prev < num_longterm_mem)
                pad_len = self.num_longterm_mem - T_prev
                padding = retrieved[:, -1:, :].expand(B, pad_len, C)
                retrieved = torch.cat([retrieved, padding], dim=1)
        # else: T_prev == num_longterm_mem, no pooling needed

        return self.out_proj(retrieved)

    def update(
        self,
        x: torch.Tensor,
        state: MemoryState,
    ) -> MemoryState:
        """
        Update memory based on surprise (gradient of MSE loss w.r.t. MLP weights).

        From paper eq (12): Loss = |M(k) - v|^2
        Surprise = -grad(Loss) w.r.t. MLP weights
        Update: weights += lr * momentum(surprise) (with decay)

        Uses:
        - Batched matmul for efficient per-token gradient computation (replaces vmap)
        - parallel_momentum (cumsum trick) for O(n) parallel momentum across tokens

        Args:
            x: Current segment output [B, T, C]
            state: Current memory state

        Returns:
            new_state: Updated memory state (includes storing x for next segment's retrieval)
        """
        _b, _t, _c = x.shape

        # Project to keys and values for storage - ensure contiguous
        x_detached = x.detach().contiguous()
        keys = self.key_proj(x_detached).contiguous()  # [B, T, C]
        values = self.value_proj(x_detached).contiguous()  # [B, T, C]

        # Use batched matmul (fast) or vmap (fallback)
        if self._use_batched_matmul and self.config.memory_depth == 2:
            # Fast path: analytical gradient computation using batched matmul
            # This replaces O(B*T) small vmap operations with O(1) large einsum operations
            all_grads = self._compute_gradients_batched(keys, values, state.weights)
        else:
            # Fallback: vmap for non-standard MLP depths
            if self._batched_grad_fn is None:
                def single_token_loss(
                    params: dict[str, torch.Tensor], k: torch.Tensor, v: torch.Tensor
                ) -> torch.Tensor:
                    k = k.unsqueeze(0).unsqueeze(0)
                    v = v.unsqueeze(0).unsqueeze(0)
                    pred = torch.func.functional_call(self.memory_mlp, params, (k,))
                    return F.mse_loss(pred, v)

                grad_fn = torch.func.grad(single_token_loss)
                params_in_dims = dict.fromkeys(state.weights, 0)
                self._batched_grad_fn = torch.func.vmap(
                    torch.func.vmap(grad_fn, in_dims=(None, 0, 0)),
                    in_dims=(params_in_dims, 0, 0),
                )

            with torch.enable_grad():
                params_for_grad = {
                    name: w.contiguous().clone().requires_grad_(True)
                    for name, w in state.weights.items()
                }
                all_grads = self._batched_grad_fn(params_for_grad, keys, values)

        # Apply updates with parallel momentum
        # Check if we can use the fused Triton kernel
        first_grad = next(iter(all_grads.values()))
        use_fused = (
            _TRITON_AVAILABLE
            and first_grad.is_cuda
            and triton_fused_weight_update is not None
        )

        if use_fused:
            # Fused Triton kernel: momentum + weight update per parameter
            new_weights: dict[str, torch.Tensor] = {}
            new_last_momentum: dict[str, torch.Tensor] = {}

            for name in state.weights:
                # Ensure contiguous inputs for Triton kernel
                grads = all_grads[name].contiguous()
                weights_in = state.weights[name].contiguous()
                momentum_in = state.last_momentum[name].contiguous()

                updated_weights, updated_momentum = triton_fused_weight_update(
                    weights_in,
                    grads,
                    momentum_in,
                    self.lr,
                    self.momentum,
                    self.decay,
                )
                new_weights[name] = updated_weights.detach().contiguous()
                new_last_momentum[name] = updated_momentum.detach().contiguous()
        else:
            # Fallback: separate momentum + weight update
            new_weights = {}
            new_last_momentum = {}

            for name in state.weights:
                grads = all_grads[name].contiguous()

                if _TRITON_AVAILABLE and grads.is_cuda:
                    # Triton kernel for momentum only
                    final_momentum = triton_momentum_update(
                        grads,
                        self.momentum,
                        state.last_momentum[name].contiguous(),
                    )
                else:
                    # Parallel scan via cumsum trick
                    momentum_values = parallel_momentum(
                        grads,
                        self.momentum,
                        prev_momentum=state.last_momentum[name],
                    )
                    final_momentum = momentum_values[:, -1]  # [B, *param_shape]

                new_last_momentum[name] = final_momentum.detach().contiguous()

                # Weight update: w = decay*w - lr*momentum
                decay_factor = 1 - self.decay
                updated_weights = decay_factor * state.weights[name] - self.lr * final_momentum
                new_weights[name] = updated_weights.detach().contiguous()

        return MemoryState(
            weights=new_weights,
            last_momentum=new_last_momentum,
            last_segment_output=x.detach().contiguous(),  # Store current segment for next retrieval
            step=state.step + 1,
        )


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with prefix-LM masking support.

    For Titans MAC architecture:
    - Prefix tokens (memory + persistent) can attend to each other fully
    - Current segment tokens can attend to prefix + causally to each other

    Uses FlexAttention when available (CUDA + PyTorch 2.5+) for better performance.
    Falls back to standard SDPA or manual attention otherwise.
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            msg = f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
            raise ValueError(msg)

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head  # type: ignore[assignment]
        self.n_embd = config.n_embd  # type: ignore[assignment]
        self.dropout = config.dropout  # type: ignore[assignment]

        self.flash = hasattr(F, "scaled_dot_product_attention")  # type: ignore[assignment]
        self.use_flex_attn = _FLEX_ATTENTION_AVAILABLE  # type: ignore[assignment]

        # Cache for block masks (avoid recreating every forward pass)
        self._block_mask_cache: dict[tuple, object] = {}
        # Cache for SDPA float masks
        self._sdpa_mask_cache: dict[tuple, torch.Tensor] = {}

    def _get_block_mask(
        self,
        seq_len: int,
        prefix_len: int,
        device: torch.device,
        document_ids: torch.Tensor | None = None,
    ):
        """Get or create a block mask for the given configuration."""
        cache_key = (seq_len, prefix_len, document_ids is not None)

        if cache_key not in self._block_mask_cache:
            if document_ids is not None:
                # Document-aware causal mask for packed sequences
                mask_fn = document_causal_mask(document_ids)
            elif prefix_len > 0:
                # Prefix-LM mask
                mask_fn = prefix_lm_mask(prefix_len)
            else:
                # Standard causal mask
                mask_fn = causal_mask

            block_mask = create_block_mask(
                mask_fn,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
            )
            self._block_mask_cache[cache_key] = block_mask

        return self._block_mask_cache[cache_key]

    def forward(
        self,
        x: torch.Tensor,
        prefix_len: int = 0,
        packed_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with prefix-LM attention mask.

        Args:
            x: Input tensor [B, T, C] where T = prefix_len + segment_len
            prefix_len: Number of prefix tokens (memory + persistent)
                       These tokens can attend to each other fully.
            packed_mask: Optional [B, T, T] boolean mask for packed sequences.
                        True = can attend, False = cannot attend.
                        If provided, overrides prefix_len behavior.
                        (Used when FlexAttention is not available)
            document_ids: Optional [T] tensor of document IDs for packed sequences.
                         Used with FlexAttention for efficient document masking.

        Returns:
            Output tensor [B, T, C]
        """
        b, t, c = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)

        # Try FlexAttention first (most efficient on CUDA)
        if self.use_flex_attn and x.is_cuda and packed_mask is None:
            # Use FlexAttention with block mask
            block_mask = self._get_block_mask(t, prefix_len, x.device, document_ids)
            y = flex_attention(q, k, v, block_mask=block_mask)

        elif packed_mask is not None:
            # Use provided packed attention mask (fallback for non-CUDA or complex masks)
            # Convert boolean mask (True=attend) to float mask (0=attend, -inf=block)
            # packed_mask: [B, T, T] -> need [B, 1, T, T] for broadcasting
            attn_mask = packed_mask.unsqueeze(1)  # [B, 1, T, T]
            float_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            float_mask.masked_fill_(~attn_mask, float("-inf"))

            if self.flash:
                # Flash attention with explicit mask
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=float_mask,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False,  # We provide explicit mask
                )
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att + float_mask
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v

        elif self.flash and prefix_len == 0:
            # Standard causal attention (no prefix) - use SDPA
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        elif self.flash and prefix_len > 0:
            # Prefix-LM attention using SDPA with cached explicit mask
            cache_key = (t, prefix_len, x.device, q.dtype)
            if cache_key not in self._sdpa_mask_cache:
                # Build prefix-LM mask: prefix bidirectional, suffix causal + can see prefix
                mask = torch.ones(t, t, device=x.device, dtype=torch.bool)
                # Prefix tokens can attend to all prefix tokens
                mask[:prefix_len, :prefix_len] = False
                # Suffix tokens can attend to all prefix tokens
                mask[prefix_len:, :prefix_len] = False
                # Suffix tokens have causal attention among themselves
                suffix_len = t - prefix_len
                if suffix_len > 0:
                    suffix_causal = torch.triu(
                        torch.ones(suffix_len, suffix_len, device=x.device, dtype=torch.bool),
                        diagonal=1,
                    )
                    mask[prefix_len:, prefix_len:] = suffix_causal

                # Convert to float mask for SDPA (True -> -inf, False -> 0)
                float_mask = torch.zeros(t, t, device=x.device, dtype=q.dtype)
                float_mask.masked_fill_(mask, float("-inf"))
                self._sdpa_mask_cache[cache_key] = float_mask

            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=self._sdpa_mask_cache[cache_key],
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,  # We provide explicit mask
            )
        else:
            # CPU fallback - manual attention
            mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TitansBlock(nn.Module):
    """
    Memory as Context (MAC) Transformer Block.

    Fixed version:
    - Uses per-sequence memory state
    - Proper prefix-LM attention masking
    - Updates memory during both training and inference
    - FIXED: Causal memory retrieval
    """

    def __init__(self, config: TitansConfig, has_memory: bool = True) -> None:
        super().__init__()
        self.config = config  # type: ignore[assignment]
        self.has_memory = has_memory
        self.num_longterm_mem = config.num_longterm_mem if has_memory else 0
        self.num_persist_mem = config.num_persist_mem if has_memory else 0
        self.update_memory = True  # Can be disabled during training for speed

        # Neural memory module (only if this layer has memory)
        if has_memory:
            self.memory = NeuralMemory(config)
            self.persist_mem = nn.Parameter(
                torch.randn(1, config.num_persist_mem, config.n_embd) * 0.02
            )
        else:
            self.memory = None  # type: ignore[assignment]
            self.persist_mem = None  # type: ignore[assignment]

        # Layer norms
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        # Attention and MLP
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState | None:
        """Initialize memory state for this block."""
        if not self.has_memory:
            return None
        return self.memory.init_state(batch_size, device)

    def reset_state(self, state: MemoryState | None) -> None:
        """Reset memory state in-place."""
        if self.has_memory and state is not None:
            self.memory.reset_state(state)

    def forward(
        self,
        x: torch.Tensor,
        memory_state: MemoryState | None,
        packed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MemoryState | None]:
        """
        Forward pass with memory integration.

        Args:
            x: Input tensor [B, T, C]
            memory_state: Current memory state (None if no memory)
            packed_mask: Optional [B, T, T] attention mask for packed sequences

        Returns:
            output: Transformed tensor [B, T, C]
            new_state: Updated memory state (None if no memory)
        """
        # If no memory, just do standard transformer block
        if not self.has_memory:
            x = x + self.attn(self.ln_1(x), packed_mask=packed_mask)
            x = x + self.mlp(self.ln_2(x))
            return x, None

        b, _t, _c = x.shape

        # 1. Retrieve from long-term memory (uses PREVIOUS segment, not current - FIXED)
        mem_context = self.memory(x, memory_state)  # [B, num_longterm_mem, C]

        # 2. Expand persistent memory for batch
        persist = self.persist_mem.expand(b, -1, -1)  # [B, num_persist_mem, C]

        # 3. Concatenate: [memory_context, persistent_memory, current_segment]
        prefix_len = self.num_longterm_mem + self.num_persist_mem
        context = torch.cat([mem_context, persist, x], dim=1)  # [B, prefix_len + T, C]

        # 4. Self-attention with prefix-LM mask
        # Note: For packed sequences with memory, we need to expand the mask
        # to include prefix tokens (memory can attend to all, segment uses packed_mask)
        if packed_mask is not None:
            # Expand packed_mask to include prefix tokens
            # Prefix tokens can attend to themselves and be attended by all
            t = x.shape[1]
            full_len = prefix_len + t
            expanded_mask = torch.zeros(b, full_len, full_len, dtype=torch.bool, device=x.device)
            # Prefix tokens can attend to all prefix tokens (bidirectional)
            expanded_mask[:, :prefix_len, :prefix_len] = True
            # Segment tokens can attend to prefix
            expanded_mask[:, prefix_len:, :prefix_len] = True
            # Segment tokens use packed_mask for attending to each other
            expanded_mask[:, prefix_len:, prefix_len:] = packed_mask
            attn_out = self.attn(self.ln_1(context), packed_mask=expanded_mask)
        else:
            attn_out = self.attn(self.ln_1(context), prefix_len=prefix_len)

        # 5. Extract only the current segment positions (residual connection)
        x = x + attn_out[:, prefix_len:]

        # 6. MLP with residual
        x = x + self.mlp(self.ln_2(x))

        # 7. Update memory based on this segment (stores x for next segment's retrieval)
        new_state = self.memory.update(x, memory_state) if self.update_memory else memory_state

        return x, new_state


class TitansGPT(nn.Module):
    """
    Full Titans GPT model with segmented processing and neural memory.

    Fixed version:
    - Proper memory state management
    - Memory updates during training
    - Correct state passing between segments
    - FIXED: Causal memory retrieval
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config  # type: ignore[assignment]

        # Determine which layers have memory
        if config.memory_layer == -2:
            # All layers have memory (backwards compat)
            memory_layers = set(range(config.n_layer))
        elif config.memory_layer == -1:
            # Default: only middle layer has memory (like lucidrains)
            memory_layers = {config.n_layer // 2}
        else:
            # Specific layer
            memory_layers = {config.memory_layer}

        self.memory_layers = memory_layers

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList(
                    [
                        TitansBlock(config, has_memory=(i in memory_layers))
                        for i in range(config.n_layer)
                    ]
                ),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer["wte"].weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Special scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        mem_layers_str = ",".join(str(i) for i in sorted(memory_layers))
        print(
            f"TitansGPT initialized with {self.get_num_params() / 1e6:.2f}M parameters (memory on layer {mem_layers_str})"
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, *, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer["wpe"].weight.numel()  # type: ignore[call-non-callable]
        return n_params

    def init_memory_states(self, batch_size: int, device: torch.device) -> list[MemoryState | None]:
        """Initialize memory states for all layers (None for layers without memory)."""
        return [block.init_state(batch_size, device) for block in self.transformer["h"]]  # type: ignore[not-iterable]

    def reset_memory_states(self, states: list[MemoryState | None]) -> None:
        """Reset memory states in-place - avoids reallocation."""
        for block, state in zip(self.transformer["h"], states):  # type: ignore[not-iterable]
            block.reset_state(state)

    def set_memory_update(self, enabled: bool) -> None:
        """Enable/disable memory updates. Disable during training for speed."""
        for block in self.transformer["h"]:  # type: ignore[not-iterable]
            block.update_memory = enabled

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        memory_states: list[MemoryState | None] | None = None,
        position_ids: torch.Tensor | None = None,
        packed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[MemoryState | None]]:
        """
        Forward pass with segmented processing.

        Args:
            idx: Input token indices [B, T]
            targets: Target token indices [B, T] (optional, for training)
            memory_states: List of memory states per layer (initialized if None)
            position_ids: Optional [B, T] position indices for packed sequences.
                         If None, uses sequential positions 0..T-1.
            packed_mask: Optional [B, T, T] attention mask for packed sequences.
                        True = can attend, False = cannot attend.

        Returns:
            logits: Output logits [B, T, vocab_size]
            loss: Cross-entropy loss (if targets provided)
            memory_states: Updated memory states
        """
        device = idx.device
        b, t = idx.shape
        segment_len = self.config.segment_len

        if t > self.config.block_size:
            msg = f"Sequence length {t} exceeds block size {self.config.block_size}"
            raise ValueError(msg)

        # Initialize memory states if needed
        if memory_states is None:
            memory_states = self.init_memory_states(b, device)

        # Process in segments
        num_segments = (t + segment_len - 1) // segment_len
        all_logits: list[torch.Tensor] = []

        for seg_idx in range(num_segments):
            start = seg_idx * segment_len
            end = min(start + segment_len, t)
            seg_tokens = idx[:, start:end]

            # Token embeddings
            tok_emb = self.transformer["wte"](seg_tokens)

            # Position embeddings - use provided position_ids or sequential
            if position_ids is not None:
                seg_pos = position_ids[:, start:end]
                pos_emb = self.transformer["wpe"](seg_pos)
            else:
                pos = torch.arange(start, end, dtype=torch.long, device=device)
                pos_emb = self.transformer["wpe"](pos)

            x = self.transformer["drop"](tok_emb + pos_emb)

            # Extract segment mask if provided
            seg_mask = None
            if packed_mask is not None:
                seg_mask = packed_mask[:, start:end, start:end]

            # Process through transformer blocks, updating memory states
            new_states: list[MemoryState | None] = []
            for i, block in enumerate(self.transformer["h"]):  # type: ignore[arg-type]
                x, new_state = block(x, memory_states[i], packed_mask=seg_mask)
                new_states.append(new_state)
            memory_states = new_states

            # Final layer norm
            x = self.transformer["ln_f"](x)

            # Project to vocabulary
            logits = self.lm_head(x)
            all_logits.append(logits)

        # Concatenate all segment logits
        final_logits = torch.cat(all_logits, dim=1)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Use fused Triton cross-entropy when available (CUDA only)
            if _TRITON_AVAILABLE and triton_cross_entropy is not None and final_logits.is_cuda:
                # triton_cross_entropy handles ignore_index=-1 internally
                loss = triton_cross_entropy(
                    final_logits.view(-1, final_logits.size(-1)),
                    targets.view(-1),
                )
            else:
                loss = F.cross_entropy(
                    final_logits.view(-1, final_logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1,
                )

        return final_logits, loss, memory_states

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        """Configure optimizer with weight decay."""
        decay: set[str] = set()
        no_decay: set[str] = set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, _p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                # Skip lm_head.weight as it's tied to wte.weight
                if fpn == "lm_head.weight":
                    continue

                # Check no_decay conditions first (they take priority)
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif "memory_mlp" in fpn or "persist_mem" in fpn or "init_query" in fpn:
                    # Memory MLP weights are templates for per-sequence memory
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

        param_dict = dict(self.named_parameters())
        inter_params = decay & no_decay
        if inter_params:
            msg = f"Parameters in both decay/no_decay: {inter_params}"
            raise ValueError(msg)

        missing = param_dict.keys() - (decay | no_decay)
        no_decay.update(missing)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

        use_fused = device_type == "cuda"
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        memory_states: list[MemoryState] | None = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively with memory."""
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )

            logits, _, memory_states = self(idx_cond, memory_states=memory_states)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# Quick test
if __name__ == "__main__":
    test_config = TitansConfig(
        block_size=256,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        segment_len=32,
    )
    model = TitansGPT(test_config)

    # Test forward pass
    test_x = torch.randint(0, 1000, (2, 64))
    test_logits, test_loss, test_states = model(test_x, targets=test_x)
    print(f"Logits shape: {test_logits.shape}")
    if test_loss is not None:
        print(f"Loss: {test_loss.item():.4f}")

    # Test that memory states are updated
    print(f"Memory state step: {test_states[0].step}")

    # Test generation
    prompt = torch.randint(0, 1000, (1, 8))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
