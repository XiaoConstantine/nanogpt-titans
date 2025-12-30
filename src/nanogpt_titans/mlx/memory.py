"""
MLX Neural Memory implementations for TITANS architecture.

This module provides MLX-native implementations of:
- MLXMemoryState: Per-sequence memory state
- MLXNeuralMemory: Neural memory with test-time learning
- MLXCMSState: State for multi-level memory
- MLXContinuumMemorySystem: Multi-frequency memory system

Matches PyTorch HOPE architecture exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class MLXMemoryState:
    """
    Per-sequence memory state - stores MLP weights as the memory.

    Matches PyTorch MemoryState from model.py.
    """

    # MLP weights per batch item: {name: [B, *param_shape]}
    weights: dict[str, mx.array]
    # Last momentum value for each weight: {name: [B, *param_shape]}
    last_momentum: dict[str, mx.array]
    # Last segment's output for causal retrieval: [B, T, C] or None
    last_segment_output: mx.array | None = None
    step: int = 0


@dataclass
class MLXCMSState:
    """State for ContinuumMemorySystem - stores state for each level."""

    level_states: list[MLXMemoryState]
    step: int = 0


class MLXNeuralMemory(nn.Module):
    """
    Neural Memory module with Test-Time Learning (TTL).

    Matches PyTorch NeuralMemory exactly:
    - Memory IS the MLP weights (stored per-sequence in MemoryState)
    - Surprise = gradient of MSE loss w.r.t. MLP weights
    - Memory update = weight update with momentum and decay
    - Retrieval = forward pass through MLP with per-sequence weights

    FIXED: Retrieval now uses previous segment's output (stored in state)
    to avoid leaking future token information.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        expansion: int = 2,
        memory_lr: float = 0.01,
        memory_momentum: float = 0.9,
        memory_decay: float = 0.001,
        adaptive: bool = True,
        lr_max: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.hidden_dim = dim * expansion

        # Hyperparameters for test-time learning (used when adaptive=False)
        self.lr = memory_lr
        self.momentum = memory_momentum
        self.decay = memory_decay

        # Adaptive memory parameters (matches PyTorch model.py)
        self.adaptive = adaptive
        self.lr_max = lr_max

        # Projections (matches PyTorch)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Template MLP weights - these are cloned per-sequence as initial memory
        # For a 2-layer MLP: h = silu(x @ W0.T), out = h @ W1.T
        # W0: [hidden_dim, dim], W1: [dim, hidden_dim]
        # NOTE: Using Linear layers to make weights trainable parameters
        self._template_mlp_w0 = nn.Linear(dim, self.hidden_dim, bias=False)
        self._template_mlp_w1 = nn.Linear(self.hidden_dim, dim, bias=False)

        # Initialize with small weights
        self._template_mlp_w0.weight = mx.random.normal((self.hidden_dim, dim)) * 0.02
        self._template_mlp_w1.weight = mx.random.normal((dim, self.hidden_dim)) * 0.02

        # Learned initial query for first segment (when no previous output exists)
        self.init_query = mx.random.normal((1, 1, dim)) * 0.02

        # Adaptive memory projections (when enabled)
        if self.adaptive:
            self.to_lr = nn.Linear(dim, 1)
            self.to_momentum = nn.Linear(dim, 1)
            self.to_decay = nn.Linear(dim, 1)

            # Initialize with small weights for stable training
            self.to_lr.weight = mx.zeros(self.to_lr.weight.shape)
            self.to_momentum.weight = mx.zeros(self.to_momentum.weight.shape)
            self.to_decay.weight = mx.zeros(self.to_decay.weight.shape)

            # Bias initialized for reasonable defaults (matches PyTorch model.py:399-404)
            self.to_lr.bias = mx.array([0.0])  # sigmoid(0)=0.5 -> lr = 0.5*lr_max
            self.to_momentum.bias = mx.array([2.0])  # sigmoid(2)≈0.88
            self.to_decay.bias = mx.array([-4.0])  # sigmoid(-4)≈0.018

    def init_state(self, batch_size: int) -> MLXMemoryState:
        """Initialize memory state - clone MLP weights for each batch item."""
        # Get template weights from Linear layers (trainable parameters)
        # w0 has shape [hidden_dim, dim], w1 has shape [dim, hidden_dim]
        template_w0 = self._template_mlp_w0.weight  # [H, C]
        template_w1 = self._template_mlp_w1.weight  # [C, H]

        # Expand template weights to [B, *param_shape]
        w0 = mx.broadcast_to(template_w0[None, :, :], (batch_size, self.hidden_dim, self.dim))
        w1 = mx.broadcast_to(template_w1[None, :, :], (batch_size, self.dim, self.hidden_dim))

        # Stop gradient - state weights are updated by test-time learning,
        # not by backprop. Template weights receive gradients via internal loss.
        weights = {
            "w0": mx.stop_gradient(w0),
            "w1": mx.stop_gradient(w1),
        }

        # Initialize momentum to zero
        last_momentum = {
            "w0": mx.zeros_like(weights["w0"]),
            "w1": mx.zeros_like(weights["w1"]),
        }

        return MLXMemoryState(
            weights=weights, last_momentum=last_momentum, last_segment_output=None, step=0
        )

    def _batched_mlp_forward(self, x: mx.array, weights: dict[str, mx.array]) -> mx.array:
        """
        Batched forward pass through memory MLP.

        Args:
            x: Input [B, T, C]
            weights: Per-batch weights {'w0': [B, H, C], 'w1': [B, C, H]}

        Returns:
            Output [B, T, C]
        """
        W0 = weights["w0"]  # [B, H, C]
        W1 = weights["w1"]  # [B, C, H]

        # Layer 0: h = silu(x @ W0.T)
        h_pre = mx.matmul(x, mx.transpose(W0, axes=(0, 2, 1)))  # [B, T, H]
        h = mx.sigmoid(h_pre) * h_pre  # SiLU activation

        # Layer 1: out = h @ W1.T
        out = mx.matmul(h, mx.transpose(W1, axes=(0, 2, 1)))  # [B, T, C]

        return out

    def _compute_gradients(
        self, keys: mx.array, values: mx.array, weights: dict[str, mx.array]
    ) -> dict[str, mx.array]:
        """
        Compute gradients of MSE loss w.r.t. MLP weights.

        MSE Loss = mean(|M(keys) - values|^2)

        Args:
            keys: Input keys [B, T, C]
            values: Target values [B, T, C]
            weights: Per-batch MLP weights

        Returns:
            Gradients {name: [B, *param_shape]} (aggregated over T)
        """
        _B, _T, C = keys.shape

        W0 = weights["w0"]  # [B, H, C]
        W1 = weights["w1"]  # [B, C, H]

        # Forward pass
        h_pre = mx.matmul(keys, mx.transpose(W0, axes=(0, 2, 1)))  # [B, T, H]
        sig_h = mx.sigmoid(h_pre)
        h = h_pre * sig_h  # SiLU
        pred = mx.matmul(h, mx.transpose(W1, axes=(0, 2, 1)))  # [B, T, C]

        # Backward pass
        # dL/d_pred = 2 * (pred - values) / C
        d_pred = (2.0 / C) * (pred - values)  # [B, T, C]

        # dL/dW1 = d_pred.T @ h = [B, C, T] @ [B, T, H] = [B, C, H]
        dW1 = mx.matmul(mx.transpose(d_pred, axes=(0, 2, 1)), h)  # [B, C, H]

        # dL/dh = d_pred @ W1 = [B, T, C] @ [B, C, H] = [B, T, H]
        dh = mx.matmul(d_pred, W1)  # [B, T, H]

        # SiLU backward: d_silu/d_x = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        silu_grad = sig_h * (1 + h_pre * (1 - sig_h))
        dh_pre = dh * silu_grad  # [B, T, H]

        # dL/dW0 = dh_pre.T @ keys = [B, H, T] @ [B, T, C] = [B, H, C]
        dW0 = mx.matmul(mx.transpose(dh_pre, axes=(0, 2, 1)), keys)  # [B, H, C]

        return {"w0": dW0, "w1": dW1}

    def compute_internal_loss(self, x: mx.array, _state: MLXMemoryState) -> mx.array:
        """
        Compute internal loss: ||M(keys) - values||^2 + adaptive param regularization

        This is the reconstruction error that teaches the TEMPLATE weights
        to store/retrieve patterns. From Titans paper Eq. 9.

        Also includes gradient path through adaptive params (to_lr, to_momentum, to_decay)
        so they receive training signal.

        Args:
            x: Input hidden states [B, T, C]
            state: Current memory state (unused - we use template weights)

        Returns:
            Scalar internal loss
        """
        _B, _T, _C = x.shape

        # Clip input for numerical stability
        x_clipped = mx.clip(x, -10.0, 10.0)

        # Project to keys and values
        keys = self.key_proj(x_clipped)
        values = self.value_proj(x_clipped)

        # Clip projections
        keys = mx.clip(keys, -10.0, 10.0)
        values = mx.clip(values, -10.0, 10.0)

        # Use TEMPLATE weights (trainable parameters) for internal loss
        # This trains the template so init_state produces better initial memory
        W0 = self._template_mlp_w0.weight  # [H, C]
        W1 = self._template_mlp_w1.weight  # [C, H]

        # Forward through template MLP (non-batched, then broadcast)
        # Layer 0: h = silu(keys @ W0.T)
        h_pre = mx.matmul(keys, W0.T)  # [B, T, H]
        h = mx.sigmoid(h_pre) * h_pre  # SiLU

        # Layer 1: pred = h @ W1.T
        pred = mx.matmul(h, W1.T)  # [B, T, C]

        # Clip prediction
        pred = mx.clip(pred, -10.0, 10.0)

        # MSE loss for reconstruction
        diff = pred - values
        recon_loss = mx.mean(diff * diff)

        # =========================================================================
        # Adaptive parameter training: DIRECT gradient path through to_lr etc.
        # =========================================================================
        if self.adaptive:
            # Direct use of adaptive params in loss - this MUST produce gradients
            # Simply add the raw output of the adaptive projections to the loss
            lr_out = self.to_lr(x_clipped)  # [B, T, 1]
            mom_out = self.to_momentum(x_clipped)  # [B, T, 1]
            decay_out = self.to_decay(x_clipped)  # [B, T, 1]

            # L2 regularization on adaptive outputs - gives direct gradients
            # This encourages adaptive params to produce reasonable values
            adaptive_reg = (
                mx.mean(lr_out**2) + mx.mean(mom_out**2) + mx.mean(decay_out**2)
            ) * 0.001  # Small weight

            # Also train query_proj
            query_out = self.query_proj(x_clipped)
            query_reg = mx.mean(query_out**2) * 0.001

            total_loss = recon_loss + adaptive_reg + query_reg
        else:
            total_loss = recon_loss

        # Clip final loss to prevent gradient explosion
        total_loss = mx.minimum(total_loss, mx.array(100.0))

        return total_loss

    def __call__(self, x: mx.array, state: MLXMemoryState) -> mx.array:
        """
        Retrieve from memory using PREVIOUS segment's output (causal).

        Args:
            x: Input tensor [B, T, C] - current segment
            state: Current memory state (contains previous segment's output)

        Returns:
            Retrieved memory [B, T, C]
        """
        B, T, C = x.shape

        # Use previous segment's output for retrieval (causal)
        if state.last_segment_output is not None:
            query_source = state.last_segment_output
        else:
            # First segment: use learned initial query
            query_source = mx.broadcast_to(self.init_query, (B, 1, C))

        # Project to queries
        queries = self.query_proj(query_source)

        # Retrieve using per-batch memory MLP weights
        retrieved = self._batched_mlp_forward(queries, state.weights)

        # Pool to match sequence length T if needed
        T_prev = retrieved.shape[1]
        if T_prev != T:
            if T_prev > T:
                # Truncate
                retrieved = retrieved[:, :T, :]
            else:
                # Repeat last token to fill
                padding = mx.broadcast_to(retrieved[:, -1:, :], (B, T - T_prev, C))
                retrieved = mx.concatenate([retrieved, padding], axis=1)

        return self.out_proj(retrieved)

    def update(self, x: mx.array, state: MLXMemoryState) -> MLXMemoryState:
        """
        Update memory based on surprise (gradient of MSE loss w.r.t. MLP weights).

        Args:
            x: Current segment output [B, T, C]
            state: Current memory state

        Returns:
            new_state: Updated memory state
        """
        # Project to keys and values for storage
        keys = self.key_proj(x)
        values = self.value_proj(x)

        # Compute gradients (aggregated over T)
        grads = self._compute_gradients(keys, values, state.weights)

        # Compute adaptive parameters if enabled
        if self.adaptive:
            adaptive_lr = mx.sigmoid(self.to_lr(x)) * self.lr_max  # [B, T, 1]
            adaptive_momentum = mx.sigmoid(self.to_momentum(x))  # [B, T, 1]
            adaptive_decay = mx.sigmoid(self.to_decay(x))  # [B, T, 1]

            # Average across tokens
            lr_param = mx.mean(adaptive_lr, axis=1)[:, 0]  # [B]
            mom_param = mx.mean(adaptive_momentum, axis=1)[:, 0]  # [B]
            decay_param = mx.mean(adaptive_decay, axis=1)[:, 0]  # [B]
        else:
            lr_param = self.lr
            mom_param = self.momentum
            decay_param = self.decay

        # Apply momentum and update weights
        new_weights = {}
        new_momentum = {}

        for name in state.weights:
            g = grads[name]
            m_prev = state.last_momentum[name]
            w_prev = state.weights[name]

            # Momentum update: m = momentum * m_prev + (1 - momentum) * g
            if self.adaptive:
                # Broadcast [B] to [B, 1, 1] for weight shape
                mom_expanded = mom_param.reshape((-1,) + (1,) * (g.ndim - 1))
                lr_expanded = lr_param.reshape((-1,) + (1,) * (g.ndim - 1))
                decay_expanded = decay_param.reshape((-1,) + (1,) * (g.ndim - 1))

                m = mom_expanded * m_prev + (1 - mom_expanded) * g
                decay_factor = 1 - decay_expanded
                w = decay_factor * w_prev - lr_expanded * m
            else:
                m = mom_param * m_prev + (1 - mom_param) * g
                decay_factor = 1 - decay_param
                w = decay_factor * w_prev - lr_param * m

            # Clamp for numerical stability
            w = mx.clip(w, -10.0, 10.0)

            new_weights[name] = w
            new_momentum[name] = m

        return MLXMemoryState(
            weights=new_weights,
            last_momentum=new_momentum,
            last_segment_output=x,  # Store for next causal retrieval
            step=state.step + 1,
        )


class MLXContinuumMemorySystem(nn.Module):
    """
    Multi-frequency memory system from Nested Learning.

    Different memory levels update at different rates:
    - Level 0: Updates every segment (fast, working memory)
    - Level 1: Updates every 4 segments (medium, episodic)
    - Level 2: Updates every 16 segments (slow, semantic)

    This matches the PyTorch ContinuumMemorySystem exactly.
    """

    def __init__(
        self,
        dim: int,
        num_levels: int = 3,
        update_frequencies: tuple = (1, 4, 16),
        memory_depth: int = 2,
        memory_expansion: int = 2,
        adaptive: bool = True,
        lr_max: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels
        self.update_frequencies = update_frequencies

        # Create a memory module for each level
        self.memories = [
            MLXNeuralMemory(
                dim,
                depth=memory_depth,
                expansion=memory_expansion,
                adaptive=adaptive,
                lr_max=lr_max,
            )
            for _ in range(num_levels)
        ]

        # Learnable weights for combining levels (initialized uniform)
        self.level_weights = mx.zeros((num_levels,))

    def init_state(self, batch_size: int) -> MLXCMSState:
        """Initialize memory state for all levels."""
        level_states = [mem.init_state(batch_size) for mem in self.memories]
        return MLXCMSState(level_states=level_states, step=0)

    def __call__(self, hidden_states: mx.array, state: MLXCMSState) -> mx.array:
        """
        Retrieve from all memory levels and combine.

        Args:
            hidden_states: Current segment [B, T, C]
            state: CMS state containing per-level memory states

        Returns:
            Combined memory output [B, T, C]
        """
        # Compute softmax weights
        weights = mx.softmax(self.level_weights)

        # Retrieve from each level and combine
        level_outputs = []
        for i, mem in enumerate(self.memories):
            level_outputs.append(mem(hidden_states, state.level_states[i]))

        # Stack and apply weights: [num_levels, B, T, C]
        stacked = mx.stack(level_outputs, axis=0)

        # Weight and sum
        weights_expanded = weights.reshape(-1, 1, 1, 1)
        combined = mx.sum(weights_expanded * stacked, axis=0)  # [B, T, C]

        return combined

    def compute_internal_loss(self, x: mx.array, state: MLXCMSState) -> mx.array:
        """
        Compute internal loss for CMS (uses first/fastest level).

        Args:
            x: Input hidden states [B, T, C]
            state: Current CMS state

        Returns:
            Scalar internal loss
        """
        # Use first memory level for internal loss
        return self.memories[0].compute_internal_loss(x, state.level_states[0])

    def update(self, hidden_states: mx.array, state: MLXCMSState) -> MLXCMSState:
        """
        Update memory levels based on their frequencies.

        Args:
            hidden_states: Current segment output [B, T, C]
            state: Current CMS state

        Returns:
            Updated CMS state
        """
        new_level_states = []
        for i, (mem, freq) in enumerate(zip(self.memories, self.update_frequencies)):
            # Update this level if step is a multiple of its frequency
            if state.step % freq == 0:
                new_state = mem.update(hidden_states, state.level_states[i])
            else:
                new_state = state.level_states[i]
            new_level_states.append(new_state)

        return MLXCMSState(level_states=new_level_states, step=state.step + 1)
