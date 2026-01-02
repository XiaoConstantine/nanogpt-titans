"""Muon optimizer for MLX. https://github.com/KellerJordan/Muon"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn


def zeropower_via_newtonschulz5(G: mx.array, steps: int = 5, eps: float = 1e-7) -> mx.array:
    """Newton-Schulz iteration to compute G @ (G.T @ G)^{-1/2}."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.astype(mx.bfloat16)
    X = X / (mx.linalg.norm(X) + eps)
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X.astype(G.dtype)


class Muon:
    """Muon optimizer for MLX. Use for hidden layer weights (2D matrices only)."""

    def __init__(self, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay
        self.state: dict[int, mx.array] = {}

    def apply_gradients(self, grads: dict, params: dict) -> dict:
        """Apply Muon update to parameters."""
        new_params = {}
        for k, g in grads.items():
            p = params[k]
            param_id = id(p)

            # Weight decay
            if self.weight_decay > 0:
                p = p * (1 - self.lr * self.weight_decay)

            # Newton-Schulz for 2D+ tensors
            if g.ndim >= 2:
                orig_shape = g.shape
                if g.ndim > 2:
                    g = g.reshape(g.shape[0], -1)
                g = zeropower_via_newtonschulz5(g, steps=self.ns_steps)
                if len(orig_shape) > 2:
                    g = g.reshape(orig_shape)

            # Momentum
            if param_id not in self.state:
                self.state[param_id] = mx.zeros_like(p)
            buf = self.state[param_id]
            buf = self.momentum * buf + g
            self.state[param_id] = buf

            # Nesterov
            if self.nesterov:
                g = g + self.momentum * buf

            new_params[k] = p - self.lr * g
        return new_params


def create_muon_optimizer(model: nn.Module, muon_lr: float = 0.02, adam_lr: float = 6e-4,
                          weight_decay: float = 0.1) -> tuple:
    """Create hybrid Muon + AdamW optimizers for MLX model."""
    from mlx.optimizers import AdamW

    muon_patterns = {"c_attn.weight", "c_proj.weight", "c_fc.weight", "query_proj.weight",
                     "key_proj.weight", "value_proj.weight", "out_proj.weight"}

    muon_keys, adam_keys = [], []
    for k, v in model.parameters().items():
        is_muon = v.ndim == 2 and any(pat in k for pat in muon_patterns)
        if is_muon:
            muon_keys.append(k)
        else:
            adam_keys.append(k)

    muon_opt = Muon(lr=muon_lr, weight_decay=weight_decay)
    adam_opt = AdamW(learning_rate=adam_lr, weight_decay=weight_decay)

    print(f"Muon: {len(muon_keys)} params, lr={muon_lr}")
    print(f"AdamW: {len(adam_keys)} params, lr={adam_lr}")

    return muon_opt, adam_opt, muon_keys, adam_keys


def step(model: nn.Module, grads: dict, muon_opt: Muon, adam_opt, muon_keys: list, adam_keys: list):
    """Apply hybrid optimizer step."""
    params = model.parameters()

    # Split grads
    muon_grads = {k: grads[k] for k in muon_keys if k in grads}
    adam_grads = {k: grads[k] for k in adam_keys if k in grads}

    # Apply updates
    muon_params = {k: params[k] for k in muon_keys}
    adam_params = {k: params[k] for k in adam_keys}

    new_muon = muon_opt.apply_gradients(muon_grads, muon_params)
    new_adam = adam_opt.apply_gradients(adam_grads, adam_params)

    # Merge and update
    new_params = {**new_muon, **new_adam}
    model.update(new_params)
