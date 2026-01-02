"""Muon optimizer. https://github.com/KellerJordan/Muon"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Newton-Schulz iteration to compute G @ (G.T @ G)^{-1/2}."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon optimizer for hidden layer weights (2D matrices only)."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            lr, momentum, nesterov = group["lr"], group["momentum"], group["nesterov"]
            ns_steps, wd = group["ns_steps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if wd > 0:
                    p.mul_(1 - lr * wd)

                if g.dim() >= 2:
                    orig_shape = g.shape
                    if g.dim() > 2:
                        g = g.view(g.size(0), -1)
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                    if len(orig_shape) > 2:
                        g = g.view(orig_shape)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g + momentum * buf if nesterov else buf

                p.add_(g, alpha=-lr)
        return loss
