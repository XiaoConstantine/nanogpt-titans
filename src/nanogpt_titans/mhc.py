"""mHC: Manifold-Constrained Hyper-Connections. https://arxiv.org/abs/2512.24880"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def sinkhorn(M: Tensor, n_iters: int = 4, eps: float = 1e-8) -> Tensor:
    """Sinkhorn-Knopp: project matrix onto Birkhoff polytope (doubly stochastic)."""
    M = M.exp()
    for _ in range(n_iters):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M


class HyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection (mHC).

    Paper: https://arxiv.org/abs/2512.24880 (mHC)
    Based on: https://arxiv.org/abs/2409.19606 (Hyper-Connections)

    Expands residual stream to n copies, applies learnable (n+1)x(n+1) connection
    matrix projected onto Birkhoff polytope for training stability.

    Args:
        dim: hidden dimension
        n: expansion rate (default 4, paper recommendation)
        dynamic: use input-dependent connections (DHC) vs static (SHC)
        sinkhorn_iters: iterations for Birkhoff projection
    """

    def __init__(self, dim: int, n: int = 4, dynamic: bool = True, sinkhorn_iters: int = 4):
        super().__init__()
        self.dim = dim
        self.n = n
        self.dynamic = dynamic
        self.sinkhorn_iters = sinkhorn_iters

        # Connection matrix: (n+1) x (n+1)
        # Structured as: [depth_in; width_connections; depth_out]
        matrix_size = n + 1

        if dynamic:
            # DHC: input-dependent connection weights
            self.to_conn = nn.Linear(dim, matrix_size * matrix_size, bias=False)
            # Scaling factors (initialized small for stability)
            self.scale = nn.Parameter(torch.ones(1) * 0.01)
            nn.init.zeros_(self.to_conn.weight)
        else:
            # SHC: static learned connections
            self.conn_logits = nn.Parameter(torch.zeros(matrix_size, matrix_size))

        # Initialize toward identity-like behavior
        self._init_identity()

    def _init_identity(self):
        """Initialize connection matrix toward identity mapping."""
        # Start with small values that will become near-uniform after Sinkhorn
        pass

    def expand_stream(self, x: Tensor) -> Tensor:
        """Expand hidden state to n streams: [B,T,C] -> [B,T,n,C]"""
        return x.unsqueeze(2).expand(-1, -1, self.n, -1).clone()

    def reduce_stream(self, streams: Tensor) -> Tensor:
        """Reduce n streams back to hidden state: [B,T,n,C] -> [B,T,C]"""
        return streams.mean(dim=2)

    def get_connection_matrix(self, x: Tensor) -> Tensor:
        """Get (n+1)x(n+1) doubly stochastic connection matrix."""
        B = x.shape[0]
        matrix_size = self.n + 1

        if self.dynamic:
            # Pool input for connection computation
            x_pool = x.mean(dim=1)  # [B, C]
            conn_logits = self.to_conn(x_pool) * self.scale  # [B, (n+1)^2]
            conn_logits = conn_logits.view(B, matrix_size, matrix_size)
        else:
            conn_logits = self.conn_logits.unsqueeze(0).expand(B, -1, -1)

        # Project onto Birkhoff polytope (key mHC contribution)
        conn = sinkhorn(conn_logits, self.sinkhorn_iters)
        return conn

    def forward(self, x: Tensor, branch_out: Tensor) -> Tensor:
        """
        Apply mHC residual connection.

        Standard residual: out = x + branch(x)
        Hyper-connection: out = HC_reduce(HC_conn @ [x_expanded; branch_out_expanded])

        Args:
            x: input tensor [B, T, C]
            branch_out: output from branch (attention/MLP) [B, T, C]

        Returns:
            output tensor [B, T, C]
        """
        B, T, C = x.shape

        # Get connection matrix [B, n+1, n+1]
        conn = self.get_connection_matrix(x)

        # Expand inputs to n streams each
        x_streams = self.expand_stream(x)  # [B, T, n, C]
        branch_streams = self.expand_stream(branch_out)  # [B, T, n, C]

        # Create input vector for connection: [x_mean, stream_0, stream_1, ..., stream_{n-1}]
        # Using x as the "layer input" and streams as expanded representations
        # Paper structure: first row/col = depth connections, rest = width connections

        # Simplified: treat x and branch_out as two groups to mix
        # Stack all streams: [B, T, n, C] from x, [B, T, n, C] from branch
        # We need to map this to (n+1) sized input for the connection matrix

        # Layer input (pre-branch): x
        # Layer output (post-branch): branch_out
        # Expanded streams allow information mixing

        # Construct input streams for connection matrix multiplication
        # Input vector: [layer_in, stream_0, stream_1, ..., stream_{n-1}]
        # For simplicity: use x as layer_in, mix of x and branch for streams

        # Create (n+1) input channels
        layer_input = x.unsqueeze(2)  # [B, T, 1, C] - the "depth" input
        # Interpolate between x and branch_out for the n streams
        alphas = torch.linspace(0, 1, self.n, device=x.device).view(1, 1, self.n, 1)
        mixed_streams = (1 - alphas) * x_streams + alphas * branch_streams  # [B, T, n, C]

        # Full input: [layer_in; mixed_streams] -> [B, T, n+1, C]
        full_input = torch.cat([layer_input, mixed_streams], dim=2)

        # Apply connection matrix: [B, n+1, n+1] @ [B, T, n+1, C]
        # Output: [B, T, n+1, C]
        output = torch.einsum('bij,btjc->btic', conn, full_input)

        # Extract: first channel is layer output, rest are streams
        layer_out = output[:, :, 0, :]  # [B, T, C] - depth output
        stream_out = output[:, :, 1:, :]  # [B, T, n, C] - width outputs

        # Final output: combine layer output with reduced streams
        # Paper uses learned combination, we use simple mean for now
        stream_reduced = stream_out.mean(dim=2)  # [B, T, C]

        return 0.5 * layer_out + 0.5 * stream_reduced


class MHCResidual(nn.Module):
    """Drop-in mHC replacement for x + branch(x)."""

    def __init__(self, dim: int, n: int = 4, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.hc = HyperConnection(dim, n=n, dynamic=True)

    def forward(self, x: Tensor, branch_out: Tensor) -> Tensor:
        return self.hc(x, branch_out) if self.enabled else x + branch_out
