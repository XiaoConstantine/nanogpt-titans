"""Tests for MLX mHC implementation."""

import pytest

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")

if MLX_AVAILABLE:
    from nanogpt_titans.mhc_mlx import HyperConnection, MHCResidual, sinkhorn


class TestSinkhorn:
    """Tests for Sinkhorn-Knopp algorithm."""

    def test_doubly_stochastic_rows(self):
        """Test rows sum to approximately 1."""
        M = mx.random.normal((4, 5, 5))
        ds = sinkhorn(M, n_iters=10)
        mx.eval(ds)
        row_sums = mx.sum(ds, axis=-1)
        # Use 1e-3 tolerance - sufficient for training stability
        assert mx.allclose(row_sums, mx.ones_like(row_sums), atol=1e-3), (
            f"Row sums should be ~1, got {row_sums}"
        )

    def test_doubly_stochastic_cols(self):
        """Test columns sum to approximately 1."""
        M = mx.random.normal((4, 5, 5))
        ds = sinkhorn(M, n_iters=10)
        mx.eval(ds)
        col_sums = mx.sum(ds, axis=-2)
        assert mx.allclose(col_sums, mx.ones_like(col_sums), atol=1e-4), (
            f"Col sums should be ~1, got {col_sums}"
        )

    def test_positive_entries(self):
        """Test all entries are positive after Sinkhorn."""
        M = mx.random.normal((2, 5, 5))
        ds = sinkhorn(M, n_iters=8)
        mx.eval(ds)
        assert mx.all(ds > 0), "All entries should be positive"

    def test_convergence_with_more_iters(self):
        """Test more iterations improve doubly stochastic property."""
        M = mx.random.normal((2, 5, 5))

        ds4 = sinkhorn(M, n_iters=4)
        ds8 = sinkhorn(M, n_iters=8)
        mx.eval(ds4, ds8)

        # Measure deviation from doubly stochastic
        def deviation(m):
            row_dev = mx.mean(mx.abs(mx.sum(m, axis=-1) - 1))
            col_dev = mx.mean(mx.abs(mx.sum(m, axis=-2) - 1))
            return row_dev + col_dev

        dev4 = deviation(ds4)
        dev8 = deviation(ds8)
        mx.eval(dev4, dev8)

        assert dev8 <= dev4, "More iterations should reduce deviation"

    def test_batched_input(self):
        """Test Sinkhorn works with batched inputs."""
        M = mx.random.normal((8, 5, 5))
        ds = sinkhorn(M, n_iters=6)
        mx.eval(ds)
        assert ds.shape == (8, 5, 5)


class TestHyperConnection:
    """Tests for HyperConnection module."""

    def test_output_shape(self):
        """Test forward pass produces correct output shape."""
        dim, n = 64, 4
        B, T = 2, 16
        hc = HyperConnection(dim=dim, n=n, dynamic=True)

        x = mx.random.normal((B, T, dim))
        branch_out = mx.random.normal((B, T, dim))

        output = hc(x, branch_out)
        mx.eval(output)

        assert output.shape == (B, T, dim), f"Expected {(B, T, dim)}, got {output.shape}"

    def test_connection_matrix_shape(self):
        """Test connection matrix has correct shape (n+1, n+1)."""
        dim, n = 64, 4
        B, T = 2, 16
        hc = HyperConnection(dim=dim, n=n, dynamic=True)

        x = mx.random.normal((B, T, dim))
        conn = hc.get_connection_matrix(x)
        mx.eval(conn)

        assert conn.shape == (B, n + 1, n + 1), f"Expected {(B, n+1, n+1)}, got {conn.shape}"

    def test_connection_matrix_doubly_stochastic(self):
        """Test connection matrix is doubly stochastic."""
        dim, n = 64, 4
        B, T = 2, 16
        hc = HyperConnection(dim=dim, n=n, dynamic=True, sinkhorn_iters=10)

        x = mx.random.normal((B, T, dim))
        conn = hc.get_connection_matrix(x)
        mx.eval(conn)

        row_sums = mx.sum(conn, axis=-1)
        col_sums = mx.sum(conn, axis=-2)

        assert mx.allclose(row_sums, mx.ones_like(row_sums), atol=1e-3), (
            f"Row sums: {row_sums}"
        )
        assert mx.allclose(col_sums, mx.ones_like(col_sums), atol=1e-3), (
            f"Col sums: {col_sums}"
        )

    def test_static_mode(self):
        """Test static (SHC) mode works."""
        dim, n = 64, 4
        B, T = 2, 16
        hc = HyperConnection(dim=dim, n=n, dynamic=False)

        x = mx.random.normal((B, T, dim))
        branch_out = mx.random.normal((B, T, dim))

        output = hc(x, branch_out)
        mx.eval(output)

        assert output.shape == (B, T, dim)

    def test_expand_stream_shape(self):
        """Test expand_stream produces correct shape."""
        dim, n = 64, 4
        B, T = 2, 16
        hc = HyperConnection(dim=dim, n=n)

        x = mx.random.normal((B, T, dim))
        expanded = hc.expand_stream(x)
        mx.eval(expanded)

        assert expanded.shape == (B, T, n, dim), f"Expected {(B, T, n, dim)}, got {expanded.shape}"

    def test_reduce_stream_shape(self):
        """Test reduce_stream produces correct shape."""
        dim, n = 64, 4
        B, T = 2, 16
        hc = HyperConnection(dim=dim, n=n)

        streams = mx.random.normal((B, T, n, dim))
        reduced = hc.reduce_stream(streams)
        mx.eval(reduced)

        assert reduced.shape == (B, T, dim), f"Expected {(B, T, dim)}, got {reduced.shape}"

    def test_no_nan_output(self):
        """Test no NaN values in output."""
        dim, n = 64, 4
        B, T = 2, 16
        hc = HyperConnection(dim=dim, n=n, dynamic=True)

        x = mx.random.normal((B, T, dim))
        branch_out = mx.random.normal((B, T, dim))

        output = hc(x, branch_out)
        mx.eval(output)

        assert not mx.any(mx.isnan(output)), "Output contains NaN values"

    def test_different_n_values(self):
        """Test different expansion rates."""
        dim = 64
        B, T = 2, 16

        for n in [2, 4, 8]:
            hc = HyperConnection(dim=dim, n=n, dynamic=True)
            x = mx.random.normal((B, T, dim))
            branch_out = mx.random.normal((B, T, dim))

            output = hc(x, branch_out)
            mx.eval(output)

            assert output.shape == (B, T, dim), f"Failed for n={n}"

            conn = hc.get_connection_matrix(x)
            mx.eval(conn)
            assert conn.shape == (B, n + 1, n + 1), f"Conn shape wrong for n={n}"

    def test_gradients_flow(self):
        """Test gradients flow through the module."""
        dim, n = 32, 4
        B, T = 2, 8
        hc = HyperConnection(dim=dim, n=n, dynamic=True)

        x = mx.random.normal((B, T, dim))
        branch_out = mx.random.normal((B, T, dim))

        def loss_fn(hc, x, branch_out):
            output = hc(x, branch_out)
            return mx.mean(output ** 2)

        loss, grads = mx.value_and_grad(loss_fn)(hc, x, branch_out)
        mx.eval(loss, grads)

        assert not mx.isnan(loss), "Loss is NaN"
        # Check that to_conn has gradients (grads is a dict in MLX)
        assert "to_conn" in grads, "No gradient for to_conn"
        assert "weight" in grads["to_conn"], "No gradient for to_conn.weight"
        assert not mx.all(grads["to_conn"]["weight"] == 0), "Gradients are all zero"


class TestMHCResidual:
    """Tests for MHCResidual drop-in replacement."""

    def test_enabled_mode(self):
        """Test MHCResidual in enabled mode."""
        dim = 64
        B, T = 2, 16
        mhc = MHCResidual(dim=dim, n=4, enabled=True)

        x = mx.random.normal((B, T, dim))
        branch_out = mx.random.normal((B, T, dim))

        output = mhc(x, branch_out)
        mx.eval(output)

        assert output.shape == (B, T, dim)

    def test_disabled_mode(self):
        """Test MHCResidual in disabled mode (fallback to standard residual)."""
        dim = 64
        B, T = 2, 16
        mhc = MHCResidual(dim=dim, n=4, enabled=False)

        x = mx.random.normal((B, T, dim))
        branch_out = mx.random.normal((B, T, dim))

        output = mhc(x, branch_out)
        expected = x + branch_out
        mx.eval(output, expected)

        assert mx.allclose(output, expected), "Disabled mode should be standard residual"

    def test_drop_in_replacement(self):
        """Test MHCResidual works as drop-in for standard residual."""
        dim = 64
        B, T = 2, 16

        x = mx.random.normal((B, T, dim))
        branch_out = mx.random.normal((B, T, dim))

        # Standard residual
        standard_out = x + branch_out

        # MHC residual
        mhc = MHCResidual(dim=dim, n=4, enabled=True)
        mhc_out = mhc(x, branch_out)
        mx.eval(standard_out, mhc_out)

        # Shapes should match
        assert mhc_out.shape == standard_out.shape

        # Values will differ (mHC learns different combination)
        # but both should be valid
        assert not mx.any(mx.isnan(mhc_out))


class TestIntegration:
    """Integration tests."""

    def test_sequential_calls(self):
        """Test multiple sequential forward passes."""
        dim, n = 64, 4
        B, T = 2, 16
        hc = HyperConnection(dim=dim, n=n, dynamic=True)

        x = mx.random.normal((B, T, dim))

        # Multiple sequential passes
        for _ in range(5):
            branch_out = mx.random.normal((B, T, dim))
            x = hc(x, branch_out)
            mx.eval(x)

        assert x.shape == (B, T, dim)
        assert not mx.any(mx.isnan(x))

    def test_batch_size_one(self):
        """Test with batch size 1."""
        dim, n = 64, 4
        B, T = 1, 16
        hc = HyperConnection(dim=dim, n=n, dynamic=True)

        x = mx.random.normal((B, T, dim))
        branch_out = mx.random.normal((B, T, dim))

        output = hc(x, branch_out)
        mx.eval(output)

        assert output.shape == (B, T, dim)

    def test_single_token(self):
        """Test with single token sequence."""
        dim, n = 64, 4
        B, T = 2, 1
        hc = HyperConnection(dim=dim, n=n, dynamic=True)

        x = mx.random.normal((B, T, dim))
        branch_out = mx.random.normal((B, T, dim))

        output = hc(x, branch_out)
        mx.eval(output)

        assert output.shape == (B, T, dim)

    def test_numerical_stability_large_inputs(self):
        """Test numerical stability with large input values."""
        dim, n = 64, 4
        B, T = 2, 16
        hc = HyperConnection(dim=dim, n=n, dynamic=True)

        # Large input values
        x = mx.random.normal((B, T, dim)) * 100
        branch_out = mx.random.normal((B, T, dim)) * 100

        output = hc(x, branch_out)
        mx.eval(output)

        assert not mx.any(mx.isnan(output)), "NaN with large inputs"
        assert not mx.any(mx.isinf(output)), "Inf with large inputs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
