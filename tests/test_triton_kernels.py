"""
Tests for Triton kernel optimizations.

These tests verify that the optimized Triton kernels produce
the same results as the reference PyTorch implementations.

Run with: uv run pytest tests/test_triton_kernels.py -v
"""

import pytest
import torch

from nanogpt_titans.model import (
    TitansConfig,
    NeuralMemory,
    parallel_momentum,
)


# Check if CUDA and Triton are available
CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = False
triton_momentum_update = None
triton_fused_weight_update = None
triton_batched_weight_update = None

if CUDA_AVAILABLE:
    try:
        from nanogpt_titans.triton_kernels import (
            triton_batched_weight_update,
            triton_fused_weight_update,
            triton_momentum_update,
        )
        TRITON_AVAILABLE = True
    except ImportError:
        pass


# --- Triton Momentum Kernel Tests ---

@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
class TestTritonMomentumKernel:
    """Tests for the Triton momentum update kernel."""

    def test_matches_pytorch_sequential(self):
        """Triton kernel should match sequential PyTorch loop."""
        torch.manual_seed(42)

        B, T, D = 4, 64, 384
        momentum_coef = 0.9

        grads = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
        prev_momentum = torch.randn(B, D, device="cuda", dtype=torch.float32)

        # PyTorch reference (sequential loop)
        m_ref = prev_momentum.clone()
        for t in range(T):
            m_ref = momentum_coef * m_ref + (1 - momentum_coef) * grads[:, t]

        # Triton kernel
        m_triton = triton_momentum_update(grads, momentum_coef, prev_momentum)

        torch.testing.assert_close(m_ref, m_triton, rtol=1e-4, atol=1e-4)

    def test_matches_parallel_momentum(self):
        """Triton kernel should match parallel_momentum for final value."""
        torch.manual_seed(123)

        B, T, D = 4, 64, 128
        momentum_coef = 0.9

        grads = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
        prev_momentum = torch.randn(B, D, device="cuda", dtype=torch.float32)

        # parallel_momentum returns all timesteps, take last one
        all_momentum = parallel_momentum(grads.cpu(), momentum_coef, prev_momentum.cpu())
        final_parallel = all_momentum[:, -1].cuda()

        # Triton returns only final
        final_triton = triton_momentum_update(grads, momentum_coef, prev_momentum)

        torch.testing.assert_close(final_parallel, final_triton, rtol=1e-4, atol=1e-4)

    def test_various_shapes(self):
        """Test with various tensor shapes."""
        torch.manual_seed(42)
        momentum_coef = 0.9

        shapes = [
            (1, 32, 64),
            (8, 64, 256),
            (2, 128, 384),
            (4, 16, 768),
        ]

        for B, T, D in shapes:
            grads = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
            prev = torch.randn(B, D, device="cuda", dtype=torch.float32)

            result = triton_momentum_update(grads, momentum_coef, prev)

            assert result.shape == (B, D), f"Wrong shape for {(B, T, D)}"
            assert not torch.isnan(result).any(), f"NaN for shape {(B, T, D)}"

    def test_momentum_coefficients(self):
        """Test with various momentum coefficients."""
        torch.manual_seed(42)
        B, T, D = 4, 32, 64

        grads = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
        prev = torch.randn(B, D, device="cuda", dtype=torch.float32)

        for momentum_coef in [0.1, 0.5, 0.9, 0.99]:
            result = triton_momentum_update(grads, momentum_coef, prev)
            assert not torch.isnan(result).any()
            assert not torch.isinf(result).any()


# --- Fused Weight Update Kernel Tests ---

@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
class TestFusedWeightUpdateKernel:
    """Tests for the fused momentum + weight update kernel."""

    def test_matches_separate_operations(self):
        """Fused kernel should match separate momentum + weight update."""
        torch.manual_seed(42)

        B, T, D = 4, 64, 384
        momentum_coef = 0.9
        lr = 0.01
        decay = 0.001

        grads = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
        weights_in = torch.randn(B, D, device="cuda", dtype=torch.float32)
        prev_momentum = torch.randn(B, D, device="cuda", dtype=torch.float32)

        # Reference: separate operations
        m_ref = prev_momentum.clone()
        for t in range(T):
            m_ref = momentum_coef * m_ref + (1 - momentum_coef) * grads[:, t]
        decay_factor = 1 - decay
        weights_ref = decay_factor * weights_in - lr * m_ref

        # Fused Triton kernel
        weights_out, momentum_out = triton_fused_weight_update(
            weights_in, grads, prev_momentum, lr, momentum_coef, decay
        )

        torch.testing.assert_close(weights_ref, weights_out, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(m_ref, momentum_out, rtol=1e-4, atol=1e-4)

    def test_weight_update_uses_subtraction(self):
        """
        Verify weight update uses subtraction (gradient descent).

        Formula: w_new = decay * w_old - lr * momentum
        """
        torch.manual_seed(42)
        B, T, D = 2, 32, 64
        momentum_coef = 0.9
        lr = 0.1
        decay = 0.0  # No decay for simpler test

        # Positive gradients
        grads = torch.ones(B, T, D, device="cuda", dtype=torch.float32)
        weights_in = torch.ones(B, D, device="cuda", dtype=torch.float32)
        prev_momentum = torch.zeros(B, D, device="cuda", dtype=torch.float32)

        weights_out, _ = triton_fused_weight_update(
            weights_in, grads, prev_momentum, lr, momentum_coef, decay
        )

        # With positive grads and subtraction, weights should decrease
        assert (weights_out < weights_in).all(), \
            "Weights should decrease with positive gradients"

    def test_decay_shrinks_weights(self):
        """Verify decay factor shrinks weights."""
        torch.manual_seed(42)
        B, T, D = 2, 32, 64
        momentum_coef = 0.9
        lr = 0.0  # No learning to isolate decay effect
        decay = 0.1

        grads = torch.zeros(B, T, D, device="cuda", dtype=torch.float32)
        weights_in = torch.ones(B, D, device="cuda", dtype=torch.float32) * 10
        prev_momentum = torch.zeros(B, D, device="cuda", dtype=torch.float32)

        weights_out, _ = triton_fused_weight_update(
            weights_in, grads, prev_momentum, lr, momentum_coef, decay
        )

        expected = (1 - decay) * weights_in
        torch.testing.assert_close(weights_out, expected, rtol=1e-5, atol=1e-5)


# --- Batched Weight Update Kernel Tests ---

@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
class TestBatchedWeightUpdateKernel:
    """Tests for the batched weight update kernel (single kernel for all params)."""

    def test_matches_per_param_kernel(self):
        """Batched kernel should match per-parameter fused kernel."""
        torch.manual_seed(42)

        B, T = 4, 64
        momentum_coef = 0.9
        lr = 0.01
        decay = 0.001

        # Simulate multiple parameters with different shapes
        param_shapes = {
            "layer1.weight": (128, 64),
            "layer1.bias": (128,),
            "layer2.weight": (64, 128),
            "layer2.bias": (64,),
        }

        weights_dict = {}
        grads_dict = {}
        momentum_dict = {}

        for name, shape in param_shapes.items():
            weights_dict[name] = torch.randn(B, *shape, device="cuda", dtype=torch.float32)
            grads_dict[name] = torch.randn(B, T, *shape, device="cuda", dtype=torch.float32)
            momentum_dict[name] = torch.randn(B, *shape, device="cuda", dtype=torch.float32)

        # Reference: per-parameter fused kernel
        ref_weights = {}
        ref_momentum = {}
        for name in param_shapes:
            w, m = triton_fused_weight_update(
                weights_dict[name],
                grads_dict[name],
                momentum_dict[name],
                lr,
                momentum_coef,
                decay,
            )
            ref_weights[name] = w
            ref_momentum[name] = m

        # Batched kernel: single kernel for all params
        batched_weights, batched_momentum = triton_batched_weight_update(
            weights_dict, grads_dict, momentum_dict, lr, momentum_coef, decay
        )

        # Verify all parameters match
        for name in param_shapes:
            torch.testing.assert_close(
                ref_weights[name], batched_weights[name], rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(
                ref_momentum[name], batched_momentum[name], rtol=1e-4, atol=1e-4
            )

    def test_single_param(self):
        """Batched kernel should work with a single parameter."""
        torch.manual_seed(42)

        B, T, D = 4, 64, 256
        momentum_coef = 0.9
        lr = 0.01
        decay = 0.001

        weights_dict = {"weight": torch.randn(B, D, device="cuda", dtype=torch.float32)}
        grads_dict = {"weight": torch.randn(B, T, D, device="cuda", dtype=torch.float32)}
        momentum_dict = {"weight": torch.randn(B, D, device="cuda", dtype=torch.float32)}

        new_weights, new_momentum = triton_batched_weight_update(
            weights_dict, grads_dict, momentum_dict, lr, momentum_coef, decay
        )

        assert "weight" in new_weights
        assert new_weights["weight"].shape == (B, D)
        assert not torch.isnan(new_weights["weight"]).any()

    def test_preserves_param_order(self):
        """Batched kernel should preserve parameter order in output dicts."""
        torch.manual_seed(42)

        B, T = 2, 32
        param_names = ["z_param", "a_param", "m_param"]  # Non-alphabetical

        weights_dict = {}
        grads_dict = {}
        momentum_dict = {}

        for name in param_names:
            weights_dict[name] = torch.randn(B, 64, device="cuda", dtype=torch.float32)
            grads_dict[name] = torch.randn(B, T, 64, device="cuda", dtype=torch.float32)
            momentum_dict[name] = torch.randn(B, 64, device="cuda", dtype=torch.float32)

        new_weights, new_momentum = triton_batched_weight_update(
            weights_dict, grads_dict, momentum_dict, 0.01, 0.9, 0.001
        )

        assert list(new_weights.keys()) == param_names
        assert list(new_momentum.keys()) == param_names


# --- Weight Update Equivalence Tests ---

class TestWeightUpdateEquivalence:
    """
    Tests verifying the weight update formula change is mathematically equivalent.

    Old: surprises = -grads; weights += lr * momentum(surprises)
    New: weights -= lr * momentum(grads)
    """

    def test_negation_removal_equivalence_cpu(self):
        """
        Verify new formula matches old on CPU.

        Note: Equivalence holds when starting from zero momentum (clean start).
        With non-zero prev_momentum, the two approaches would diverge because
        the momentum history would be inconsistent with the gradient signs.
        """
        torch.manual_seed(42)
        B, T, D = 2, 16, 32
        momentum_coef = 0.9
        lr = 0.01
        decay = 0.001

        grads = torch.randn(B, T, D)
        weights = torch.randn(B, D)
        # Start from zero momentum (clean start) for equivalence to hold
        prev_momentum = torch.zeros(B, D)

        # OLD approach: negate grads, then add
        surprises = -grads
        momentum_old = parallel_momentum(surprises, momentum_coef, prev_momentum)
        final_momentum_old = momentum_old[:, -1]
        weights_old = (1 - decay) * weights + lr * final_momentum_old

        # NEW approach: use grads directly, then subtract
        momentum_new = parallel_momentum(grads, momentum_coef, prev_momentum)
        final_momentum_new = momentum_new[:, -1]
        weights_new = (1 - decay) * weights - lr * final_momentum_new

        torch.testing.assert_close(weights_old, weights_new, rtol=1e-5, atol=1e-5)


# --- NeuralMemory Integration Tests ---

class TestNeuralMemoryTritonIntegration:
    """Tests for NeuralMemory using Triton kernels when available."""

    @pytest.fixture
    def config(self):
        return TitansConfig(
            block_size=256,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_embd=64,
            segment_len=32,
            num_persist_mem=2,
            num_longterm_mem=4,
            memory_depth=2,
        )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_memory_update_produces_valid_output(self, config):
        """Memory update should work regardless of Triton availability."""
        memory = NeuralMemory(config).cuda()
        B, T, C = 2, config.segment_len, config.n_embd

        state = memory.init_state(B, torch.device("cuda"))
        x = torch.randn(B, T, C, device="cuda")

        new_state = memory.update(x, state)

        # Weights should have changed
        for name in state.weights:
            assert not torch.allclose(state.weights[name], new_state.weights[name])

        # No NaN/Inf
        for name, w in new_state.weights.items():
            assert not torch.isnan(w).any(), f"NaN in {name}"
            assert not torch.isinf(w).any(), f"Inf in {name}"


# --- Segment Length Tests ---

class TestSegmentLength:
    """Tests related to segment_len=128 default."""

    def test_default_segment_len_is_128(self):
        """Verify default segment_len is 128."""
        config = TitansConfig()
        assert config.segment_len == 128

    def test_segment_128_correct_updates(self):
        """With segment_len=128, verify correct number of updates."""
        from nanogpt_titans.model import TitansGPT

        config = TitansConfig(
            block_size=512,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_embd=64,
            segment_len=128,
        )
        model = TitansGPT(config)
        B, T = 2, 256  # 2 segments with len=128

        x = torch.randint(0, config.vocab_size, (B, T))
        _, _, states = model(x, targets=x)

        memory_layer_idx = config.n_layer // 2
        assert states[memory_layer_idx].step == 2  # 256/128 = 2 segments


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
