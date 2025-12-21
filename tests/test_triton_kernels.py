"""
Tests for Triton kernel optimizations.

These tests verify that the optimized Triton kernels produce
the same results as the reference PyTorch implementations.

Run with: uv run pytest tests/test_triton_kernels.py -v
"""

import pytest
import torch

from nanogpt_titans.model import (
    NeuralMemory,
    TitansConfig,
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


# --- Cross-Entropy Kernel Tests ---

triton_cross_entropy = None
triton_layer_norm = None
triton_linear_silu = None
triton_mse_grad = None

if CUDA_AVAILABLE:
    try:
        from nanogpt_titans.triton_kernels import (
            triton_cross_entropy,
            triton_layer_norm,
            triton_linear_silu,
            triton_mse_grad,
        )
    except ImportError:
        pass


@pytest.mark.skipif(not TRITON_AVAILABLE or triton_cross_entropy is None, reason="Triton not available")
class TestTritonCrossEntropy:
    """Tests for the Triton fused cross-entropy kernel."""

    def test_matches_pytorch_basic(self):
        """Triton cross-entropy should match PyTorch F.cross_entropy."""
        torch.manual_seed(42)

        N, V = 128, 1000  # batch size, vocab size
        logits = torch.randn(N, V, device="cuda", dtype=torch.float32, requires_grad=True)
        targets = torch.randint(0, V, (N,), device="cuda")

        # PyTorch reference
        logits_ref = logits.clone().detach().requires_grad_(True)
        loss_ref = torch.nn.functional.cross_entropy(logits_ref, targets)

        # Triton
        loss_triton = triton_cross_entropy(logits, targets)

        # Forward should match
        torch.testing.assert_close(loss_ref, loss_triton, rtol=1e-3, atol=1e-3)

        # Backward should match
        loss_ref.backward()
        loss_triton.backward()
        torch.testing.assert_close(logits_ref.grad, logits.grad, rtol=1e-3, atol=1e-3)

    def test_ignore_index(self):
        """Test that ignore_index=-1 is handled correctly."""
        torch.manual_seed(42)

        N, V = 64, 500
        logits = torch.randn(N, V, device="cuda", dtype=torch.float32)
        targets = torch.randint(0, V, (N,), device="cuda")

        # Set some targets to -1 (ignore)
        targets[::3] = -1

        # PyTorch reference
        loss_ref = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)

        # Triton
        loss_triton = triton_cross_entropy(logits, targets)

        torch.testing.assert_close(loss_ref, loss_triton, rtol=1e-3, atol=1e-3)

    def test_3d_input(self):
        """Test with 3D logits input (flattened internally)."""
        torch.manual_seed(42)

        B, T, V = 4, 64, 256
        logits = torch.randn(B, T, V, device="cuda", dtype=torch.float32)
        targets = torch.randint(0, V, (B, T), device="cuda")

        # PyTorch reference (need to flatten)
        loss_ref = torch.nn.functional.cross_entropy(
            logits.view(-1, V), targets.view(-1)
        )

        # Triton (handles flattening internally)
        loss_triton = triton_cross_entropy(logits, targets)

        torch.testing.assert_close(loss_ref, loss_triton, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not TRITON_AVAILABLE or triton_layer_norm is None, reason="Triton not available")
class TestTritonLayerNorm:
    """Tests for the Triton fused layer norm kernel."""

    def test_matches_pytorch(self):
        """Triton layer norm should match PyTorch."""
        torch.manual_seed(42)

        B, T, C = 4, 64, 384
        x = torch.randn(B, T, C, device="cuda", dtype=torch.float32)
        weight = torch.randn(C, device="cuda", dtype=torch.float32)
        bias = torch.randn(C, device="cuda", dtype=torch.float32)

        # PyTorch reference
        y_ref = torch.nn.functional.layer_norm(x, (C,), weight, bias)

        # Triton
        y_triton = triton_layer_norm(x, weight, bias)

        # Relaxed tolerance for GPU architecture differences
        torch.testing.assert_close(y_ref, y_triton, rtol=1e-2, atol=1e-2)

    def test_without_bias(self):
        """Test layer norm without bias."""
        torch.manual_seed(42)

        B, T, C = 4, 64, 256
        x = torch.randn(B, T, C, device="cuda", dtype=torch.float32)
        weight = torch.randn(C, device="cuda", dtype=torch.float32)

        # PyTorch reference
        y_ref = torch.nn.functional.layer_norm(x, (C,), weight, None)

        # Triton
        y_triton = triton_layer_norm(x, weight, None)

        # Relaxed tolerance for GPU architecture differences
        torch.testing.assert_close(y_ref, y_triton, rtol=1e-2, atol=1e-2)

    def test_2d_input(self):
        """Test with 2D input."""
        torch.manual_seed(42)

        N, C = 128, 512
        x = torch.randn(N, C, device="cuda", dtype=torch.float32)
        weight = torch.randn(C, device="cuda", dtype=torch.float32)
        bias = torch.randn(C, device="cuda", dtype=torch.float32)

        # PyTorch reference
        y_ref = torch.nn.functional.layer_norm(x, (C,), weight, bias)

        # Triton
        y_triton = triton_layer_norm(x, weight, bias)

        # Relaxed tolerance for GPU architecture differences
        torch.testing.assert_close(y_ref, y_triton, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not TRITON_AVAILABLE or triton_linear_silu is None, reason="Triton not available")
class TestTritonLinearSiLU:
    """Tests for the Triton fused linear + SiLU kernel."""

    def test_matches_pytorch(self):
        """Triton linear + SiLU should match PyTorch."""
        torch.manual_seed(42)

        B, T, K, N = 4, 32, 256, 512
        x = torch.randn(B, T, K, device="cuda", dtype=torch.float16)
        weight = torch.randn(N, K, device="cuda", dtype=torch.float16)
        bias = torch.randn(N, device="cuda", dtype=torch.float16)

        # PyTorch reference
        y_ref = torch.nn.functional.silu(torch.nn.functional.linear(x, weight, bias))

        # Triton
        y_triton = triton_linear_silu(x, weight, bias)

        # Use looser tolerance for fp16
        torch.testing.assert_close(y_ref, y_triton, rtol=1e-2, atol=1e-2)

    def test_without_bias(self):
        """Test linear + SiLU without bias."""
        torch.manual_seed(42)

        B, T, K, N = 4, 32, 128, 256
        x = torch.randn(B, T, K, device="cuda", dtype=torch.float16)
        weight = torch.randn(N, K, device="cuda", dtype=torch.float16)

        # PyTorch reference
        y_ref = torch.nn.functional.silu(torch.nn.functional.linear(x, weight, None))

        # Triton
        y_triton = triton_linear_silu(x, weight, None)

        torch.testing.assert_close(y_ref, y_triton, rtol=1e-2, atol=1e-2)

    def test_2d_input(self):
        """Test with 2D input."""
        torch.manual_seed(42)

        N, K, M = 128, 256, 512
        x = torch.randn(N, K, device="cuda", dtype=torch.float16)
        weight = torch.randn(M, K, device="cuda", dtype=torch.float16)
        bias = torch.randn(M, device="cuda", dtype=torch.float16)

        # PyTorch reference
        y_ref = torch.nn.functional.silu(torch.nn.functional.linear(x, weight, bias))

        # Triton
        y_triton = triton_linear_silu(x, weight, bias)

        torch.testing.assert_close(y_ref, y_triton, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not TRITON_AVAILABLE or triton_mse_grad is None, reason="Triton not available")
class TestTritonMSEGrad:
    """Tests for the Triton MSE gradient kernel."""

    def test_matches_pytorch(self):
        """Triton MSE gradient should match PyTorch."""
        torch.manual_seed(42)

        B, T, C = 4, 64, 256
        pred = torch.randn(B, T, C, device="cuda", dtype=torch.float32, requires_grad=True)
        target = torch.randn(B, T, C, device="cuda", dtype=torch.float32)

        # PyTorch reference: grad of MSE = 2 * (pred - target) / n
        pred_ref = pred.clone().detach().requires_grad_(True)
        loss = torch.nn.functional.mse_loss(pred_ref, target)
        loss.backward()
        grad_ref = pred_ref.grad

        # Triton
        grad_triton = triton_mse_grad(pred.detach(), target)

        torch.testing.assert_close(grad_ref, grad_triton, rtol=1e-4, atol=1e-4)

    def test_various_shapes(self):
        """Test with various tensor shapes."""
        torch.manual_seed(42)

        shapes = [
            (128,),
            (64, 128),
            (4, 32, 64),
            (2, 8, 16, 32),
        ]

        for shape in shapes:
            pred = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
            target = torch.randn(shape, device="cuda", dtype=torch.float32)

            # PyTorch reference
            pred_ref = pred.clone().detach().requires_grad_(True)
            loss = torch.nn.functional.mse_loss(pred_ref, target)
            loss.backward()
            grad_ref = pred_ref.grad

            # Triton
            grad_triton = triton_mse_grad(pred.detach(), target)

            torch.testing.assert_close(
                grad_ref, grad_triton, rtol=1e-4, atol=1e-4,
                msg=f"Shape {shape} failed"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
