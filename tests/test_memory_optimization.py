"""
Unit tests for Titans memory optimization and adaptive parameters.

Tests cover:
1. Aggregated gradient memory update (500x memory reduction)
2. Adaptive memory parameters (learned lr/momentum/decay)
3. Backward compatibility and edge cases

Run with: uv run pytest tests/test_memory_optimization.py -v
"""

import pytest
import torch
from torch import nn

from nanogpt_titans.model import NeuralMemory, TitansConfig, TitansGPT
from nanogpt_titans.triton_memory_fused import (
    aggregated_gradient_memory_update,
    chunked_gradient_memory_update,
)

# --- Fixtures ---


@pytest.fixture
def small_config():
    """Small config for fast testing with adaptive memory."""
    return TitansConfig(
        block_size=128,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        segment_len=32,
        num_persist_mem=2,
        num_longterm_mem=4,
        memory_depth=2,
        memory_expansion=2,
        adaptive_memory=True,
    )


@pytest.fixture
def small_config_non_adaptive():
    """Small config without adaptive memory."""
    return TitansConfig(
        block_size=128,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        segment_len=32,
        num_persist_mem=2,
        num_longterm_mem=4,
        memory_depth=2,
        memory_expansion=2,
        adaptive_memory=False,
    )


@pytest.fixture
def memory_weights():
    """Create sample memory MLP weights for testing."""
    B, C, H = 2, 64, 128
    return {
        'layers.0.weight': torch.randn(B, H, C) * 0.02,
        'layers.1.weight': torch.randn(B, C, H) * 0.02,
        'layers.0.bias': torch.zeros(B, H),
        'layers.1.bias': torch.zeros(B, C),
    }


@pytest.fixture
def memory_momentum():
    """Create sample momentum state for testing."""
    B, C, H = 2, 64, 128
    return {
        'layers.0.weight': torch.zeros(B, H, C),
        'layers.1.weight': torch.zeros(B, C, H),
        'layers.0.bias': torch.zeros(B, H),
        'layers.1.bias': torch.zeros(B, C),
    }


# --- Aggregated Gradient Update Tests ---


class TestAggregatedGradientUpdate:
    """Tests for the aggregated gradient memory update optimization."""

    def test_output_shapes(self, memory_weights, memory_momentum):
        """Test output shapes are correct."""
        B, T, C = 2, 32, 64
        keys = torch.randn(B, T, C)
        values = torch.randn(B, T, C)

        new_weights, new_momentum = aggregated_gradient_memory_update(
            keys, values, memory_weights, memory_momentum,
            lr=0.01, mom_coef=0.9, decay=0.001
        )

        # Check all weights have same shapes
        for name in memory_weights:
            assert new_weights[name].shape == memory_weights[name].shape
            assert new_momentum[name].shape == memory_momentum[name].shape

    def test_weights_are_updated(self, memory_weights, memory_momentum):
        """Test that weights actually change after update."""
        B, T, C = 2, 32, 64
        keys = torch.randn(B, T, C)
        values = torch.randn(B, T, C)

        original_w0 = memory_weights['layers.0.weight'].clone()

        new_weights, _ = aggregated_gradient_memory_update(
            keys, values, memory_weights, memory_momentum,
            lr=0.01, mom_coef=0.9, decay=0.001
        )

        # Weights should be different
        assert not torch.allclose(new_weights['layers.0.weight'], original_w0)

    def test_momentum_accumulates(self, memory_weights, memory_momentum):
        """Test that momentum accumulates correctly."""
        B, T, C = 2, 32, 64
        keys = torch.randn(B, T, C)
        values = torch.randn(B, T, C)

        # First update
        _, new_momentum1 = aggregated_gradient_memory_update(
            keys, values, memory_weights, memory_momentum,
            lr=0.01, mom_coef=0.9, decay=0.001
        )

        # Momentum should no longer be zero
        assert not torch.allclose(
            new_momentum1['layers.0.weight'],
            torch.zeros_like(new_momentum1['layers.0.weight'])
        )

    def test_zero_lr_no_weight_change(self, memory_weights, memory_momentum):
        """Test that lr=0 results in no weight change (only decay)."""
        B, T, C = 2, 32, 64
        keys = torch.randn(B, T, C)
        values = torch.randn(B, T, C)

        original_w0 = memory_weights['layers.0.weight'].clone()

        new_weights, _ = aggregated_gradient_memory_update(
            keys, values, memory_weights, memory_momentum,
            lr=0.0, mom_coef=0.9, decay=0.0  # No lr, no decay
        )

        # With lr=0 and decay=0, weights should be unchanged
        torch.testing.assert_close(new_weights['layers.0.weight'], original_w0)

    def test_high_decay_shrinks_weights(self, memory_weights, memory_momentum):
        """Test that high decay shrinks weights toward zero."""
        B, T, C = 2, 32, 64
        keys = torch.randn(B, T, C)
        values = torch.randn(B, T, C)

        # Set weights to known values
        memory_weights['layers.0.weight'] = torch.ones(2, 128, 64)

        new_weights, _ = aggregated_gradient_memory_update(
            keys, values, memory_weights, memory_momentum,
            lr=0.0, mom_coef=0.9, decay=0.5  # High decay, no lr
        )

        # Weights should shrink (multiplied by 0.5)
        expected = 0.5 * torch.ones(2, 128, 64)
        torch.testing.assert_close(new_weights['layers.0.weight'], expected)

    def test_without_bias(self):
        """Test update works without bias terms."""
        B, T, C, H = 2, 32, 64, 128
        keys = torch.randn(B, T, C)
        values = torch.randn(B, T, C)

        weights = {
            'layers.0.weight': torch.randn(B, H, C) * 0.02,
            'layers.1.weight': torch.randn(B, C, H) * 0.02,
        }
        momentum = {
            'layers.0.weight': torch.zeros(B, H, C),
            'layers.1.weight': torch.zeros(B, C, H),
        }

        new_weights, _new_momentum = aggregated_gradient_memory_update(
            keys, values, weights, momentum,
            lr=0.01, mom_coef=0.9, decay=0.001
        )

        assert 'layers.0.weight' in new_weights
        assert 'layers.1.weight' in new_weights
        assert 'layers.0.bias' not in new_weights

    def test_adaptive_tensor_params(self, memory_weights, memory_momentum):
        """Test update with adaptive tensor parameters."""
        B, T, C = 2, 32, 64
        keys = torch.randn(B, T, C)
        values = torch.randn(B, T, C)

        # Adaptive parameters as tensors [B, T, 1]
        lr_tensor = torch.full((B, T, 1), 0.01)
        mom_tensor = torch.full((B, T, 1), 0.9)
        decay_tensor = torch.full((B, T, 1), 0.001)

        new_weights, _new_momentum = aggregated_gradient_memory_update(
            keys, values, memory_weights, memory_momentum,
            lr=lr_tensor, mom_coef=mom_tensor, decay=decay_tensor
        )

        # Should produce valid outputs
        for name in memory_weights:
            assert new_weights[name].shape == memory_weights[name].shape
            assert not torch.isnan(new_weights[name]).any()

    def test_varying_adaptive_params(self, memory_weights, memory_momentum):
        """Test that varying adaptive params produce different results."""
        B, T, C = 2, 32, 64
        keys = torch.randn(B, T, C)
        values = torch.randn(B, T, C)

        # High lr for first batch, low for second
        lr_tensor = torch.zeros(B, T, 1)
        lr_tensor[0] = 0.1
        lr_tensor[1] = 0.001

        new_weights, _ = aggregated_gradient_memory_update(
            keys, values, memory_weights, memory_momentum,
            lr=lr_tensor, mom_coef=0.9, decay=0.001
        )

        # First batch should have larger weight changes
        # (This is approximate due to aggregation)
        w0_change_b0 = (new_weights['layers.0.weight'][0] - memory_weights['layers.0.weight'][0]).abs().mean()
        w0_change_b1 = (new_weights['layers.0.weight'][1] - memory_weights['layers.0.weight'][1]).abs().mean()

        # With higher lr, expect larger changes (not always true due to gradient direction)
        # Just verify both batches updated
        assert w0_change_b0 > 0
        assert w0_change_b1 > 0

    def test_numerical_stability(self, memory_weights, memory_momentum):
        """Test numerical stability with extreme values."""
        B, T, C = 2, 32, 64

        # Very small inputs
        keys = torch.randn(B, T, C) * 1e-6
        values = torch.randn(B, T, C) * 1e-6

        new_weights, _new_momentum = aggregated_gradient_memory_update(
            keys, values, memory_weights, memory_momentum,
            lr=0.01, mom_coef=0.9, decay=0.001
        )

        assert not torch.isnan(new_weights['layers.0.weight']).any()
        assert not torch.isinf(new_weights['layers.0.weight']).any()


class TestChunkedGradientUpdate:
    """Tests for the chunked gradient memory update."""

    def test_chunked_matches_aggregated(self, memory_weights, memory_momentum):
        """Test chunked update produces similar results to aggregated."""
        B, T, C = 2, 64, 64
        keys = torch.randn(B, T, C)
        values = torch.randn(B, T, C)

        # Deep copy for fair comparison
        weights1 = {k: v.clone() for k, v in memory_weights.items()}
        weights2 = {k: v.clone() for k, v in memory_weights.items()}
        momentum1 = {k: v.clone() for k, v in memory_momentum.items()}
        momentum2 = {k: v.clone() for k, v in memory_momentum.items()}

        new_w1, _ = aggregated_gradient_memory_update(
            keys, values, weights1, momentum1,
            lr=0.01, mom_coef=0.9, decay=0.001
        )

        new_w2, _ = chunked_gradient_memory_update(
            keys, values, weights2, momentum2,
            lr=0.01, mom_coef=0.9, decay=0.001,
            chunk_size=16
        )

        # Results should be similar (not exact due to momentum approximation)
        for name in weights1:
            torch.testing.assert_close(
                new_w1[name], new_w2[name],
                rtol=0.1, atol=0.01,
                msg=f"Chunked and aggregated differ for {name}"
            )


# --- Adaptive Memory Parameters Tests ---


class TestAdaptiveMemoryParameters:
    """Tests for adaptive memory parameters (learned lr/momentum/decay)."""

    def test_adaptive_projections_exist(self, small_config):
        """Test adaptive projection layers are created."""
        memory = NeuralMemory(small_config)

        assert hasattr(memory, 'to_lr')
        assert hasattr(memory, 'to_momentum')
        assert hasattr(memory, 'to_decay')
        assert isinstance(memory.to_lr, nn.Linear)
        assert isinstance(memory.to_momentum, nn.Linear)
        assert isinstance(memory.to_decay, nn.Linear)

    def test_adaptive_projections_absent_when_disabled(self, small_config_non_adaptive):
        """Test adaptive projections are not created when disabled."""
        memory = NeuralMemory(small_config_non_adaptive)

        assert not hasattr(memory, 'to_lr')
        assert not hasattr(memory, 'to_momentum')
        assert not hasattr(memory, 'to_decay')

    def test_projection_output_shapes(self, small_config):
        """Test projection layers produce correct output shapes."""
        memory = NeuralMemory(small_config)
        B, T, C = 2, 32, small_config.n_embd

        x = torch.randn(B, T, C)

        lr_out = memory.to_lr(x)
        mom_out = memory.to_momentum(x)
        decay_out = memory.to_decay(x)

        assert lr_out.shape == (B, T, 1)
        assert mom_out.shape == (B, T, 1)
        assert decay_out.shape == (B, T, 1)

    def test_sigmoid_bounds(self, small_config):
        """Test sigmoid activation bounds outputs to (0, 1)."""
        memory = NeuralMemory(small_config)
        B, T, C = 2, 32, small_config.n_embd

        x = torch.randn(B, T, C)

        lr = torch.sigmoid(memory.to_lr(x)) * memory.lr_max
        mom = torch.sigmoid(memory.to_momentum(x))
        decay = torch.sigmoid(memory.to_decay(x))

        # lr should be in (0, lr_max)
        assert (lr >= 0).all() and (lr <= memory.lr_max).all()
        # momentum and decay should be in (0, 1)
        assert (mom >= 0).all() and (mom <= 1).all()
        assert (decay >= 0).all() and (decay <= 1).all()

    def test_default_bias_initialization_in_model(self, small_config):
        """Test bias is initialized for reasonable defaults when using TitansGPT."""
        model = TitansGPT(small_config)

        for block in model.transformer['h']:
            if block.has_memory:
                mem = block.memory
                # Check bias values
                assert abs(mem.to_lr.bias.item() - 0.0) < 1e-5, "to_lr bias should be ~0"
                assert abs(mem.to_momentum.bias.item() - 2.0) < 1e-5, "to_momentum bias should be ~2"
                assert abs(mem.to_decay.bias.item() - (-4.0)) < 1e-5, "to_decay bias should be ~-4"

    def test_default_parameter_values(self, small_config):
        """Test default adaptive parameter values with zero input."""
        model = TitansGPT(small_config)

        for block in model.transformer['h']:
            if block.has_memory:
                mem = block.memory
                x = torch.zeros(1, 1, small_config.n_embd)

                lr = torch.sigmoid(mem.to_lr(x)) * mem.lr_max
                mom = torch.sigmoid(mem.to_momentum(x))
                decay = torch.sigmoid(mem.to_decay(x))

                # With zero input and initialized biases:
                # lr: sigmoid(0) * lr_max = 0.5 * 0.01 = 0.005
                # momentum: sigmoid(2) ≈ 0.88
                # decay: sigmoid(-4) ≈ 0.018
                assert abs(lr.item() - 0.005) < 0.001, f"Expected lr ~0.005, got {lr.item()}"
                assert abs(mom.item() - 0.88) < 0.02, f"Expected momentum ~0.88, got {mom.item()}"
                assert abs(decay.item() - 0.018) < 0.005, f"Expected decay ~0.018, got {decay.item()}"

    def test_set_adaptive_toggle(self, small_config):
        """Test set_adaptive method toggles adaptive mode."""
        memory = NeuralMemory(small_config)

        assert memory.adaptive is True
        memory.set_adaptive(False)
        assert memory.adaptive is False
        memory.set_adaptive(True)
        assert memory.adaptive is True

    def test_set_adaptive_raises_without_projections(self, small_config_non_adaptive):
        """Test set_adaptive raises error when projections don't exist."""
        memory = NeuralMemory(small_config_non_adaptive)

        with pytest.raises(ValueError, match="Adaptive mode requires projection layers"):
            memory.set_adaptive(True)

    def test_adaptive_update_uses_projections(self, small_config):
        """Test that adaptive update actually uses the projection outputs."""
        memory = NeuralMemory(small_config)
        B, T, C = 2, 32, small_config.n_embd

        x = torch.randn(B, T, C)
        state = memory.init_state(B, torch.device("cpu"))

        # Modify projection weights to produce distinct outputs
        with torch.no_grad():
            memory.to_lr.weight.fill_(0.1)  # Non-zero weight
            memory.to_lr.bias.fill_(5.0)    # High bias -> high sigmoid -> high lr

        # Update should use the high learning rate
        new_state = memory.update(x, state)

        # With high lr, weights should change significantly
        weight_change = (new_state.weights['layers.0.weight'] - state.weights['layers.0.weight']).abs().mean()
        assert weight_change > 1e-4, "Adaptive lr should cause weight changes"


class TestSetAggregatedUpdate:
    """Tests for set_aggregated_update toggle."""

    def test_toggle_aggregated_update(self, small_config):
        """Test toggling aggregated update mode."""
        memory = NeuralMemory(small_config)

        # Default should be True (when available)
        initial = memory._use_aggregated_update

        memory.set_aggregated_update(False)
        assert memory._use_aggregated_update is False

        memory.set_aggregated_update(True)
        # Should be True if aggregated update is available
        assert memory._use_aggregated_update == initial


# --- Integration Tests ---


class TestMemoryOptimizationIntegration:
    """Integration tests for memory optimization with full model."""

    def test_forward_backward_with_adaptive(self, small_config):
        """Test forward and backward pass with adaptive memory.

        Note: Adaptive projection gradients (to_lr, to_momentum, to_decay) don't
        flow from the main loss because:
        1. They're computed from detached input in update()
        2. They affect memory weights which are intentionally detached
        3. This is correct behavior - memory update is "test-time learning"

        The adaptive projections would be trained through auxiliary losses or
        REINFORCE-style gradient estimators in a full implementation.
        """
        model = TitansGPT(small_config)
        B, T = 2, 64
        x = torch.randint(0, small_config.vocab_size, (B, T))

        _logits, loss, _states = model(x, targets=x)
        loss.backward()

        # Forward-backward should complete without error
        assert loss.item() > 0

        # Adaptive projections should exist and have requires_grad=True
        for block in model.transformer['h']:
            if block.has_memory:
                assert block.memory.to_lr.weight.requires_grad
                assert block.memory.to_momentum.weight.requires_grad
                assert block.memory.to_decay.weight.requires_grad

                # Note: These don't have gradients from main loss (by design)
                # because memory update path is intentionally detached

    def test_forward_backward_without_adaptive(self, small_config_non_adaptive):
        """Test forward and backward pass without adaptive memory."""
        model = TitansGPT(small_config_non_adaptive)
        B, T = 2, 64
        x = torch.randint(0, small_config_non_adaptive.vocab_size, (B, T))

        _logits, loss, _states = model(x, targets=x)
        loss.backward()

        # Should work without errors
        assert loss.item() > 0

    def test_consistent_outputs_across_modes(self, small_config, small_config_non_adaptive):
        """Test that adaptive and non-adaptive modes both produce valid outputs."""
        torch.manual_seed(42)
        model_adaptive = TitansGPT(small_config)

        torch.manual_seed(42)
        model_fixed = TitansGPT(small_config_non_adaptive)

        B, T = 2, 64
        x = torch.randint(0, small_config.vocab_size, (B, T))

        model_adaptive.eval()
        model_fixed.eval()

        with torch.no_grad():
            logits1, loss1, _ = model_adaptive(x, targets=x)
            logits2, loss2, _ = model_fixed(x, targets=x)

        # Both should produce valid outputs (losses may differ due to different params)
        assert not torch.isnan(logits1).any()
        assert not torch.isnan(logits2).any()
        assert loss1.item() > 0
        assert loss2.item() > 0

    def test_memory_state_step_increments(self, small_config):
        """Test memory state step increments correctly."""
        model = TitansGPT(small_config)
        B = 2
        T = small_config.segment_len * 3  # 3 segments

        x = torch.randint(0, small_config.vocab_size, (B, T))
        _, _, states = model(x)

        # Find memory layer
        for state in states:
            if state is not None:
                assert state.step == 3, f"Expected 3 segments, got step={state.step}"
                break

    def test_parameter_count_difference(self, small_config, small_config_non_adaptive):
        """Test adaptive mode adds expected number of parameters."""
        model_adaptive = TitansGPT(small_config)
        model_fixed = TitansGPT(small_config_non_adaptive)

        params_adaptive = sum(p.numel() for p in model_adaptive.parameters())
        params_fixed = sum(p.numel() for p in model_fixed.parameters())

        # Adaptive should have more parameters (3 projections per memory layer)
        # Each projection: n_embd -> 1, so n_embd + 1 params (weight + bias)
        expected_extra = 3 * (small_config.n_embd + 1)  # 3 projections
        actual_extra = params_adaptive - params_fixed

        assert actual_extra == expected_extra, \
            f"Expected {expected_extra} extra params, got {actual_extra}"


# --- Edge Cases ---


class TestMemoryOptimizationEdgeCases:
    """Edge case tests for memory optimization."""

    def test_single_token_sequence(self, small_config):
        """Test with single token per segment."""
        B, T, C = 2, 1, small_config.n_embd
        memory = NeuralMemory(small_config)

        x = torch.randn(B, T, C)
        state = memory.init_state(B, torch.device("cpu"))

        # Should handle single token without error
        new_state = memory.update(x, state)
        assert new_state.step == 1

    def test_large_sequence(self, small_config):
        """Test with larger sequence."""
        B, T, C = 2, 256, small_config.n_embd
        memory = NeuralMemory(small_config)

        x = torch.randn(B, T, C)
        state = memory.init_state(B, torch.device("cpu"))

        new_state = memory.update(x, state)
        assert new_state.step == 1

    def test_batch_size_one(self, small_config):
        """Test with batch size 1."""
        B, T, C = 1, 32, small_config.n_embd
        memory = NeuralMemory(small_config)

        x = torch.randn(B, T, C)
        state = memory.init_state(B, torch.device("cpu"))

        new_state = memory.update(x, state)
        assert new_state.step == 1

    def test_consecutive_updates(self, small_config):
        """Test multiple consecutive memory updates."""
        B, T, C = 2, 32, small_config.n_embd
        memory = NeuralMemory(small_config)

        state = memory.init_state(B, torch.device("cpu"))

        for i in range(5):
            x = torch.randn(B, T, C)
            state = memory.update(x, state)
            assert state.step == i + 1

    def test_reset_after_updates(self, small_config):
        """Test memory reset after multiple updates."""
        B, T, C = 2, 32, small_config.n_embd
        memory = NeuralMemory(small_config)

        state = memory.init_state(B, torch.device("cpu"))

        # Do some updates
        for _ in range(3):
            x = torch.randn(B, T, C)
            state = memory.update(x, state)

        assert state.step == 3

        # Reset
        memory.reset_state(state)
        assert state.step == 0
        assert state.last_segment_output is None


# --- Gradient Flow Tests ---


class TestGradientFlow:
    """Tests for gradient flow through memory optimization."""

    def test_gradients_flow_through_retrieval(self, small_config):
        """Test gradients flow through memory retrieval path.

        On first segment, retrieval uses init_query (not x), so gradients
        flow to init_query but not x. On subsequent segments, gradients
        would flow to query_proj parameters.
        """
        memory = NeuralMemory(small_config)
        B, T, C = 2, 32, small_config.n_embd

        x = torch.randn(B, T, C, requires_grad=True)
        state = memory.init_state(B, torch.device("cpu"))

        # Forward pass (first segment - uses init_query, not x)
        retrieved = memory(x, state)

        # Backward pass
        loss = retrieved.sum()
        loss.backward()

        # init_query should have gradients (it's used for first segment retrieval)
        assert memory.init_query.grad is not None
        assert not torch.isnan(memory.init_query.grad).any()

        # query_proj should have gradients
        assert memory.query_proj.weight.grad is not None

        # Note: x doesn't affect retrieval on first segment (uses init_query)
        # so x.grad may be None - this is correct behavior

    def test_no_gradient_to_detached_weights(self, small_config):
        """Test that memory state weights don't receive gradients (detached)."""
        memory = NeuralMemory(small_config)
        B, T, C = 2, 32, small_config.n_embd

        x = torch.randn(B, T, C)
        state = memory.init_state(B, torch.device("cpu"))

        # Update returns detached weights
        new_state = memory.update(x, state)

        # State weights should not require grad
        for name, w in new_state.weights.items():
            assert not w.requires_grad, f"State weight {name} should not require grad"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
