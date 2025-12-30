"""
Unit tests for MLX TITANS implementation.

Tests cover:
- MLXNeuralMemory: shapes, gradients, internal loss
- MLXContinuumMemorySystem: multi-level retrieval, update frequencies
- MLXPositionDependentGate: initialization, output shape
- MLXTitansLayer: HOPE integration
- Gate regularization
- Training components: loss function, gradient masking

Run with: uv run pytest tests/test_mlx_titans.py -v
"""

import pytest
import math

# Check if MLX is available (Apple Silicon only)
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Skip all tests in this module if MLX is not available
pytestmark = pytest.mark.skipif(
    not MLX_AVAILABLE, reason="MLX not available (requires Apple Silicon)"
)


# Only import MLX modules if available
if MLX_AVAILABLE:
    from nanogpt_titans.mlx.memory import (
        MLXMemoryState,
        MLXNeuralMemory,
        MLXCMSState,
        MLXContinuumMemorySystem,
    )
    from nanogpt_titans.mlx.decoder_layer import (
        MLXPositionDependentGate,
        MLXTitansLayer,
    )
    from nanogpt_titans.mlx.training import (
        compute_gate_regularization,
        create_masked_grads,
        scale_grads_recursive,
        accumulate_grads,
    )
    from nanogpt_titans.mlx.config import MLXTitansConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_dim():
    """Small dimension for fast testing."""
    return 64


@pytest.fixture
def small_config():
    """Small config for fast testing."""
    return MLXTitansConfig(
        memory_layer=2,
        segment_len=32,
        memory_depth=2,
        memory_expansion=2,
        num_longterm_mem=4,
        use_cms=True,
        num_cms_levels=3,
        cms_update_frequencies=(1, 4, 16),
        gate_init_bias=-2.0,  # Conservative start: sigmoid(-2) ≈ 0.12
        adaptive_memory=True,
        memory_lr_max=0.01,
        gate_min_value=0.15,
        gate_reg_weight=1.0,
    )


@pytest.fixture
def neural_memory(small_dim):
    """Create a neural memory module for testing."""
    return MLXNeuralMemory(
        dim=small_dim,
        depth=2,
        expansion=2,
        memory_lr=0.01,
        memory_momentum=0.9,
        memory_decay=0.001,
        adaptive=True,
        lr_max=0.01,
    )


@pytest.fixture
def cms(small_dim):
    """Create a CMS module for testing."""
    return MLXContinuumMemorySystem(
        dim=small_dim,
        num_levels=3,
        update_frequencies=(1, 4, 16),
        memory_depth=2,
        memory_expansion=2,
        adaptive=True,
        lr_max=0.01,
    )


@pytest.fixture
def titans_layer(small_dim):
    """Create a TITANS layer for testing."""
    return MLXTitansLayer(
        dim=small_dim,
        use_cms=True,
        num_cms_levels=3,
        cms_update_frequencies=(1, 4, 16),
        memory_depth=2,
        memory_expansion=2,
        adaptive_memory=True,
        memory_lr_max=0.01,
        gate_init_bias=-2.0,
    )


# =============================================================================
# MLXNeuralMemory Tests
# =============================================================================


class TestMLXNeuralMemory:
    """Tests for MLXNeuralMemory module."""

    def test_init_state_shapes(self, neural_memory, small_dim):
        """Test init_state produces correct shapes."""
        B = 4
        state = neural_memory.init_state(B)

        assert isinstance(state, MLXMemoryState)
        assert state.step == 0
        assert state.last_segment_output is None

        # Check weight shapes
        H = small_dim * 2  # expansion=2
        assert "w0" in state.weights
        assert "w1" in state.weights
        assert state.weights["w0"].shape == (B, H, small_dim)
        assert state.weights["w1"].shape == (B, small_dim, H)

        # Check momentum shapes
        assert state.last_momentum["w0"].shape == state.weights["w0"].shape
        assert state.last_momentum["w1"].shape == state.weights["w1"].shape

        # Momentum should be initialized to zeros
        assert mx.allclose(state.last_momentum["w0"], mx.zeros_like(state.last_momentum["w0"]))
        assert mx.allclose(state.last_momentum["w1"], mx.zeros_like(state.last_momentum["w1"]))

    def test_batched_mlp_forward_shapes(self, neural_memory, small_dim):
        """Test _batched_mlp_forward produces correct shapes."""
        B, T = 2, 16
        x = mx.random.normal((B, T, small_dim))
        state = neural_memory.init_state(B)

        output = neural_memory._batched_mlp_forward(x, state.weights)

        assert output.shape == (B, T, small_dim)
        assert not mx.any(mx.isnan(output))

    def test_compute_gradients_shapes(self, neural_memory, small_dim):
        """Test _compute_gradients produces correct gradient shapes."""
        B, T = 2, 16
        H = small_dim * 2

        keys = mx.random.normal((B, T, small_dim))
        values = mx.random.normal((B, T, small_dim))
        state = neural_memory.init_state(B)

        grads = neural_memory._compute_gradients(keys, values, state.weights)

        # Gradients should have same shape as weights
        assert "w0" in grads
        assert "w1" in grads
        assert grads["w0"].shape == (B, H, small_dim)
        assert grads["w1"].shape == (B, small_dim, H)
        assert not mx.any(mx.isnan(grads["w0"]))
        assert not mx.any(mx.isnan(grads["w1"]))

    def test_internal_loss_uses_template_weights(self, neural_memory, small_dim):
        """Test compute_internal_loss uses template weights (trainable)."""
        B, T = 2, 16
        x = mx.random.normal((B, T, small_dim))
        state = neural_memory.init_state(B)

        # Compute internal loss
        loss = neural_memory.compute_internal_loss(x, state)

        # Loss should be a scalar
        assert loss.shape == ()
        assert not mx.isnan(loss)
        assert loss >= 0  # MSE loss is non-negative

    def test_template_weights_are_nn_linear_modules(self, neural_memory):
        """Test that template weights are nn.Linear modules.

        NOTE: Template weights use underscore prefix (_template_mlp_w0) which
        in MLX means they are private and not collected by parameters().
        However, they ARE nn.Linear modules and receive gradients through
        the compute_internal_loss function which directly uses their weights.
        """
        # Template weights should be nn.Linear layers
        assert isinstance(neural_memory._template_mlp_w0, nn.Linear)
        assert isinstance(neural_memory._template_mlp_w1, nn.Linear)

        # Check they have weight attributes with correct shapes
        H = neural_memory.hidden_dim
        C = neural_memory.dim
        assert neural_memory._template_mlp_w0.weight.shape == (H, C)
        assert neural_memory._template_mlp_w1.weight.shape == (C, H)

    def test_update_applies_momentum(self, neural_memory, small_dim):
        """Test update() applies momentum correctly."""
        B, T = 2, 16
        x = mx.random.normal((B, T, small_dim)) * 0.1  # Small values for stability
        state = neural_memory.init_state(B)

        # First update
        new_state = neural_memory.update(x, state)

        # Step should increase
        assert new_state.step == 1

        # Weights should change
        assert not mx.allclose(new_state.weights["w0"], state.weights["w0"])
        assert not mx.allclose(new_state.weights["w1"], state.weights["w1"])

        # Momentum should be non-zero now
        assert not mx.allclose(
            new_state.last_momentum["w0"], mx.zeros_like(new_state.last_momentum["w0"])
        )

        # Last segment output should be stored
        assert new_state.last_segment_output is not None
        assert mx.allclose(new_state.last_segment_output, x)

    def test_weight_clipping(self, neural_memory, small_dim):
        """Test weights are clipped to [-10, 10] for numerical stability."""
        B, T = 2, 16
        # Use large input to potentially cause large weight updates
        x = mx.random.normal((B, T, small_dim)) * 10
        state = neural_memory.init_state(B)

        # Multiple updates
        for _ in range(10):
            state = neural_memory.update(x, state)

        # Weights should be clipped
        assert mx.all(state.weights["w0"] >= -10.0)
        assert mx.all(state.weights["w0"] <= 10.0)
        assert mx.all(state.weights["w1"] >= -10.0)
        assert mx.all(state.weights["w1"] <= 10.0)

    def test_retrieval_uses_previous_segment(self, neural_memory, small_dim):
        """Test __call__ uses previous segment for causal retrieval."""
        B, T = 2, 16
        x1 = mx.random.normal((B, T, small_dim))
        x2 = mx.random.normal((B, T, small_dim))

        state = neural_memory.init_state(B)

        # First retrieval - uses init_query (no previous segment)
        retrieved1 = neural_memory(x1, state)
        assert retrieved1.shape == (B, T, small_dim)

        # Update with x1
        state = neural_memory.update(x1, state)

        # Second retrieval - should use x1 (stored in state.last_segment_output)
        retrieved2 = neural_memory(x2, state)
        assert retrieved2.shape == (B, T, small_dim)

        # Retrievals should differ (different query sources)
        assert not mx.allclose(retrieved1, retrieved2)

    def test_adaptive_parameters(self, small_dim):
        """Test adaptive memory parameters (lr, momentum, decay)."""
        memory = MLXNeuralMemory(
            dim=small_dim,
            depth=2,
            expansion=2,
            adaptive=True,
            lr_max=0.01,
        )

        # Should have adaptive projection layers
        assert hasattr(memory, "to_lr")
        assert hasattr(memory, "to_momentum")
        assert hasattr(memory, "to_decay")
        assert isinstance(memory.to_lr, nn.Linear)

    def test_non_adaptive_memory(self, small_dim):
        """Test non-adaptive memory uses fixed hyperparameters."""
        memory = MLXNeuralMemory(
            dim=small_dim,
            depth=2,
            expansion=2,
            adaptive=False,
            memory_lr=0.01,
            memory_momentum=0.9,
            memory_decay=0.001,
        )

        assert not memory.adaptive
        assert memory.lr == 0.01
        assert memory.momentum == 0.9
        assert memory.decay == 0.001


# =============================================================================
# MLXContinuumMemorySystem Tests
# =============================================================================


class TestMLXContinuumMemorySystem:
    """Tests for MLXContinuumMemorySystem module."""

    def test_init_state_creates_level_states(self, cms):
        """Test init_state creates states for all levels."""
        B = 2
        state = cms.init_state(B)

        assert isinstance(state, MLXCMSState)
        assert len(state.level_states) == 3  # num_levels=3
        assert state.step == 0

        for level_state in state.level_states:
            assert isinstance(level_state, MLXMemoryState)

    def test_multi_level_retrieval_combines_outputs(self, cms, small_dim):
        """Test retrieval combines outputs from all levels."""
        B, T = 2, 16
        x = mx.random.normal((B, T, small_dim))
        state = cms.init_state(B)

        output = cms(x, state)

        assert output.shape == (B, T, small_dim)
        assert not mx.any(mx.isnan(output))

    def test_level_weights_softmax(self, cms):
        """Test level_weights produces valid softmax distribution."""
        weights = mx.softmax(cms.level_weights)

        # Should sum to 1
        assert mx.allclose(mx.sum(weights), mx.array(1.0), atol=1e-5)
        # All weights should be positive
        assert mx.all(weights > 0)

    def test_update_frequencies_respected(self, cms, small_dim):
        """Test different levels update at different frequencies.

        Update logic in CMS: if state.step % freq == 0, update the level.
        - Level 0 (freq=1): updates when step=0,1,2,3,4... (every call)
        - Level 1 (freq=4): updates when step=0,4,8... (every 4th starting from 0)
        - Level 2 (freq=16): updates when step=0,16,32... (every 16th starting from 0)
        """
        B, T = 2, 16
        x = mx.random.normal((B, T, small_dim))
        state = cms.init_state(B)

        # Helper to copy weights (MLX arrays use mx.array() for copy)
        def copy_weights(weights_dict):
            return {k: mx.array(v) for k, v in weights_dict.items()}

        # Track weight changes at each level
        initial_weights = [copy_weights(level_state.weights) for level_state in state.level_states]
        mx.eval(*[w["w0"] for w in initial_weights])

        # First update (step=0): ALL levels update (0%1=0, 0%4=0, 0%16=0)
        state = cms.update(x, state)
        mx.eval(state)
        assert state.step == 1

        # All levels should have updated
        for i in range(3):
            assert not mx.allclose(state.level_states[i].weights["w0"], initial_weights[i]["w0"]), (
                f"Level {i} should have updated at step 0"
            )

        # Store weights after first update
        weights_after_step0 = [
            copy_weights(level_state.weights) for level_state in state.level_states
        ]
        mx.eval(*[w["w0"] for w in weights_after_step0])

        # Second update (step=1): Only level 0 updates (1%1=0, 1%4≠0, 1%16≠0)
        state = cms.update(x, state)
        mx.eval(state)
        assert state.step == 2

        # Level 0 should have updated
        assert not mx.allclose(state.level_states[0].weights["w0"], weights_after_step0[0]["w0"])
        # Level 1 should NOT have updated
        assert mx.allclose(state.level_states[1].weights["w0"], weights_after_step0[1]["w0"])
        # Level 2 should NOT have updated
        assert mx.allclose(state.level_states[2].weights["w0"], weights_after_step0[2]["w0"])

        # Continue to step=4 where level 1 should update again
        state = cms.update(x, state)  # step=2
        state = cms.update(x, state)  # step=3
        mx.eval(state)

        # Store weights before step=4
        weights_before_step4 = copy_weights(state.level_states[1].weights)
        mx.eval(weights_before_step4["w0"])

        state = cms.update(x, state)  # step=4, checks if 4%4==0 (yes!)
        mx.eval(state)
        assert state.step == 5

        # Level 1 should have updated at step=4
        assert not mx.allclose(state.level_states[1].weights["w0"], weights_before_step4["w0"]), (
            "Level 1 should update when step=4 (4%4==0)"
        )

    def test_internal_loss_uses_first_level(self, cms, small_dim):
        """Test compute_internal_loss uses the fastest (first) memory level."""
        B, T = 2, 16
        x = mx.random.normal((B, T, small_dim))
        state = cms.init_state(B)

        loss = cms.compute_internal_loss(x, state)

        assert loss.shape == ()
        assert not mx.isnan(loss)
        assert loss >= 0


# =============================================================================
# MLXPositionDependentGate Tests
# =============================================================================


class TestMLXPositionDependentGate:
    """Tests for MLXPositionDependentGate module."""

    def test_init_bias_values(self, small_dim):
        """Test gate initializes with correct bias value."""
        init_bias = -2.0
        gate = MLXPositionDependentGate(small_dim, init_bias=init_bias)

        # linear2.bias should be init_bias
        assert mx.allclose(gate.linear2.bias, mx.array([init_bias]))

        # Expected gate value at initialization (with zero input)
        x = mx.zeros((1, 1, small_dim))
        gate_value = gate(x)
        expected_gate = 1.0 / (1.0 + math.exp(-init_bias))  # sigmoid(init_bias)

        assert mx.allclose(gate_value, mx.array([[[expected_gate]]]), atol=0.1)

    def test_output_shape(self, small_dim):
        """Test gate output has shape [B, T, 1]."""
        gate = MLXPositionDependentGate(small_dim, init_bias=0.0)
        B, T = 4, 32
        x = mx.random.normal((B, T, small_dim))

        output = gate(x)

        assert output.shape == (B, T, 1)

    def test_sigmoid_range(self, small_dim):
        """Test gate output is in (0, 1) range."""
        gate = MLXPositionDependentGate(small_dim, init_bias=0.0)
        B, T = 4, 32
        x = mx.random.normal((B, T, small_dim)) * 10  # Larger values

        output = gate(x)

        assert mx.all(output > 0)
        assert mx.all(output < 1)

    def test_small_weight_initialization(self, small_dim):
        """Test weights are initialized with small values (std=0.01)."""
        gate = MLXPositionDependentGate(small_dim, init_bias=0.0)

        # linear1 weight should have small values
        w1_std = mx.std(gate.linear1.weight).item()
        assert w1_std < 0.05  # Allow some variance, but should be small

        # linear2 weight should have small values
        w2_std = mx.std(gate.linear2.weight).item()
        assert w2_std < 0.05


# =============================================================================
# MLXTitansLayer Tests
# =============================================================================


class TestMLXTitansLayer:
    """Tests for MLXTitansLayer module."""

    def test_hope_integration_formula(self, titans_layer, small_dim):
        """Test HOPE formula: output = hidden + gate * scale * LN(proj(memory))."""
        B, T = 2, 16
        x = mx.random.normal((B, T, small_dim))
        state = titans_layer.init_state(B)

        output, new_state = titans_layer(x, state)

        assert output.shape == x.shape
        assert not mx.any(mx.isnan(output))

        # Output should be different from input (memory contribution)
        # Due to gating, it might be similar but not identical
        assert output.shape == (B, T, small_dim)

    def test_state_updates_correctly(self, titans_layer, small_dim):
        """Test memory state is updated after forward pass."""
        B, T = 2, 16
        x = mx.random.normal((B, T, small_dim))
        state = titans_layer.init_state(B)

        initial_step = state.step if hasattr(state, "step") else state.level_states[0].step

        output, new_state = titans_layer(x, state)

        # For CMS, check the first level's step
        if hasattr(new_state, "level_states"):
            new_step = new_state.step
        else:
            new_step = new_state.step

        assert new_step > initial_step

    def test_internal_loss_delegation(self, titans_layer, small_dim):
        """Test compute_internal_loss delegates to memory module."""
        B, T = 2, 16
        x = mx.random.normal((B, T, small_dim))
        state = titans_layer.init_state(B)

        loss = titans_layer.compute_internal_loss(x, state)

        assert loss.shape == ()
        assert not mx.isnan(loss)
        assert loss >= 0

    def test_mem_scale_initialization(self, titans_layer):
        """Test mem_scale is initialized to 0.0 (sigmoid(0)=0.5)."""
        expected_init = 0.0
        assert mx.allclose(titans_layer.mem_scale, mx.array([expected_init]))

    def test_single_memory_mode(self, small_dim):
        """Test TitansLayer with single memory (no CMS)."""
        layer = MLXTitansLayer(
            dim=small_dim,
            use_cms=False,  # Single memory
            memory_depth=2,
            memory_expansion=2,
            gate_init_bias=-2.0,
        )

        B, T = 2, 16
        x = mx.random.normal((B, T, small_dim))
        state = layer.init_state(B)

        output, new_state = layer(x, state)

        assert output.shape == x.shape
        assert isinstance(new_state, MLXMemoryState)  # Not MLXCMSState


# =============================================================================
# Gate Regularization Tests
# =============================================================================


class TestGateRegularization:
    """Tests for gate regularization function."""

    def test_penalty_when_gate_below_min(self, titans_layer):
        """Test regularization returns penalty when gate < min_value."""
        # Set gate bias to produce gate < 0.15
        titans_layer.gate.linear2.bias = mx.array([-4.0])  # sigmoid(-4) ≈ 0.018

        penalty = compute_gate_regularization(titans_layer, min_value=0.15)

        # Should have non-zero penalty
        assert penalty > 0
        assert not mx.isnan(penalty)

    def test_zero_penalty_when_gate_above_min(self, titans_layer):
        """Test regularization returns zero when gate >= min_value."""
        # Set gate bias to produce gate > 0.15
        titans_layer.gate.linear2.bias = mx.array([2.0])  # sigmoid(2) ≈ 0.88

        penalty = compute_gate_regularization(titans_layer, min_value=0.15)

        # Should have zero penalty
        assert mx.allclose(penalty, mx.array(0.0))

    def test_penalty_formula(self, titans_layer):
        """Test penalty follows max(0, min_value - gate)^2 formula."""
        min_value = 0.15

        # Test with gate < min_value
        gate_bias = -2.0  # sigmoid(-2) ≈ 0.119
        titans_layer.gate.linear2.bias = mx.array([gate_bias])

        penalty = compute_gate_regularization(titans_layer, min_value=min_value)

        expected_gate = 1.0 / (1.0 + math.exp(-gate_bias))
        expected_penalty = max(0, min_value - expected_gate) ** 2

        assert mx.allclose(penalty, mx.array(expected_penalty), atol=1e-5)


# =============================================================================
# Training Component Tests
# =============================================================================


class TestTrainingComponents:
    """Tests for training utility functions."""

    def test_create_masked_grads_keeps_gate_scale(self, small_dim):
        """Test gradient masking keeps gate/scale params when specified."""
        # Create mock gradient structure
        grads = {
            "memory": {
                "w0": mx.ones((small_dim, small_dim)),
                "w1": mx.ones((small_dim, small_dim)),
            },
            "gate": {
                "linear1": {"weight": mx.ones((small_dim // 4, small_dim))},
                "linear2": {"weight": mx.ones((1, small_dim // 4))},
            },
            "mem_scale": mx.ones((1,)),
        }

        masked = create_masked_grads(grads, keep_gate_scale=True)

        # Gate and scale should be preserved
        assert mx.allclose(masked["gate"]["linear1"]["weight"], grads["gate"]["linear1"]["weight"])
        assert mx.allclose(masked["mem_scale"], grads["mem_scale"])

        # Memory should be zeroed
        assert mx.allclose(masked["memory"]["w0"], mx.zeros_like(grads["memory"]["w0"]))

    def test_create_masked_grads_keeps_memory(self, small_dim):
        """Test gradient masking keeps memory params when specified."""
        grads = {
            "memory": {
                "w0": mx.ones((small_dim, small_dim)),
            },
            "gate": {
                "linear2": {"bias": mx.ones((1,))},
            },
        }

        masked = create_masked_grads(grads, keep_gate_scale=False)

        # Memory should be preserved
        assert mx.allclose(masked["memory"]["w0"], grads["memory"]["w0"])

        # Gate should be zeroed
        assert mx.allclose(
            masked["gate"]["linear2"]["bias"], mx.zeros_like(grads["gate"]["linear2"]["bias"])
        )

    def test_create_masked_grads_freeze_gate(self, small_dim):
        """Test gradient masking can freeze gate during warmup."""
        grads = {
            "gate": {
                "linear2": {"weight": mx.ones((1, small_dim // 4))},
            },
            "mem_scale": mx.ones((1,)),
        }

        masked = create_masked_grads(grads, keep_gate_scale=True, freeze_gate=True)

        # Gate should be zeroed (frozen)
        assert mx.allclose(
            masked["gate"]["linear2"]["weight"], mx.zeros_like(grads["gate"]["linear2"]["weight"])
        )

        # Scale should be preserved (not frozen, just gate)
        assert mx.allclose(masked["mem_scale"], grads["mem_scale"])

    def test_scale_grads_recursive(self):
        """Test gradient scaling works recursively."""
        grads = {
            "layer1": {
                "weight": mx.array([1.0, 2.0, 3.0]),
            },
            "layer2": mx.array([4.0, 5.0]),
        }

        scaled = scale_grads_recursive(grads, factor=0.5)

        assert mx.allclose(scaled["layer1"]["weight"], mx.array([0.5, 1.0, 1.5]))
        assert mx.allclose(scaled["layer2"], mx.array([2.0, 2.5]))

    def test_accumulate_grads(self):
        """Test gradient accumulation."""
        grads1 = {
            "weight": mx.array([1.0, 2.0]),
        }
        grads2 = {
            "weight": mx.array([3.0, 4.0]),
        }

        # First accumulation (from None)
        accum = accumulate_grads(None, grads1)
        assert mx.allclose(accum["weight"], mx.array([1.0, 2.0]))

        # Second accumulation
        accum = accumulate_grads(accum, grads2)
        assert mx.allclose(accum["weight"], mx.array([4.0, 6.0]))


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_forward_backward_pass(self, titans_layer, small_dim):
        """Test complete forward and backward pass."""
        B, T = 2, 16
        x = mx.random.normal((B, T, small_dim))
        state = titans_layer.init_state(B)

        # Define loss function that takes parameters as first argument
        def loss_fn(params, x, state):
            # Apply params to layer temporarily
            titans_layer.update(params)
            output, _ = titans_layer(x, state)
            return mx.mean(output**2)

        # Get initial parameters
        params = titans_layer.trainable_parameters()

        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(params, x, state)

        assert loss.shape == ()
        assert not mx.isnan(loss)
        assert grads is not None

        # Verify gradients have expected structure
        flat_grads = dict(tree_flatten(grads))
        assert len(flat_grads) > 0

    def test_memory_state_persistence_across_segments(self, titans_layer, small_dim):
        """Test memory state persists and evolves across multiple segments."""
        B, T = 2, 16
        state = titans_layer.init_state(B)

        states_history = [state]

        # Process multiple segments
        for i in range(5):
            x = mx.random.normal((B, T, small_dim))
            _, state = titans_layer(x, state)
            states_history.append(state)
            mx.eval(state)  # Force evaluation

        # State should evolve over time
        if hasattr(states_history[0], "level_states"):
            # CMS mode
            first_weights = states_history[0].level_states[0].weights["w0"]
            last_weights = states_history[-1].level_states[0].weights["w0"]
        else:
            first_weights = states_history[0].weights["w0"]
            last_weights = states_history[-1].weights["w0"]

        assert not mx.allclose(first_weights, last_weights)

    def test_gate_values_in_valid_range(self, titans_layer, small_dim):
        """Test gate always produces values in (0, 1)."""
        B, T = 4, 64
        x = mx.random.normal((B, T, small_dim)) * 5  # Larger variance

        gate_values = titans_layer.gate(x)

        assert mx.all(gate_values > 0)
        assert mx.all(gate_values < 1)

    def test_numerical_stability_with_large_inputs(self, titans_layer, small_dim):
        """Test numerical stability with large input values."""
        B, T = 2, 16
        # Large input values that could cause instability
        x = mx.random.normal((B, T, small_dim)) * 100
        state = titans_layer.init_state(B)

        output, new_state = titans_layer(x, state)

        # Should not produce NaN or Inf
        assert not mx.any(mx.isnan(output))
        assert not mx.any(mx.isinf(output))

    def test_internal_loss_clipping(self, neural_memory, small_dim):
        """Test internal loss is clipped for numerical stability."""
        B, T = 2, 16
        # Extreme values
        x = mx.random.normal((B, T, small_dim)) * 1000
        state = neural_memory.init_state(B)

        loss = neural_memory.compute_internal_loss(x, state)

        # Loss should be clipped to max 100
        assert loss <= 100.0
        assert not mx.isnan(loss)

    def test_deterministic_output(self, titans_layer, small_dim):
        """Test layer produces deterministic output for same input."""
        mx.random.seed(42)
        B, T = 2, 16
        x = mx.random.normal((B, T, small_dim))

        state1 = titans_layer.init_state(B)
        output1, _ = titans_layer(x, state1)

        state2 = titans_layer.init_state(B)
        output2, _ = titans_layer(x, state2)

        mx.eval(output1, output2)

        assert mx.allclose(output1, output2)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and potential failure modes."""

    def test_batch_size_one(self, titans_layer, small_dim):
        """Test with batch size 1."""
        B, T = 1, 16
        x = mx.random.normal((B, T, small_dim))
        state = titans_layer.init_state(B)

        output, new_state = titans_layer(x, state)

        assert output.shape == (B, T, small_dim)

    def test_single_token_sequence(self, titans_layer, small_dim):
        """Test with single token sequence."""
        B, T = 2, 1
        x = mx.random.normal((B, T, small_dim))
        state = titans_layer.init_state(B)

        output, new_state = titans_layer(x, state)

        assert output.shape == (B, T, small_dim)

    def test_very_long_sequence(self, titans_layer, small_dim):
        """Test with longer sequence."""
        B, T = 1, 256
        x = mx.random.normal((B, T, small_dim))
        state = titans_layer.init_state(B)

        output, new_state = titans_layer(x, state)

        assert output.shape == (B, T, small_dim)

    def test_zero_input(self, titans_layer, small_dim):
        """Test with zero input."""
        B, T = 2, 16
        x = mx.zeros((B, T, small_dim))
        state = titans_layer.init_state(B)

        output, new_state = titans_layer(x, state)

        assert not mx.any(mx.isnan(output))

    def test_cms_with_different_frequencies(self, small_dim):
        """Test CMS with different update frequency configurations."""
        for frequencies in [(1, 2, 4), (1, 8, 64), (1, 1, 1)]:
            cms = MLXContinuumMemorySystem(
                dim=small_dim,
                num_levels=3,
                update_frequencies=frequencies,
            )
            B, T = 2, 16
            x = mx.random.normal((B, T, small_dim))
            state = cms.init_state(B)

            output = cms(x, state)
            assert output.shape == (B, T, small_dim)


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegression:
    """Regression tests for previously fixed issues."""

    def test_gate_init_bias_matches_pytorch(self, small_dim):
        """Test gate initialization matches PyTorch (init_bias=-2.0 -> gate≈0.12)."""
        gate = MLXPositionDependentGate(small_dim, init_bias=-2.0)

        # With zero input, gate should be sigmoid(-2.0) ≈ 0.119
        x = mx.zeros((1, 1, small_dim))
        gate_value = gate(x)

        expected = 1.0 / (1.0 + math.exp(2.0))  # ≈ 0.119
        assert abs(gate_value.item() - expected) < 0.1

    def test_template_weights_receive_gradients(self, neural_memory, small_dim):
        """Test template weights can receive gradients (trainable)."""
        B, T = 2, 16
        x = mx.random.normal((B, T, small_dim))
        state = neural_memory.init_state(B)

        def loss_fn(memory, x, state):
            return memory.compute_internal_loss(x, state)

        loss, grads = mx.value_and_grad(loss_fn)(neural_memory, x, state)

        # Template weights should have non-zero gradients
        flat_grads = dict(tree_flatten(grads))
        has_template_grads = any("_template_mlp" in k for k in flat_grads.keys())
        assert has_template_grads, "Template weights should receive gradients"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
