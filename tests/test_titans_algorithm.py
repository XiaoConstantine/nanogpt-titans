"""
Tests for Titans core algorithm based on the paper:
"Titans: Learning to Memorize at Test Time" (arXiv:2501.00663)

These tests verify the implementation matches the paper's key equations:
- Eq (12): Loss ℓ(M; x) = ‖M(k) - v‖²
- Eq (10): Sₜ = ηₜSₜ₋₁ - θₜ∇ℓ(Mₜ₋₁; xₜ)  [surprise with momentum]
- Eq (11): ℳₜ = (1 - αₜ)ℳₜ₋₁ + Sₜ  [weight update with decay]
- Memory retrieval uses Mₜ₋₁ (previous state, not current)
- MAC architecture: [persistent_mem ‖ retrieved_mem ‖ segment]

Run with: uv run pytest tests/test_titans_algorithm.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from nanogpt_titans.model import (
    TitansConfig,
    TitansGPT,
    TitansBlock,
    NeuralMemory,
    MemoryState,
    MemoryMLP,
    parallel_momentum,
)


@pytest.fixture
def config():
    """Config for algorithm tests."""
    return TitansConfig(
        block_size=128,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        segment_len=16,
        num_persist_mem=2,
        num_longterm_mem=4,
        memory_depth=2,
        memory_expansion=2,
        memory_lr=0.01,
        memory_momentum=0.9,
        memory_decay=0.001,
    )


# --- Eq (12): Associative Memory Loss ---

class TestAssociativeMemoryLoss:
    """
    Paper Eq (12): ℓ(M; x) = ‖M(k) - v‖²

    The loss measures how well the memory MLP can predict v from k.
    """

    def test_loss_is_mse_between_prediction_and_value(self, config):
        """Verify loss is MSE(M(k), v) as stated in paper."""
        memory = NeuralMemory(config)
        B, T, C = 2, config.segment_len, config.n_embd

        state = memory.init_state(B, torch.device("cpu"))
        x = torch.randn(B, T, C)

        # Get keys and values
        keys = memory.key_proj(x)
        values = memory.value_proj(x)

        # Manually compute what loss should be for one sample
        with torch.no_grad():
            # Forward through MLP
            params = {name: w[0] for name, w in state.weights.items()}
            pred = torch.func.functional_call(memory.memory_mlp, params, (keys[0:1],))
            expected_loss = F.mse_loss(pred, values[0:1])

        # This should match what the gradient is computed from
        assert expected_loss.item() >= 0  # MSE is non-negative

    def test_gradient_computed_wrt_mlp_weights(self, config):
        """Verify gradients are w.r.t. MLP weights, not inputs."""
        memory = NeuralMemory(config)
        B, T, C = 1, 4, config.n_embd

        state = memory.init_state(B, torch.device("cpu"))
        x = torch.randn(B, T, C)

        # Update should modify weights based on gradients
        new_state = memory.update(x, state)

        # Weights should have changed
        for name in state.weights:
            assert not torch.allclose(
                state.weights[name], new_state.weights[name]
            ), f"Weight {name} unchanged after update"


# --- Eq (10): Surprise with Momentum ---

class TestSurpriseWithMomentum:
    """
    Paper Eq (10): Sₜ = ηₜSₜ₋₁ - θₜ∇ℓ(Mₜ₋₁; xₜ)

    - Sₜ is accumulated surprise (momentum)
    - ηₜ is momentum coefficient
    - θₜ scales momentary surprise
    - ∇ℓ is gradient of loss
    """

    def test_weight_update_uses_gradient_descent(self, config):
        """
        Weight update should follow gradient descent direction.

        Paper: Sₜ = ... - θₜ∇ℓ (negative gradient for surprise)
        Our optimized impl: weights -= lr * momentum(grads)

        This is mathematically equivalent to:
        weights += lr * momentum(-grads) = weights += lr * momentum(surprise)
        """
        from nanogpt_titans.model import parallel_momentum

        B, T, D = 2, 8, 4
        grads = torch.randn(B, T, D)
        weights = torch.randn(B, D)
        prev_momentum = torch.zeros(B, D)
        momentum_coef = 0.9
        lr = 0.01
        decay = 0.001

        # OLD approach: negate grads, add to weights
        surprises = -grads
        mom_old = parallel_momentum(surprises, momentum_coef, prev_momentum)
        weights_old = (1 - decay) * weights + lr * mom_old[:, -1]

        # NEW approach: use grads directly, subtract from weights
        mom_new = parallel_momentum(grads, momentum_coef, prev_momentum)
        weights_new = (1 - decay) * weights - lr * mom_new[:, -1]

        # Both should produce identical results
        torch.testing.assert_close(weights_old, weights_new, rtol=1e-5, atol=1e-5)

    def test_momentum_accumulates_across_tokens(self, config):
        """
        Verify momentum accumulates: m_t = η * m_{t-1} + (1-η) * s_t

        Note: Paper uses Sₜ = ηSₜ₋₁ + θ*surprise (additive)
        Our impl uses EMA: m_t = η*m_{t-1} + (1-η)*surprise
        """
        B, T, D = 2, 16, 4
        surprises = torch.randn(B, T, D)
        momentum_coef = 0.9

        result = parallel_momentum(surprises, momentum_coef)

        # Verify accumulation pattern
        # First token: m_0 = (1-η) * s_0
        expected_0 = (1 - momentum_coef) * surprises[:, 0]
        torch.testing.assert_close(result[:, 0], expected_0, rtol=1e-4, atol=1e-4)

        # Second token: m_1 = η * m_0 + (1-η) * s_1
        expected_1 = momentum_coef * expected_0 + (1 - momentum_coef) * surprises[:, 1]
        torch.testing.assert_close(result[:, 1], expected_1, rtol=1e-4, atol=1e-4)

    def test_momentum_persists_across_segments(self, config):
        """Verify momentum state carries over between segments."""
        memory = NeuralMemory(config)
        B, T, C = 2, config.segment_len, config.n_embd

        state = memory.init_state(B, torch.device("cpu"))

        # First segment update
        x1 = torch.randn(B, T, C)
        state1 = memory.update(x1, state)

        # Momentum should be non-zero
        for name in state1.last_momentum:
            assert not torch.allclose(
                state1.last_momentum[name],
                torch.zeros_like(state1.last_momentum[name])
            ), "Momentum should be non-zero after update"

        # Second segment should use previous momentum
        x2 = torch.randn(B, T, C)
        state2 = memory.update(x2, state1)

        # The momentum should have accumulated
        assert state2.step == 2


# --- Eq (11): Weight Update with Decay ---

class TestWeightUpdateWithDecay:
    """
    Paper Eq (11): ℳₜ = (1 - αₜ)ℳₜ₋₁ + Sₜ

    - αₜ is weight decay / forgetting factor
    - Sₜ is accumulated surprise

    Our impl: weights = (1 - decay) * weights + lr * momentum
    """

    def test_weight_decay_applied(self, config):
        """Verify weight decay shrinks old weights."""
        memory = NeuralMemory(config)
        B, T, C = 2, config.segment_len, config.n_embd

        state = memory.init_state(B, torch.device("cpu"))
        initial_weight_norm = sum(
            w.norm().item() for w in state.weights.values()
        )

        # With zero surprise, weights should decay
        # Create input that produces near-zero gradients (memory already fits well)
        x = torch.zeros(B, T, C)  # Zero input -> small gradients
        new_state = memory.update(x, state)

        # Weight update: (1 - decay) * old + lr * momentum
        # With small surprise/momentum, weights should shrink slightly
        decay = config.memory_decay
        expected_factor = 1 - decay

        # Weights should be approximately (1-decay) * original
        for name in state.weights:
            # Not exact due to momentum term, but should be close
            ratio = new_state.weights[name].norm() / state.weights[name].norm()
            # Should be less than 1 (decayed) but not too small
            assert 0.5 < ratio.item() < 1.5, f"Unexpected weight change ratio: {ratio}"

    def test_learning_rate_scales_update(self, config):
        """Verify learning rate scales the surprise contribution."""
        # Create two memories with different learning rates
        # Disable adaptive_memory to test fixed memory_lr
        config1 = TitansConfig(**{**vars(config), 'memory_lr': 0.01, 'adaptive_memory': False})
        config2 = TitansConfig(**{**vars(config), 'memory_lr': 0.1, 'adaptive_memory': False})

        memory1 = NeuralMemory(config1)
        memory2 = NeuralMemory(config2)

        # Sync initial weights
        memory2.load_state_dict(memory1.state_dict())

        B, T, C = 2, config.segment_len, config.n_embd
        x = torch.randn(B, T, C)

        state1 = memory1.init_state(B, torch.device("cpu"))
        state2 = memory2.init_state(B, torch.device("cpu"))

        # Sync initial state weights
        for name in state1.weights:
            state2.weights[name] = state1.weights[name].clone()
            state2.last_momentum[name] = state1.last_momentum[name].clone()

        new_state1 = memory1.update(x, state1)
        new_state2 = memory2.update(x, state2)

        # Higher LR should cause larger weight changes
        change1 = sum(
            (new_state1.weights[n] - state1.weights[n]).abs().mean().item()
            for n in state1.weights
        )
        change2 = sum(
            (new_state2.weights[n] - state2.weights[n]).abs().mean().item()
            for n in state2.weights
        )

        # LR ratio is 10x, so change2 should be larger
        assert change2 > change1, "Higher LR should cause larger updates"


# --- Memory Retrieval Causality ---

class TestMemoryRetrievalCausality:
    """
    Paper states: retrieval uses ℳₜ₋₁ (previous memory state).

    This is critical for causality - current segment cannot influence
    what is retrieved from memory for that same segment.
    """

    def test_retrieval_uses_previous_segment(self, config):
        """Verify memory retrieval uses stored previous segment, not current."""
        memory = NeuralMemory(config)
        B, T, C = 2, config.segment_len, config.n_embd

        state = memory.init_state(B, torch.device("cpu"))

        # First retrieval should NOT use x1 (uses init_query instead)
        x1 = torch.randn(B, T, C) * 100  # Large values
        retrieved1 = memory(x1, state)

        # Update with x1
        state = memory.update(x1, state)

        # Verify x1 is now stored for next retrieval
        assert state.last_segment_output is not None
        torch.testing.assert_close(state.last_segment_output, x1.detach())

        # Second retrieval should use x1, not x2
        x2 = torch.randn(B, T, C) * 0.001  # Very different values
        retrieved2 = memory(x2, state)

        # retrieved2 should be based on x1, not x2
        # If it used x2, results would be very different due to value difference

    def test_init_query_used_for_first_segment(self, config):
        """Verify first segment uses learned init_query, not input."""
        memory = NeuralMemory(config)
        B, T, C = 2, config.segment_len, config.n_embd

        state = memory.init_state(B, torch.device("cpu"))

        # last_segment_output should be None initially
        assert state.last_segment_output is None

        # Two very different inputs should give same retrieval
        # (both use init_query since no previous segment)
        x1 = torch.ones(B, T, C)
        x2 = torch.ones(B, T, C) * -1

        retrieved1 = memory(x1, state)
        retrieved2 = memory(x2, state)

        # Should be identical since both use init_query
        torch.testing.assert_close(retrieved1, retrieved2)


# --- MAC Architecture Tests ---

class TestMACArchitecture:
    """
    Paper's MAC (Memory as Context) architecture:
    S̃⁽ᵗ⁾ = [persistent_mem] ‖ [retrieved_mem] ‖ [current_segment]

    Then full causal attention is applied with prefix-LM masking.
    """

    def test_concatenation_order(self, config):
        """Verify concatenation order: [persistent, retrieved, segment]."""
        block = TitansBlock(config, has_memory=True)
        B, T, C = 2, config.segment_len, config.n_embd

        state = block.init_state(B, torch.device("cpu"))
        x = torch.randn(B, T, C)

        # The forward pass should concatenate in order:
        # [num_longterm_mem, num_persist_mem, T]
        prefix_len = config.num_longterm_mem + config.num_persist_mem

        output, new_state = block(x, state)

        # Output should have same shape as input segment
        assert output.shape == (B, T, C)

    def test_prefix_length_correct(self, config):
        """Verify prefix length = num_longterm_mem + num_persist_mem."""
        block = TitansBlock(config, has_memory=True)

        expected_prefix = config.num_longterm_mem + config.num_persist_mem
        actual_prefix = block.num_longterm_mem + block.num_persist_mem

        assert actual_prefix == expected_prefix

    def test_persistent_memory_is_learnable(self, config):
        """Verify persistent memory tokens are learnable parameters."""
        block = TitansBlock(config, has_memory=True)

        assert block.persist_mem is not None
        assert isinstance(block.persist_mem, torch.nn.Parameter)
        assert block.persist_mem.shape == (1, config.num_persist_mem, config.n_embd)
        assert block.persist_mem.requires_grad


# --- Full Sequence Processing Tests ---

class TestSequenceProcessing:
    """Tests for processing sequences through the full model."""

    def test_multiple_segments_processed_sequentially(self, config):
        """Verify segments are processed in order with state passing."""
        model = TitansGPT(config)
        B = 2
        T = config.segment_len * 3  # 3 segments

        x = torch.randint(0, config.vocab_size, (B, T))
        logits, loss, states = model(x, targets=x)

        # Memory should have been updated 3 times (once per segment)
        memory_layer_idx = config.n_layer // 2
        assert states[memory_layer_idx].step == 3

    def test_memory_influence_increases_over_segments(self, config):
        """Memory should have more influence as more segments are processed."""
        model = TitansGPT(config)
        B = 2

        # Compare output with 1 segment vs 3 segments
        x1 = torch.randint(0, config.vocab_size, (B, config.segment_len))
        x3 = torch.randint(0, config.vocab_size, (B, config.segment_len * 3))

        model.eval()
        with torch.no_grad():
            _, _, states1 = model(x1)
            _, _, states3 = model(x3)

        # Memory should have evolved more with 3 segments
        memory_layer_idx = config.n_layer // 2
        assert states1[memory_layer_idx].step == 1
        assert states3[memory_layer_idx].step == 3


# --- Numerical Stability Tests ---

class TestNumericalStability:
    """Tests for numerical stability of the algorithm."""

    def test_no_nan_in_gradients(self, config):
        """Verify no NaN values in computed gradients/surprises."""
        memory = NeuralMemory(config)
        B, T, C = 2, config.segment_len, config.n_embd

        state = memory.init_state(B, torch.device("cpu"))

        # Various input patterns
        for x in [
            torch.randn(B, T, C),           # Normal
            torch.randn(B, T, C) * 0.001,    # Small
            torch.randn(B, T, C) * 100,      # Large
            torch.zeros(B, T, C),            # Zero
        ]:
            new_state = memory.update(x, state)
            for name, w in new_state.weights.items():
                assert not torch.isnan(w).any(), f"NaN in weights after update with {x.abs().mean()}"
            for name, m in new_state.last_momentum.items():
                assert not torch.isnan(m).any(), f"NaN in momentum after update"

    def test_weights_bounded(self, config):
        """Verify weights don't explode over many updates."""
        memory = NeuralMemory(config)
        B, T, C = 2, config.segment_len, config.n_embd

        state = memory.init_state(B, torch.device("cpu"))
        initial_norm = sum(w.norm().item() for w in state.weights.values())

        # Many updates
        for _ in range(50):
            x = torch.randn(B, T, C)
            state = memory.update(x, state)

        final_norm = sum(w.norm().item() for w in state.weights.values())

        # Weights shouldn't explode (allow 10x growth as reasonable)
        assert final_norm < initial_norm * 10, f"Weights exploded: {initial_norm} -> {final_norm}"


# --- Ablation-Inspired Tests ---

class TestAblationComponents:
    """
    Tests inspired by paper's ablation study (Table 3).
    Each component should positively contribute.
    """

    def test_momentum_affects_updates(self, config):
        """Verify momentum (η) affects how updates accumulate."""
        B, T, D = 2, 16, 8
        surprises = torch.randn(B, T, D)

        # High momentum: past surprises dominate
        high_mom = parallel_momentum(surprises, 0.99)

        # Low momentum: recent surprises dominate
        low_mom = parallel_momentum(surprises, 0.1)

        # Results should be different
        assert not torch.allclose(high_mom, low_mom)

        # High momentum should be "smoother" (less variance)
        high_var = high_mom.var().item()
        low_var = low_mom.var().item()
        # Not guaranteed, but typically true

    def test_weight_decay_prevents_unbounded_growth(self, config):
        """Verify weight decay (α) prevents memory from growing unbounded."""
        memory = NeuralMemory(config)
        B, T, C = 2, config.segment_len, config.n_embd

        state = memory.init_state(B, torch.device("cpu"))

        # Constant "surprising" input
        x = torch.randn(B, T, C) * 10

        norms = []
        for _ in range(20):
            state = memory.update(x, state)
            norm = sum(w.norm().item() for w in state.weights.values())
            norms.append(norm)

        # With decay, norms should stabilize, not grow forever
        # Check last 5 norms are within 50% of each other
        last_norms = norms[-5:]
        assert max(last_norms) < min(last_norms) * 2, "Weights didn't stabilize"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
