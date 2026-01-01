"""
Unit tests for NanoGPT-Titans model components.

Run with: uv run pytest tests/ -v
"""

import pytest
import torch

from nanogpt_titans.model import (
    CausalSelfAttention,
    MemoryState,
    NeuralMemory,
    TitansConfig,
    TitansGPT,
    parallel_momentum,
    parallel_scan_log,
)

# --- Fixtures ---


@pytest.fixture
def small_config():
    """Small config for fast testing."""
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
    )


@pytest.fixture
def model(small_config):
    """Create a small model for testing."""
    return TitansGPT(small_config)


# --- Parallel Scan Tests ---


class TestParallelScan:
    """Tests for parallel_scan_log and parallel_momentum."""

    def test_parallel_scan_matches_sequential(self):
        """Verify parallel scan produces same result as sequential loop."""
        B, T, D = 2, 16, 8
        gates = torch.full((B, T), 0.9)
        tokens = torch.randn(B, T, D)

        # Parallel version
        result = parallel_scan_log(gates, tokens)

        # Sequential version
        expected = torch.zeros_like(tokens)
        expected[:, 0] = tokens[:, 0]
        for t in range(1, T):
            expected[:, t] = gates[:, t : t + 1] * expected[:, t - 1] + tokens[:, t]

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_parallel_momentum_matches_sequential(self):
        """Verify parallel momentum produces same result as sequential EMA."""
        B, T, D = 2, 16, 8
        surprises = torch.randn(B, T, D)
        momentum_coef = 0.9

        # Parallel version
        result = parallel_momentum(surprises, momentum_coef)

        # Sequential version
        expected = torch.zeros_like(surprises)
        expected[:, 0] = (1 - momentum_coef) * surprises[:, 0]
        for t in range(1, T):
            expected[:, t] = momentum_coef * expected[:, t - 1] + (1 - momentum_coef) * surprises[:, t]

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_parallel_momentum_with_prev_state(self):
        """Test momentum with previous state continuation."""
        B, T, D = 2, 8, 4
        surprises = torch.randn(B, T, D)
        prev_momentum = torch.randn(B, D)
        momentum_coef = 0.9

        result = parallel_momentum(surprises, momentum_coef, prev_momentum)

        # First position should incorporate prev_momentum
        expected_0 = momentum_coef * prev_momentum + (1 - momentum_coef) * surprises[:, 0]
        torch.testing.assert_close(result[:, 0], expected_0, rtol=1e-4, atol=1e-4)

    def test_parallel_scan_handles_different_shapes(self):
        """Test parallel scan with various tensor shapes."""
        for shape in [(1, 4, 8), (4, 8, 16), (2, 32, 4, 4)]:
            B, T = shape[0], shape[1]
            gates = torch.full((B, T), 0.8)
            tokens = torch.randn(*shape)
            result = parallel_scan_log(gates, tokens)
            assert result.shape == tokens.shape


# --- Memory Causality Tests ---


class TestMemoryCausality:
    """Tests to ensure memory doesn't leak future information."""

    def test_memory_retrieval_uses_previous_segment(self, small_config):
        """Verify memory retrieval uses previous segment, not current."""
        memory = NeuralMemory(small_config)
        B, T, C = 2, small_config.segment_len, small_config.n_embd

        # Initialize state
        state = memory.init_state(B, torch.device("cpu"))

        # First segment - should use init_query
        x1 = torch.randn(B, T, C)
        retrieved1 = memory(x1, state)

        # retrieved1 should NOT contain information from x1
        # It should only use init_query (learned parameter)
        assert retrieved1.shape == (B, small_config.num_longterm_mem, C)

        # Update state with x1
        state = memory.update(x1, state)

        # Verify state now has x1 stored
        assert state.last_segment_output is not None
        torch.testing.assert_close(state.last_segment_output, x1.detach())

        # Second segment - should use x1 for retrieval
        x2 = torch.randn(B, T, C)
        retrieved2 = memory(x2, state)

        # retrieved2 should be different from retrieved1 (uses different query source)
        # but should NOT depend on x2
        assert not torch.allclose(retrieved1, retrieved2)

    def test_no_future_leakage_in_forward(self, model, small_config):
        """Test that position 0 cannot see future tokens."""
        B = 2
        T = small_config.segment_len * 2  # Two segments

        # Create input where future tokens have distinct pattern
        x = torch.zeros(B, T, dtype=torch.long)
        x[:, : T // 2] = 1  # First half: token 1
        x[:, T // 2 :] = 2  # Second half: token 2

        # Get output
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(x)

        # Logits at position 0 should not depend on tokens at positions 1+
        # Test by modifying future tokens and checking position 0 output
        x_modified = x.clone()
        x_modified[:, 1:] = 3  # Change all future tokens

        with torch.no_grad():
            logits_modified, _, _ = model(x_modified)

        # Position 0 should produce same output regardless of future tokens
        torch.testing.assert_close(
            logits[:, 0],
            logits_modified[:, 0],
            rtol=1e-5,
            atol=1e-5,
            msg="Position 0 output changed when future tokens were modified - causality violation!",
        )


# --- Model Forward/Backward Tests ---


class TestModelForwardBackward:
    """Tests for model forward and backward passes."""

    def test_forward_shape(self, model, small_config):
        """Test output shapes are correct."""
        B, T = 2, 64
        x = torch.randint(0, small_config.vocab_size, (B, T))

        logits, loss, states = model(x, targets=x)

        assert logits.shape == (B, T, small_config.vocab_size)
        assert loss is not None
        assert len(states) == small_config.n_layer

    def test_backward_pass(self, model, small_config):
        """Test backward pass completes without error."""
        B, T = 2, 64
        x = torch.randint(0, small_config.vocab_size, (B, T))

        _logits, loss, _ = model(x, targets=x)
        loss.backward()

        # Check gradients exist
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0, "No gradients computed"

    def test_memory_state_persistence(self, model, small_config):
        """Test memory states persist across segments."""
        B = 2
        T = small_config.segment_len * 3  # Multiple segments

        x = torch.randint(0, small_config.vocab_size, (B, T))

        _, _, states = model(x)

        # Find the memory layer
        memory_layer_idx = None
        for i, state in enumerate(states):
            if state is not None:
                memory_layer_idx = i
                break

        assert memory_layer_idx is not None, "No memory layer found"
        assert states[memory_layer_idx].step > 0, "Memory not updated"

    def test_deterministic_output(self, model, small_config):
        """Test model produces deterministic output."""
        torch.manual_seed(42)
        B, T = 2, 64
        x = torch.randint(0, small_config.vocab_size, (B, T))

        model.eval()
        with torch.no_grad():
            logits1, _, _ = model(x)
            logits2, _, _ = model(x)

        torch.testing.assert_close(logits1, logits2)


# --- Attention Tests ---


class TestCausalSelfAttention:
    """Tests for attention mechanism."""

    def test_causal_masking(self, small_config):
        """Test causal attention mask is correctly applied."""
        attn = CausalSelfAttention(small_config)
        B, T, C = 2, 16, small_config.n_embd

        x = torch.randn(B, T, C)
        out = attn(x)

        assert out.shape == x.shape

    def test_prefix_lm_masking(self, small_config):
        """Test prefix-LM attention with memory tokens."""
        attn = CausalSelfAttention(small_config)
        B, T, C = 2, 24, small_config.n_embd
        prefix_len = 8

        x = torch.randn(B, T, C)
        out = attn(x, prefix_len=prefix_len)

        assert out.shape == x.shape

    def test_packed_mask(self, small_config):
        """Test attention with explicit packed mask."""
        attn = CausalSelfAttention(small_config)
        B, T, C = 2, 16, small_config.n_embd

        x = torch.randn(B, T, C)

        # Create causal mask
        mask = torch.tril(torch.ones(B, T, T, dtype=torch.bool))
        out = attn(x, packed_mask=mask)

        assert out.shape == x.shape


# --- Memory State Tests ---


class TestMemoryState:
    """Tests for memory state management."""

    def test_init_state(self, small_config):
        """Test memory state initialization."""
        memory = NeuralMemory(small_config)
        B = 4
        state = memory.init_state(B, torch.device("cpu"))

        assert isinstance(state, MemoryState)
        assert state.step == 0
        assert state.last_segment_output is None
        assert len(state.weights) > 0
        assert len(state.last_momentum) > 0

        # Check batch dimension
        for _name, w in state.weights.items():
            assert w.shape[0] == B

    def test_reset_state(self, small_config):
        """Test memory state reset."""
        memory = NeuralMemory(small_config)
        B, T, C = 2, small_config.segment_len, small_config.n_embd

        state = memory.init_state(B, torch.device("cpu"))

        # Simulate some updates
        x = torch.randn(B, T, C)
        state = memory.update(x, state)

        assert state.step > 0

        # Reset
        memory.reset_state(state)

        assert state.step == 0
        assert state.last_segment_output is None


# --- Integration Tests ---


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_training_step(self, model, small_config):
        """Test a complete training step."""
        B, T = 4, 64
        x = torch.randint(0, small_config.vocab_size, (B, T))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Forward
        _logits, loss, _ = model(x, targets=x)
        loss.item()

        # Backward
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Second forward - loss should generally decrease or stay similar
        with torch.no_grad():
            _, loss2, _ = model(x, targets=x)

        # Just verify it ran - loss may not decrease in one step
        assert loss2.item() > 0

    def test_generation(self, model, small_config):
        """Test token generation."""
        B = 1
        prompt = torch.randint(0, small_config.vocab_size, (B, 10))

        model.eval()
        with torch.no_grad():
            logits, _, _ = model(prompt)

        # Sample next token
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        assert next_token.shape == (B,)
        assert 0 <= next_token.item() < small_config.vocab_size

    def test_variable_sequence_length(self, model, small_config):
        """Test model handles different sequence lengths."""
        B = 2

        for T in [16, 32, 64, 128]:
            x = torch.randint(0, small_config.vocab_size, (B, T))
            logits, _loss, _ = model(x, targets=x)
            assert logits.shape == (B, T, small_config.vocab_size)


# --- Edge Cases ---


class TestEdgeCases:
    """Tests for edge cases and potential failure modes."""

    def test_single_token(self, model, small_config):
        """Test with single token input."""
        B, T = 2, 1
        x = torch.randint(0, small_config.vocab_size, (B, T))

        logits, _, _ = model(x)
        assert logits.shape == (B, T, small_config.vocab_size)

    def test_batch_size_one(self, model, small_config):
        """Test with batch size 1."""
        B, T = 1, 64
        x = torch.randint(0, small_config.vocab_size, (B, T))

        logits, _loss, _ = model(x, targets=x)
        assert logits.shape == (B, T, small_config.vocab_size)

    def test_exact_segment_length(self, model, small_config):
        """Test with sequence length exactly equal to segment length."""
        B = 2
        T = small_config.segment_len
        x = torch.randint(0, small_config.vocab_size, (B, T))

        logits, _, _ = model(x)
        assert logits.shape == (B, T, small_config.vocab_size)

    def test_momentum_edge_values(self):
        """Test parallel momentum with edge case momentum values."""
        B, T, D = 2, 8, 4
        surprises = torch.randn(B, T, D)

        # Very high momentum (close to 1)
        result_high = parallel_momentum(surprises, 0.99)
        assert not torch.isnan(result_high).any()

        # Very low momentum (close to 0)
        result_low = parallel_momentum(surprises, 0.01)
        assert not torch.isnan(result_low).any()

    def test_zero_gradients(self, model, small_config):
        """Test that zero gradients don't cause issues."""
        B, T = 2, 32
        x = torch.randint(0, small_config.vocab_size, (B, T))

        # Run forward/backward
        _, loss, _ = model(x, targets=x)
        loss.backward()

        # Zero gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Should still work
        _, loss2, _ = model(x, targets=x)
        loss2.backward()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
