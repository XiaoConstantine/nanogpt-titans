"""
Tests for torch.compile compatibility and graph break optimizations.

These tests verify that:
1. The model works correctly with torch.compile
2. @torch.compiler.disable decorators don't break functionality
3. Training/eval mode behavior is correct with fused CE
4. Memory operations work correctly after decorator changes

Run with: uv run pytest tests/test_compile.py -v
"""

import pytest
import torch

from nanogpt_titans.model import (
    MemoryState,
    NeuralMemory,
    TitansConfig,
    TitansGPT,
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


# --- Compiler Disable Decorator Tests ---


class TestCompilerDisableDecorators:
    """Tests that @torch.compiler.disable decorators work correctly."""

    def test_memory_init_state_works(self, small_config):
        """Test NeuralMemory.init_state works with decorator."""
        memory = NeuralMemory(small_config)
        B = 4

        state = memory.init_state(B, torch.device("cpu"))

        assert isinstance(state, MemoryState)
        assert state.step == 0
        assert len(state.weights) > 0
        # Verify batch dimension
        for _name, w in state.weights.items():
            assert w.shape[0] == B

    def test_memory_reset_state_works(self, small_config):
        """Test NeuralMemory.reset_state works with decorator."""
        memory = NeuralMemory(small_config)
        B, T, C = 2, small_config.segment_len, small_config.n_embd

        state = memory.init_state(B, torch.device("cpu"))
        x = torch.randn(B, T, C)
        state = memory.update(x, state)

        assert state.step > 0

        memory.reset_state(state)

        assert state.step == 0
        assert state.last_segment_output is None

    def test_memory_forward_works(self, small_config):
        """Test NeuralMemory.forward works with decorator."""
        memory = NeuralMemory(small_config)
        B, T, C = 2, small_config.segment_len, small_config.n_embd

        state = memory.init_state(B, torch.device("cpu"))
        x = torch.randn(B, T, C)

        retrieved = memory(x, state)

        assert retrieved.shape == (B, small_config.num_longterm_mem, C)
        assert not torch.isnan(retrieved).any()

    def test_memory_update_works(self, small_config):
        """Test NeuralMemory.update works with decorator."""
        memory = NeuralMemory(small_config)
        B, T, C = 2, small_config.segment_len, small_config.n_embd

        state = memory.init_state(B, torch.device("cpu"))
        x = torch.randn(B, T, C)

        new_state = memory.update(x, state)

        assert new_state.step == state.step + 1
        assert new_state.last_segment_output is not None

    def test_model_init_memory_states_works(self, model, small_config):
        """Test TitansGPT.init_memory_states works with decorator."""
        B = 2

        states = model.init_memory_states(B, torch.device("cpu"))

        assert len(states) == small_config.n_layer
        # At least one layer should have memory
        assert any(s is not None for s in states)

    def test_model_reset_memory_states_works(self, model, small_config):
        """Test TitansGPT.reset_memory_states works with decorator."""
        B, T = 2, 64
        x = torch.randint(0, small_config.vocab_size, (B, T))

        _, _, states = model(x)

        # Find memory state and verify step > 0
        memory_state = next(s for s in states if s is not None)
        assert memory_state.step > 0

        model.reset_memory_states(states)

        # Verify reset worked
        memory_state = next(s for s in states if s is not None)
        assert memory_state.step == 0


# --- Training Mode Behavior Tests ---


class TestTrainingModeBehavior:
    """Tests for training vs eval mode behavior."""

    def test_training_mode_forward(self, model, small_config):
        """Test forward pass in training mode."""
        B, T = 2, 64
        x = torch.randint(0, small_config.vocab_size, (B, T))

        model.train()
        logits, loss, _states = model(x, targets=x)

        assert loss is not None
        assert loss.item() > 0
        assert not torch.isnan(logits).any()

    def test_eval_mode_forward(self, model, small_config):
        """Test forward pass in eval mode."""
        B, T = 2, 64
        x = torch.randint(0, small_config.vocab_size, (B, T))

        model.eval()
        with torch.no_grad():
            logits, _loss, _states = model(x, targets=x)

        assert logits.shape == (B, T, small_config.vocab_size)
        assert not torch.isnan(logits).any()

    def test_training_backward_pass(self, model, small_config):
        """Test backward pass completes in training mode."""
        B, T = 2, 64
        x = torch.randint(0, small_config.vocab_size, (B, T))

        model.train()
        _logits, loss, _ = model(x, targets=x)
        loss.backward()

        # Verify gradients exist
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_training_eval_consistency(self, model, small_config):
        """Test that loss is similar between train and eval modes."""
        B, T = 2, 64
        x = torch.randint(0, small_config.vocab_size, (B, T))

        # Training mode
        model.train()
        _, loss_train, _ = model(x, targets=x)

        # Eval mode (no dropout)
        model.eval()
        with torch.no_grad():
            _, loss_eval, _ = model(x, targets=x)

        # Losses should be in same ballpark (dropout=0 by default)
        assert abs(loss_train.item() - loss_eval.item()) < 1.0


# --- Memory Operations Correctness ---


class TestMemoryOperationsCorrectness:
    """Tests that memory operations produce correct results after changes."""

    def test_memory_update_changes_weights(self, small_config):
        """Verify memory update actually modifies weights."""
        memory = NeuralMemory(small_config)
        B, T, C = 2, small_config.segment_len, small_config.n_embd

        state = memory.init_state(B, torch.device("cpu"))
        x = torch.randn(B, T, C)

        # Store original weights
        orig_weights = {k: v.clone() for k, v in state.weights.items()}

        # Update
        new_state = memory.update(x, state)

        # Weights should have changed
        for name in orig_weights:
            assert not torch.allclose(orig_weights[name], new_state.weights[name])

    def test_memory_retrieval_varies_with_state(self, small_config):
        """Verify retrieval produces different results with different states."""
        memory = NeuralMemory(small_config)
        B, T, C = 2, small_config.segment_len, small_config.n_embd

        state = memory.init_state(B, torch.device("cpu"))
        x = torch.randn(B, T, C)

        # First retrieval (with initial state)
        retrieved1 = memory(x, state)

        # Update state
        state = memory.update(x, state)

        # Second retrieval (with updated state)
        retrieved2 = memory(x, state)

        # Results should differ
        assert not torch.allclose(retrieved1, retrieved2)

    def test_multiple_segment_processing(self, model, small_config):
        """Test processing multiple segments maintains correctness."""
        B = 2
        T = small_config.segment_len * 3  # Three segments
        x = torch.randint(0, small_config.vocab_size, (B, T))

        model.eval()
        with torch.no_grad():
            logits, _, states = model(x)

        assert logits.shape == (B, T, small_config.vocab_size)

        # Memory should have been updated multiple times
        memory_state = next(s for s in states if s is not None)
        assert memory_state.step == 3  # One update per segment


# --- torch.compile Tests (CUDA only) ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile tests")
class TestTorchCompile:
    """Tests for torch.compile compatibility (requires CUDA)."""

    def test_compile_forward(self, small_config):
        """Test compiled model forward pass."""
        model = TitansGPT(small_config).cuda()
        model = torch.compile(model)

        B, T = 2, 64
        x = torch.randint(0, small_config.vocab_size, (B, T)).cuda()

        logits, loss, _states = model(x, targets=x)

        assert logits.shape == (B, T, small_config.vocab_size)
        assert loss is not None

    def test_compile_backward(self, small_config):
        """Test compiled model backward pass."""
        model = TitansGPT(small_config).cuda()
        model = torch.compile(model)

        B, T = 2, 64
        x = torch.randint(0, small_config.vocab_size, (B, T)).cuda()

        _logits, loss, _ = model(x, targets=x)
        loss.backward()

        # Verify gradients exist
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_compile_multiple_iterations(self, small_config):
        """Test compiled model over multiple training iterations."""
        model = TitansGPT(small_config).cuda()
        model = torch.compile(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        B, T = 2, 64
        x = torch.randint(0, small_config.vocab_size, (B, T)).cuda()

        losses = []
        for _ in range(3):
            _logits, loss, _ = model(x, targets=x)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # All losses should be valid
        assert all(v > 0 and float("inf") != v for v in losses)

    def test_compile_with_different_sequence_lengths(self, small_config):
        """Test compiled model with varying sequence lengths."""
        model = TitansGPT(small_config).cuda()
        # Use dynamic=True for variable shapes
        model = torch.compile(model, dynamic=True)

        B = 2
        for T in [32, 64, 96]:
            x = torch.randint(0, small_config.vocab_size, (B, T)).cuda()
            logits, _, _ = model(x)
            assert logits.shape == (B, T, small_config.vocab_size)


# --- Regression Tests ---


class TestRegressions:
    """Regression tests for specific issues."""

    def test_no_nan_in_outputs(self, model, small_config):
        """Ensure no NaN values in model outputs."""
        B, T = 2, 64
        x = torch.randint(0, small_config.vocab_size, (B, T))

        for _ in range(5):  # Multiple iterations
            logits, loss, _ = model(x, targets=x)

            assert not torch.isnan(logits).any(), "NaN in logits"
            assert not torch.isnan(loss), "NaN in loss"
            assert not torch.isinf(logits).any(), "Inf in logits"
            assert not torch.isinf(loss), "Inf in loss"

    def test_memory_state_not_corrupted(self, model, small_config):
        """Ensure memory states don't get corrupted over iterations."""
        B = 2
        T = small_config.segment_len * 2

        states = model.init_memory_states(B, torch.device("cpu"))

        for _ in range(5):
            x = torch.randint(0, small_config.vocab_size, (B, T))
            _, _, states = model(x, memory_states=states)

            # Check states are valid
            memory_state = next(s for s in states if s is not None)
            for name, w in memory_state.weights.items():
                assert not torch.isnan(w).any(), f"NaN in {name}"
                assert not torch.isinf(w).any(), f"Inf in {name}"

    def test_gradients_flow_correctly(self, model, small_config):
        """Ensure gradients flow to all trainable parameters."""
        B, T = 2, 64
        x = torch.randint(0, small_config.vocab_size, (B, T))

        _, loss, _ = model(x, targets=x)
        loss.backward()

        # Check key parameter groups have gradients
        has_embedding_grad = model.transformer["wte"].weight.grad is not None
        has_lm_head_grad = model.lm_head.weight.grad is not None

        assert has_embedding_grad, "No gradient for embeddings"
        assert has_lm_head_grad, "No gradient for lm_head"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
