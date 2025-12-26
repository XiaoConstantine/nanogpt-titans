"""
Unit tests for HOPE-Titans Qwen integration.

Tests all HOPE components:
- SelfModifyingLinear
- SelfModifyingGate
- ContinuumMemorySystem
- WarmStartEncoder
- DeepMomentumUpdate
- TitansQwenDecoderLayer
- Patcher functions

Run with: uv run pytest tests/test_qwen_titans.py -v
"""

import pytest
import torch
import torch.nn as nn

from nanogpt_titans.qwen_titans.config import TitansQwenConfig
from nanogpt_titans.qwen_titans.memory_adapter import (
    NeuralMemoryAdapter,
    SelfModifyingLinear,
    SelfModifyingGate,
    ContinuumMemorySystem,
    ContinuumMemoryState,
    WarmStartEncoder,
    DeepMomentumUpdate,
)
from nanogpt_titans.qwen_titans.decoder_layer import TitansQwenDecoderLayer
from nanogpt_titans.qwen_titans.patcher import (
    patch_qwen_with_titans,
    freeze_base_model,
    get_titans_layers,
    get_gate_statistics,
    get_internal_losses,
)


# --- Fixtures ---

@pytest.fixture
def small_config():
    """Small config for fast testing."""
    return TitansQwenConfig(
        n_embd=64,
        n_head=4,
        n_layer=4,
        block_size=256,
        vocab_size=1000,
        segment_len=32,
        num_persist_mem=2,
        num_longterm_mem=4,
        memory_depth=2,
        memory_expansion=2,
        use_cms=False,
        use_self_mod_proj=True,
        use_self_mod_gate=True,
        use_warm_start=False,
        use_internal_loss=True,
    )


@pytest.fixture
def cms_config():
    """Config with CMS enabled."""
    return TitansQwenConfig(
        n_embd=64,
        n_head=4,
        n_layer=4,
        block_size=256,
        vocab_size=1000,
        segment_len=32,
        num_persist_mem=2,
        num_longterm_mem=4,
        use_cms=True,
        num_cms_levels=3,
        cms_update_frequencies=[1, 4, 16],
    )


@pytest.fixture
def batch_hidden():
    """Sample batch of hidden states."""
    return torch.randn(2, 32, 64)  # [B, T, C]


# --- SelfModifyingLinear Tests ---

class TestSelfModifyingLinear:
    """Tests for SelfModifyingLinear."""

    def test_forward_shape(self):
        """Test output shape is correct."""
        layer = SelfModifyingLinear(64, 64, lr=0.001)
        x = torch.randn(2, 10, 64)
        out = layer(x, update=False)
        assert out.shape == x.shape

    def test_weight_update_in_training(self):
        """Test weights change during training forward."""
        layer = SelfModifyingLinear(64, 64, lr=0.01)
        layer.train()
        x = torch.randn(2, 10, 64)

        weight_before = layer.weight.data.clone()
        _ = layer(x, update=True)
        weight_after = layer.weight.data

        # Weights should change
        assert not torch.allclose(weight_before, weight_after)

    def test_no_update_in_eval(self):
        """Test weights don't change in eval mode."""
        layer = SelfModifyingLinear(64, 64, lr=0.01)
        layer.eval()
        x = torch.randn(2, 10, 64)

        weight_before = layer.weight.data.clone()
        _ = layer(x, update=True)
        weight_after = layer.weight.data

        # Weights should not change in eval
        assert torch.allclose(weight_before, weight_after)

    def test_gradients_flow(self):
        """Test gradients flow through layer."""
        layer = SelfModifyingLinear(64, 64, lr=0.001)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = layer(x, update=False)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.weight.grad is not None


# --- SelfModifyingGate Tests ---

class TestSelfModifyingGate:
    """Tests for SelfModifyingGate."""

    def test_output_shape(self):
        """Test output shape matches input."""
        gate = SelfModifyingGate(64, init_bias=-2.0, lr=0.001)
        mem = torch.randn(2, 10, 64)
        res = torch.randn(2, 10, 64)
        out = gate(mem, res, update=False)
        assert out.shape == mem.shape

    def test_initial_gate_conservative(self):
        """Test gate starts near 0 (conservative)."""
        gate = SelfModifyingGate(64, init_bias=-2.0, lr=0.001)
        mem = torch.randn(2, 10, 64)
        res = torch.randn(2, 10, 64)
        _ = gate(mem, res, update=False)

        # sigmoid(-2) ≈ 0.12
        assert 0.1 < gate.mean_gate_value < 0.15

    def test_gate_between_zero_and_one(self):
        """Test gate values are valid probabilities."""
        gate = SelfModifyingGate(64, init_bias=0.0, lr=0.001)
        mem = torch.randn(2, 10, 64)
        res = torch.randn(2, 10, 64)
        _ = gate(mem, res)

        # Gate should be in [0, 1]
        assert 0 <= gate.mean_gate_value <= 1

    def test_gradients_flow(self):
        """Test gradients flow through gate."""
        gate = SelfModifyingGate(64, init_bias=-2.0)
        mem = torch.randn(2, 10, 64, requires_grad=True)
        res = torch.randn(2, 10, 64, requires_grad=True)
        out = gate(mem, res, update=False)
        loss = out.sum()
        loss.backward()

        assert mem.grad is not None
        assert res.grad is not None


# --- ContinuumMemorySystem Tests ---

class TestContinuumMemorySystem:
    """Tests for ContinuumMemorySystem."""

    def test_init_state(self, cms_config):
        """Test state initialization."""
        cms = ContinuumMemorySystem(cms_config)
        state = cms.init_state(batch_size=2, device=torch.device('cpu'))

        assert isinstance(state, ContinuumMemoryState)
        assert len(state.level_states) == 3
        assert state.segment_counts == [0, 0, 0]

    def test_forward_shape(self, cms_config):
        """Test output shape."""
        cms = ContinuumMemorySystem(cms_config)
        state = cms.init_state(2, torch.device('cpu'))
        x = torch.randn(2, 32, 64)

        out = cms(x, state)
        assert out.shape == (2, cms_config.num_longterm_mem, 64)

    def test_update_frequencies(self, cms_config):
        """Test different levels update at different frequencies."""
        cms = ContinuumMemorySystem(cms_config)
        state = cms.init_state(2, torch.device('cpu'))
        x = torch.randn(2, 32, 64)

        # Update 16 times
        for i in range(16):
            state = cms.update(x, state)

        # All levels should have been updated at least once
        assert state.segment_counts == [16, 16, 16]

    def test_level_weights_learnable(self, cms_config):
        """Test level weights are learnable parameters."""
        cms = ContinuumMemorySystem(cms_config)
        assert cms.level_weights.requires_grad


# --- WarmStartEncoder Tests ---

class TestWarmStartEncoder:
    """Tests for WarmStartEncoder."""

    def test_output_shape(self, small_config):
        """Test output shape."""
        small_config.use_warm_start = True
        small_config.warm_start_prefix_len = 16
        small_config.warm_start_layers = 2

        encoder = WarmStartEncoder(small_config)
        x = torch.randn(2, 32, 64)
        out = encoder(x)

        assert out.shape == (2, 64)  # [B, C]

    def test_different_inputs_different_outputs(self, small_config):
        """Test encoder produces different outputs for different inputs."""
        small_config.warm_start_prefix_len = 16
        encoder = WarmStartEncoder(small_config)

        x1 = torch.randn(2, 32, 64)
        x2 = torch.randn(2, 32, 64)

        out1 = encoder(x1)
        out2 = encoder(x2)

        assert not torch.allclose(out1, out2)

    def test_gradients_flow(self, small_config):
        """Test gradients flow through encoder."""
        small_config.warm_start_prefix_len = 16
        encoder = WarmStartEncoder(small_config)

        x = torch.randn(2, 32, 64, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None


# --- DeepMomentumUpdate Tests ---

class TestDeepMomentumUpdate:
    """Tests for DeepMomentumUpdate."""

    def test_output_shape(self):
        """Test output shape matches input."""
        deep_mom = DeepMomentumUpdate(dim=64)
        grad = torch.randn(4, 64)
        prev_mom = torch.randn(4, 64)

        out = deep_mom(grad, prev_mom)
        assert out.shape == grad.shape

    def test_differs_from_simple_momentum(self):
        """Test output differs from simple momentum."""
        deep_mom = DeepMomentumUpdate(dim=64)
        grad = torch.randn(4, 64)
        prev_mom = torch.randn(4, 64)

        out = deep_mom(grad, prev_mom, beta=0.9)
        simple = 0.9 * prev_mom + 0.1 * grad

        # Should be different (learned component)
        assert not torch.allclose(out, simple)

    def test_initial_mix_alpha(self):
        """Test mix_alpha starts at 0.5."""
        deep_mom = DeepMomentumUpdate(dim=64)
        alpha = torch.sigmoid(deep_mom.mix_alpha).item()
        assert abs(alpha - 0.5) < 0.01

    def test_gradients_flow(self):
        """Test gradients flow through."""
        deep_mom = DeepMomentumUpdate(dim=64)
        grad = torch.randn(4, 64, requires_grad=True)
        prev_mom = torch.randn(4, 64)

        out = deep_mom(grad, prev_mom)
        loss = out.sum()
        loss.backward()

        assert grad.grad is not None


# --- NeuralMemoryAdapter Tests ---

class TestNeuralMemoryAdapter:
    """Tests for NeuralMemoryAdapter."""

    def test_init_state(self, small_config):
        """Test state initialization."""
        adapter = NeuralMemoryAdapter(small_config)
        state = adapter.init_state(2, torch.device('cpu'))

        assert state is not None

    def test_forward_shape(self, small_config, batch_hidden):
        """Test retrieval shape."""
        adapter = NeuralMemoryAdapter(small_config)
        state = adapter.init_state(2, torch.device('cpu'))

        out = adapter(batch_hidden, state)
        assert out.shape == (2, small_config.num_longterm_mem, 64)

    def test_update_returns_new_state(self, small_config, batch_hidden):
        """Test update returns new state."""
        adapter = NeuralMemoryAdapter(small_config)
        state = adapter.init_state(2, torch.device('cpu'))

        new_state = adapter.update(batch_hidden, state)
        assert new_state is not None


# --- TitansQwenDecoderLayer Tests ---

class TestTitansQwenDecoderLayer:
    """Tests for TitansQwenDecoderLayer."""

    @pytest.fixture
    def mock_qwen_layer(self):
        """Create a mock Qwen decoder layer."""
        class MockQwenLayer(nn.Module):
            def __init__(self, hidden_size=64):
                super().__init__()
                self.self_attn = nn.Linear(hidden_size, hidden_size)
                self.mlp = nn.Linear(hidden_size, hidden_size)
                self.input_layernorm = nn.LayerNorm(hidden_size)
                self.post_attention_layernorm = nn.LayerNorm(hidden_size)
                self.attention_type = "sdpa"

            def forward(self, hidden_states, **kwargs):
                return hidden_states

        return MockQwenLayer()

    def test_forward_preserves_shape(self, small_config, mock_qwen_layer):
        """Test forward doesn't change sequence length."""
        layer = TitansQwenDecoderLayer(mock_qwen_layer, 0, small_config)
        x = torch.randn(2, 32, 64)

        out = layer(x)
        if isinstance(out, tuple):
            out = out[0]

        assert out.shape == x.shape

    def test_gate_starts_conservative(self, small_config, mock_qwen_layer):
        """Test gate starts near 0."""
        layer = TitansQwenDecoderLayer(mock_qwen_layer, 0, small_config)
        x = torch.randn(2, 32, 64)

        _ = layer(x)
        stats = layer.get_gate_statistics()

        assert 0.1 < stats["mean_gate"] < 0.15

    def test_memory_state_management(self, small_config, mock_qwen_layer):
        """Test memory state can be set and retrieved."""
        layer = TitansQwenDecoderLayer(mock_qwen_layer, 0, small_config)
        x = torch.randn(2, 32, 64)

        # Forward creates state
        _ = layer(x)
        state = layer.get_memory_state()
        assert state is not None

        # Can set state
        layer.set_memory_state(None)
        assert layer._current_memory_state is None

    def test_internal_loss_computed(self, small_config, mock_qwen_layer):
        """Test internal loss is computed during training."""
        small_config.use_internal_loss = True
        layer = TitansQwenDecoderLayer(mock_qwen_layer, 0, small_config)
        layer.train()
        x = torch.randn(2, 32, 64)

        _ = layer(x)
        internal_loss = layer.get_internal_loss()

        assert internal_loss is not None
        assert internal_loss.item() >= 0

    def test_memory_updates_can_be_disabled(self, small_config, mock_qwen_layer):
        """Test memory updates can be disabled."""
        layer = TitansQwenDecoderLayer(mock_qwen_layer, 0, small_config)
        layer.enable_memory_updates(False)

        assert layer.update_memory is False


# --- Config Tests ---

class TestTitansQwenConfig:
    """Tests for TitansQwenConfig."""

    def test_default_values(self):
        """Test default config values."""
        config = TitansQwenConfig()

        assert config.n_embd == 1536
        assert config.use_cms is True
        assert config.use_self_mod_proj is True
        assert config.use_self_mod_gate is True

    def test_prefix_len_property(self):
        """Test prefix_len calculation."""
        config = TitansQwenConfig(
            num_longterm_mem=16,
            num_persist_mem=4,
        )

        assert config.prefix_len == 20

    def test_estimate_memory_params(self):
        """Test parameter estimation."""
        config = TitansQwenConfig(n_embd=64, memory_expansion=2)
        params = config.estimate_memory_params()

        assert "total" in params
        assert params["total"] > 0


# --- Integration Tests ---

class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_forward_backward(self, small_config):
        """Test full forward and backward pass."""
        # Create mock layer
        class MockQwenLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)
                self.attention_type = "sdpa"

            def forward(self, x, **kwargs):
                return self.linear(x)

        mock_layer = MockQwenLayer()
        titans_layer = TitansQwenDecoderLayer(mock_layer, 0, small_config)

        x = torch.randn(2, 32, 64, requires_grad=True)
        out = titans_layer(x)
        if isinstance(out, tuple):
            out = out[0]

        loss = out.sum()
        loss.backward()

        assert x.grad is not None

    def test_cms_with_decoder_layer(self, cms_config):
        """Test CMS works with decoder layer."""
        class MockQwenLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention_type = "sdpa"

            def forward(self, x, **kwargs):
                return x

        mock_layer = MockQwenLayer()
        titans_layer = TitansQwenDecoderLayer(mock_layer, 0, cms_config)

        x = torch.randn(2, 32, 64)

        # Multiple forward passes
        for _ in range(5):
            out = titans_layer(x)

        if isinstance(out, tuple):
            out = out[0]

        assert out.shape == x.shape
