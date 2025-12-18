"""
Titans: Learning to Memorize at Test Time
Implementation of Memory as Context (MAC) architecture on top of nanoGPT.

Based on: https://arxiv.org/abs/2501.00663

Fixed version addressing:
- Proper prefix-LM attention masking
- Per-sequence memory state (not global)
- Memory updates during training
- Correct state passing between segments
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TitansConfig:
    """Configuration for TitansGPT model."""

    # Base GPT config (nanoGPT defaults)
    block_size: int = 2048
    vocab_size: int = 50304  # GPT-2 vocab size (50257 rounded for efficiency)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True

    # Titans-specific config
    segment_len: int = 64  # tokens per segment
    memory_depth: int = 2  # MLP layers in neural memory
    num_persist_mem: int = 4  # persistent memory tokens
    num_longterm_mem: int = 16  # long-term memory tokens retrieved
    memory_lr: float = 0.01  # learning rate for surprise-based updates
    memory_momentum: float = 0.9  # momentum for past surprise
    memory_decay: float = 0.001  # forgetting/decay factor


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch doesn't support bias=False directly)."""

    def __init__(self, ndim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


@dataclass
class MemoryState:
    """Per-sequence memory state."""

    memory_tokens: torch.Tensor  # [B, num_longterm_mem, C] - the actual memory content
    momentum_buffer: torch.Tensor  # [B, num_longterm_mem, C] - surprise momentum
    step: int = 0


class NeuralMemory(nn.Module):
    """
    Neural Memory Module that learns at test time.

    Fixed version:
    - initial_memory is a learned parameter (template)
    - Actual memory is stored in MemoryState (per-sequence)
    - Updates modify state, not global parameters
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config  # type: ignore[assignment]
        self.n_embd = config.n_embd  # type: ignore[assignment]
        self.num_longterm_mem = config.num_longterm_mem  # type: ignore[assignment]

        # Memory MLP: transforms memory tokens for retrieval
        hidden_dim = config.n_embd * 4
        layers: list[nn.Module] = []
        in_dim = config.n_embd
        for _ in range(config.memory_depth - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim, bias=config.bias),
                    nn.SiLU(),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, config.n_embd, bias=config.bias))
        self.memory_mlp = nn.Sequential(*layers)

        # Initial memory tokens (learned template, cloned per-sequence)
        self.initial_memory = nn.Parameter(
            torch.randn(1, config.num_longterm_mem, config.n_embd) * 0.02
        )

        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Hyperparameters for test-time learning
        self.lr = config.memory_lr  # type: ignore[assignment]
        self.momentum = config.memory_momentum  # type: ignore[assignment]
        self.decay = config.memory_decay  # type: ignore[assignment]

    def init_state(self, batch_size: int, _device: torch.device) -> MemoryState:
        """Initialize memory state for a new sequence."""
        # Clone initial memory for each batch item (detach to make it mutable state)
        memory_tokens = self.initial_memory.expand(batch_size, -1, -1).clone()
        momentum_buffer = torch.zeros_like(memory_tokens)
        return MemoryState(
            memory_tokens=memory_tokens,
            momentum_buffer=momentum_buffer,
            step=0,
        )

    def forward(
        self,
        _x: torch.Tensor,
        state: MemoryState,
    ) -> torch.Tensor:
        """
        Retrieve from memory.

        Args:
            x: Input tensor [B, T, C]
            state: Current memory state

        Returns:
            retrieved: Memory context [B, num_longterm_mem, C]
        """
        # Transform memory through MLP
        memory_out = self.memory_mlp(state.memory_tokens)
        return self.out_proj(memory_out)

    def update(
        self,
        x: torch.Tensor,
        state: MemoryState,
    ) -> MemoryState:
        """
        Update memory based on surprise.

        The update rule (from paper):
        - Surprise = how unexpected is the input (measured by reconstruction error gradient)
        - Memory is updated with: new = decay * old + lr * surprise * input_signal

        Args:
            x: Current segment output [B, T, C]
            state: Current memory state

        Returns:
            new_state: Updated memory state
        """
        _b, _t, _c = x.shape  # Validate shape

        # Compute "surprise" - gradient magnitude of reconstruction error
        # Use the segment mean as the signal to memorize
        input_signal = x.mean(dim=1, keepdim=True)  # [B, 1, C]

        # Compute reconstruction error (how well can memory predict the input?)
        with torch.enable_grad():
            x_for_grad = input_signal.detach().requires_grad_(True)
            memory_pred = self.memory_mlp(state.memory_tokens).mean(dim=1, keepdim=True)
            recon_loss = F.mse_loss(memory_pred, x_for_grad)
            grad = torch.autograd.grad(recon_loss, x_for_grad, create_graph=False)[0]
            surprise = grad.abs().mean(dim=-1, keepdim=True)

        # Update momentum (exponential moving average of surprise)
        new_momentum = self.momentum * state.momentum_buffer + (1 - self.momentum) * surprise

        # Compute memory update
        # Expand input signal to match memory shape
        update_signal = input_signal.expand(-1, self.num_longterm_mem, -1)

        # Apply update: new_memory = decay * old_memory + lr * surprise * signal
        decay_factor = 1 - self.decay
        new_memory = decay_factor * state.memory_tokens + self.lr * surprise * update_signal

        return MemoryState(
            memory_tokens=new_memory,
            momentum_buffer=new_momentum,
            step=state.step + 1,
        )


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with prefix-LM masking support.

    For Titans MAC architecture:
    - Prefix tokens (memory + persistent) can attend to each other fully
    - Current segment tokens can attend to prefix + causally to each other
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            msg = f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
            raise ValueError(msg)

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head  # type: ignore[assignment]
        self.n_embd = config.n_embd  # type: ignore[assignment]
        self.dropout = config.dropout  # type: ignore[assignment]

        self.flash = hasattr(F, "scaled_dot_product_attention")  # type: ignore[assignment]

    def forward(
        self,
        x: torch.Tensor,
        prefix_len: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass with prefix-LM attention mask.

        Args:
            x: Input tensor [B, T, C] where T = prefix_len + segment_len
            prefix_len: Number of prefix tokens (memory + persistent)
                       These tokens can attend to each other fully.

        Returns:
            Output tensor [B, T, C]
        """
        b, t, c = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)

        if self.flash and prefix_len == 0:
            # Standard causal attention (no prefix)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # Build prefix-LM mask
            # Shape: [T, T] where:
            # - prefix tokens (0:prefix_len) can attend to all prefix tokens
            # - segment tokens (prefix_len:) can attend to prefix + causal within segment
            mask = torch.zeros(t, t, device=x.device, dtype=torch.bool)

            if prefix_len > 0:
                # Segment tokens cannot attend to future segment tokens
                segment_len = t - prefix_len
                if segment_len > 0:
                    # Create causal mask for segment portion
                    segment_mask = torch.triu(
                        torch.ones(segment_len, segment_len, device=x.device, dtype=torch.bool),
                        diagonal=1,
                    )
                    mask[prefix_len:, prefix_len:] = segment_mask
            else:
                # Pure causal mask
                mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)

            # Apply attention with mask
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TitansBlock(nn.Module):
    """
    Memory as Context (MAC) Transformer Block.

    Fixed version:
    - Uses per-sequence memory state
    - Proper prefix-LM attention masking
    - Updates memory during both training and inference
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config  # type: ignore[assignment]
        self.num_longterm_mem = config.num_longterm_mem  # type: ignore[assignment]
        self.num_persist_mem = config.num_persist_mem  # type: ignore[assignment]

        # Neural memory module
        self.memory = NeuralMemory(config)

        # Persistent memory (learnable, input-independent task knowledge)
        self.persist_mem = nn.Parameter(
            torch.randn(1, config.num_persist_mem, config.n_embd) * 0.02
        )

        # Layer norms
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        # Attention and MLP
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        """Initialize memory state for this block."""
        return self.memory.init_state(batch_size, device)

    def forward(
        self,
        x: torch.Tensor,
        memory_state: MemoryState,
    ) -> tuple[torch.Tensor, MemoryState]:
        """
        Forward pass with memory integration.

        Args:
            x: Input tensor [B, T, C]
            memory_state: Current memory state

        Returns:
            output: Transformed tensor [B, T, C]
            new_state: Updated memory state
        """
        b, _t, _c = x.shape

        # 1. Retrieve from long-term memory
        mem_context = self.memory(x, memory_state)  # [B, num_longterm_mem, C]

        # 2. Expand persistent memory for batch
        persist = self.persist_mem.expand(b, -1, -1)  # [B, num_persist_mem, C]

        # 3. Concatenate: [memory_context, persistent_memory, current_segment]
        prefix_len = self.num_longterm_mem + self.num_persist_mem
        context = torch.cat([mem_context, persist, x], dim=1)  # [B, prefix_len + T, C]

        # 4. Self-attention with prefix-LM mask
        attn_out = self.attn(self.ln_1(context), prefix_len=prefix_len)

        # 5. Extract only the current segment positions (residual connection)
        x = x + attn_out[:, prefix_len:]

        # 6. MLP with residual
        x = x + self.mlp(self.ln_2(x))

        # 7. Update memory based on this segment
        new_state = self.memory.update(x, memory_state)

        return x, new_state


class TitansGPT(nn.Module):
    """
    Full Titans GPT model with segmented processing and neural memory.

    Fixed version:
    - Proper memory state management
    - Memory updates during training
    - Correct state passing between segments
    """

    def __init__(self, config: TitansConfig) -> None:
        super().__init__()
        self.config = config  # type: ignore[assignment]

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([TitansBlock(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer["wte"].weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Special scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"TitansGPT initialized with {self.get_num_params() / 1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, *, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer["wpe"].weight.numel()  # type: ignore[call-non-callable]
        return n_params

    def init_memory_states(self, batch_size: int, device: torch.device) -> list[MemoryState]:
        """Initialize memory states for all layers."""
        return [block.init_state(batch_size, device) for block in self.transformer["h"]]  # type: ignore[not-iterable]

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        memory_states: list[MemoryState] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[MemoryState]]:
        """
        Forward pass with segmented processing.

        Args:
            idx: Input token indices [B, T]
            targets: Target token indices [B, T] (optional, for training)
            memory_states: List of memory states per layer (initialized if None)

        Returns:
            logits: Output logits [B, T, vocab_size]
            loss: Cross-entropy loss (if targets provided)
            memory_states: Updated memory states
        """
        device = idx.device
        b, t = idx.shape
        segment_len = self.config.segment_len

        if t > self.config.block_size:
            msg = f"Sequence length {t} exceeds block size {self.config.block_size}"
            raise ValueError(msg)

        # Initialize memory states if needed
        if memory_states is None:
            memory_states = self.init_memory_states(b, device)

        # Process in segments
        num_segments = (t + segment_len - 1) // segment_len
        all_logits: list[torch.Tensor] = []

        for seg_idx in range(num_segments):
            start = seg_idx * segment_len
            end = min(start + segment_len, t)
            seg_tokens = idx[:, start:end]

            # Token and position embeddings
            tok_emb = self.transformer["wte"](seg_tokens)
            pos = torch.arange(start, end, dtype=torch.long, device=device)
            pos_emb = self.transformer["wpe"](pos)
            x = self.transformer["drop"](tok_emb + pos_emb)

            # Process through transformer blocks, updating memory states
            new_states: list[MemoryState] = []
            for i, block in enumerate(self.transformer["h"]):  # type: ignore[arg-type]
                x, new_state = block(x, memory_states[i])
                new_states.append(new_state)
            memory_states = new_states

            # Final layer norm
            x = self.transformer["ln_f"](x)

            # Project to vocabulary
            logits = self.lm_head(x)
            all_logits.append(logits)

        # Concatenate all segment logits
        final_logits = torch.cat(all_logits, dim=1)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                final_logits.view(-1, final_logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return final_logits, loss, memory_states

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        """Configure optimizer with weight decay."""
        decay: set[str] = set()
        no_decay: set[str] = set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, _p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                # Skip lm_head.weight as it's tied to wte.weight
                if fpn == "lm_head.weight":
                    continue

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif (
                    (pn.endswith("weight") and isinstance(m, blacklist_weight_modules))
                    or "initial_memory" in fpn
                    or "persist_mem" in fpn
                ):
                    no_decay.add(fpn)

        param_dict = dict(self.named_parameters())
        inter_params = decay & no_decay
        if inter_params:
            msg = f"Parameters in both decay/no_decay: {inter_params}"
            raise ValueError(msg)

        missing = param_dict.keys() - (decay | no_decay)
        no_decay.update(missing)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

        use_fused = device_type == "cuda"
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        memory_states: list[MemoryState] | None = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively with memory."""
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )

            logits, _, memory_states = self(idx_cond, memory_states=memory_states)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# Quick test
if __name__ == "__main__":
    test_config = TitansConfig(
        block_size=256,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        segment_len=32,
    )
    model = TitansGPT(test_config)

    # Test forward pass
    test_x = torch.randint(0, 1000, (2, 64))
    test_logits, test_loss, test_states = model(test_x, targets=test_x)
    print(f"Logits shape: {test_logits.shape}")
    if test_loss is not None:
        print(f"Loss: {test_loss.item():.4f}")

    # Test that memory states are updated
    print(f"Memory state step: {test_states[0].step}")

    # Test generation
    prompt = torch.randint(0, 1000, (1, 8))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
