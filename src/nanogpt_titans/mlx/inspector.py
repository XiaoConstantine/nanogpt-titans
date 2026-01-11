"""
Data Structure Inspector for MLX Training.

This module lets you SEE what's happening during training:
- Tensor shapes, values, statistics
- Gradient flow and magnitudes
- Memory states and updates
- Attention patterns
- Activation distributions

The goal is educational: understand the math behind training.
"""

import mlx.core as mx
import numpy as np
from dataclasses import dataclass, field
from typing import Any
import json
from pathlib import Path


@dataclass
class TensorStats:
    """Statistics for a single tensor - see the numbers!"""
    name: str
    shape: tuple
    dtype: str
    min_val: float
    max_val: float
    mean: float
    std: float
    num_zeros: int
    num_nans: int
    num_infs: int
    l2_norm: float

    def __str__(self) -> str:
        return (
            f"{self.name}\n"
            f"  Shape: {self.shape} | dtype: {self.dtype}\n"
            f"  Range: [{self.min_val:.4f}, {self.max_val:.4f}]\n"
            f"  Mean: {self.mean:.4f} | Std: {self.std:.4f}\n"
            f"  L2 Norm: {self.l2_norm:.4f}\n"
            f"  Zeros: {self.num_zeros} | NaNs: {self.num_nans} | Infs: {self.num_infs}"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "min": self.min_val,
            "max": self.max_val,
            "mean": self.mean,
            "std": self.std,
            "l2_norm": self.l2_norm,
            "zeros": self.num_zeros,
            "nans": self.num_nans,
            "infs": self.num_infs,
        }


@dataclass
class GradientFlow:
    """Track gradient magnitudes through layers."""
    layer_name: str
    param_name: str
    grad_norm: float
    param_norm: float
    ratio: float  # grad_norm / param_norm - how much is this param changing?

    def __str__(self) -> str:
        return (
            f"{self.layer_name}/{self.param_name}: "
            f"grad={self.grad_norm:.2e} param={self.param_norm:.2e} "
            f"ratio={self.ratio:.2e}"
        )


@dataclass
class MemoryState:
    """Snapshot of neural memory state."""
    level: int  # CMS level (0, 1, 2)
    weights_norm: float
    momentum_norm: float
    update_magnitude: float
    surprise_value: float  # How "surprising" was the input?


@dataclass
class TrainingSnapshot:
    """Complete snapshot of training state at one step."""
    step: int
    loss: float
    learning_rate: float
    tensor_stats: dict[str, TensorStats] = field(default_factory=dict)
    gradient_flow: list[GradientFlow] = field(default_factory=list)
    memory_states: list[MemoryState] = field(default_factory=list)
    activations: dict[str, TensorStats] = field(default_factory=dict)
    attention_entropy: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"STEP {self.step} | Loss: {self.loss:.4f} | LR: {self.learning_rate:.2e}",
            f"{'='*60}",
        ]

        if self.tensor_stats:
            lines.append("\n--- Tensor Stats ---")
            for stats in self.tensor_stats.values():
                lines.append(str(stats))

        if self.gradient_flow:
            lines.append("\n--- Gradient Flow ---")
            # Sort by ratio to see which params are updating most
            sorted_grads = sorted(self.gradient_flow, key=lambda x: x.ratio, reverse=True)
            for gf in sorted_grads[:10]:  # Top 10
                lines.append(str(gf))

        if self.memory_states:
            lines.append("\n--- Memory States ---")
            for ms in self.memory_states:
                lines.append(
                    f"  Level {ms.level}: weights={ms.weights_norm:.4f} "
                    f"momentum={ms.momentum_norm:.4f} "
                    f"update={ms.update_magnitude:.2e} "
                    f"surprise={ms.surprise_value:.4f}"
                )

        return "\n".join(lines)


class Inspector:
    """
    The main inspector class - your window into training.

    Usage:
        inspector = Inspector(verbose=True)

        # During training:
        inspector.inspect_tensor("embeddings", embeddings)
        inspector.inspect_gradients(grads, params)
        inspector.inspect_memory(memory_state)

        # Get snapshot:
        snapshot = inspector.snapshot(step, loss, lr)
        print(snapshot.summary())
    """

    def __init__(
        self,
        verbose: bool = True,
        save_history: bool = True,
        history_dir: str = "training_inspection",
    ):
        self.verbose = verbose
        self.save_history = save_history
        self.history_dir = Path(history_dir)

        if save_history:
            self.history_dir.mkdir(parents=True, exist_ok=True)

        # Current step data
        self._tensor_stats: dict[str, TensorStats] = {}
        self._gradient_flow: list[GradientFlow] = []
        self._memory_states: list[MemoryState] = []
        self._activations: dict[str, TensorStats] = {}
        self._attention_entropy: list[float] = []

        # History for plotting
        self.history: list[TrainingSnapshot] = []

    def _compute_stats(self, name: str, tensor: mx.array) -> TensorStats:
        """Compute comprehensive statistics for a tensor."""
        # Force evaluation to get actual values
        mx.eval(tensor)

        # Convert to numpy for easier stats
        arr = np.array(tensor)

        return TensorStats(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            num_zeros=int(np.sum(arr == 0)),
            num_nans=int(np.sum(np.isnan(arr))),
            num_infs=int(np.sum(np.isinf(arr))),
            l2_norm=float(np.linalg.norm(arr)),
        )

    def inspect_tensor(self, name: str, tensor: mx.array) -> TensorStats:
        """
        Inspect a tensor - see its shape, range, distribution.

        This is the core "see the math" function!
        """
        stats = self._compute_stats(name, tensor)
        self._tensor_stats[name] = stats

        if self.verbose:
            print(f"\n[INSPECT] {stats}")

        return stats

    def inspect_gradients(
        self,
        grads: dict[str, Any],
        params: dict[str, Any],
        prefix: str = ""
    ) -> list[GradientFlow]:
        """
        Inspect gradient flow through the model.

        Shows which parameters are getting updated and by how much.
        """
        flows = []

        def traverse(g_dict, p_dict, path=""):
            if isinstance(g_dict, mx.array):
                mx.eval(g_dict, p_dict)
                g_arr = np.array(g_dict)
                p_arr = np.array(p_dict)

                grad_norm = float(np.linalg.norm(g_arr))
                param_norm = float(np.linalg.norm(p_arr))
                ratio = grad_norm / (param_norm + 1e-8)

                parts = path.rsplit("/", 1)
                layer = parts[0] if len(parts) > 1 else ""
                param = parts[-1]

                flow = GradientFlow(
                    layer_name=prefix + layer,
                    param_name=param,
                    grad_norm=grad_norm,
                    param_norm=param_norm,
                    ratio=ratio,
                )
                flows.append(flow)

            elif isinstance(g_dict, dict):
                for k, v in g_dict.items():
                    new_path = f"{path}/{k}" if path else k
                    if k in p_dict:
                        traverse(v, p_dict[k], new_path)

        traverse(grads, params)
        self._gradient_flow.extend(flows)

        if self.verbose and flows:
            print(f"\n[GRADIENTS] Found {len(flows)} parameter gradients")
            # Show top 5 by ratio
            sorted_flows = sorted(flows, key=lambda x: x.ratio, reverse=True)
            for f in sorted_flows[:5]:
                print(f"  {f}")

        return flows

    def inspect_memory(
        self,
        weights: mx.array,
        momentum: mx.array | None = None,
        level: int = 0,
        surprise: float = 0.0,
        prev_weights: mx.array | None = None,
    ) -> MemoryState:
        """
        Inspect neural memory state.

        Shows how memory is evolving during training.
        """
        mx.eval(weights)
        weights_norm = float(np.linalg.norm(np.array(weights)))

        momentum_norm = 0.0
        if momentum is not None:
            mx.eval(momentum)
            momentum_norm = float(np.linalg.norm(np.array(momentum)))

        update_magnitude = 0.0
        if prev_weights is not None:
            mx.eval(prev_weights)
            diff = np.array(weights) - np.array(prev_weights)
            update_magnitude = float(np.linalg.norm(diff))

        state = MemoryState(
            level=level,
            weights_norm=weights_norm,
            momentum_norm=momentum_norm,
            update_magnitude=update_magnitude,
            surprise_value=surprise,
        )
        self._memory_states.append(state)

        if self.verbose:
            print(f"\n[MEMORY L{level}] weights={weights_norm:.4f} "
                  f"momentum={momentum_norm:.4f} "
                  f"update={update_magnitude:.2e}")

        return state

    def inspect_activations(
        self,
        name: str,
        activations: mx.array,
    ) -> TensorStats:
        """
        Inspect layer activations.

        Useful for detecting dead neurons, saturation, etc.
        """
        stats = self._compute_stats(f"act/{name}", activations)
        self._activations[name] = stats

        if self.verbose:
            # Check for potential issues
            issues = []
            if stats.num_zeros > stats.shape[0] * 0.5:
                issues.append("Many zeros (dead neurons?)")
            if stats.std < 0.01:
                issues.append("Low variance (saturation?)")
            if stats.num_nans > 0:
                issues.append(f"{stats.num_nans} NaNs!")

            issue_str = f" ⚠️ {', '.join(issues)}" if issues else ""
            print(f"[ACTIVATION] {name}: mean={stats.mean:.4f} "
                  f"std={stats.std:.4f}{issue_str}")

        return stats

    def inspect_attention(
        self,
        attention_weights: mx.array,
        layer_idx: int = 0,
    ) -> float:
        """
        Inspect attention patterns.

        Returns entropy - low entropy = focused attention, high = diffuse.
        """
        mx.eval(attention_weights)
        attn = np.array(attention_weights)

        # Compute entropy per head, average
        # attn shape: [batch, heads, seq, seq] or [heads, seq, seq]
        if len(attn.shape) == 4:
            attn = attn[0]  # Take first batch

        # Entropy: -sum(p * log(p))
        eps = 1e-8
        attn_clipped = np.clip(attn, eps, 1.0)
        entropy = -np.sum(attn_clipped * np.log(attn_clipped), axis=-1)
        avg_entropy = float(np.mean(entropy))

        self._attention_entropy.append(avg_entropy)

        if self.verbose:
            print(f"[ATTENTION L{layer_idx}] entropy={avg_entropy:.4f} "
                  f"(low=focused, high=diffuse)")

        return avg_entropy

    def snapshot(
        self,
        step: int,
        loss: float,
        learning_rate: float,
    ) -> TrainingSnapshot:
        """
        Create a snapshot of current training state.

        Call this at the end of each step to record everything.
        """
        snapshot = TrainingSnapshot(
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            tensor_stats=self._tensor_stats.copy(),
            gradient_flow=self._gradient_flow.copy(),
            memory_states=self._memory_states.copy(),
            activations=self._activations.copy(),
            attention_entropy=self._attention_entropy.copy(),
        )

        self.history.append(snapshot)

        # Save to disk
        if self.save_history:
            self._save_snapshot(snapshot)

        # Clear current step data
        self._tensor_stats.clear()
        self._gradient_flow.clear()
        self._memory_states.clear()
        self._activations.clear()
        self._attention_entropy.clear()

        return snapshot

    def _save_snapshot(self, snapshot: TrainingSnapshot) -> None:
        """Save snapshot to JSON for later analysis."""
        data = {
            "step": snapshot.step,
            "loss": snapshot.loss,
            "lr": snapshot.learning_rate,
            "tensors": {k: v.to_dict() for k, v in snapshot.tensor_stats.items()},
            "gradients": [
                {
                    "layer": g.layer_name,
                    "param": g.param_name,
                    "grad_norm": g.grad_norm,
                    "param_norm": g.param_norm,
                    "ratio": g.ratio,
                }
                for g in snapshot.gradient_flow
            ],
            "memory": [
                {
                    "level": m.level,
                    "weights_norm": m.weights_norm,
                    "momentum_norm": m.momentum_norm,
                    "update": m.update_magnitude,
                    "surprise": m.surprise_value,
                }
                for m in snapshot.memory_states
            ],
            "attention_entropy": snapshot.attention_entropy,
        }

        path = self.history_dir / f"step_{snapshot.step:06d}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self, last_n: int = 10) -> None:
        """Print summary of recent training history."""
        if not self.history:
            print("No history yet!")
            return

        recent = self.history[-last_n:]

        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY (last {len(recent)} steps)")
        print(f"{'='*60}")

        losses = [s.loss for s in recent]
        print(f"\nLoss: {losses[-1]:.4f} (min={min(losses):.4f}, max={max(losses):.4f})")

        if recent[-1].gradient_flow:
            avg_ratio = np.mean([g.ratio for g in recent[-1].gradient_flow])
            print(f"Avg grad/param ratio: {avg_ratio:.2e}")

        if recent[-1].memory_states:
            for ms in recent[-1].memory_states:
                print(f"Memory L{ms.level}: weights={ms.weights_norm:.4f}")


def demo_inspector():
    """Quick demo of the inspector."""
    print("=== Inspector Demo ===\n")

    inspector = Inspector(verbose=True, save_history=False)

    # Simulate a training step
    fake_embeddings = mx.random.normal((32, 512, 768))
    inspector.inspect_tensor("embeddings", fake_embeddings)

    fake_hidden = mx.random.normal((32, 512, 768)) * 0.1
    inspector.inspect_activations("layer_0", fake_hidden)

    fake_attn = mx.softmax(mx.random.normal((8, 512, 512)), axis=-1)
    inspector.inspect_attention(fake_attn, layer_idx=0)

    fake_weights = mx.random.normal((768, 768))
    inspector.inspect_memory(fake_weights, level=0, surprise=0.42)

    snapshot = inspector.snapshot(step=1, loss=2.34, learning_rate=1e-4)
    print(snapshot.summary())


if __name__ == "__main__":
    demo_inspector()
