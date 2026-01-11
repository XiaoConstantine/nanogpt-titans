"""
Real-time Training Visualizer for MLX.

Provides live plots and visualizations during training:
- Loss curves
- Gradient flow
- Memory evolution
- Attention patterns
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
import json
import time

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will be saved as JSON only.")

try:
    import mlx.core as mx
except ImportError:
    mx = None


@dataclass
class TrainingMetrics:
    """Container for all training metrics."""
    steps: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    mem_scales: list = field(default_factory=list)
    gate_values: dict = field(default_factory=dict)  # layer_idx -> list of values
    cms_weights: dict = field(default_factory=dict)  # level -> list of values
    grad_norms: dict = field(default_factory=dict)   # param_name -> list of norms
    memory_norms: dict = field(default_factory=dict) # level -> list of weight norms
    learning_rates: list = field(default_factory=list)
    step_times: list = field(default_factory=list)
    internal_losses: list = field(default_factory=list)
    generated_samples: list = field(default_factory=list)


class Visualizer:
    """
    Real-time training visualizer.

    Usage:
        viz = Visualizer(output_dir="./training_viz")

        # During training loop:
        viz.log_step(step, loss, mem_scale, gate_values, ...)
        viz.log_gradients(grads, params)
        viz.log_generation(step, prompt, generated_text)

        # Periodically:
        viz.update_plots()

        # At end:
        viz.save_final_report()
    """

    def __init__(
        self,
        output_dir: str = "./training_viz",
        plot_every: int = 10,
        max_history: int = 10000,
        figsize: tuple = (16, 12),
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.plot_every = plot_every
        self.max_history = max_history
        self.figsize = figsize

        self.metrics = TrainingMetrics()
        self._last_plot_step = -1
        self._start_time = time.time()

        # For tracking gradient flow
        self._grad_history = deque(maxlen=100)

    def log_step(
        self,
        step: int,
        loss: float,
        mem_scale: float,
        gate_values: dict[int, float],
        cms_weights: list[float] | None = None,
        lr: float = 0.0,
        step_time: float = 0.0,
        internal_loss: float = 0.0,
    ) -> None:
        """Log metrics for a training step."""
        self.metrics.steps.append(step)
        self.metrics.losses.append(loss)
        self.metrics.mem_scales.append(mem_scale)
        self.metrics.learning_rates.append(lr)
        self.metrics.step_times.append(step_time)
        self.metrics.internal_losses.append(internal_loss)

        # Gate values per layer
        for layer_idx, gate_val in gate_values.items():
            if layer_idx not in self.metrics.gate_values:
                self.metrics.gate_values[layer_idx] = []
            self.metrics.gate_values[layer_idx].append(gate_val)

        # CMS weights
        if cms_weights:
            for i, w in enumerate(cms_weights):
                if i not in self.metrics.cms_weights:
                    self.metrics.cms_weights[i] = []
                self.metrics.cms_weights[i].append(w)

        # Trim history if needed
        if len(self.metrics.steps) > self.max_history:
            self._trim_history()

    def log_gradients(
        self,
        grads: dict,
        params: dict,
        prefix: str = "",
    ) -> dict[str, float]:
        """
        Log gradient statistics.

        Returns dict of param_name -> grad_norm for inspection.
        """
        grad_stats = {}

        def traverse(g_dict, p_dict, path=""):
            if mx is not None and isinstance(g_dict, mx.array):
                mx.eval(g_dict)
                g_arr = np.array(g_dict)
                p_arr = np.array(p_dict) if isinstance(p_dict, mx.array) else p_dict

                grad_norm = float(np.linalg.norm(g_arr))
                param_norm = float(np.linalg.norm(p_arr)) if isinstance(p_arr, np.ndarray) else 1.0

                name = f"{prefix}{path}" if prefix else path
                grad_stats[name] = {
                    "grad_norm": grad_norm,
                    "param_norm": param_norm,
                    "ratio": grad_norm / (param_norm + 1e-8),
                }

                # Track in history
                if name not in self.metrics.grad_norms:
                    self.metrics.grad_norms[name] = []
                self.metrics.grad_norms[name].append(grad_norm)

            elif isinstance(g_dict, dict):
                for k, v in g_dict.items():
                    new_path = f"{path}.{k}" if path else k
                    if isinstance(p_dict, dict) and k in p_dict:
                        traverse(v, p_dict[k], new_path)

        traverse(grads, params)
        self._grad_history.append(grad_stats)

        return grad_stats

    def log_memory_state(
        self,
        weights: "mx.array",
        level: int = 0,
    ) -> float:
        """Log memory weight norms."""
        if mx is not None:
            mx.eval(weights)
            norm = float(np.linalg.norm(np.array(weights)))
        else:
            norm = 0.0

        if level not in self.metrics.memory_norms:
            self.metrics.memory_norms[level] = []
        self.metrics.memory_norms[level].append(norm)

        return norm

    def log_generation(
        self,
        step: int,
        prompt: str,
        generated: str,
        loss_at_step: float = 0.0,
    ) -> None:
        """Log a generated sample."""
        self.metrics.generated_samples.append({
            "step": step,
            "prompt": prompt,
            "generated": generated,
            "loss": loss_at_step,
            "timestamp": time.time() - self._start_time,
        })

    def update_plots(self, force: bool = False) -> None:
        """Update visualization plots."""
        if not HAS_MATPLOTLIB:
            return

        if not self.metrics.steps:
            return

        current_step = self.metrics.steps[-1]
        if not force and current_step - self._last_plot_step < self.plot_every:
            return

        self._last_plot_step = current_step

        # Create figure with subplots
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Loss curve (top left, spans 2 cols)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_loss(ax1)

        # 2. Memory scale (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_mem_scale(ax2)

        # 3. Gate values (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_gates(ax3)

        # 4. CMS weights (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_cms_weights(ax4)

        # 5. Gradient flow (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_gradient_flow(ax5)

        # 6. Learning rate (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_learning_rate(ax6)

        # 7. Step time (bottom center)
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_step_time(ax7)

        # 8. Memory norms (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_memory_norms(ax8)

        # Title
        fig.suptitle(
            f"Training Progress - Step {current_step} | "
            f"Loss: {self.metrics.losses[-1]:.4f}",
            fontsize=14,
            fontweight='bold'
        )

        # Save
        plt.savefig(self.output_dir / "training_progress.png", dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Also save metrics as JSON
        self._save_metrics_json()

    def _plot_loss(self, ax) -> None:
        """Plot loss curve."""
        steps = self.metrics.steps
        losses = self.metrics.losses

        ax.plot(steps, losses, 'b-', linewidth=1.5, label='Loss')

        # Add smoothed line
        if len(losses) > 20:
            window = min(50, len(losses) // 4)
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], smoothed, 'r-', linewidth=2, alpha=0.7, label=f'Smoothed ({window})')

        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Show min/max/current
        if losses:
            ax.axhline(y=min(losses), color='g', linestyle='--', alpha=0.5, label=f'Min: {min(losses):.4f}')

    def _plot_mem_scale(self, ax) -> None:
        """Plot memory scale evolution."""
        steps = self.metrics.steps
        scales = self.metrics.mem_scales

        ax.plot(steps, scales, 'g-', linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Scale')
        ax.set_title('Memory Scale')
        ax.grid(True, alpha=0.3)

    def _plot_gates(self, ax) -> None:
        """Plot gate values per layer."""
        steps = self.metrics.steps

        colors = plt.cm.viridis(np.linspace(0, 1, max(len(self.metrics.gate_values), 1)))

        for i, (layer_idx, values) in enumerate(sorted(self.metrics.gate_values.items())):
            # Align with steps
            plot_steps = steps[:len(values)]
            ax.plot(plot_steps, values, color=colors[i], linewidth=1.5, label=f'L{layer_idx}')

        ax.set_xlabel('Step')
        ax.set_ylabel('Gate Value')
        ax.set_title('Gate Values')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def _plot_cms_weights(self, ax) -> None:
        """Plot CMS level weights."""
        if not self.metrics.cms_weights:
            ax.text(0.5, 0.5, 'CMS not enabled', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('CMS Weights')
            return

        steps = self.metrics.steps
        colors = ['#e41a1c', '#377eb8', '#4daf4a']  # Red, Blue, Green
        labels = ['Fast (1x)', 'Medium (4x)', 'Slow (16x)']

        for i, (level, values) in enumerate(sorted(self.metrics.cms_weights.items())):
            plot_steps = steps[:len(values)]
            color = colors[i] if i < len(colors) else f'C{i}'
            label = labels[i] if i < len(labels) else f'Level {level}'
            ax.plot(plot_steps, values, color=color, linewidth=1.5, label=label)

        ax.set_xlabel('Step')
        ax.set_ylabel('Weight')
        ax.set_title('CMS Level Weights')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def _plot_gradient_flow(self, ax) -> None:
        """Plot gradient norms for key parameters."""
        if not self.metrics.grad_norms:
            ax.text(0.5, 0.5, 'No gradients logged', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Gradient Flow')
            return

        # Select important params
        important_keys = []
        for key in self.metrics.grad_norms:
            if any(k in key.lower() for k in ['mem_scale', 'gate', 'mlp', 'weight']):
                important_keys.append(key)

        # Limit to top 5
        important_keys = important_keys[:5]

        steps = self.metrics.steps
        for key in important_keys:
            values = self.metrics.grad_norms[key]
            plot_steps = steps[:len(values)]
            # Shorten label
            short_key = key.split('.')[-2:]
            ax.semilogy(plot_steps, values, linewidth=1, label='.'.join(short_key), alpha=0.7)

        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient Norm (log)')
        ax.set_title('Gradient Flow')
        if important_keys:
            ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)

    def _plot_learning_rate(self, ax) -> None:
        """Plot learning rate schedule."""
        steps = self.metrics.steps
        lrs = self.metrics.learning_rates

        ax.semilogy(steps, lrs, 'purple', linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('LR (log)')
        ax.set_title('Learning Rate')
        ax.grid(True, alpha=0.3)

    def _plot_step_time(self, ax) -> None:
        """Plot step time."""
        steps = self.metrics.steps
        times = self.metrics.step_times

        if not times:
            return

        # Convert to ms
        times_ms = [t * 1000 for t in times]

        ax.plot(steps, times_ms, 'orange', linewidth=1, alpha=0.5)

        # Smoothed
        if len(times_ms) > 10:
            window = 10
            smoothed = np.convolve(times_ms, np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], smoothed, 'darkorange', linewidth=2)

        ax.set_xlabel('Step')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Step Time (avg: {np.mean(times_ms):.1f}ms)')
        ax.grid(True, alpha=0.3)

    def _plot_memory_norms(self, ax) -> None:
        """Plot memory weight norms."""
        if not self.metrics.memory_norms:
            ax.text(0.5, 0.5, 'No memory data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Memory Norms')
            return

        steps = self.metrics.steps

        for level, norms in sorted(self.metrics.memory_norms.items()):
            plot_steps = steps[:len(norms)]
            ax.plot(plot_steps, norms, linewidth=1.5, label=f'Level {level}')

        ax.set_xlabel('Step')
        ax.set_ylabel('Norm')
        ax.set_title('Memory Weight Norms')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def plot_attention_heatmap(
        self,
        attention_weights,
        tokens: list[str] | None = None,
        layer_idx: int = 0,
        max_heads: int = 4,
        max_seq: int = 32,
        save_path: str | None = None,
    ):
        """
        Plot attention heatmap showing which tokens attend to which.

        The math: attention = softmax(Q @ K.T / sqrt(d_k))

        Args:
            attention_weights: [batch, n_heads, seq, seq] attention matrix
            tokens: List of token strings for labels
            layer_idx: Which layer this is from
            max_heads: Maximum number of heads to show
            max_seq: Maximum sequence length to show (for readability)
            save_path: Where to save the plot
        """
        if not HAS_MATPLOTLIB:
            return

        if mx is not None:
            mx.eval(attention_weights)
            attn = np.array(attention_weights)
        else:
            attn = np.array(attention_weights)

        # Handle shape
        if len(attn.shape) == 4:
            attn = attn[0]  # First batch

        n_heads = min(attn.shape[0], max_heads)
        seq_len = min(attn.shape[1], max_seq)
        attn = attn[:n_heads, :seq_len, :seq_len]

        # Create figure
        fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
        if n_heads == 1:
            axes = [axes]

        for head_idx, ax in enumerate(axes):
            im = ax.imshow(attn[head_idx], cmap='Blues', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'Head {head_idx}', fontsize=10)
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')

            # Add token labels if provided
            if tokens and len(tokens) >= seq_len:
                short_tokens = [t[:6] for t in tokens[:seq_len]]
                ax.set_xticks(range(0, seq_len, max(1, seq_len // 8)))
                ax.set_xticklabels([short_tokens[i] for i in range(0, seq_len, max(1, seq_len // 8))],
                                   rotation=45, ha='right', fontsize=7)
                ax.set_yticks(range(0, seq_len, max(1, seq_len // 8)))
                ax.set_yticklabels([short_tokens[i] for i in range(0, seq_len, max(1, seq_len // 8))],
                                   fontsize=7)

        fig.colorbar(im, ax=axes, shrink=0.6, label='Attention Weight')
        fig.suptitle(f'Layer {layer_idx} Attention: softmax(QK^T / √d)', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / f'attention_layer{layer_idx}.png', dpi=100, bbox_inches='tight')
        plt.close(fig)

        return fig

    def plot_gradient_flow(
        self,
        grad_norms: list[tuple[str, float]],
        step: int = 0,
        save_path: str | None = None,
    ):
        """
        Plot gradient flow as bar chart - shows learning signal per layer.

        The math: gradients = d(Loss)/d(weights) via chain rule

        Args:
            grad_norms: List of (layer_name, gradient_norm) tuples
            step: Current training step
            save_path: Where to save the plot
        """
        if not HAS_MATPLOTLIB:
            return

        if not grad_norms:
            return

        # Sort by layer order (try to extract layer number)
        def get_layer_order(name):
            import re
            match = re.search(r'blocks\.(\d+)', name)
            if match:
                return int(match.group(1))
            if 'wte' in name or 'wpe' in name:
                return -1
            if 'lm_head' in name:
                return 999
            return 500

        sorted_grads = sorted(grad_norms, key=lambda x: get_layer_order(x[0]))

        names = [g[0] for g in sorted_grads]
        norms = [g[1] for g in sorted_grads]

        # Shorten names for display
        short_names = []
        for name in names:
            if 'blocks.' in name:
                # e.g., "blocks.0.mlp.c_proj.weight" -> "L0.mlp.proj"
                parts = name.replace('blocks.', 'L').replace('.weight', '').replace('c_', '')
                short_names.append(parts[:15])
            else:
                short_names.append(name[:15])

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 5))

        # Color bars by magnitude
        colors = ['#d62728' if n > 1.0 else '#2ca02c' if n > 0.01 else '#7f7f7f' for n in norms]

        bars = ax.bar(range(len(norms)), norms, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_yscale('log')
        ax.set_xticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Gradient Norm (log scale)')
        ax.set_xlabel('Layer / Parameter')
        ax.set_title(f'Step {step}: Gradient Flow (∂Loss/∂W via backprop)')

        # Add threshold lines
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Large gradient')
        ax.axhline(y=0.01, color='orange', linestyle='--', alpha=0.5, label='Small gradient')
        ax.axhline(y=0.0001, color='gray', linestyle='--', alpha=0.5, label='Vanishing')
        ax.legend(loc='upper right', fontsize=8)

        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / f'gradient_flow_step{step}.png', dpi=100, bbox_inches='tight')
        plt.close(fig)

        return fig

    def plot_embedding_pca(
        self,
        embeddings,
        tokens: list[str] | None = None,
        n_components: int = 2,
        save_path: str | None = None,
    ):
        """
        Plot PCA of token embeddings - shows semantic relationships.

        The math: PCA finds principal components (eigenvectors of covariance matrix)

        Args:
            embeddings: [n_tokens, embedding_dim] matrix
            tokens: List of token strings for labels
            save_path: Where to save
        """
        if not HAS_MATPLOTLIB:
            return

        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("sklearn required for PCA visualization")
            return

        if mx is not None:
            mx.eval(embeddings)
            emb = np.array(embeddings)
        else:
            emb = np.array(embeddings)

        # Limit number of points for clarity
        max_points = 100
        if len(emb) > max_points:
            indices = np.random.choice(len(emb), max_points, replace=False)
            emb = emb[indices]
            if tokens:
                tokens = [tokens[i] for i in indices]

        # PCA
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(emb)

        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=range(len(reduced)),
                            cmap='viridis', alpha=0.7, s=50)

        # Add token labels
        if tokens:
            for i, (x, y) in enumerate(reduced):
                ax.annotate(tokens[i][:8], (x, y), fontsize=7, alpha=0.7)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('Token Embeddings: PCA Projection\n(Similar tokens should cluster together)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'embeddings_pca.png', dpi=100, bbox_inches='tight')
        plt.close(fig)

        return fig

    def _trim_history(self) -> None:
        """Trim history to max_history."""
        excess = len(self.metrics.steps) - self.max_history
        if excess <= 0:
            return

        self.metrics.steps = self.metrics.steps[excess:]
        self.metrics.losses = self.metrics.losses[excess:]
        self.metrics.mem_scales = self.metrics.mem_scales[excess:]
        self.metrics.learning_rates = self.metrics.learning_rates[excess:]
        self.metrics.step_times = self.metrics.step_times[excess:]
        self.metrics.internal_losses = self.metrics.internal_losses[excess:]

        for key in self.metrics.gate_values:
            self.metrics.gate_values[key] = self.metrics.gate_values[key][excess:]
        for key in self.metrics.cms_weights:
            self.metrics.cms_weights[key] = self.metrics.cms_weights[key][excess:]
        for key in self.metrics.grad_norms:
            if len(self.metrics.grad_norms[key]) > excess:
                self.metrics.grad_norms[key] = self.metrics.grad_norms[key][excess:]

    def _save_metrics_json(self) -> None:
        """Save metrics to JSON."""
        data = {
            "steps": self.metrics.steps,
            "losses": self.metrics.losses,
            "mem_scales": self.metrics.mem_scales,
            "gate_values": {str(k): v for k, v in self.metrics.gate_values.items()},
            "cms_weights": {str(k): v for k, v in self.metrics.cms_weights.items()},
            "learning_rates": self.metrics.learning_rates,
            "generated_samples": self.metrics.generated_samples,
        }

        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(data, f)

    def print_gradient_summary(self, top_n: int = 10) -> None:
        """Print summary of gradient statistics."""
        if not self._grad_history:
            print("No gradient history available")
            return

        latest = self._grad_history[-1]

        # Sort by ratio (grad/param)
        sorted_grads = sorted(
            latest.items(),
            key=lambda x: x[1]["ratio"],
            reverse=True
        )

        print(f"\n{'─'*60}")
        print(f"Gradient Summary (top {top_n} by update ratio)")
        print(f"{'─'*60}")
        print(f"{'Parameter':<40} {'Grad Norm':>12} {'Ratio':>10}")
        print(f"{'─'*60}")

        for name, stats in sorted_grads[:top_n]:
            short_name = name[-38:] if len(name) > 38 else name
            print(f"{short_name:<40} {stats['grad_norm']:>12.2e} {stats['ratio']:>10.2e}")

    def print_sample_generations(self, last_n: int = 3) -> None:
        """Print recent generated samples."""
        samples = self.metrics.generated_samples[-last_n:]

        if not samples:
            print("No generated samples yet")
            return

        print(f"\n{'='*60}")
        print("Recent Generations")
        print(f"{'='*60}")

        for s in samples:
            print(f"\nStep {s['step']} (loss={s['loss']:.4f}):")
            print(f"  Prompt: {s['prompt']}")
            print(f"  Output: {s['generated']}")

    def save_final_report(self) -> None:
        """Save final training report."""
        self.update_plots(force=True)

        # Create summary report
        report = []
        report.append("=" * 60)
        report.append("TRAINING REPORT")
        report.append("=" * 60)

        if self.metrics.steps:
            report.append(f"\nTotal steps: {len(self.metrics.steps)}")
            report.append(f"Final loss: {self.metrics.losses[-1]:.4f}")
            report.append(f"Best loss: {min(self.metrics.losses):.4f}")
            report.append(f"Loss improvement: {self.metrics.losses[0]:.4f} -> {self.metrics.losses[-1]:.4f}")

        if self.metrics.gate_values:
            report.append("\nGate values (final):")
            for layer_idx, values in sorted(self.metrics.gate_values.items()):
                report.append(f"  Layer {layer_idx}: {values[-1]:.4f}")

        if self.metrics.cms_weights:
            report.append("\nCMS weights (final):")
            for level, values in sorted(self.metrics.cms_weights.items()):
                report.append(f"  Level {level}: {values[-1]:.4f}")

        if self.metrics.generated_samples:
            report.append(f"\nGenerated {len(self.metrics.generated_samples)} samples during training")
            report.append("\nLast generation:")
            last = self.metrics.generated_samples[-1]
            report.append(f"  Step {last['step']}: {last['generated'][:100]}...")

        report_text = "\n".join(report)
        print(report_text)

        with open(self.output_dir / "report.txt", "w") as f:
            f.write(report_text)

        print(f"\nVisualization saved to: {self.output_dir}/")


def generate_sample(
    model,
    tokenizer,
    prompt: str = "The",
    max_tokens: int = 30,
    temperature: float = 0.8,
) -> str:
    """Generate a text sample from the model."""
    if mx is None:
        return "[MLX not available]"

    tokens = tokenizer.encode(prompt)
    generated = mx.array(tokens).reshape(1, -1)

    for _ in range(max_tokens):
        logits = model(generated)
        mx.eval(logits)

        # Sample from last position
        last_logits = logits[0, -1] / temperature
        probs = mx.softmax(last_logits, axis=-1)

        # Sample (or just take argmax for deterministic)
        next_token = mx.argmax(probs).item()

        # Stop on EOS
        if hasattr(tokenizer, 'eos_token_id') and next_token == tokenizer.eos_token_id:
            break

        generated = mx.concatenate([generated, mx.array([[next_token]])], axis=1)

    return tokenizer.decode(generated[0].tolist())
