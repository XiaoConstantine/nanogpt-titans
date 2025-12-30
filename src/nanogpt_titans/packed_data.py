"""
Padding-Free Packing for efficient training.

Based on unsloth's packing technique:
- Packs multiple sequences into single batches without padding
- Uses attention masks / position IDs to prevent cross-sequence attention
- Maximizes GPU utilization by eliminating wasted compute on pad tokens

Supports:
- Document-aware packing (respects EOT boundaries)
- Variable-length sequence handling
- Flash Attention 2 compatible (via cu_seqlens)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class PackedBatch:
    """A packed batch containing multiple sequences."""

    # Packed token IDs [B, max_seq_len] - no padding, densely packed
    input_ids: torch.Tensor
    # Target IDs [B, max_seq_len]
    targets: torch.Tensor
    # Position IDs [B, max_seq_len] - resets at document boundaries
    position_ids: torch.Tensor
    # Attention mask [B, max_seq_len, max_seq_len] or None for Flash Attention
    attention_mask: torch.Tensor | None
    # Cumulative sequence lengths for Flash Attention [num_seqs + 1]
    cu_seqlens: torch.Tensor | None
    # Number of actual sequences packed in this batch
    num_sequences: int
    # Total tokens (excluding padding if any)
    total_tokens: int


class PackedDataset:
    """
    Dataset that packs multiple sequences efficiently.

    Instead of padding short sequences, we concatenate them and use
    position IDs / attention masks to maintain sequence boundaries.
    """

    def __init__(
        self,
        data_path: Path,
        block_size: int,
        batch_size: int,
        eot_token: int = 50256,  # GPT-2 EOT token
        *,
        shuffle: bool = True,
        seed: int = 1337,
    ) -> None:
        """
        Initialize packed dataset.

        Args:
            data_path: Path to .bin file with tokenized data
            block_size: Maximum sequence length
            batch_size: Number of packed sequences per batch
            eot_token: End-of-text token ID (document separator)
            shuffle: Whether to shuffle document order
            seed: Random seed for shuffling
        """
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.eot_token = eot_token
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # Memory-map the data
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)

        # Find document boundaries (EOT positions)
        self._find_document_boundaries()

    def _find_document_boundaries(self) -> None:
        """Find all document boundaries (EOT token positions)."""
        # Find EOT positions - this marks end of each document
        eot_positions = np.where(self.data == self.eot_token)[0]

        # Create document spans: (start, end) for each document
        self.documents: list[tuple[int, int]] = []
        prev_end = 0

        for eot_pos in eot_positions:
            doc_start = prev_end
            doc_end = eot_pos + 1  # Include the EOT token
            if doc_end - doc_start > 1:  # Skip empty documents
                self.documents.append((doc_start, doc_end))
            prev_end = doc_end

        # Handle last document if no EOT at end
        if prev_end < len(self.data):
            self.documents.append((prev_end, len(self.data)))

        print(f"Found {len(self.documents)} documents in {self.data_path}")
        print(f"Average document length: {self.total_tokens / len(self.documents):.1f} tokens")

    def _pack_sequences(
        self,
        doc_indices: list[int],
        device: torch.device,
    ) -> PackedBatch:
        """
        Pack multiple documents into a single batch.

        Uses bin-packing to efficiently fill the batch without padding.
        """
        packed_ids: list[list[int]] = [[] for _ in range(self.batch_size)]
        packed_positions: list[list[int]] = [[] for _ in range(self.batch_size)]
        sequence_lengths: list[list[int]] = [[] for _ in range(self.batch_size)]

        # Greedy bin-packing: assign each document to the batch row with most space
        for doc_idx in doc_indices:
            start, end = self.documents[doc_idx]
            doc_tokens = self.data[start:end].astype(np.int64).tolist()
            doc_len = len(doc_tokens)

            if doc_len > self.block_size:
                # Truncate long documents
                doc_tokens = doc_tokens[: self.block_size]
                doc_len = self.block_size

            # Find batch row with most remaining space
            remaining_space = [self.block_size - len(row) for row in packed_ids]
            best_row = max(range(self.batch_size), key=lambda i: remaining_space[i])

            if remaining_space[best_row] >= doc_len:
                # Add document to this row
                packed_ids[best_row].extend(doc_tokens)
                # Position IDs reset for each document
                packed_positions[best_row].extend(range(doc_len))
                sequence_lengths[best_row].append(doc_len)
            # If no space, document is skipped (can happen with very long docs)

        # Convert to tensors and pad to block_size
        input_ids = torch.zeros(self.batch_size, self.block_size, dtype=torch.long)
        targets = torch.zeros(self.batch_size, self.block_size, dtype=torch.long)
        position_ids = torch.zeros(self.batch_size, self.block_size, dtype=torch.long)

        total_tokens = 0
        num_sequences = 0

        for i in range(self.batch_size):
            row_len = len(packed_ids[i])
            if row_len > 0:
                input_ids[i, :row_len] = torch.tensor(packed_ids[i])
                # Targets are shifted by 1
                targets[i, : row_len - 1] = torch.tensor(packed_ids[i][1:])
                targets[i, row_len - 1 :] = -1  # Ignore last token and padding
                position_ids[i, :row_len] = torch.tensor(packed_positions[i])
                total_tokens += row_len
                num_sequences += len(sequence_lengths[i])

        # Build attention mask that prevents cross-document attention
        # For each row, documents can only attend within themselves
        attention_mask = self._build_packed_attention_mask(
            sequence_lengths, self.block_size, device
        )

        # Build cu_seqlens for Flash Attention (optional)
        cu_seqlens = self._build_cu_seqlens(sequence_lengths, device)

        return PackedBatch(
            input_ids=input_ids.to(device),
            targets=targets.to(device),
            position_ids=position_ids.to(device),
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            num_sequences=num_sequences,
            total_tokens=total_tokens,
        )

    def _build_packed_attention_mask(
        self,
        sequence_lengths: list[list[int]],
        max_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build attention mask for packed sequences.

        Each position can only attend to positions within the same document.
        Uses causal masking within each document.

        Returns:
            mask: [B, max_len, max_len] boolean mask (True = can attend)
        """
        batch_size = len(sequence_lengths)
        mask = torch.zeros(batch_size, max_len, max_len, dtype=torch.bool, device=device)

        for b in range(batch_size):
            pos = 0
            for seq_len in sequence_lengths[b]:
                # Create causal mask for this document
                # Each token can attend to itself and previous tokens in same doc
                for i in range(seq_len):
                    mask[b, pos + i, pos : pos + i + 1] = True
                pos += seq_len

        return mask

    def _build_cu_seqlens(
        self,
        sequence_lengths: list[list[int]],
        device: torch.device,
    ) -> torch.Tensor | None:
        """
        Build cumulative sequence lengths for Flash Attention 2.

        This is used with flash_attn_varlen_func for variable-length sequences.

        Returns:
            cu_seqlens: [total_seqs + 1] cumulative lengths
        """
        # Flatten all sequence lengths
        all_lengths = []
        for row_lengths in sequence_lengths:
            all_lengths.extend(row_lengths)

        if not all_lengths:
            return None

        # Cumulative sum starting from 0
        cu_seqlens = [0]
        for length in all_lengths:
            cu_seqlens.append(cu_seqlens[-1] + length)

        return torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    def __iter__(self) -> Iterator[PackedBatch]:
        """Iterate over packed batches."""
        doc_indices = list(range(len(self.documents)))
        if self.shuffle:
            self.rng.shuffle(doc_indices)

        # Process documents in chunks
        docs_per_batch = self.batch_size * 2  # Pack ~2 docs per row on average
        for i in range(0, len(doc_indices), docs_per_batch):
            chunk = doc_indices[i : i + docs_per_batch]
            if chunk:
                yield self._pack_sequences(chunk, torch.device("cpu"))


class PackedDataLoader:
    """
    Efficient data loader with padding-free packing.

    Features:
    - Packs multiple documents per batch row
    - Respects document boundaries (no cross-doc attention)
    - Maximizes GPU utilization
    - Supports prefetching
    """

    def __init__(
        self,
        data_path: Path,
        block_size: int,
        batch_size: int,
        device: torch.device,
        *,
        eot_token: int = 50256,
        shuffle: bool = True,
        seed: int = 1337,
        prefetch: bool = True,
    ) -> None:
        self.data_path = data_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.eot_token = eot_token
        self.shuffle = shuffle
        self.seed = seed
        self.prefetch = prefetch

        # Load data
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)

        # Find document boundaries
        self._find_boundaries()

        # Initialize state
        self.rng = np.random.default_rng(seed)
        self._reset_indices()

        # Prefetch buffer
        self._prefetch_batch: PackedBatch | None = None

    def _find_boundaries(self) -> None:
        """Find document boundaries efficiently."""
        eot_mask = self.data == self.eot_token
        eot_positions = np.nonzero(eot_mask)[0]

        self.doc_starts: list[int] = [0]
        self.doc_ends: list[int] = []

        for pos in eot_positions:
            self.doc_ends.append(pos + 1)
            if pos + 1 < self.total_tokens:
                self.doc_starts.append(pos + 1)

        # Handle final document
        if not self.doc_ends or self.doc_ends[-1] < self.total_tokens:
            self.doc_ends.append(self.total_tokens)

        self.num_docs = len(self.doc_ends)
        print(f"PackedDataLoader: {self.num_docs} documents, {self.total_tokens:,} tokens")

    def _reset_indices(self) -> None:
        """Reset document indices for new epoch."""
        self.doc_indices = np.arange(self.num_docs)
        if self.shuffle:
            self.rng.shuffle(self.doc_indices)
        self.current_idx = 0

    def _get_next_docs(self, n: int) -> list[int]:
        """Get next n document indices, wrapping if needed."""
        if self.current_idx + n > self.num_docs:
            self._reset_indices()

        indices = self.doc_indices[self.current_idx : self.current_idx + n].tolist()
        self.current_idx += n
        return indices

    def get_batch(self) -> PackedBatch:
        """Get a packed batch."""
        if self._prefetch_batch is not None:
            batch = self._prefetch_batch
            self._prefetch_batch = None
            # Start prefetching next batch
            if self.prefetch:
                self._start_prefetch()
            return batch

        return self._create_batch()

    def _start_prefetch(self) -> None:
        """Prefetch next batch (simple sync version for now)."""
        self._prefetch_batch = self._create_batch()

    def _create_batch(self) -> PackedBatch:
        """Create a packed batch from documents."""
        # Get enough documents to fill batch (estimate ~2 docs per row)
        target_docs = self.batch_size * 3
        doc_indices = self._get_next_docs(target_docs)

        # Pack documents into batch rows using greedy bin-packing
        packed_ids: list[list[int]] = [[] for _ in range(self.batch_size)]
        packed_positions: list[list[int]] = [[] for _ in range(self.batch_size)]
        seq_lengths: list[list[int]] = [[] for _ in range(self.batch_size)]

        for doc_idx in doc_indices:
            start = self.doc_starts[doc_idx]
            end = self.doc_ends[doc_idx]
            doc_len = end - start

            if doc_len > self.block_size:
                doc_len = self.block_size
                end = start + doc_len

            # Find row with most space
            remaining = [self.block_size - len(row) for row in packed_ids]
            best_row = max(range(self.batch_size), key=lambda i: remaining[i])

            if remaining[best_row] >= doc_len:
                tokens = self.data[start:end].astype(np.int64).tolist()
                packed_ids[best_row].extend(tokens)
                packed_positions[best_row].extend(range(doc_len))
                seq_lengths[best_row].append(doc_len)

        # Convert to tensors
        input_ids = torch.zeros(self.batch_size, self.block_size, dtype=torch.long)
        targets = torch.full((self.batch_size, self.block_size), -1, dtype=torch.long)
        position_ids = torch.zeros(self.batch_size, self.block_size, dtype=torch.long)

        total_tokens = 0
        num_seqs = 0

        for i in range(self.batch_size):
            row_len = len(packed_ids[i])
            if row_len > 0:
                input_ids[i, :row_len] = torch.tensor(packed_ids[i])
                if row_len > 1:
                    targets[i, : row_len - 1] = torch.tensor(packed_ids[i][1:])
                position_ids[i, :row_len] = torch.tensor(packed_positions[i])
                total_tokens += row_len
                num_seqs += len(seq_lengths[i])

        # Build attention mask
        attention_mask = self._build_attention_mask(seq_lengths)

        # Move to device
        device = self.device
        if "cuda" in str(device):
            input_ids = input_ids.pin_memory().to(device, non_blocking=True)
            targets = targets.pin_memory().to(device, non_blocking=True)
            position_ids = position_ids.pin_memory().to(device, non_blocking=True)
            attention_mask = attention_mask.pin_memory().to(device, non_blocking=True)
        else:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            position_ids = position_ids.to(device)
            attention_mask = attention_mask.to(device)

        return PackedBatch(
            input_ids=input_ids,
            targets=targets,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cu_seqlens=None,  # TODO: Add for Flash Attention varlen
            num_sequences=num_seqs,
            total_tokens=total_tokens,
        )

    def _build_attention_mask(self, seq_lengths: list[list[int]]) -> torch.Tensor:
        """Build causal attention mask respecting document boundaries."""
        mask = torch.zeros(self.batch_size, self.block_size, self.block_size, dtype=torch.bool)

        for b in range(self.batch_size):
            pos = 0
            for seq_len in seq_lengths[b]:
                # Causal mask within this document
                for i in range(seq_len):
                    mask[b, pos + i, pos : pos + i + 1] = True
                pos += seq_len

        return mask


def get_packed_batch(
    _split: str,
    loader: PackedDataLoader,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get a packed batch compatible with training loop.

    Returns:
        input_ids: [B, T]
        targets: [B, T]
        position_ids: [B, T]
        attention_mask: [B, T, T]
    """
    batch = loader.get_batch()
    return batch.input_ids, batch.targets, batch.position_ids, batch.attention_mask


# Quick test
if __name__ == "__main__":
    from pathlib import Path

    # Test with a data file if it exists
    test_path = Path("data/shakespeare/train.bin")
    if test_path.exists():
        loader = PackedDataLoader(
            data_path=test_path,
            block_size=256,
            batch_size=4,
            device=torch.device("cpu"),
            shuffle=True,
        )

        batch = loader.get_batch()
        print(f"Input IDs shape: {batch.input_ids.shape}")
        print(f"Targets shape: {batch.targets.shape}")
        print(f"Position IDs shape: {batch.position_ids.shape}")
        print(f"Attention mask shape: {batch.attention_mask.shape}")
        print(f"Num sequences: {batch.num_sequences}")
        print(f"Total tokens: {batch.total_tokens}")
        print(f"Utilization: {batch.total_tokens / (4 * 256) * 100:.1f}%")
    else:
        print("Test data not found. Run: python -m nanogpt_titans.prepare_data shakespeare")
