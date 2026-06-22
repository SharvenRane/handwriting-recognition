"""Training loop helpers for the CRNN with CTC loss."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from .ctc import greedy_decode
from .model import CRNN
from .vocab import Vocabulary, BLANK_INDEX


def _ctc_targets(targets: Sequence[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten variable length targets into the format nn.CTCLoss expects."""
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat = torch.tensor([idx for t in targets for idx in t], dtype=torch.long)
    return flat, lengths


def train_overfit(
    model: CRNN,
    images: torch.Tensor,
    targets: Sequence[Sequence[int]],
    vocab: Vocabulary,
    steps: int = 300,
    lr: float = 1e-2,
) -> List[str]:
    """Train the model to overfit a tiny batch and return the decoded strings.

    The function runs full batch gradient descent with CTC loss for a fixed
    number of steps, then greedy decodes the final predictions. It is meant for
    tiny sets where memorisation is the goal, for example in tests.
    """
    device = next(model.parameters()).device
    images = images.to(device)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    flat_targets, target_lengths = _ctc_targets(targets)
    flat_targets = flat_targets.to(device)

    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        log_probs = model(images)  # (T, B, C)
        t_axis = log_probs.size(0)
        batch = log_probs.size(1)
        input_lengths = torch.full((batch,), t_axis, dtype=torch.long)
        loss = criterion(log_probs, flat_targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        log_probs = model(images)
        decoded_ids = greedy_decode(log_probs, blank=BLANK_INDEX)
    return [vocab.decode_ids(ids) for ids in decoded_ids]
