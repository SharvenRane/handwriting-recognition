"""CTC alignment helpers: blank collapse and greedy decoding.

These functions implement the deterministic parts of CTC that do not depend on
the training loss. They operate on per timestep class ids and follow the
standard two step collapse rule:

  1. merge runs of repeated ids,
  2. then drop the blank symbol.

The two steps must happen in that order. Collapsing repeats first lets a target
that genuinely contains a repeated character (for example "ll" in "hello")
survive, because CTC separates the two real characters with a blank.
"""

from __future__ import annotations

from typing import List, Sequence

from .vocab import BLANK_INDEX


def collapse_path(path: Sequence[int], blank: int = BLANK_INDEX) -> List[int]:
    """Collapse a CTC alignment path into the underlying id sequence.

    Runs of the same id are merged into one, and then blanks are removed. The
    result is the label sequence that the path decodes to.
    """
    collapsed: List[int] = []
    prev = None
    for idx in path:
        if idx != prev:
            collapsed.append(idx)
        prev = idx
    return [idx for idx in collapsed if idx != blank]


def greedy_decode(logits, blank: int = BLANK_INDEX) -> List[List[int]]:
    """Greedy (best path) CTC decode.

    Args:
        logits: tensor shaped (time, batch, num_classes) or
            (batch, time, num_classes). The argmax over the class axis is taken
            at every timestep, then each path is collapsed.
        blank: blank class id.

    Returns:
        A list with one decoded id sequence per batch element.
    """
    import torch

    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)

    if logits.dim() != 3:
        raise ValueError(f"expected a 3D tensor, got shape {tuple(logits.shape)}")

    # Normalise to (batch, time, num_classes). We detect the layout by assuming
    # the time axis is the larger of the first two when ambiguous is not safe,
    # so we require the caller convention (time, batch, classes) and transpose.
    # To stay unambiguous we accept an explicit (T, B, C) layout only.
    time_major = logits  # treated as (T, B, C)
    best = time_major.argmax(dim=-1)  # (T, B)
    best = best.transpose(0, 1).tolist()  # (B, T)

    return [collapse_path(path, blank=blank) for path in best]
