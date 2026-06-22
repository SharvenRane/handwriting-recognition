"""Synthetic handwriting style image generation.

This is a fully offline, deterministic generator. It renders short text strings
onto a fixed height grayscale canvas using a bundled bitmap font drawn by hand
in code, then applies light jitter so the images look hand written rather than
crisp. No external fonts, no downloads, no network.

The renderer is deliberately simple but real: every supported character has its
own stroke pattern, so different strings produce genuinely different images and
a model has to read the strokes to recover the text.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .vocab import Vocabulary


IMAGE_HEIGHT = 32
CHAR_WIDTH = 16
PAD = 2


# A 5x3 stroke grid per character. Rows go top to bottom, each row is 3 cells.
# A "1" means the cell is inked. This is a tiny readable font defined in code.
_GLYPHS = {
    "a": ["010", "101", "111", "101", "101"],
    "b": ["110", "101", "110", "101", "110"],
    "c": ["011", "100", "100", "100", "011"],
    "d": ["110", "101", "101", "101", "110"],
    "e": ["111", "100", "110", "100", "111"],
    "f": ["111", "100", "110", "100", "100"],
    "g": ["011", "100", "101", "101", "011"],
    "h": ["101", "101", "111", "101", "101"],
    "i": ["111", "010", "010", "010", "111"],
    "j": ["111", "001", "001", "101", "010"],
    "k": ["101", "110", "100", "110", "101"],
    "l": ["100", "100", "100", "100", "111"],
    "m": ["101", "111", "111", "101", "101"],
    "n": ["101", "111", "111", "111", "101"],
    "o": ["010", "101", "101", "101", "010"],
    "p": ["110", "101", "110", "100", "100"],
    "q": ["010", "101", "101", "011", "001"],
    "r": ["110", "101", "110", "101", "101"],
    "s": ["011", "100", "010", "001", "110"],
    "t": ["111", "010", "010", "010", "010"],
    "u": ["101", "101", "101", "101", "111"],
    "v": ["101", "101", "101", "010", "010"],
    "w": ["101", "101", "111", "111", "101"],
    "x": ["101", "101", "010", "101", "101"],
    "y": ["101", "101", "010", "010", "010"],
    "z": ["111", "001", "010", "100", "111"],
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["110", "001", "010", "100", "111"],
    "3": ["110", "001", "010", "001", "110"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "110", "001", "110"],
    "6": ["011", "100", "110", "101", "010"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["010", "101", "010", "101", "010"],
    "9": ["010", "101", "011", "001", "110"],
    " ": ["000", "000", "000", "000", "000"],
}


def supported_characters() -> List[str]:
    """Characters the renderer knows how to draw."""
    return list(_GLYPHS.keys())


def _render_glyph(ch: str, rng: np.random.Generator) -> np.ndarray:
    """Render a single character into a CHAR_WIDTH wide, IMAGE_HEIGHT tall tile."""
    grid = _GLYPHS[ch]
    tile = np.zeros((IMAGE_HEIGHT, CHAR_WIDTH), dtype=np.float32)

    rows = len(grid)
    cols = len(grid[0])
    # Leave a margin so strokes do not touch the tile edge.
    cell_h = (IMAGE_HEIGHT - 2 * PAD) / rows
    cell_w = (CHAR_WIDTH - 2 * PAD) / cols

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != "1":
                continue
            y0 = int(PAD + r * cell_h)
            y1 = int(PAD + (r + 1) * cell_h)
            x0 = int(PAD + c * cell_w)
            x1 = int(PAD + (c + 1) * cell_w)
            tile[y0:y1, x0:x1] = 1.0
    return tile


def render_text(text: str, seed: int = 0, jitter: bool = True) -> np.ndarray:
    """Render a string into a single grayscale image.

    Args:
        text: characters to draw, all must be in supported_characters().
        seed: controls the per image jitter so renders are reproducible.
        jitter: when True, applies small noise and an intensity wobble so the
            image looks hand written rather than a clean bitmap.

    Returns:
        A float32 array shaped (IMAGE_HEIGHT, CHAR_WIDTH * len(text)) with values
        in [0, 1] where 1 is ink.
    """
    rng = np.random.default_rng(seed)
    tiles = [_render_glyph(ch, rng) for ch in text]
    if not tiles:
        return np.zeros((IMAGE_HEIGHT, CHAR_WIDTH), dtype=np.float32)
    image = np.concatenate(tiles, axis=1)

    if jitter:
        # Intensity wobble per pixel and a touch of additive noise on the ink.
        wobble = rng.uniform(0.75, 1.0, size=image.shape).astype(np.float32)
        image = image * wobble
        noise = rng.normal(0.0, 0.03, size=image.shape).astype(np.float32)
        image = np.clip(image + noise * (image > 0), 0.0, 1.0)
    return image


def make_dataset(
    texts: Sequence[str],
    vocab: Vocabulary,
    seed: int = 0,
    jitter: bool = True,
) -> Tuple["object", List[List[int]], List[str]]:
    """Build a padded tensor batch from a list of strings.

    Returns:
        images: float tensor shaped (batch, 1, height, max_width). Shorter
            strings are right padded with background so the batch is rectangular.
        targets: list of label id lists, one per string.
        texts: the original strings, echoed back for convenience.
    """
    import torch

    rendered = [render_text(t, seed=seed + i, jitter=jitter) for i, t in enumerate(texts)]
    max_w = max(r.shape[1] for r in rendered)
    batch = np.zeros((len(rendered), 1, IMAGE_HEIGHT, max_w), dtype=np.float32)
    for i, r in enumerate(rendered):
        batch[i, 0, :, : r.shape[1]] = r

    images = torch.from_numpy(batch)
    targets = [vocab.encode(t) for t in texts]
    return images, targets, list(texts)
