"""Tests for the deterministic CTC collapse and greedy decode."""

import torch

from src.ctc import collapse_path, greedy_decode
from src.vocab import Vocabulary, BLANK_INDEX


def test_collapse_merges_repeats_then_drops_blank():
    # ids: a=1, b=2, blank=0. Path a a blank a b b -> a a b.
    path = [1, 1, 0, 1, 2, 2]
    assert collapse_path(path) == [1, 1, 2]


def test_collapse_preserves_double_letter_via_blank():
    # The canonical "hello" style case: a real double letter survives only when
    # the two copies are separated by a blank in the alignment.
    vocab = Vocabulary(list("helo"))
    h, e, l, o = (vocab.char_to_id(c) for c in "helo")
    # h e l blank l o, with frame repeats, must decode back to "hello".
    path = [h, h, e, l, l, BLANK_INDEX, l, o, o]
    ids = collapse_path(path)
    assert vocab.decode_ids(ids) == "hello"


def test_collapse_without_blank_merges_double_letter():
    # Without the separating blank, the two l frames merge into one, which is
    # exactly why CTC needs the blank between identical characters.
    vocab = Vocabulary(list("helo"))
    h, e, l, o = (vocab.char_to_id(c) for c in "helo")
    path = [h, e, l, l, o]
    assert vocab.decode_ids(collapse_path(path)) == "helo"


def test_all_blank_path_decodes_to_empty():
    assert collapse_path([0, 0, 0]) == []


def test_greedy_decode_known_alignment():
    # Build logits (T, B, C) where the argmax path is a known alignment.
    vocab = Vocabulary(list("ab"))
    a = vocab.char_to_id("a")
    b = vocab.char_to_id("b")
    # frames: a a blank b  -> collapses to a b
    frames = [a, a, BLANK_INDEX, b]
    T = len(frames)
    C = vocab.num_classes
    logits = torch.full((T, 1, C), -10.0)
    for t, idx in enumerate(frames):
        logits[t, 0, idx] = 10.0
    decoded = greedy_decode(logits)
    assert len(decoded) == 1
    assert vocab.decode_ids(decoded[0]) == "ab"


def test_greedy_decode_batch():
    vocab = Vocabulary(list("ab"))
    a = vocab.char_to_id("a")
    b = vocab.char_to_id("b")
    seqs = [[a, a, BLANK_INDEX, a], [b, BLANK_INDEX, b, b]]
    T = 4
    C = vocab.num_classes
    logits = torch.full((T, 2, C), -10.0)
    for batch_idx, frames in enumerate(seqs):
        for t, idx in enumerate(frames):
            logits[t, batch_idx, idx] = 10.0
    decoded = greedy_decode(logits)
    assert vocab.decode_ids(decoded[0]) == "aa"
    assert vocab.decode_ids(decoded[1]) == "bb"
