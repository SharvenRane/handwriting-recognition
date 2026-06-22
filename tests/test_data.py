"""Tests for the synthetic handwriting renderer."""

import numpy as np

from src.data import (
    IMAGE_HEIGHT,
    CHAR_WIDTH,
    render_text,
    make_dataset,
    supported_characters,
)
from src.vocab import Vocabulary


def test_render_shape_and_range():
    img = render_text("abc", seed=1)
    assert img.shape == (IMAGE_HEIGHT, CHAR_WIDTH * 3)
    assert img.min() >= 0.0
    assert img.max() <= 1.0


def test_render_is_deterministic_for_same_seed():
    a = render_text("hello", seed=5)
    b = render_text("hello", seed=5)
    assert np.array_equal(a, b)


def test_different_strings_render_differently():
    a = render_text("abc", seed=0, jitter=False)
    b = render_text("xyz", seed=0, jitter=False)
    assert not np.array_equal(a, b)


def test_space_renders_blank_column():
    img = render_text(" ", seed=0, jitter=False)
    assert img.sum() == 0.0


def test_ink_present_for_letters():
    img = render_text("a", seed=0, jitter=False)
    assert img.sum() > 0.0


def test_make_dataset_pads_to_rectangle():
    vocab = Vocabulary(supported_characters())
    images, targets, texts = make_dataset(["ab", "abcd"], vocab, jitter=False)
    assert images.shape[0] == 2
    assert images.shape[1] == 1
    assert images.shape[2] == IMAGE_HEIGHT
    # Width is driven by the longest string.
    assert images.shape[3] == CHAR_WIDTH * 4
    assert targets[0] == vocab.encode("ab")
    assert targets[1] == vocab.encode("abcd")
    assert texts == ["ab", "abcd"]
