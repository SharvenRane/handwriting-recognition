"""Tests for the character vocabulary."""

import pytest

from src.vocab import Vocabulary, default_vocabulary, BLANK_INDEX


def test_blank_is_index_zero():
    vocab = Vocabulary(list("abc"))
    # No character maps to the blank id.
    for ch in "abc":
        assert vocab.char_to_id(ch) != BLANK_INDEX


def test_num_classes_includes_blank():
    vocab = Vocabulary(list("abc"))
    assert vocab.num_classes == 4  # 3 characters plus blank


def test_encode_decode_roundtrip():
    vocab = default_vocabulary()
    text = "hello world 42"
    ids = vocab.encode(text)
    assert vocab.decode_ids(ids) == text


def test_decode_skips_blank():
    vocab = Vocabulary(list("ab"))
    a = vocab.char_to_id("a")
    b = vocab.char_to_id("b")
    assert vocab.decode_ids([a, BLANK_INDEX, b]) == "ab"


def test_duplicate_characters_collapse():
    vocab = Vocabulary(list("aabbc"))
    assert vocab.num_classes == 4  # a b c plus blank


def test_rejects_multichar_entries():
    with pytest.raises(ValueError):
        Vocabulary(["ab"])
