"""Tests for the CRNN model and the overfitting training loop."""

import torch

from src.data import make_dataset, supported_characters, CHAR_WIDTH
from src.model import CRNN
from src.train import train_overfit
from src.vocab import Vocabulary


def test_forward_shape_is_time_batch_classes():
    vocab = Vocabulary(supported_characters())
    model = CRNN(num_classes=vocab.num_classes)
    images, _, _ = make_dataset(["abc", "de"], vocab, jitter=False)
    out = model(images)
    # (time, batch, classes)
    assert out.dim() == 3
    assert out.size(1) == 2
    assert out.size(2) == vocab.num_classes


def test_forward_output_is_log_probabilities():
    vocab = Vocabulary(supported_characters())
    model = CRNN(num_classes=vocab.num_classes)
    images, _, _ = make_dataset(["ab"], vocab, jitter=False)
    out = model(images)
    probs = out.exp().sum(dim=-1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)


def test_time_axis_tracks_image_width():
    vocab = Vocabulary(supported_characters())
    model = CRNN(num_classes=vocab.num_classes)
    short, _, _ = make_dataset(["ab"], vocab, jitter=False)
    longer, _, _ = make_dataset(["abcde"], vocab, jitter=False)
    out_short = model(short)
    out_long = model(longer)
    # A wider image yields more CTC timesteps.
    assert out_long.size(0) > out_short.size(0)


def test_overfit_tiny_set_decodes_targets():
    torch.manual_seed(0)
    vocab = Vocabulary(supported_characters())
    texts = ["cat", "dog", "fox"]
    images, targets, _ = make_dataset(texts, vocab, seed=0, jitter=False)
    model = CRNN(num_classes=vocab.num_classes, hidden_size=64)
    decoded = train_overfit(model, images, targets, vocab, steps=400, lr=1e-2)
    assert decoded == texts


def test_overfit_recovers_double_letter_word():
    torch.manual_seed(0)
    vocab = Vocabulary(supported_characters())
    texts = ["hello"]
    images, targets, _ = make_dataset(texts, vocab, seed=3, jitter=False)
    model = CRNN(num_classes=vocab.num_classes, hidden_size=64)
    decoded = train_overfit(model, images, targets, vocab, steps=500, lr=1e-2)
    # The model must learn the CTC blank between the two l characters.
    assert decoded == ["hello"]
