# Handwriting Recognition with CRNN and CTC

A compact, self contained implementation of handwritten text recognition. It pairs
a small convolutional recurrent network (CRNN) with the CTC loss so that the model
can read a line of text without any per character bounding boxes. Everything runs
on CPU and there is nothing to download. The handwriting images are generated in
code, so the whole project is reproducible from a clean checkout.

## What is inside

The pipeline has four real pieces that fit together the way a production OCR stack
would, just shrunk down so it trains in seconds.

**Vocabulary.** `src/vocab.py` maps characters to integer ids. Id 0 is reserved for
the CTC blank symbol and every real character starts at id 1. Encoding a string and
decoding a list of ids back to text are exact inverses.

**Synthetic handwriting.** `src/data.py` renders short strings onto a fixed height
grayscale canvas. Each supported character has its own stroke pattern defined in a
tiny bitmap font, so two different strings always produce two different images. A
light intensity wobble and a touch of noise make the renders look hand drawn rather
than crisp, and the renderer is seeded so a given string and seed always produce the
same image.

**CRNN.** `src/model.py` is the usual convolutional recurrent design. Convolutions
reduce the image and pool the height down to a single row, which turns the width into
a time axis. A bidirectional GRU reads that sequence and a linear head emits a class
score at every timestep. The forward pass returns log probabilities shaped
`(time, batch, classes)`, which is exactly what `nn.CTCLoss` and the greedy decoder
expect.

**CTC collapse and decoding.** `src/ctc.py` holds the deterministic half of CTC. The
collapse rule runs in two steps and the order matters: first merge runs of repeated
ids, then drop the blank. Collapsing repeats before removing blanks is what lets a
genuine double letter like the two l characters in "hello" survive, because CTC keeps
a blank between the two real characters so they do not merge into one.

## Training

`src/train.py` wires the model to `nn.CTCLoss` and runs full batch gradient descent.
The `train_overfit` helper trains on a tiny set until the model memorises it, then
greedy decodes the predictions and hands back the recovered strings. This is the
behaviour the tests lean on.

A minimal run looks like this:

```python
from src.vocab import Vocabulary
from src.data import make_dataset, supported_characters
from src.model import CRNN
from src.train import train_overfit

vocab = Vocabulary(supported_characters())
texts = ["cat", "dog", "fox"]
images, targets, _ = make_dataset(texts, vocab, jitter=False)
model = CRNN(num_classes=vocab.num_classes)
decoded = train_overfit(model, images, targets, vocab, steps=400, lr=1e-2)
print(decoded)  # ['cat', 'dog', 'fox']
```

## Tests

The test suite checks behaviour, not just shapes.

* The CTC collapse is verified on known alignments, including the double letter case
  where the blank is what keeps "hello" from collapsing to "helo".
* The greedy decoder is checked against hand built logits whose argmax path is a known
  alignment, in both single and batched form.
* The renderer is checked for determinism, for distinct output on distinct strings, and
  for correct padding when strings of different length share a batch.
* The model is trained to overfit a small set and must decode its inputs back to the
  exact target strings, which exercises the convolutions, the GRU, the CTC loss, and the
  decoder end to end.

Run them with:

```
python -m pytest tests/ -q
```

On the development machine all 23 tests pass on CPU in roughly nine seconds.

## Layout

```
src/
  vocab.py    character to id mapping with a reserved blank
  data.py     synthetic handwriting renderer and batching
  model.py    the CRNN
  ctc.py      blank collapse and greedy decode
  train.py    CTC training loop and the overfit helper
tests/        pytest behaviour checks
```

## Notes

The bitmap font is intentionally small and readable rather than realistic. It exists so
the project stays offline and deterministic while still giving the model real strokes to
learn from. Swapping in a richer renderer or a real handwriting dataset would not touch
the model, the loss, or the decoder, since those operate on tensors and ids rather than
on any particular image source.
