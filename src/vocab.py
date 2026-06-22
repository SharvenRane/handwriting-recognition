"""Character vocabulary for CTC based recognition.

Index 0 is reserved for the CTC blank symbol. Real characters start at index 1.
"""

from __future__ import annotations

from typing import Iterable, List


BLANK_INDEX = 0


class Vocabulary:
    """Maps characters to integer ids and back.

    The blank symbol used by CTC always lives at index 0 and has no character
    associated with it. Every other character gets a contiguous id starting at 1.
    """

    def __init__(self, characters: Iterable[str]):
        chars: List[str] = []
        seen = set()
        for ch in characters:
            if len(ch) != 1:
                raise ValueError(f"vocabulary entries must be single characters, got {ch!r}")
            if ch in seen:
                continue
            seen.add(ch)
            chars.append(ch)
        self.characters = chars
        # id 0 is blank, so characters map to 1..N
        self._char_to_id = {ch: i + 1 for i, ch in enumerate(chars)}
        self._id_to_char = {i + 1: ch for i, ch in enumerate(chars)}

    @property
    def num_classes(self) -> int:
        """Total number of CTC classes including the blank symbol."""
        return len(self.characters) + 1

    def char_to_id(self, ch: str) -> int:
        return self._char_to_id[ch]

    def id_to_char(self, idx: int) -> str:
        return self._id_to_char[idx]

    def encode(self, text: str) -> List[int]:
        """Turn a string into a list of class ids (no blanks inserted)."""
        return [self._char_to_id[ch] for ch in text]

    def decode_ids(self, ids: Iterable[int]) -> str:
        """Turn a list of class ids into a string, skipping the blank id."""
        out = []
        for idx in ids:
            if idx == BLANK_INDEX:
                continue
            out.append(self._id_to_char[idx])
        return "".join(out)


def default_vocabulary() -> Vocabulary:
    """A small vocabulary covering lowercase letters, digits and a space."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    digits = "0123456789"
    return Vocabulary(list(letters) + list(digits) + [" "])
