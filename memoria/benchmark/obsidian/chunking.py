"""
Split long section text into pieces that fit the embedding limit.

Memoria's Embedder encodes only the first ``max_chars`` characters of a
record, so a long section would lose everything after that point in the
vector index.
"""

from __future__ import annotations

from typing import Iterator


def split_text(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks of at most ``max_chars`` characters.

    Whole lines are kept together where they fit. A line longer than
    ``max_chars`` is broken at the last space before the limit, or at the
    limit itself when it has no space.
    """

    chunks: list[str] = []
    current = ""

    for line in _lines(text, max_chars):
        candidate = f"{current}\n{line}" if current else line

        if len(candidate) > max_chars:
            chunks.append(current.strip())
            candidate = line

        current = candidate

    chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def _lines(text: str, max_chars: int) -> Iterator[str]:
    """Yield the lines of text, breaking any line longer than max_chars."""

    for line in text.splitlines():
        while len(line) > max_chars:
            cut = line.rfind(" ", 1, max_chars + 1)

            if cut == -1:
                cut = max_chars

            yield line[:cut]
            line = line[cut:].lstrip()

        yield line
