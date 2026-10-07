"""
Word tokenizer shared by stored memories and queries.

BM25, phrase search and token-overlap ranking compare these tokens
directly, so memories and queries must be split the same way.
"""

from __future__ import annotations

import re
from typing import List

_WORD = re.compile(r"\w+")

# Hangul, kana and Han text attaches particles and compounds without spaces.
_CJK_RUN = re.compile(
    r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]+"
    r"|[\u3040-\u30ff]+"
    r"|[\u3400-\u4dbf\u4e00-\u9fff]+"
)


def tokenize(text: str) -> List[str]:
    """
    Split normalized text into tokens.

    Tokens are runs of letters and digits, so punctuation never sticks to a
    word. Hangul, kana and Han runs are separated from neighbouring Latin
    letters or digits ("gx10에서" gives "gx10" and "에서") and indexed as
    overlapping character bigrams ("플러그를" gives "플러", "러그", "그를"), so
    a word still matches when a different particle is attached to it.
    """

    if not text:
        return []

    tokens: List[str] = []

    for word in _WORD.findall(text):
        position = 0

        for run in _CJK_RUN.finditer(word):
            if run.start() > position:
                tokens.append(word[position:run.start()])

            tokens.extend(_bigrams(run.group()))
            position = run.end()

        if position < len(word):
            tokens.append(word[position:])

    return tokens


def _bigrams(run: str) -> List[str]:
    if len(run) == 1:
        return [run]

    return [run[i:i + 2] for i in range(len(run) - 1)]
