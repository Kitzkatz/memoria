import unicodedata
from typing import Optional


def fold_case(text: str) -> str:
    """
    Normalize Unicode (NFKC) and convert to lowercase.

    NFKC (Compatibility Composition) maps compatibility characters to their
    standard form and recomposes them. This handles:
    - Ligatures (ﬁ → fi)
    - Full-width forms (Ｔ → t)
    - Special characters (ℌ → h)

    Precomposed characters such as Hangul syllables stay intact. NFKD would
    split them into conjoining jamo, which embedding models and tokenizers
    treat as unrelated text.

    This is used for case-insensitive text matching.

    Args:
        text: Input string to normalize

    Returns:
        Normalized lowercase string, or empty string if input is None
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Normalize and lower
    normalized = unicodedata.normalize('NFKC', text).lower()

    # Optional: strip whitespace
    return normalized.strip()
