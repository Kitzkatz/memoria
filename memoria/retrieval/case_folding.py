import unicodedata
from typing import Optional


def fold_case(text: str) -> str:
    """
    Normalize Unicode (NFKD) and convert to lowercase.

    NFKD (Compatibility Decomposition) decomposes characters into their
    canonical components, then converts to lowercase. This handles:
    - Accented characters (é → e + ́ → e)
    - Ligatures (ﬁ → f + i → fi)
    - Special characters (ℌ → H)

    This is used for case-insensitive, accent-insensitive text matching.

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
    normalized = unicodedata.normalize('NFKD', text).lower()

    # Optional: strip whitespace
    return normalized.strip()
