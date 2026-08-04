import unicodedata

def fold_case(text: str) -> str:
    """
    Normalize Unicode (NFKD) and convert to lowercase.
    Handles accents, special characters, and ligatures.
    """
    return unicodedata.normalize('NFKD', text).lower()
