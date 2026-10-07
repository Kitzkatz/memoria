from retrieval.tokenizer import tokenize


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def test_tokenize_splits_punctuation_scripts_and_hangul():
    tokens = tokenize("gx10-03에서 `8110`, gateway. 플러그를")

    require(
        tokens == ["gx10", "03", "에서", "8110", "gateway", "플러", "러그", "그를"],
        f"Unexpected tokens: {tokens}",
    )


def test_tokenize_indexes_kana_and_han_as_bigrams():
    tokens = tokenize("データベース 太淸劍法")

    require(
        tokens == ["デー", "ータ", "タベ", "ベー", "ース", "太淸", "淸劍", "劍法"],
        f"Unexpected tokens: {tokens}",
    )
