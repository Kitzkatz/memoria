from retrieval.case_folding import fold_case


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def test_fold_case_keeps_precomposed_characters():
    for text in ("한국어 메모", "データベース"):
        folded = fold_case(text)

        require(
            folded == text,
            f"{text!r} was decomposed: {[hex(ord(c)) for c in folded]}",
        )


def test_fold_case_maps_compatibility_forms():
    require(
        fold_case("ＴＥＳＴ ﬁle ℌ") == "test file h",
        "Compatibility characters were not normalized.",
    )
