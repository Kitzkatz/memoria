from ranking.bm25_ranker import BM25


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def test_build_from_memories_scores_real_ids():
    bm25 = BM25()

    indexed = bm25.build_from_memories([
        {"id": 5, "tokens": ["solar", "panel"]},
        {"id": 7, "tokens": []},
        {"id": 9, "tokens": ["water", "pump"]},
    ])

    scores = bm25.score_ids(["pump"], [5, 9])

    require(indexed == 2, f"Expected 2 indexed memories, got {indexed}.")
    require(
        scores[9] > 0 and scores[5] == 0,
        f"BM25 scored the wrong memory: {scores}",
    )
