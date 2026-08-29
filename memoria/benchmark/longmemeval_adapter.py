#!/usr/bin/env python3
"""
LongMemEval-S adapter.

Responsibilities:
- Per-question haystack isolation.
- Store every non-empty turn with session/turn metadata.
- Query Memoria and preserve ranked candidates.
- Evaluate session-level and turn-level retrieval.
- Exclude abstention questions from retrieval metrics.
- Maintain reproducible per-question caches.
- Record dataset/model/settings/commit metadata.
- Support dense, BM25, fusion, and full-ranking ablations.

Important:
- Retrieval workers remain responsible for candidate generation.
- Ranking remains responsible for candidate ordering.
- The adapter does not implement retrieval or ranking logic itself.
"""

import json
import hashlib
import os
import sys
import time
import shutil
import subprocess
import pickle
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import debug
from shared.memory_interface import MemoryInterface
from benchmark.batch_loader import BatchLoader
from cache.config import settings
from ranking.adaptive_weighter import adaptive_weighter_pipeline


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

K_VALUES = (1, 3, 5, 10, 30, 50)


# ----------------------------------------------------------------------
# Reproducibility helpers
# ----------------------------------------------------------------------

def compute_checksum(filepath):
    sha256 = hashlib.sha256()

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_git_commit():
    """Return current git commit when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------
# Dataset helpers
# ----------------------------------------------------------------------

def load_questions_file(
    questions_path="benchmark_output/longmemeval_questions.json"
):
    """Load extracted question metadata when available."""

    path = Path(questions_path)

    if not path.exists():
        print(
            f"Warning: Questions file not found at {path}. "
            "Answer-text matching will be limited."
        )
        return None

    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load questions file: {e}")
        return None


def get_question_metadata(question, questions_data):
    """Return extracted metadata for a question when available."""

    if not questions_data:
        return None

    question_id = question.get("question_id")

    for item in questions_data:
        if item.get("question_id") == question_id:
            return item

    return None


def get_expected_text(question, questions_data):
    """Return expected answer text from either dataset or extracted metadata."""

    expected = question.get("expected")

    if expected:
        return expected

    metadata = get_question_metadata(question, questions_data)

    if metadata:
        return metadata.get("expected", "")

    return ""


def is_abstention(question, questions_data=None):
    """
    LongMemEval uses '_abs' as the abstention marker.
    """
    # Check for explicit abstention marker
    if question.get("_abs", False):
        return True
    
    # Check expected text
    expected = get_expected_text(question, questions_data)
    if expected and "don't know" in expected.lower():
        return True
    
    # Check if expected IDs contain '_abs'
    expected_ids = question.get("answer_session_ids", [])
    for eid in expected_ids:
        if "_abs" in str(eid):
            return True
    
    return False


# ----------------------------------------------------------------------
# Metadata construction
# ----------------------------------------------------------------------

def build_texts_and_metadata(
    question,
    haystack_session_ids=None,
    questions_data=None,
    q_idx=0,
):
    """
    Extract every non-empty turn and attach deterministic metadata.

    Gold sessions are taken from answer_session_ids.

    Gold turns are identified separately using expected-answer text when
    possible. We do NOT mark every turn in an answer session as a gold turn.
    """

    texts = []
    metadatas = []

    answer_session_ids = set(
        question.get("answer_session_ids", [])
    )

    question_id = question.get(
        "question_id",
        f"q_{q_idx}",
    )

    session_ids = (
        haystack_session_ids
        if haystack_session_ids is not None
        else []
    )

    expected_text = get_expected_text(
        question,
        questions_data,
    )

    if q_idx == 0:
        print(
            f"\n[DEBUG] question_id={question_id}, "
            f"answer_ids={sorted(answer_session_ids)}"
        )
        print(
            f"[DEBUG] haystack_session_ids count: "
            f"{len(session_ids)}"
        )
        print(
            f"[DEBUG] expected_text="
            f"'{expected_text[:80] if expected_text else 'None'}'"
        )
        print(
            f"[DEBUG] Number of sessions: "
            f"{len(question.get('haystack_sessions', []))}"
        )
        print("-" * 60)

    for session_idx, session_data in enumerate(
        question.get("haystack_sessions", [])
    ):
        session_id = None

        # --------------------------------------------------------------
        # Resolve session ID deterministically.
        # --------------------------------------------------------------

        if (
            session_idx < len(session_ids)
            and session_ids[session_idx]
        ):
            session_id = session_ids[session_idx]

        if not session_id and isinstance(session_data, dict):
            session_id = (
                session_data.get("id")
                or session_data.get("session_id")
            )

        if not session_id:
            session_id = (
                f"distractor_{question_id}_{session_idx}"
            )

        if isinstance(session_data, dict):
            turns = session_data.get("turns", [])
        else:
            turns = session_data or []

        # --------------------------------------------------------------
        # Identify answer turns.
        #
        # We deliberately do not make every turn in an answer session
        # relevant. The expected answer should identify the actual turn.
        # If expected_text is unavailable, we fall back to the first turn.
        # --------------------------------------------------------------

        answer_turn_indices = set()

        if (
            session_id in answer_session_ids
            and expected_text
        ):
            for turn_idx, turn in enumerate(turns):
                content = (
                    turn.get("content", "")
                    if isinstance(turn, dict)
                    else str(turn)
                )

                if not content:
                    continue

                if expected_text.strip() in content:
                    answer_turn_indices.add(turn_idx)

        # If the expected text is unavailable, we retain session-level
        # gold but do NOT fabricate turn-level gold.
        has_answer_session = (
            session_id in answer_session_ids
        )

        for turn_idx, turn in enumerate(turns):
            if isinstance(turn, dict):
                content = turn.get("content", "")
                role = turn.get("role", "unknown")
            else:
                content = str(turn)
                role = "unknown"

            if not content or not content.strip():
                continue

            turn_id = f"{session_id}_{turn_idx}"

            texts.append(content)

            # Determine if this is a gold turn
            is_gold = False
            if expected_text:
                # Use the text-matching method
                is_gold = turn_idx in answer_turn_indices
            elif has_answer_session:
                # Fallback: mark the first turn of any answer session as gold
                is_gold = (turn_idx == 0)

            metadatas.append({
                "session_id": session_id,
                "turn_id": turn_id,
                "turn_index": turn_idx,
                "role": role,
                "has_answer_session": has_answer_session,
                "is_gold_turn": is_gold,
            })

    return texts, metadatas


# ----------------------------------------------------------------------
# Retrieval evaluation
# ----------------------------------------------------------------------

def _dcg(relevances):
    """Standard discounted cumulative gain."""

    total = 0.0

    for rank, relevance in enumerate(
        relevances,
        start=1,
    ):
        if relevance:
            total += 1.0 / __import__("math").log2(rank + 1)

    return total


def evaluate_retrieval(
    ranked_results,
    gold_session_ids,
    gold_turn_ids,
    k_values=K_VALUES,
):
    """
    Compute retrieval metrics.

    Session level:
        - recall_any@k
        - recall_all@k
        - ndcg@k

    Turn level:
        - recall_any@k
        - recall_all@k
        - ndcg@k

    Session ranking is de-duplicated by first occurrence.

    Turn ranking preserves individual retrieved turns.
    """

    gold_session_set = set(gold_session_ids)
    gold_turn_set = set(gold_turn_ids)

    # --------------------------------------------------------------
    # Session ranking
    # --------------------------------------------------------------

    ranked_sessions = []
    seen_sessions = set()

    for result in ranked_results:
        metadata = result.get("metadata", {})
        session_id = metadata.get("session_id")

        if (
            session_id
            and session_id not in seen_sessions
        ):
            seen_sessions.add(session_id)
            ranked_sessions.append(session_id)

    # --------------------------------------------------------------
    # Turn ranking
    # --------------------------------------------------------------

    ranked_turns = []

    for result in ranked_results:
        metadata = result.get("metadata", {})
        turn_id = metadata.get("turn_id")

        if turn_id:
            ranked_turns.append(turn_id)

    metrics = {
        "session": {},
        "turn": {},
    }

    for mode, ranked_ids, gold_set in (
        (
            "session",
            ranked_sessions,
            gold_session_set,
        ),
        (
            "turn",
            ranked_turns,
            gold_turn_set,
        ),
    ):
        recall_any = {}
        recall_all = {}
        ndcg = {}

        for k in k_values:
            top_k = ranked_ids[:k]

            # ----------------------------------------------------------
            # Recall any
            # ----------------------------------------------------------

            found_any = any(
                item in gold_set
                for item in top_k
            )

            # ----------------------------------------------------------
            # Recall all
            # ----------------------------------------------------------

            found_all = (
                bool(gold_set)
                and gold_set.issubset(set(top_k))
            )

            # ----------------------------------------------------------
            # NDCG
            # ----------------------------------------------------------

            if not gold_set:
                ndcg_value = None
            else:
                relevances = [
                    1 if item in gold_set else 0
                    for item in top_k
                ]

                dcg = _dcg(relevances)

                ideal_relevances = [
                    1
                    for _ in range(
                        min(k, len(gold_set))
                    )
                ]

                idcg = _dcg(
                    ideal_relevances
                )

                ndcg_value = (
                    dcg / idcg
                    if idcg > 0
                    else 0.0
                )

            recall_any[k] = found_any
            recall_all[k] = found_all
            ndcg[k] = ndcg_value

        metrics[mode] = {
            "recall_any": recall_any,
            "recall_all": recall_all,
            "ndcg_any": ndcg,
        }

    return metrics


# ----------------------------------------------------------------------
# Cache manifests
# ----------------------------------------------------------------------

def get_cache_signature():
    """
    Return settings that materially affect cached retrieval data.
    """

    return {
        "embedding_model": str(
            getattr(
                settings,
                "EMBEDDING_MODEL",
                "",
            )
        ),
        "embedding_dimension": int(
            getattr(
                settings,
                "VECTOR_DIM",
                0,
            )
        ),
    }


def write_cache_manifest(
    cache_dir,
    question_id,
    dataset_hash,
    memory_count,
    vector_count,
):
    """Write a reproducible per-question cache manifest."""

    manifest_path = (
        cache_dir
        / f"{question_id}.manifest.json"
    )

    manifest = {
        "question_id": question_id,
        "dataset_hash": dataset_hash,
        **get_cache_signature(),
        "memory_count": memory_count,
        "vector_count": vector_count,
        "created_at": utc_now(),
    }

    with open(manifest_path, "w") as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    return manifest


def validate_cache_manifest(
    cache_dir,
    question_id,
    dataset_hash,
):
    """Validate cache against dataset and embedding identity."""

    manifest_path = (
        cache_dir
        / f"{question_id}.manifest.json"
    )

    if not manifest_path.exists():
        return False

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        if manifest.get("dataset_hash") != dataset_hash:
            return False

        signature = get_cache_signature()

        if (
            manifest.get("embedding_model")
            != signature["embedding_model"]
        ):
            return False

        if (
            manifest.get("embedding_dimension")
            != signature["embedding_dimension"]
        ):
            return False

        return True

    except Exception:
        return False


# ----------------------------------------------------------------------
# Legacy retrieval check
# ----------------------------------------------------------------------

def check_retrieval(
    response,
    expected_session_ids,
):
    """
    Legacy compatibility check.

    Returns:
        (found, rank)
    """

    expected = set(expected_session_ids)

    results = response.get(
        "results",
        [],
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        metadata = result.get(
            "metadata",
            {},
        )

        if metadata.get("session_id") in expected:
            return True, rank

    return False, None


# ----------------------------------------------------------------------
# Database / index helpers
# ----------------------------------------------------------------------

def clear_db_fast(controller):
    """Fast truncate clear between isolated questions."""

    db = controller.system.db

    if (
        hasattr(db, "conn")
        and hasattr(db.conn, "conn")
    ):
        conn = db.conn.conn
    else:
        conn = getattr(
            db,
            "conn",
            None,
        )

        if conn is None:
            raise AttributeError(
                "No connection found on db object"
            )

    conn.execute(
        "PRAGMA foreign_keys = OFF"
    )

    tables = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        """
    ).fetchall()

    for (table_name,) in tables:
        conn.execute(
            f'DELETE FROM "{table_name}"'
        )

        # sqlite_sequence may not exist.
        try:
            conn.execute(
                "DELETE FROM sqlite_sequence "
                "WHERE name=?",
                (table_name,),
            )
        except Exception:
            pass

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    conn.commit()

    controller.system.vector_store.reset()
    controller.system.embedding_cache.clear()

    debug(
        "[LongMemEval] DB cleared."
    )


def rebuild_indices_from_db(system):
    """
    Rebuild indexes from the currently loaded DB.

    BM25 receives explicit memory IDs so corpus positions cannot be
    confused with database IDs.
    """

    if (
        hasattr(system, "inverted_index")
        and system.inverted_index
    ):
        system.inverted_index.build()

    if (
        hasattr(system, "bm25_ranker")
        and system.bm25_ranker
    ):
        rows = system.db.fetch_all()

        bm25_rows = [
            row
            for row in rows
            if row.get("tokens")
        ]

        if bm25_rows:
            corpus_tokens = [
                row["tokens"]
                for row in bm25_rows
            ]

            memory_ids = [
                row["id"]
                for row in bm25_rows
            ]

            system.bm25_ranker.build(
                corpus_tokens,
                doc_ids=memory_ids,
            )
        else:
            system.bm25_ranker.build(
                [],
                doc_ids=[],
            )


def rebuild_bm25_from_db(system):
    """Build BM25 from current DB (no caching)."""
    if not hasattr(system, "bm25_ranker") or system.bm25_ranker is None:
        return

    rows = system.db.fetch_all()
    bm25_rows = [row for row in rows if row.get("tokens")]

    if bm25_rows:
        corpus_tokens = [row["tokens"] for row in bm25_rows]
        memory_ids = [row["id"] for row in bm25_rows]
        system.bm25_ranker.build(corpus_tokens, doc_ids=memory_ids)
        debug(f"[BM25] Built on {len(corpus_tokens)} memories")
    else:
        system.bm25_ranker.build([], doc_ids=[])


def save_bm25_cache(bm25_ranker, cache_path):
    """Save BM25 index to disk."""
    with open(cache_path, "wb") as f:
        pickle.dump({
            "corpus_tokens": bm25_ranker.corpus,
            "doc_ids": bm25_ranker.doc_ids,
            "idf": bm25_ranker.idf,
            "avg_doc_length": bm25_ranker.avg_doc_length,
            "k1": bm25_ranker.k1,
            "b": bm25_ranker.b,
        }, f)


def load_bm25_cache(bm25_ranker, cache_path):
    """Load BM25 index from disk."""
    if not Path(cache_path).exists():
        return False
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        bm25_ranker.corpus = data["corpus_tokens"]
        bm25_ranker.doc_ids = data["doc_ids"]
        bm25_ranker.idf = data["idf"]
        bm25_ranker.avg_doc_length = data["avg_doc_length"]
        if "k1" in data:
            bm25_ranker.k1 = data["k1"]
        if "b" in data:
            bm25_ranker.b = data["b"]
        return True
    except Exception as e:
        print(f"BM25 cache load error: {e}")
        return False

def build_faiss_from_texts(
    controller,
    texts,
    mem_ids,
):
    """
    Compute embeddings and populate FAISS.
    """

    extractor = controller.system.extractor

    records = [
        extractor.extract(text)
        for text in texts
    ]

    normalized_texts = [
        record.normalized_text
        for record in records
    ]

    vectors = (
        controller.system.embedder
        .embed_many(normalized_texts)
    )

    controller.system.vector_store.add_many(
        mem_ids,
        vectors,
        persist=False,
    )

    controller.system.embedding_cache.add_many(
        mem_ids,
        vectors,
    )

    return vectors


# ----------------------------------------------------------------------
# Ablation configuration
# ----------------------------------------------------------------------

def configure_retrieval_mode(mode):
    """
    Configure retrieval/ranking ablation.

    Important:
    'fusion' is not treated as a retrieval worker. The router should
    still submit the underlying retrieval sources and combine their
    candidates through its normal fusion path.
    """

    if mode == "dense":
        settings.WORKERS_TO_USE = [
            "faiss"
        ]
        settings.RANKING_ENABLED = False

    elif mode == "bm25":
        settings.WORKERS_TO_USE = [
            "bm25"
        ]
        settings.RANKING_ENABLED = False

    elif mode == "raw":
        settings.WORKERS_TO_USE = [
            "faiss",
            "bm25",
            
        ]
        settings.RANKING_ENABLED = False

    elif mode == "fusion":
        # FusionWorker runs FAISS + BM25 internally and applies RRF
        settings.WORKERS_TO_USE = ["fusion"]
        settings.RANKING_ENABLED = False
        # Ensure RRF is enabled via the fusion worker
        if hasattr(settings, "RRF_K"):
            settings.RRF_K = 10  # Optimal value from your tuning
        

    elif mode == "full":
        settings.WORKERS_TO_USE = [
            "faiss",
            "bm25",
            "graph",
            "phrase",
            "attribute",
        ]
        settings.RANKING_ENABLED = True

    else:
        raise ValueError(
            f"Unknown retrieval mode: {mode}"
        )


# ----------------------------------------------------------------------
# Main benchmark
# ----------------------------------------------------------------------

def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="longmemeval_s_cleaned.json",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--skip-embedding",
        action="store_true",
    )

    parser.add_argument(
        "--no-parallel",
        action="store_true",
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run-weights",
        action="store_true",
    )

    parser.add_argument(
        "--step-size",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--cache-dir",
        default="cache/faiss_indices",
    )

    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=str,
    )

    parser.add_argument(
        "--retrieval-mode",
        choices=[
            "dense",
            "bm25",
            "raw",
            "fusion",
            "full",
        ],
        default="fusion",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(
            f"Error: Dataset not found: "
            f"{dataset_path}"
        )
        sys.exit(1)

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    print(
        f"[LongMemEval] Loading dataset: "
        f"{dataset_path}"
    )

    with open(dataset_path, "r") as f:
        questions = json.load(f)

    if args.limit:
        questions = questions[:args.limit]

    dataset_hash = compute_checksum(
        dataset_path
    )

    print(
        f"[LongMemEval] "
        f"{len(questions)} questions loaded"
    )

    print(
        f"[LongMemEval] Dataset checksum: "
        f"{dataset_hash}"
    )

    print(
        f"[LongMemEval] Retrieval mode: "
        f"{args.retrieval_mode}"
    )

    configure_retrieval_mode(
        args.retrieval_mode
    )

    settings.CONTEXT_TOKEN_BUDGET = 10000

    questions_data = load_questions_file()

    # --------------------------------------------------------------
    # Memory system
    # --------------------------------------------------------------

    memory = MemoryInterface()
    loader = BatchLoader(memory)
    controller = memory.controller

    print(
        "Embedder model:",
        controller.system.embedder.model,
    )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Counters
    # --------------------------------------------------------------

    records = []

    retrieval_hits = 0
    total_memories = 0

    abstention_count = 0
    retrieval_evaluable_count = 0

    start_time = time.perf_counter()

    # --------------------------------------------------------------
    # Questions
    # --------------------------------------------------------------

    for q_idx, question in enumerate(
        questions
    ):
        q_id = question.get(
            "question_id",
            f"q_{q_idx}",
        )

        question_text = question.get(
            "question",
            "",
        )

        answer_ids = list(
            question.get(
                "answer_session_ids",
                [],
            )
        )

        # ----------------------------------------------------------
        # Abstention
        # ----------------------------------------------------------

        if is_abstention(
            question,
            questions_data,
        ):
            abstention_count += 1

            print(
                f"\n[Question "
                f"{q_idx + 1}/"
                f"{len(questions)}] "
                f"SKIP (abstention)"
            )

            records.append({
                "query": question_text,
                "expected": get_expected_text(
                    question,
                    questions_data,
                ),
                "expected_ids": answer_ids,
                "expected_rank": None,
                "retrieved": False,
                "candidates": [],
                "runtime_ms": 0.0,
                "diagnostics": {},
                "metrics": None,
                "abstention": True,
            })

            continue

        retrieval_evaluable_count += 1

        print(
            f"\n[Question "
            f"{q_idx + 1}/"
            f"{len(questions)}] "
            f"{question_text[:60]}..."
        )

        # ----------------------------------------------------------
        # Build isolated haystack
        # ----------------------------------------------------------

        haystack_session_ids = question.get(
            "haystack_session_ids",
            [],
        )

        texts, metadatas = (
            build_texts_and_metadata(
                question,
                haystack_session_ids=(
                    haystack_session_ids
                ),
                questions_data=questions_data,
                q_idx=q_idx,
            )
        )

        haystack_mem_count = len(texts)
        total_memories += haystack_mem_count

        print(
            f"    Haystack: "
            f"{haystack_mem_count} memories",
            end="",
            flush=True,
        )

        # ----------------------------------------------------------
        # Cache
        # ----------------------------------------------------------

        db_cache_path = (
            cache_dir
            / f"db_{q_id}.sqlite"
        )

        faiss_cache_path = (
            cache_dir
            / f"faiss_{q_id}.index"
        )

        cache_valid = (
            validate_cache_manifest(
                cache_dir,
                q_id,
                dataset_hash,
            )
        )

        # Ensure BM25 ranker exists before caching
        if not hasattr(controller.system, "bm25_ranker") or controller.system.bm25_ranker is None:
            from ranking.bm25_ranker import BM25
            controller.system.bm25_ranker = BM25()
            print("\n    BM25 ranker created", end="", flush=True)

        db_cache_exists = (
            db_cache_path.exists()
            and not args.rebuild_cache
            and cache_valid
        )

        faiss_cache_exists = (
            faiss_cache_path.exists()
            and not args.rebuild_cache
            and cache_valid
        )

        skip_embedding_build = False
        skip_db_insert = False

        # ---- Load DB cache ----
        if db_cache_exists:
            print(
                f"\n    Loading cached DB "
                f"for {q_id}...",
                end="",
                flush=True,
            )

            try:
                if hasattr(
                    controller.system.db,
                    "_conn",
                ):
                    controller.system.db._conn.close()

                shutil.copy2(
                    str(db_cache_path),
                    settings.DB_PATH,
                )

                from db.connection import DBConnection

                controller.system.db._conn = (
                    DBConnection()
                )

                rebuild_indices_from_db(
                    controller.system
                )

                controller.system.vector_store.reset()

                skip_db_insert = True

                print(
                    " loaded",
                    end="",
                    flush=True,
                )

            except Exception as e:
                print(
                    f" failed ({e}), rebuilding",
                    end="",
                    flush=True,
                )

                skip_db_insert = False

                # Any FAISS index loaded without its matching cached DB
                # is unsafe because memory IDs may differ.
                controller.system.vector_store.reset()
                controller.system.embedding_cache.clear()
                skip_embedding_build = False
                faiss_cache_exists = False

        # ---- BM25 Cache ----
        bm25_cache_path = cache_dir / f"bm25_{q_id}.pkl"
        skip_bm25_build = False

        # Load BM25 cache if DB was loaded from cache and BM25 cache exists
        if skip_db_insert and bm25_cache_path.exists() and not args.rebuild_cache and cache_valid:
            if load_bm25_cache(controller.system.bm25_ranker, bm25_cache_path):
                skip_bm25_build = True
                print("\n    BM25 cache loaded", end="", flush=True)
            else:
                # Cache exists but failed to load — rebuild
                rebuild_bm25_from_db(controller.system)
                save_bm25_cache(controller.system.bm25_ranker, bm25_cache_path)
                print("\n    BM25 cache rebuilt (load failed)", end="", flush=True)
        elif skip_db_insert and not bm25_cache_path.exists():
            # No cache exists, build from loaded DB
            rebuild_bm25_from_db(controller.system)
            save_bm25_cache(controller.system.bm25_ranker, bm25_cache_path)
            print("\n    BM25 cache built (first time)", end="", flush=True)

        # ----------------------------------------------------------
        # Load FAISS
        # ----------------------------------------------------------

        if faiss_cache_exists:
            print(
                f"\n    Loading cached FAISS "
                f"for {q_id}...",
                end="",
                flush=True,
            )

            try:
                controller.system.vector_store.load_from_file(
                    str(faiss_cache_path)
                )

                skip_embedding_build = True

                print(
                    " loaded",
                    end="",
                    flush=True,
                )

            except Exception as e:
                print(
                    f" failed ({e}), rebuilding",
                    end="",
                    flush=True,
                )

                skip_embedding_build = False

        # ----------------------------------------------------------
        # Build FAISS if DB exists but FAISS doesn't
        # ----------------------------------------------------------

        faiss_built = False

        if (
            not args.skip_embedding
            and not faiss_cache_exists
            and skip_db_insert
        ):
            print(
                f"\n    Building FAISS "
                f"for {q_id}...",
                end="",
                flush=True,
            )

            try:
                rows = (
                    controller.system.db
                    .fetch_all()
                )

                mem_ids = [
                    row["id"]
                    for row in rows
                ]

                if len(mem_ids) == len(texts):
                    build_faiss_from_texts(
                        controller,
                        texts,
                        mem_ids,
                    )

                    controller.system.vector_store.save_to_file(
                        str(faiss_cache_path)
                    )

                    write_cache_manifest(
                        cache_dir,
                        q_id,
                        dataset_hash,
                        len(mem_ids),
                        len(texts),
                    )

                    faiss_built = True
                    skip_embedding_build = True

                    print(
                        " built and saved",
                        end="",
                        flush=True,
                    )

                else:
                    print(
                        f" ID count mismatch: "
                        f"{len(mem_ids)} vs "
                        f"{len(texts)}, "
                        f"rebuilding DB+FAISS",
                        end="",
                        flush=True,
                    )

                    skip_db_insert = False

            except Exception as e:
                print(
                    f" failed ({e}), "
                    f"will rebuild DB+FAISS",
                    end="",
                    flush=True,
                )

                skip_db_insert = False

        # ----------------------------------------------------------
        # Store
        # ----------------------------------------------------------

        store_time = 0.0
        count = 0

        if not skip_db_insert:
            store_start = time.perf_counter()

            count = loader.insert_batch(
                texts,
                metadatas=metadatas,
                batch_size=args.batch_size,
                skip_embedding=args.skip_embedding,
                parallel_extract=(
                    not args.no_parallel
                ),
                max_workers=args.workers,
                skip_embedding_build=(
                    skip_embedding_build
                ),
            )

            store_time = (
                time.perf_counter()
                - store_start
            )

            # Build and save BM25 cache from newly inserted data
            if not args.rebuild_cache and controller.system.bm25_ranker:
                rebuild_bm25_from_db(controller.system)
                save_bm25_cache(controller.system.bm25_ranker, bm25_cache_path)
                print("\n    BM25 cache built from insert", end="", flush=True)

            # ------------------------------------------------------
            # Save DB cache
            # ------------------------------------------------------

            if not args.rebuild_cache:
                print(
                    f"\n    Saving DB cache "
                    f"for {q_id}...",
                    end="",
                    flush=True,
                )

                try:
                    shutil.copy2(
                        settings.DB_PATH,
                        str(db_cache_path),
                    )

                    print(
                        " saved",
                        end="",
                        flush=True,
                    )

                except Exception as e:
                    print(
                        f" failed ({e})",
                        end="",
                        flush=True,
                    )

                # --------------------------------------------------
                # Save FAISS cache
                # --------------------------------------------------

                if (
                    not args.skip_embedding
                    and not skip_embedding_build
                ):
                    print(
                        f"\n    Saving FAISS cache "
                        f"for {q_id}...",
                        end="",
                        flush=True,
                    )

                    try:
                        controller.system.vector_store.save_to_file(
                            str(faiss_cache_path)
                        )

                        write_cache_manifest(
                            cache_dir,
                            q_id,
                            dataset_hash,
                            count,
                            len(texts),
                        )

                        print(
                            " saved",
                            end="",
                            flush=True,
                        )

                    except Exception as e:
                        print(
                            f" failed ({e})",
                            end="",
                            flush=True,
                        )

        else:
            count = haystack_mem_count

            if (
                not faiss_cache_exists
                and not faiss_built
                and not args.skip_embedding
            ):
                print(
                    "\n    FAISS missing and "
                    "could not build; "
                    "consider --rebuild-cache"
                )

        # ----------------------------------------------------------
        # Query
        # ----------------------------------------------------------

        query_start = time.perf_counter()

        response = controller.recall(
            question_text
        )

        query_time = (
            time.perf_counter()
            - query_start
        )

        raw_candidates = response.get(
            "results",
            [],
        )

        if raw_candidates:
            first_meta = (
                raw_candidates[0]
                .get("metadata", {})
            )

            print(
                f"\n    [DEBUG] First "
                f"candidate metadata: "
                f"{first_meta}"
            )

            print(
                f"    [DEBUG] Expected "
                f"session IDs: "
                f"{answer_ids}"
            )

        else:
            print(
                "\n    [DEBUG] "
                "No candidates returned!"
            )

        # ----------------------------------------------------------
        # Legacy session hit
        # ----------------------------------------------------------

        found, rank = check_retrieval(
            response,
            answer_ids,
        )

        if found:
            retrieval_hits += 1

        # ----------------------------------------------------------
        # Gold turn IDs
        #
        # IMPORTANT:
        # Only explicitly matched answer turns are gold.
        # We do not use "all turns in answer session".
        # ----------------------------------------------------------

        gold_turn_ids = set()

        for metadata in metadatas:
            if metadata.get(
                "is_gold_turn",
                False,
            ):
                turn_id = metadata.get(
                    "turn_id"
                )

                if turn_id:
                    gold_turn_ids.add(
                        turn_id
                    )

        metrics = evaluate_retrieval(
            raw_candidates,
            answer_ids,
            gold_turn_ids,
            K_VALUES,
        )

        # ----------------------------------------------------------
        # Record
        # ----------------------------------------------------------

        records.append({
            "query": question_text,
            "expected": get_expected_text(
                question,
                questions_data,
            ),
            "expected_ids": answer_ids,
            "expected_rank": rank,
            "retrieved": found,
            "candidates": raw_candidates,
            "runtime_ms": query_time * 1000,
            "diagnostics": response.get(
                "diagnostics",
                {},
            ),
            "metrics": metrics,
            "abstention": False,
        })

        print(
            f", inserted {count} "
            f"in {store_time:.2f}s, "
            f"queried in "
            f"{query_time:.2f}s",
            end="",
            flush=True,
        )

        # ----------------------------------------------------------
        # Isolation cleanup
        # ----------------------------------------------------------

        if not skip_db_insert:
            clear_db_fast(
                controller
            )
        else:
            controller.system.vector_store.reset()
            controller.system.embedding_cache.clear()

            debug(
                "[LongMemEval] DB cache used, "
                "skipping clear."
            )

        print(
            ", cleared"
        )

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    commit = get_git_commit()

    settings_snapshot = {
        key: str(value)
        for key, value in vars(settings).items()
        if (
            not key.startswith("_")
            and not callable(value)
        )
    }

    output = {
        "benchmark": "LongMemEval-S",
        "question_count": len(questions),
        "retrieval_evaluable": (
            retrieval_evaluable_count
        ),
        "abstentions_excluded": (
            abstention_count
        ),
        "records": records,
        "dataset_checksum": dataset_hash,
        "embedder": settings.EMBEDDING_MODEL,
        "embedding_dimension": settings.VECTOR_DIM,
        "commit": commit,
        "retrieval_mode": args.retrieval_mode,
        "settings": settings_snapshot,
        "command": " ".join(sys.argv),
        "elapsed_seconds": elapsed,
    }

    # ------------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------------

    if args.output:
        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    else:
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        output_dir = Path(
            "benchmark_output/results"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / f"longmemeval_{timestamp}.json"
        )

    with open(output_path, "w") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Reproducibility manifest
    # ------------------------------------------------------------------

    manifest_path = (
        output_path.with_suffix(
            ".manifest.json"
        )
    )

    manifest = {
        "benchmark": "LongMemEval-S",

        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_hash,
        },

        "run": {
            "started": None,
            "finished": utc_now(),
            "command": " ".join(sys.argv),
            "arguments": vars(args),
            "git_commit": commit,
        },

        "system": {
            "embedder": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.VECTOR_DIM,
            "chat_model": settings.CHAT_MODEL,
            "vector_store": (
                type(
                    controller.system.vector_store
                ).__name__
            ),
            "database": "sqlite",
        },

        "retrieval": {
            "mode": args.retrieval_mode,
            "sources": list(
                getattr(
                    settings,
                    "WORKERS_TO_USE",
                    [],
                )
            ),
            "candidate_limit": getattr(
                settings,
                "TOP_K",
                None,
            ),
        },

        "ranking": {
            "enabled": getattr(
                settings,
                "RANKING_ENABLED",
                False,
            ),
            "mmr": getattr(
                settings,
                "MMR_ENABLED",
                False,
            ),
        },

        "cache": {
            "enabled": True,
            "rebuild": args.rebuild_cache,
            "directory": str(cache_dir),
            "manifest": "per-question",
            "signature": get_cache_signature(),
        },

        "evaluation": {
            "k_values": list(K_VALUES),
            "abstentions_excluded": (
                abstention_count
            ),
            "retrieval_evaluable": (
                retrieval_evaluable_count
            ),
            "metrics": [
                "session.recall_any",
                "session.recall_all",
                "session.ndcg",
                "turn.recall_any",
                "turn.recall_all",
                "turn.ndcg",
            ],
        },

        "settings_snapshot": settings_snapshot,
    }

    with open(manifest_path, "w") as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "[LongMemEval] Benchmark Complete"
    )

    print(
        "=" * 60
    )

    print(
        f"Total questions: "
        f"{len(questions)}"
    )

    print(
        f"Abstentions excluded: "
        f"{abstention_count}"
    )

    print(
        f"Retrieval evaluable: "
        f"{retrieval_evaluable_count}"
    )

    print(
        f"Total memories: "
        f"{total_memories}"
    )

    if retrieval_evaluable_count:
        legacy_rate = (
            retrieval_hits
            / retrieval_evaluable_count
            * 100
        )

        print(
            f"Retrieval hits "
            f"(legacy any): "
            f"{retrieval_hits}/"
            f"{retrieval_evaluable_count} "
            f"({legacy_rate:.2f}%)"
        )

    print(
        f"Total time: "
        f"{elapsed:.2f}s"
    )

    if len(questions) > 0:
        print(
            f"Avg time per question: "
            f"{elapsed / len(questions):.2f}s"
        )

    print(
        "=" * 60
    )

    print(
        f"\nResults saved to: "
        f"{output_path}"
    )

    print(
        f"Manifest saved to: "
        f"{manifest_path}"
    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    if args.optimize:
        print(
            "\n[Optimizer] "
            "Running adaptive weight adjustment..."
        )

        result = adaptive_weighter_pipeline(
            benchmark_file=str(
                output_path
            ),
            dry_run=args.dry_run_weights,
            step_size=args.step_size,
        )

        if result:
            print(
                "[Optimizer] "
                "Weight adjustment complete."
            )
        else:
            print(
                "[Optimizer] "
                "No adjustment made "
                "(no deltas or error)."
            )


if __name__ == "__main__":
    main()
