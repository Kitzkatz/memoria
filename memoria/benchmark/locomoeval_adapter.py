#!/usr/bin/env python3
"""
LoCoMo-S adapter — stores conversations, queries, and evaluates retrieval.

Per-conversation isolation:
    store conversation -> query all questions -> clear

Supports:
    - SQLite DB caching
    - FAISS caching
    - BM25 caching
    - automatic cache versioning
    - batched ingestion
    - optional parallel extraction
    - temporal-worker routing

Temporal path:

    query
      ↓
    temporal detection
      ↓
    standalone temporal retrieval
      ↓
    temporal-scored candidates
      ↓
    downstream ranking / response

The adapter does not perform ranking itself.
"""

import hashlib
import json
import pickle
import shutil
import sqlite3
import sys
import time
import re

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


sys.path.insert(0, str(Path(__file__).parent.parent))


from core.logger import debug, info
from shared.memory_interface import MemoryInterface
from benchmark.batch_loader import BatchLoader
from benchmark.result_formatter import (
    build_record,
    build_output,
    write_output,
)
from cache.config import settings
from ranking.adaptive_weighter import adaptive_weighter_pipeline


# Increment only when cached DB/index structures become incompatible.
CACHE_VERSION = 1

# ================================================================
# TOGGLE EVIDENCE CORRECTION
# ================================================================

APPLY_EVIDENCE_CORRECTIONS = False   # Set to True to enable broken‑question fixing


# ================================================================
# DATASET
# ================================================================

def compute_checksum(filepath: Path) -> str:
    sha256 = hashlib.sha256()

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_locomo_dataset(input_path: Path) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_haystack_texts_and_metadata(
    entry: Dict[str, Any],
):
    """
    Extract every conversation turn while preserving:

        dia_id
        session_idx
        turn_idx
        session_key

    Session/turn metadata is retained so temporal components can use
    conversation ordering independently of creation timestamps.
    """

    texts = []
    metadatas = []

    conversation = entry.get("conversation", {})

    if not conversation:
        return texts, metadatas, []

    session_keys = sorted(
        key
        for key in conversation.keys()
        if key.startswith("session_")
    )

    for session_idx, session_key in enumerate(session_keys):

        turns = conversation.get(session_key, [])

        if not isinstance(turns, list):
            continue

        for turn_idx, turn in enumerate(turns):

            if not isinstance(turn, dict):
                continue

            dia_id = turn.get("dia_id", "")
            text = turn.get("text", "")

            if not text or not text.strip():
                continue

            texts.append(text)

            metadatas.append(
                {
                    "dia_id": dia_id,
                    "session_idx": session_idx,
                    "turn_idx": turn_idx,
                    "session_key": session_key,
                }
            )

    return texts, metadatas, session_keys


# ================================================================
# TEMPORAL DETECTION
# ================================================================

_TEMPORAL_KEYWORDS = (
    "before",
    "after",
    "during",
    "between",
    "since",
    "until",
    "most recent",
    "last",
    "first",
    "previous",
    "next",
    "today",
    "yesterday",
    "tomorrow",
    "last week",
    "last month",
    "ago",
    "from now",
    "in the past",
    "over the last",
    "earlier",
    "later",
    "recent",
    "recently",
    "currently",
    "previously",
    "how long",
    "session",
    "sessions",
    "conv",
    "conversation",
    "turn",
)

# ================================================================
# ABSTENTION / ADVERSARIAL DETECTION
# ================================================================

def is_category_5_abstention(question: Dict[str, Any]) -> bool:
    """
    Detect LoCoMo Category 5 adversarial questions.

    Category 5 questions are unanswerable by design. The official
    evaluation excludes them entirely.

    Returns True if this question should be skipped.
    """
    # Primary: check category field
    if question.get("category") == 5:
        return True

    # Fallback: check for _abs marker (if present)
    if question.get("_abs", False):
        return True

    # Fallback: check answer text for abstention markers
    answer = question.get("answer", "")
    if isinstance(answer, str) and "don't know" in answer.lower():
        return True

    return False


def is_temporal_query(query_text: str) -> bool:
    """
    Lightweight routing detector.

    This intentionally does NOT attempt to resolve temporal meaning.
    TemporalParser remains the authority for actual temporal constraints.

    The detector only decides whether the temporal worker should be
    included in the retrieval pipeline.
    """

    if not query_text:
        return False

    query_lower = query_text.lower()

    return any(
        keyword in query_lower
        for keyword in _TEMPORAL_KEYWORDS
    )


# ================================================================
# REFERENCE TIME
# ================================================================

def resolve_reference_time(
    question: Dict[str, Any],
) -> datetime:
    """
    Resolve a deterministic temporal anchor for benchmark queries.

    Priority:

        reference_time
        query_time
        conversation_time
        current UTC time

    LoCoMo adapters may supply benchmark-specific timestamps in any
    of the first three fields.
    """

    value = (
        question.get("reference_time")
        or question.get("query_time")
        or question.get("conversation_time")
    )

    if value is None:
        return datetime.now(timezone.utc)

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):

        try:
            parsed = datetime.fromisoformat(value)

            if parsed.tzinfo is None:
                parsed = parsed.replace(
                    tzinfo=timezone.utc
                )

            return parsed

        except ValueError:
            debug(
                f"[LoCoMo] Invalid reference time: {value!r}"
            )

    return datetime.now(timezone.utc)


# ================================================================
# RETRIEVAL EVALUATION
# ================================================================

def check_retrieval(
    response: Dict[str, Any],
    expected_evidence_ids: List[str],
    expected_answer_text: str = "",
):
    """
    Determine whether expected evidence was retrieved.

    Primary signal:
        dia_id ∈ expected evidence IDs

    Fallback:
        expected answer text appears in retrieved text.
    """

    results = response.get("results", [])

    # ------------------------------------------------------------
    # Evidence ID match
    # ------------------------------------------------------------

    if expected_evidence_ids:

        expected = set(expected_evidence_ids)

        for rank, result in enumerate(
            results,
            start=1,
        ):

            metadata = result.get(
                "metadata",
                {},
            )

            dia_id = metadata.get("dia_id")

            if dia_id and dia_id in expected:
                return True, rank

    # ------------------------------------------------------------
    # Text fallback
    # ------------------------------------------------------------

    if expected_answer_text:

        expected = str(
            expected_answer_text
        ).strip().lower()

        if expected:

            for rank, result in enumerate(
                results,
                start=1,
            ):

                text = str(
                    result.get("text", "")
                ).lower()

                if expected in text:
                    return True, rank

    return False, None


# ================================================================
# DATABASE RESET
# ================================================================

def clear_db_fast(controller):
    """
    Fast database reset between isolated LoCoMo conversations.
    """

    db = controller.system.db

    if hasattr(db, "conn") and hasattr(
        db.conn,
        "conn",
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

        # sqlite_sequence does not exist for every database schema.
        try:
            conn.execute(
                """
                DELETE FROM sqlite_sequence
                WHERE name=?
                """,
                (table_name,),
            )
        except sqlite3.OperationalError:
            pass

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    conn.commit()

    controller.system.vector_store.reset()
    controller.system.embedding_cache.clear()

    debug("[LoCoMo] DB cleared.")


# ================================================================
# INDEX REBUILD
# ================================================================

def rebuild_indices_from_db(system):
    """
    Rebuild secondary indexes from the currently loaded DB.

    BM25 receives explicit memory IDs so corpus positions cannot be
    confused with database IDs.
    """

    if (
        hasattr(system, "inverted_index")
        and system.inverted_index
    ):
        system.inverted_index.build()

    rebuild_bm25_from_db(system)


def rebuild_bm25_from_db(system):
    """
    Build BM25 directly from the active DB.
    """

    if (
        not hasattr(system, "bm25_ranker")
        or system.bm25_ranker is None
    ):
        return

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

        debug(
            f"[BM25] Built on "
            f"{len(corpus_tokens)} memories"
        )

    else:

        system.bm25_ranker.build(
            [],
            doc_ids=[],
        )


# ================================================================
# BM25 CACHE
# ================================================================

def save_bm25_cache(
    bm25_ranker,
    cache_path: Path,
):
    """
    Persist the BM25 state required to restore the current corpus.
    """

    with open(cache_path, "wb") as f:

        pickle.dump(
            {
                "corpus_tokens": bm25_ranker.corpus,
                "doc_ids": bm25_ranker.doc_ids,
                "idf": bm25_ranker.idf,
                "avg_doc_length": bm25_ranker.avg_doc_length,
                "k1": bm25_ranker.k1,
                "b": bm25_ranker.b,
            },
            f,
        )


def load_bm25_cache(
    bm25_ranker,
    cache_path: Path,
) -> bool:

    if not cache_path.exists():
        return False

    try:

        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        bm25_ranker.corpus = data[
            "corpus_tokens"
        ]

        bm25_ranker.doc_ids = data[
            "doc_ids"
        ]

        bm25_ranker.idf = data[
            "idf"
        ]

        bm25_ranker.avg_doc_length = data[
            "avg_doc_length"
        ]

        if "k1" in data:
            bm25_ranker.k1 = data["k1"]

        if "b" in data:
            bm25_ranker.b = data["b"]

        return True

    except Exception as exc:

        print(
            f"BM25 cache load error: {exc}"
        )

        return False


# ================================================================
# CACHE VERSION
# ================================================================

def get_cached_version(
    db_path: Path,
) -> int:

    try:

        conn = sqlite3.connect(
            str(db_path)
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT value
            FROM cache_meta
            WHERE key='version'
            """
        )

        row = cursor.fetchone()

        conn.close()

        return (
            int(row[0])
            if row
            else 0
        )

    except Exception:
        return 0


def set_cached_version(
    db_path: Path,
    version: int,
):
    try:

        conn = sqlite3.connect(
            str(db_path)
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )

        cursor.execute(
            """
            INSERT OR REPLACE INTO cache_meta
                (key, value)
            VALUES
                ('version', ?)
            """,
            (str(version),),
        )

        conn.commit()
        conn.close()

    except Exception as exc:

        debug(
            f"[LoCoMo] Failed to set cache version: "
            f"{exc}"
        )


# ================================================================
# TEMPORAL ROUTING
# ================================================================

def configure_workers_for_query(
    question_text: str,
    original_workers,
):
    """
    Return the worker set appropriate for this query.

    Temporal detection happens here for logging, but actual routing
    happens in _query.py based on _detect_temporal().
    """

    temporal_enabled = getattr(
        settings,
        "USE_TEMPORAL_WORKER",
        False,
    )

    if not temporal_enabled:
        return list(original_workers), False

    if not is_temporal_query(
        question_text
    ):
        return list(original_workers), False

    # Return the flag for logging - routing happens in _query.py
    workers = list(original_workers)

    return workers, True


# ================================================================
# RECORD BUILDER WITH CATEGORY FIELDS
# ================================================================

def build_record_with_metadata(
    query: str,
    expected: str,
    expected_ids: List[str],
    expected_rank: Optional[int],
    retrieved: bool,
    candidates: List[Dict],
    runtime_ms: float,
    diagnostics: Dict,
    category: Optional[int] = None,
    category_name: Optional[str] = None,
    question_id: Optional[str] = None,
    is_adversarial: bool = False,
    is_broken: bool = False,
    excluded_reason: Optional[str] = None,
) -> Dict:
    """
    Build a record with all metadata fields including category and exclusion info.
    """
    record = {
        "query": query,
        "expected": expected,
        "expected_ids": expected_ids,
        "expected_rank": expected_rank,
        "retrieved": retrieved,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "runtime_ms": runtime_ms,
        "diagnostics": diagnostics,
        "category": category,
        "category_name": category_name,
        "question_id": question_id,
        "is_adversarial": is_adversarial,
        "is_broken": is_broken,
        "excluded_reason": excluded_reason,
    }
    return record


# ================================================================
# MAIN
# ================================================================

def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="locomo10.json",
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

    # ---- NEW: category filter ----
    parser.add_argument(
        "--category",
        type=int,
        default=None,
        help="Filter questions by category (1-5). If not set, all categories are run.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    dataset_path = Path(
        args.dataset
    )

    if not dataset_path.exists():

        print(
            f"Error: Dataset not found: "
            f"{dataset_path}"
        )

        sys.exit(1)

    print(
        f"[LoCoMo] Loading dataset: "
        f"{dataset_path}"
    )

    data = load_locomo_dataset(
        dataset_path
    )

    if args.limit:
        data = data[:args.limit]

    print(
        f"[LoCoMo] {len(data)} "
        f"conversations loaded"
    )

    print(
        "[LoCoMo] Dataset checksum: "
        f"{compute_checksum(dataset_path)}"
    )

    # ============================================================
    # LOAD EVIDENCE CORRECTIONS FROM ERRORS.JSON
    # ============================================================

    errors_path = Path("errors.json")
    evidence_corrections = {}  # cited_evidence -> correct_evidence
    broken_question_ids = set()
    if APPLY_EVIDENCE_CORRECTIONS and errors_path.exists():
        try:
            with open(errors_path, "r") as f:
                errors_data = json.load(f)
                for error in errors_data:
                    cited = error.get("cited_evidence", [])
                    correct = error.get("correct_evidence", [])
                    qid = error.get("question_id")

                    # ONLY track questions where cited != correct (real broken)
                    if cited != correct and qid:
                        broken_question_ids.add(qid)

                        # Map each cited evidence to correct evidence for this question
                        # Handle cases where cited is a string with semicolon
                        if isinstance(cited, str) and ';' in cited:
                            cited = [c.strip() for c in cited.split(';')]
                        if isinstance(correct, str) and ';' in correct:
                            correct = [c.strip() for c in correct.split(';')]

                        # Ensure both are lists
                        if not isinstance(cited, list):
                            cited = [cited]
                        if not isinstance(correct, list):
                            correct = [correct]

                        # Map each cited to the corresponding correct
                        for i, ev in enumerate(cited):
                            if i < len(correct):
                                evidence_corrections[ev] = correct[i]
                            elif correct:
                                evidence_corrections[ev] = correct[0]

            print(f"[LoCoMo] Loaded {len(evidence_corrections)} evidence corrections from errors.json")
            print(f"[LoCoMo] {len(broken_question_ids)} real broken questions identified")
            print(f"[LoCoMo] Sample corrections: {list(evidence_corrections.items())[:10]}")
        except Exception as e:
            print(f"[LoCoMo] Warning: Could not load errors.json: {e}")
    elif APPLY_EVIDENCE_CORRECTIONS:
        print("[LoCoMo] Warning: errors.json not found — no evidence corrections applied")
    else:
        print("[LoCoMo] Evidence corrections disabled by APPLY_EVIDENCE_CORRECTIONS")

    # ============================================================

    # Large benchmark context budget.
    settings.CONTEXT_TOKEN_BUDGET = 10000

    memory = MemoryInterface()
    loader = BatchLoader(memory)
    controller = memory.controller

    print(
        "Embedder model:",
        controller.system.embedder.model,
    )

    temporal_enabled = getattr(
        settings,
        "USE_TEMPORAL_WORKER",
        False,
    )

    print(
        "[LoCoMo] Temporal worker:",
        "ENABLED (standalone)"
        if temporal_enabled
        else "DISABLED",
    )

    original_workers = list(
        getattr(
            settings,
            "WORKERS_TO_USE",
            ["fusion"],
        )
    )

    print(
        "[LoCoMo] Base workers:",
        original_workers,
    )

    # ------------------------------------------------------------
    # Cache directory
    # ------------------------------------------------------------

    cache_dir = Path(
        args.cache_dir
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Benchmark state
    # ------------------------------------------------------------

    records = []

    total_questions = 0
    retrieval_hits = 0
    total_memories = 0
    temporal_questions = 0
    adversarial_skipped = 0
    corrected_count = 0

    start_time = time.perf_counter()

    # ============================================================
    # CONVERSATIONS
    # ============================================================

    for conv_idx, entry in enumerate(data):

        conversation_id = entry.get(
            "id",
            f"conv_{conv_idx}",
        )

        print(
            f"\n[Conversation "
            f"{conv_idx + 1}/{len(data)}] "
            f"{conversation_id}"
        )

        texts, metadatas, session_keys = (
            build_haystack_texts_and_metadata(
                entry
            )
        )

        total_sessions = len(session_keys)
        haystack_mem_count = len(texts)

        total_memories += (
            haystack_mem_count
        )

        print(
            f"    Haystack: "
            f"{haystack_mem_count} memories, "
            f"{total_sessions} sessions"
        )

        # --------------------------------------------------------
        # Cache paths
        # --------------------------------------------------------

        db_cache_path = (
            cache_dir
            / f"db_{conversation_id}.sqlite"
        )

        faiss_cache_path = (
            cache_dir
            / f"faiss_{conversation_id}.index"
        )

        bm25_cache_path = (
            cache_dir
            / f"bm25_{conversation_id}.pkl"
        )

        rebuild = args.rebuild_cache

        if (
            db_cache_path.exists()
            and not rebuild
        ):

            cached_version = (
                get_cached_version(
                    db_cache_path
                )
            )

            if cached_version != CACHE_VERSION:

                print(
                    "    Cache version mismatch "
                    f"({cached_version} != "
                    f"{CACHE_VERSION}), "
                    "rebuilding..."
                )

                rebuild = True

                db_cache_path.unlink(
                    missing_ok=True
                )

                faiss_cache_path.unlink(
                    missing_ok=True
                )

                bm25_cache_path.unlink(
                    missing_ok=True
                )

        db_cache_exists = (
            db_cache_path.exists()
            and not rebuild
        )

        faiss_cache_exists = (
            faiss_cache_path.exists()
            and not rebuild
        )

        bm25_cache_exists = (
            bm25_cache_path.exists()
            and not rebuild
        )

        skip_embedding_build = False
        skip_db_insert = False

        store_time = 0.0
        count = 0

        # ========================================================
        # LOAD DB CACHE (CORRECT CONNECTION HANDLING)
        # ========================================================

        if db_cache_exists:

            print(
                f"    Loading cached DB for "
                f"{conversation_id}...",
                end="",
                flush=True,
            )

            try:
                db = controller.system.db

                # Close existing connection if possible (direct _conn)
                if hasattr(db, '_conn') and db._conn is not None:
                    db._conn.close()

                # Copy the cache file
                shutil.copy2(
                    str(db_cache_path),
                    settings.DB_PATH,
                )

                # Reconnect using DBConnection (as in LongMemEval)
                from db.connection import DBConnection
                db._conn = DBConnection()

                # Rebuild indices and vector store
                rebuild_indices_from_db(controller.system)

                # Rebuild temporal index (if present)
                if hasattr(controller.system, 'temporal_index'):
                    controller.system.temporal_index.build()

                controller.system.vector_store.reset()
                skip_db_insert = True

                print(
                    " loaded",
                    end="",
                    flush=True,
                )

            except Exception as exc:

                print(
                    f" failed ({exc}), "
                    "rebuilding",
                    end="",
                    flush=True,
                )

                skip_db_insert = False

        # ========================================================
        # LOAD BM25 CACHE
        # ========================================================

        if (
            bm25_cache_exists
            and skip_db_insert
            and not args.rebuild_cache
        ):

            if load_bm25_cache(
                controller.system.bm25_ranker,
                bm25_cache_path,
            ):

                print(
                    "\n    BM25 cache loaded",
                    end="",
                    flush=True,
                )

            else:

                rebuild_bm25_from_db(
                    controller.system
                )

                save_bm25_cache(
                    controller.system.bm25_ranker,
                    bm25_cache_path,
                )

                print(
                    "\n    BM25 cache rebuilt "
                    "(load failed)",
                    end="",
                    flush=True,
                )

        elif (
            skip_db_insert
            and not bm25_cache_exists
        ):

            rebuild_bm25_from_db(
                controller.system
            )

            save_bm25_cache(
                controller.system.bm25_ranker,
                bm25_cache_path,
            )

            print(
                "\n    BM25 cache built "
                "(first time)",
                end="",
                flush=True,
            )

        # ========================================================
        # LOAD FAISS CACHE
        # ========================================================

        if (
            faiss_cache_exists
            and not args.skip_embedding
            and skip_db_insert
        ):

            print(
                f"\n    Loading cached FAISS "
                f"for {conversation_id}...",
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

            except Exception as exc:

                print(
                    f" failed ({exc}), "
                    "rebuilding",
                    end="",
                    flush=True,
                )

                skip_embedding_build = False

        # ========================================================
        # STORE
        # ========================================================

        if not skip_db_insert:

            store_start = time.perf_counter()

            count = loader.insert_batch(
                texts,
                metadatas=metadatas,
                batch_size=args.batch_size,
                skip_embedding=args.skip_embedding,
                parallel_extract=not args.no_parallel,
                max_workers=args.workers,
                skip_embedding_build=skip_embedding_build,
            )

            store_time = (
                time.perf_counter()
                - store_start
            )

            # ----------------------------------------------------
            # Save caches
            # ----------------------------------------------------

            if not args.rebuild_cache:

                print(
                    f"\n    Saving DB cache for "
                    f"{conversation_id}...",
                    end="",
                    flush=True,
                )

                try:

                    shutil.copy2(
                        settings.DB_PATH,
                        str(db_cache_path),
                    )

                    set_cached_version(
                        db_cache_path,
                        CACHE_VERSION,
                    )

                    print(
                        " saved",
                        end="",
                        flush=True,
                    )

                except Exception as exc:

                    print(
                        f" failed ({exc})",
                        end="",
                        flush=True,
                    )

                # ------------------------------------------------
                # FAISS
                # ------------------------------------------------

                if (
                    not skip_embedding_build
                    and not args.skip_embedding
                ):

                    print(
                        "    Saving FAISS cache...",
                        end="",
                        flush=True,
                    )

                    try:

                        controller.system.vector_store.save_to_file(
                            str(faiss_cache_path)
                        )

                        print(
                            " saved",
                            end="",
                            flush=True,
                        )

                    except Exception as exc:

                        print(
                            f" failed ({exc})",
                            end="",
                            flush=True,
                        )

                # ------------------------------------------------
                # BM25
                # ------------------------------------------------

                if controller.system.bm25_ranker:

                    rebuild_bm25_from_db(
                        controller.system
                    )

                    save_bm25_cache(
                        controller.system.bm25_ranker,
                        bm25_cache_path,
                    )

                    print(
                        "\n    BM25 cache saved",
                        end="",
                        flush=True,
                    )

        else:

            count = haystack_mem_count
            store_time = 0.0

        # ========================================================
        # QUESTIONS
        # ========================================================

        questions = entry.get(
            "qa",
            [],
        )

        if not questions:

            print(
                "    No questions found, "
                "skipping."
            )

            if not skip_db_insert:
                clear_db_fast(controller)

            continue

        print(
            f"    Processing "
            f"{len(questions)} questions...",
            end="",
            flush=True,
        )

        # ========================================================
        # QUESTION LOOP
        # ========================================================

        for qi, q_item in enumerate(
            questions
        ):

            question_text = q_item.get(
                "question",
                "",
            )

            evidence_ids = q_item.get(
                "evidence",
                [],
            )

            answer_text = q_item.get(
                "answer",
                "",
            )

            question_id = q_item.get(
                "question_id",
                f"q_{qi}"
            )

            category = q_item.get("category")
            category_name = q_item.get("category_name")

            # ---- Category filter ----
            if args.category is not None and category != args.category:
                continue

            if not question_text:
                continue

            # ---- SKIP CATEGORY 5 ADVERSARIAL QUESTIONS ----
            if is_category_5_abstention(q_item):
                adversarial_skipped += 1
                # Add record with exclusion metadata
                record = build_record_with_metadata(
                    query=question_text,
                    expected=evidence_ids[0] if evidence_ids else "",
                    expected_ids=evidence_ids,
                    expected_rank=None,
                    retrieved=False,
                    candidates=[],
                    runtime_ms=0.0,
                    diagnostics={"skipped": "adversarial"},
                    category=category,
                    category_name=category_name,
                    question_id=question_id,
                    is_adversarial=True,
                    is_broken=False,
                    excluded_reason="adversarial",
                )
                records.append(record)
                continue
            # --------------------------------------------------

            # ---- CORRECT REAL BROKEN QUESTIONS (cited != correct) ----
            # Only run if APPLY_EVIDENCE_CORRECTIONS is True
            if APPLY_EVIDENCE_CORRECTIONS:
                is_corrected = False
                needs_correction = False

                # Check if any evidence in this question needs correction
                for ev in evidence_ids:
                    if ev in evidence_corrections:
                        needs_correction = True
                        break

                if needs_correction:
                    corrected_evidence = []
                    for ev in evidence_ids:
                        if ev in evidence_corrections:
                            corrected_ev = evidence_corrections[ev]
                            if corrected_ev and corrected_ev != ev:
                                corrected_evidence.append(corrected_ev)
                                is_corrected = True
                            else:
                                corrected_evidence.append(ev)
                        else:
                            corrected_evidence.append(ev)

                    evidence_to_use = corrected_evidence if corrected_evidence else evidence_ids

                    if is_corrected:
                        corrected_count += 1
                else:
                    evidence_to_use = evidence_ids
            else:
                # Skip correction entirely
                evidence_to_use = evidence_ids
            # --------------------------------------------------

            total_questions += 1

            # ----------------------------------------------------
            # Resolve benchmark reference time
            # ----------------------------------------------------

            reference_time = (
                resolve_reference_time(
                    q_item
                )
            )

            # ----------------------------------------------------
            # Determine worker set
            # ----------------------------------------------------

            workers, temporal = (
                configure_workers_for_query(
                    question_text,
                    original_workers,
                )
            )

            if temporal:
                temporal_questions += 1

            # ----------------------------------------------------
            # Install query-specific workers
            # ----------------------------------------------------

            settings.WORKERS_TO_USE = workers

            if temporal:

                debug(
                    "[LoCoMo] Temporal query "
                    f"({qi + 1}/{len(questions)}): "
                    f"{question_text[:100]}"
                )

                debug(
                    "[LoCoMo] Workers: "
                    f"{workers}"
                )

                debug(
                    "[LoCoMo] Reference time: "
                    f"{reference_time.isoformat()}"
                )

                debug(
                    "[LoCoMo] Session context: "
                    f"total={total_sessions}"
                )

            # ----------------------------------------------------
            # Determine which session the query is about
            # ----------------------------------------------------

            # Try to extract session number from the question text
            # This handles "session 4", "session #4", etc.
            #
            # NOTE: session_idx is zero-indexed everywhere else in
            # this file (see build_haystack_texts_and_metadata's
            # enumerate(session_keys)). A question referring to
            # "session 4" almost certainly means the 4th session,
            # i.e. index 3 — so the parsed number is adjusted down
            # by one to match. Verify against real question/session
            # pairs in your dataset before fully trusting this.
            session_match = re.search(r'session\s*#?\s*(\d+)', question_text, re.IGNORECASE)
            if session_match:
                current_session = max(0, int(session_match.group(1)) - 1)
            else:
                # For "previous session", "3 sessions ago", etc. we need the current session
                # For LoCoMo, each question is asked in the context of the last session
                current_session = total_sessions - 1 if total_sessions > 0 else 0

            # ----------------------------------------------------
            # Set temporal context on system (where query handler looks for it)
            # ----------------------------------------------------

            temporal_context = {
                "current_session": current_session,
                "total_sessions": total_sessions,
                "reference_time": reference_time,
            }
            # Set on both controller and system to be safe
            controller._temporal_context = temporal_context
            controller.system._temporal_context = temporal_context

            # ----------------------------------------------------
            # Query
            # ----------------------------------------------------

            query_start = (
                time.perf_counter()
            )

            try:

                response = controller.recall(
                    question_text
                )

            finally:

                # Never leak query-specific
                # worker configuration.
                settings.WORKERS_TO_USE = (
                    list(original_workers)
                )

                # Clean up temporal context
                if hasattr(controller, '_temporal_context'):
                    delattr(controller, '_temporal_context')
                if hasattr(controller.system, '_temporal_context'):
                    delattr(controller.system, '_temporal_context')

            query_time = (
                time.perf_counter()
                - query_start
            ) * 1000

            # ----------------------------------------------------
            # Debug first few queries
            # ----------------------------------------------------

            if (
                response.get("results")
                and total_questions <= 3
            ):

                print(
                    f"\n    [DEBUG] "
                    f"Query: "
                    f"{question_text[:80]}"
                )

                print(
                    f"    [DEBUG] "
                    f"Temporal: {temporal}"
                )

                print(
                    f"    [DEBUG] "
                    f"Workers used: {workers}"
                )

                print(
                    f"    [DEBUG] "
                    f"Reference time: "
                    f"{reference_time.isoformat()}"
                )

                print(
                    f"    [DEBUG] "
                    f"Session context: "
                    f"current={current_session}, total={total_sessions}"
                )

                print(
                    f"    [DEBUG] "
                    f"Original evidence IDs: "
                    f"{evidence_ids}"
                )

                if APPLY_EVIDENCE_CORRECTIONS and is_corrected:
                    print(
                        f"    [DEBUG] "
                        f"CORRECTED evidence IDs: "
                        f"{evidence_to_use}"
                    )

                for idx, result in enumerate(
                    response["results"][:3]
                ):

                    metadata = result.get(
                        "metadata",
                        {},
                    )

                    print(
                        f"      Rank {idx + 1}: "
                        f"dia_id = "
                        f"{metadata.get('dia_id')}, "
                        f"session_idx = "
                        f"{metadata.get('session_idx')}"
                    )

                    if temporal:

                        print(
                            f"        temporal_score = "
                            f"{result.get('temporal_score')}"
                        )

                        print(
                            f"        temporal_matches = "
                            f"{result.get('temporal_matches')}"
                        )

            # ----------------------------------------------------
            # Evaluation (use corrected evidence)
            # ----------------------------------------------------

            found, rank = check_retrieval(
                response,
                evidence_to_use,
                answer_text,
            )

            if found:
                retrieval_hits += 1

            # Build diagnostics with correction info
            diagnostics = response.get("diagnostics", {})
            if APPLY_EVIDENCE_CORRECTIONS and is_corrected:
                diagnostics["evidence_corrected"] = True
                diagnostics["original_evidence"] = evidence_ids
                diagnostics["corrected_evidence"] = evidence_to_use

            # Use our custom record builder with metadata
            record = build_record_with_metadata(
                query=question_text,
                expected=evidence_to_use[0] if evidence_to_use else "",
                expected_ids=evidence_to_use,
                expected_rank=rank,
                retrieved=found,
                candidates=response.get("results", []),
                runtime_ms=query_time,
                diagnostics=diagnostics,
                category=category,
                category_name=category_name,
                question_id=question_id,
                is_adversarial=False,
                is_broken=False,
                excluded_reason=None,
            )

            records.append(record)

        print(" done.")

        # ========================================================
        # CLEAR CONVERSATION
        # ========================================================

        if not skip_db_insert:

            clear_db_fast(
                controller
            )

        else:

            controller.system.vector_store.reset()
            controller.system.embedding_cache.clear()

            debug(
                "[LoCoMo] DB cache used, "
                "skipping clear."
            )

    # ============================================================
    # OUTPUT
    # ============================================================

    elapsed = (
        time.perf_counter()
        - start_time
    )

    output = build_output(
        records,
        question_count=total_questions,
    )

    # ---- INJECT DATASET NOTES ----
    output["notes"] = {
        "dataset": str(dataset_path),
        "timestamp": datetime.now().isoformat(),
        "limit_hit": args.limit,
        "conversations": len(data),
        "total_questions": total_questions,
        "total_memories": total_memories,
        "temporal_questions": temporal_questions,
        "adversarial_skipped": adversarial_skipped,
        "evidence_corrected": corrected_count,
        "broken_question_ids": list(broken_question_ids),
        "cache_version": CACHE_VERSION,
        "temporal_enabled": temporal_enabled,
        "workers_used": original_workers,
        "applied_evidence_corrections": APPLY_EVIDENCE_CORRECTIONS,
    }
    # ---------------------------------

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

        output_filename = (
            f"locomoeval_{timestamp}.json"
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
            / output_filename
        )

    write_output(
        output,
        str(output_path),
    )

    # ============================================================
    # SUMMARY
    # ============================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "[LoCoMo] Benchmark Complete"
    )

    print(
        "=" * 60
    )

    print(
        f"Total questions: "
        f"{total_questions}"
    )

    print(
        f"Adversarial skipped (Cat 5): "
        f"{adversarial_skipped}"
    )

    print(
        f"Evidence corrected: "
        f"{corrected_count}"
    )

    print(
        f"Total memories: "
        f"{total_memories}"
    )

    print(
        f"Temporal questions: "
        f"{temporal_questions}"
    )

    print(
        f"Retrieval hits: "
        f"{retrieval_hits}/"
        f"{total_questions} "
        f"("
        f"{retrieval_hits / total_questions * 100:.2f}%"
        f")"
        if total_questions
        else "Retrieval hits: 0/0"
    )

    print(
        f"Total time: "
        f"{elapsed:.2f}s"
    )

    print(
        f"Avg time per question: "
        f"{elapsed / total_questions:.2f}s"
        if total_questions
        else ""
    )

    print(
        "=" * 60
    )

    print(
        f"\nResults saved to: "
        f"{output_path}"
    )

    # ============================================================
    # ADAPTIVE WEIGHTS
    # ============================================================

    if args.optimize:

        print(
            "\n[Optimizer] Running "
            "adaptive weight adjustment..."
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
                "[Optimizer] Weight "
                "adjustment complete."
            )

        else:

            print(
                "[Optimizer] No adjustment "
                "made (no deltas or error)."
            )


if __name__ == "__main__":
    main()
