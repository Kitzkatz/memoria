#!/usr/bin/env python3
"""
LoCoMo-S adapter — stores conversations, queries, and evaluates retrieval.
Per‑conversation isolation (store once, query all questions, then clear).
Supports DB + FAISS caching per conversation with automatic versioning.
"""

import json
import hashlib
import sys
import time
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import info, debug
from shared.memory_interface import MemoryInterface
from benchmark.batch_loader import BatchLoader
from benchmark.result_formatter import build_record, build_output, write_output
from cache.config import settings
from ranking.adaptive_weighter import adaptive_weighter_pipeline

# Increment when schema changes to auto-rebuild caches
CACHE_VERSION = 1


def compute_checksum(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_locomo_dataset(input_path):
    """Load LoCoMo dataset from JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_haystack_texts_and_metadata(entry):
    """
    Extract all turns from the conversation as texts with metadata.
    LoCoMo conversation is a dict with session_1, session_2, etc.
    Each turn has 'dia_id' and 'text'.
    """
    texts = []
    metadatas = []
    conversation = entry.get("conversation", {})
    if not conversation:
        return texts, metadatas

    session_keys = [k for k in conversation.keys() if k.startswith("session_")]
    for session_key in session_keys:
        turns = conversation.get(session_key, [])
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            dia_id = turn.get("dia_id", "")
            text = turn.get("text", "")
            if text and text.strip():
                texts.append(text)
                metadatas.append({"dia_id": dia_id})
    return texts, metadatas


def check_retrieval(response, expected_evidence_ids, expected_answer_text=""):
    """
    Check if any expected evidence ID appears in the candidate metadata.
    Fallback to text matching if no evidence match found.
    """
    # First: evidence ID match
    if expected_evidence_ids:
        results = response.get("results", [])
        for rank, r in enumerate(results, start=1):
            meta = r.get("metadata", {})
            dia_id = meta.get("dia_id")
            if dia_id and dia_id in expected_evidence_ids:
                return True, rank

    # Fallback: text match
    if expected_answer_text:
        expected = str(expected_answer_text).lower()
        results = response.get("results", [])
        for rank, r in enumerate(results, start=1):
            if expected in r.get("text", "").lower():
                return True, rank

    return False, None


def clear_db_fast(controller):
    """Fast truncate clear."""
    db = controller.system.db
    if hasattr(db, 'conn') and hasattr(db.conn, 'conn'):
        conn = db.conn.conn
    else:
        conn = getattr(db, 'conn', None)
        if conn is None:
            raise AttributeError("No connection found on db object")
    conn.execute("PRAGMA foreign_keys = OFF")
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    for (table_name,) in tables:
        conn.execute(f"DELETE FROM {table_name}")
        conn.execute(f"DELETE FROM sqlite_sequence WHERE name='{table_name}'")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.commit()
    controller.system.vector_store.reset()
    controller.system.embedding_cache.clear()
    debug(f"[LoCoMo] DB cleared.")


def rebuild_indices_from_db(system):
    """Rebuild inverted index and BM25 from the current DB."""
    if hasattr(system, 'inverted_index') and system.inverted_index:
        system.inverted_index.build()
    if hasattr(system, 'bm25_ranker') and system.bm25_ranker:
        rows = system.db.fetch_all()
        if rows:
            corpus_tokens = [row.get('tokens', []) for row in rows]
            system.bm25_ranker.build(corpus_tokens)
        else:
            system.bm25_ranker.build([])


def get_cached_version(db_path):
    """Read cache version from the SQLite DB if present."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM cache_meta WHERE key='version'")
        row = cursor.fetchone()
        conn.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def set_cached_version(db_path, version):
    """Write cache version to the SQLite DB."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS cache_meta (key TEXT PRIMARY KEY, value TEXT)")
        cursor.execute("INSERT OR REPLACE INTO cache_meta (key, value) VALUES ('version', ?)", (str(version),))
        conn.commit()
        conn.close()
    except Exception:
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="locomo10.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--dry-run-weights", action="store_true")
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--cache-dir", default="cache/faiss_indices")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    print(f"[LoCoMo] Loading dataset: {dataset_path}")
    data = load_locomo_dataset(dataset_path)
    if args.limit:
        data = data[:args.limit]
    print(f"[LoCoMo] {len(data)} conversations loaded")
    print(f"[LoCoMo] Dataset checksum: {compute_checksum(dataset_path)}")

    settings.CONTEXT_TOKEN_BUDGET = 10000

    memory = MemoryInterface()
    loader = BatchLoader(memory)
    controller = memory.controller
    print("Embedder model:", controller.system.embedder.model)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_questions = 0
    retrieval_hits = 0
    total_memories = 0
    start_time = time.perf_counter()

    for conv_idx, entry in enumerate(data):
        conversation_id = entry.get("id", f"conv_{conv_idx}")
        print(f"\n[Conversation {conv_idx+1}/{len(data)}] {conversation_id}")

        texts, metadatas = build_haystack_texts_and_metadata(entry)
        haystack_mem_count = len(texts)
        total_memories += haystack_mem_count
        print(f"    Haystack: {haystack_mem_count} memories")

        # ---- Check caches ----
        db_cache_path = cache_dir / f"db_{conversation_id}.sqlite"
        faiss_cache_path = cache_dir / f"faiss_{conversation_id}.index"

        rebuild = args.rebuild_cache
        if db_cache_path.exists() and not rebuild:
            cached_version = get_cached_version(db_cache_path)
            if cached_version != CACHE_VERSION:
                print(f"    Cache version mismatch ({cached_version} != {CACHE_VERSION}), rebuilding...")
                rebuild = True
                db_cache_path.unlink(missing_ok=True)
                faiss_cache_path.unlink(missing_ok=True)

        db_cache_exists = db_cache_path.exists() and not rebuild
        faiss_cache_exists = faiss_cache_path.exists() and not rebuild

        skip_embedding_build = False
        skip_db_insert = False
        store_time = 0.0
        count = 0

        # ---- Load DB cache ----
        if db_cache_exists:
            print(f"    Loading cached DB for {conversation_id}...", end="", flush=True)
            try:
                if hasattr(controller.system.db, '_conn'):
                    controller.system.db._conn.close()
                shutil.copy2(str(db_cache_path), settings.DB_PATH)
                from db.connection import DBConnection
                controller.system.db._conn = DBConnection()
                rebuild_indices_from_db(controller.system)
                controller.system.vector_store.reset()
                skip_db_insert = True
                print(" loaded", end="", flush=True)
            except Exception as e:
                print(f" failed ({e}), rebuilding", end="", flush=True)
                skip_db_insert = False

        # ---- Load FAISS cache ----
        if faiss_cache_exists and not args.skip_embedding and skip_db_insert:
            print(f"    Loading cached FAISS for {conversation_id}...", end="", flush=True)
            try:
                controller.system.vector_store.load_from_file(str(faiss_cache_path))
                skip_embedding_build = True
                print(" loaded", end="", flush=True)
            except Exception as e:
                print(f" failed ({e}), rebuilding", end="", flush=True)
                skip_embedding_build = False

        # ---- Store (or skip) ----
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
            store_time = time.perf_counter() - store_start

            if not rebuild:
                print(f"    Saving DB cache for {conversation_id}...", end="", flush=True)
                try:
                    shutil.copy2(settings.DB_PATH, str(db_cache_path))
                    set_cached_version(db_cache_path, CACHE_VERSION)
                    print(" saved", end="", flush=True)
                except Exception as e:
                    print(f" failed ({e})", end="", flush=True)

                if not skip_embedding_build and not args.skip_embedding:
                    print(f"    Saving FAISS cache for {conversation_id}...", end="", flush=True)
                    try:
                        controller.system.vector_store.save_to_file(str(faiss_cache_path))
                        print(" saved", end="", flush=True)
                    except Exception as e:
                        print(f" failed ({e})", end="", flush=True)
        else:
            count = haystack_mem_count
            store_time = 0.0

        # ---- Process questions ----
        questions = entry.get("qa", [])
        if not questions:
            print("    No questions found, skipping.")
            if not skip_db_insert:
                clear_db_fast(controller)
            continue

        print(f"    Processing {len(questions)} questions...", end="", flush=True)

        for qi, q_item in enumerate(questions):
            question_text = q_item.get("question", "")
            evidence_ids = q_item.get("evidence", [])
            answer_text = q_item.get("answer", "")
            if not question_text:
                continue

            total_questions += 1

            query_start = time.perf_counter()
            response = controller.recall(question_text)
            query_time = (time.perf_counter() - query_start) * 1000

            # DEBUG: inspect first few candidates
            if response.get("results") and total_questions <= 3:
                print(f"\n    [DEBUG] Query: {question_text[:60]}...")
                print(f"    [DEBUG] Expected evidence IDs: {evidence_ids}")
                for idx, r in enumerate(response['results'][:3]):
                    meta = r.get("metadata", {})
                    print(f"      Rank {idx+1}: dia_id = {meta.get('dia_id')}")

            found, rank = check_retrieval(response, evidence_ids, answer_text)
            if found:
                retrieval_hits += 1

            record = build_record(
                query=question_text,
                expected=evidence_ids[0] if evidence_ids else "",
                expected_ids=evidence_ids,
                expected_rank=rank,
                retrieved=found,
                candidates=response.get("results", []),
                runtime_ms=query_time,
                diagnostics=response.get("diagnostics", {})
            )
            records.append(record)

        print(f" done.")

        if not skip_db_insert:
            clear_db_fast(controller)
        else:
            controller.system.vector_store.reset()
            controller.system.embedding_cache.clear()
            debug(f"[LoCoMo] DB cache used, skipping clear.")

    elapsed = time.perf_counter() - start_time

    # ---- Output ----
    output = build_output(records, question_count=total_questions)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"locomoeval_{timestamp}.json"
        output_dir = Path("benchmark_output/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

    write_output(output, str(output_path))

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("[LoCoMo] Benchmark Complete")
    print("=" * 60)
    print(f"Total questions: {total_questions}")
    print(f"Total memories: {total_memories}")
    print(f"Retrieval hits: {retrieval_hits}/{total_questions} ({retrieval_hits/total_questions*100:.2f}%)")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg time per question: {elapsed/total_questions:.2f}s" if total_questions > 0 else "")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")

    if args.optimize:
        print("\n[Optimizer] Running adaptive weight adjustment...")
        result = adaptive_weighter_pipeline(
            benchmark_file=str(output_path),
            dry_run=args.dry_run_weights,
            step_size=args.step_size
        )
        if result:
            print("[Optimizer] Weight adjustment complete.")
        else:
            print("[Optimizer] No adjustment made (no deltas or error).")


if __name__ == "__main__":
    main()
