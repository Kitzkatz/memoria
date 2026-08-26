#!/usr/bin/env python3
"""
LongMemEval-S adapter — stores texts WITH metadata, queries, and records retrieval ranks.
Per‑question isolation: store haystack, query, clear.
Supports per‑haystack DB + FAISS caching with independent rebuilds.
"""

import json
import hashlib
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import info, debug
from shared.memory_interface import MemoryInterface
from benchmark.batch_loader import BatchLoader
from benchmark.result_formatter import build_record, build_output, write_output
from cache.config import settings
from ranking.adaptive_weighter import adaptive_weighter_pipeline


def compute_checksum(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_questions_file(questions_path="benchmark_output/longmemeval_questions.json"):
    """Load the questions file (must be generated first by extract_questions.py)."""
    path = Path(questions_path)
    if not path.exists():
        print(f"Warning: Questions file not found at {path}. Falling back to index-based assignment.")
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load questions file: {e}. Falling back.")
        return None


def build_texts_and_metadata(question, haystack_session_ids=None, questions_data=None, q_idx=0):
    """
    Extract texts and metadata.
    Uses haystack_session_ids (provided by dataset) as the primary source for session_id.
    Falls back to answer_ids by index if available, else generates a distractor ID.
    """
    texts = []
    metadatas = []
    answer_ids = question.get('answer_session_ids', [])
    question_id = question.get('question_id', '')

    # Get the list of session IDs for this question, or use empty list
    session_ids = haystack_session_ids if haystack_session_ids is not None else []

    # Find expected answer text for this question (if we have questions_data)
    expected_text = None
    if questions_data:
        for q in questions_data:
            if q.get('question_id') == question_id:
                expected_text = q.get('expected')
                break

    # ----- DEBUG: print session structure for the first question only -----
    if q_idx == 0:
        print(f"\n[DEBUG] question_id={question_id}, answer_ids={answer_ids}")
        print(f"[DEBUG] haystack_session_ids count: {len(session_ids)}")
        print(f"[DEBUG] expected_text='{expected_text[:50] if expected_text else 'None'}'")
        print(f"[DEBUG] Number of sessions: {len(question.get('haystack_sessions', []))}")
        print("-" * 60)

    for session_idx, session_data in enumerate(question.get('haystack_sessions', [])):
        session_id = None

        # Primary: use the session ID from the dataset's list
        if session_idx < len(session_ids) and session_ids[session_idx]:
            session_id = session_ids[session_idx]

        # Fallback: if no session ID from the list, try to get from session_data (if it's a dict)
        if not session_id:
            if isinstance(session_data, dict):
                session_id = session_data.get('id') or session_data.get('session_id')

        # Fallback: try to match by content (if expected_text is available)
        if not session_id and expected_text:
            # Extract turns
            if isinstance(session_data, dict):
                turns = session_data.get('turns', [])
            else:
                turns = session_data
            for turn in turns:
                content = turn.get('content', '') if isinstance(turn, dict) else turn
                if expected_text in content:
                    # This session contains the answer; assign the first answer_id
                    if answer_ids:
                        session_id = answer_ids[0]
                    else:
                        session_id = f"answer_{question_id}"
                    break

        # Final fallback: use answer_ids by index if available
        if not session_id:
            if session_idx < len(answer_ids) and answer_ids[session_idx]:
                session_id = answer_ids[session_idx]
            else:
                session_id = f"distractor_{question_id}_{session_idx}"

        # Extract turns (handle both list and dict)
        if isinstance(session_data, dict):
            turns = session_data.get('turns', [])
        else:
            turns = session_data

        for turn in turns:
            content = turn.get('content', '') if isinstance(turn, dict) else turn
            if content and content.strip():
                texts.append(content)
                metadatas.append({"session_id": session_id})

    return texts, metadatas


def check_retrieval(response, expected_session_ids):
    """
    Check if any expected session ID appears in the results.
    Returns (found, rank) where rank is 1‑indexed, or (False, None).
    """
    results = response.get("results", [])
    for rank, r in enumerate(results, start=1):
        meta = r.get("metadata", {})
        if meta.get("session_id") in expected_session_ids:
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
    debug(f"[LongMemEval] DB cleared.")


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


def build_faiss_from_texts(controller, texts, mem_ids):
    """
    Compute embeddings from texts and add to vector store with given memory IDs.
    Returns vectors.
    """
    extractor = controller.system.extractor
    records = [extractor.extract(text) for text in texts]
    normalized_texts = [r.normalized_text for r in records]
    vectors = controller.system.embedder.embed_many(normalized_texts)
    controller.system.vector_store.add_many(mem_ids, vectors, persist=False)
    controller.system.embedding_cache.add_many(mem_ids, vectors)
    return vectors


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="longmemeval_s_cleaned.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--optimize", action="store_true", help="Run adaptive weight optimization after benchmark")
    parser.add_argument("--dry-run-weights", action="store_true", help="Print weight changes without saving")
    parser.add_argument("--step-size", type=float, default=0.02, help="Step size for weight adjustment")
    parser.add_argument("--cache-dir", default="cache/faiss_indices", help="Directory to store per-haystack caches (DB + FAISS)")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuilding all caches (ignore existing)")
    parser.add_argument("--output", type=str, help="Override output file path (default: timestamped in benchmark_output/results/)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    print(f"[LongMemEval] Loading dataset: {dataset_path}")
    with open(dataset_path, "r") as f:
        questions = json.load(f)

    if args.limit:
        questions = questions[:args.limit]

    print(f"[LongMemEval] {len(questions)} questions loaded")
    print(f"[LongMemEval] Dataset checksum: {compute_checksum(dataset_path)}")

    settings.CONTEXT_TOKEN_BUDGET = 10000

    questions_data = load_questions_file()

    memory = MemoryInterface()
    loader = BatchLoader(memory)
    controller = memory.controller
    print("Embedder model:", controller.system.embedder.model)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = []
    retrieval_hits = 0
    total_memories = 0
    start_time = time.perf_counter()

    for q_idx, question in enumerate(questions):
        q_id = question.get('question_id', f'q_{q_idx}')
        question_text = question.get('question', '')
        answer_ids = question.get('answer_session_ids', [])
        print(f"\n[Question {q_idx+1}/{len(questions)}] {question_text[:60]}...")

        haystack_session_ids = question.get('haystack_session_ids', [])
        texts, metadatas = build_texts_and_metadata(
            question,
            haystack_session_ids=haystack_session_ids,
            questions_data=questions_data,
            q_idx=q_idx
        )
        haystack_mem_count = len(texts)
        total_memories += haystack_mem_count

        print(f"    Haystack: {haystack_mem_count} memories", end="", flush=True)

        # ---- Check caches ----
        db_cache_path = cache_dir / f"db_{q_id}.sqlite"
        faiss_cache_path = cache_dir / f"faiss_{q_id}.index"
        db_cache_exists = db_cache_path.exists() and not args.rebuild_cache
        faiss_cache_exists = faiss_cache_path.exists() and not args.rebuild_cache

        skip_embedding_build = False
        skip_db_insert = False

        # ---- Load DB cache if exists ----
        if db_cache_exists:
            print(f"\n    Loading cached DB for {q_id}...", end="", flush=True)
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

        # ---- Load FAISS cache (always load if exists) ----
        if faiss_cache_exists:
            print(f"\n    Loading cached FAISS for {q_id}...", end="", flush=True)
            try:
                controller.system.vector_store.load_from_file(str(faiss_cache_path))
                skip_embedding_build = True
                print(" loaded", end="", flush=True)
            except Exception as e:
                print(f" failed ({e}), rebuilding", end="", flush=True)
                skip_embedding_build = False

        # ---- Build FAISS if needed (when DB loaded but FAISS missing) ----
        faiss_built = False
        if not args.skip_embedding and not faiss_cache_exists and skip_db_insert:
            print(f"\n    Building FAISS for {q_id}...", end="", flush=True)
            try:
                rows = controller.system.db.fetch_all()
                mem_ids = [row['id'] for row in rows]
                if len(mem_ids) == len(texts):
                    build_faiss_from_texts(controller, texts, mem_ids)
                    controller.system.vector_store.save_to_file(str(faiss_cache_path))
                    faiss_built = True
                    skip_embedding_build = True
                    print(" built and saved", end="", flush=True)
                else:
                    print(f" ID count mismatch: {len(mem_ids)} vs {len(texts)}, rebuilding DB+FAISS", end="", flush=True)
                    skip_db_insert = False
            except Exception as e:
                print(f" failed ({e}), will rebuild DB+FAISS", end="", flush=True)
                skip_db_insert = False

        # ---- Store (or skip) ----
        store_time = 0.0
        count = 0
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

            if not args.rebuild_cache:
                print(f"\n    Saving DB cache for {q_id}...", end="", flush=True)
                try:
                    shutil.copy2(settings.DB_PATH, str(db_cache_path))
                    print(" saved", end="", flush=True)
                except Exception as e:
                    print(f" failed ({e})", end="", flush=True)

                if not args.skip_embedding and not skip_embedding_build:
                    print(f"\n    Saving FAISS cache for {q_id}...", end="", flush=True)
                    try:
                        controller.system.vector_store.save_to_file(str(faiss_cache_path))
                        print(" saved", end="", flush=True)
                    except Exception as e:
                        print(f" failed ({e})", end="", flush=True)
        else:
            count = haystack_mem_count
            if faiss_built:
                pass
            elif not faiss_cache_exists and not args.skip_embedding:
                print(f"\n    FAISS missing and could not build; consider --rebuild-cache")

        # ---- Query ----
        query_start = time.perf_counter()
        response = controller.recall(question_text)
        query_time = time.perf_counter() - query_start

        candidates = response.get("results", [])
        if candidates:
            first_meta = candidates[0].get("metadata", {})
            print(f"\n    [DEBUG] First candidate metadata: {first_meta}")
            print(f"    [DEBUG] Expected session IDs: {answer_ids}")
        else:
            print(f"\n    [DEBUG] No candidates returned!")

        found, rank = check_retrieval(response, answer_ids)
        if found:
            retrieval_hits += 1

        raw_candidates = response.get("results", [])
        record = build_record(
            query=question_text,
            expected="",
            expected_ids=answer_ids,
            expected_rank=rank,
            retrieved=found,
            candidates=raw_candidates,
            runtime_ms=query_time * 1000,
            diagnostics=response.get("diagnostics", {})
        )
        records.append(record)

        print(f", inserted {count} in {store_time:.2f}s, queried in {query_time:.2f}s", end="", flush=True)

        if not skip_db_insert:
            clear_db_fast(controller)
        else:
            controller.system.vector_store.reset()
            controller.system.embedding_cache.clear()
            debug(f"[LongMemEval] DB cache used, skipping clear.")
        print(", cleared")

    elapsed = time.perf_counter() - start_time

    output = build_output(records, question_count=len(questions))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"longmemeval_{timestamp}.json"
        output_dir = Path("benchmark_output/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

    write_output(output, str(output_path))

    total_questions = len(questions)
    print("\n" + "=" * 60)
    print("[LongMemEval] Benchmark Complete")
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
