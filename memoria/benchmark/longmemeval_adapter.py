#!/usr/bin/env python3
"""
LongMemEval-S adapter — stores texts WITH metadata, queries, and records retrieval ranks.
Per‑question isolation: store haystack, query, clear.
Supports per‑haystack DB + FAISS caching with independent rebuilds.
Now with official‑style session/turn metrics, abstention handling, and reproducibility manifests.
"""

import json
import hashlib
import sys
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import info, debug
from shared.memory_interface import MemoryInterface
from benchmark.batch_loader import BatchLoader
from benchmark.result_formatter import build_record, build_output, write_output
from cache.config import settings
from ranking.adaptive_weighter import adaptive_weighter_pipeline


# ----------------------------------------------------------------------
#  Constants
# ----------------------------------------------------------------------

K_VALUES = (1, 3, 5, 10, 30, 50)


# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------

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


def is_abstention(question):
    """Check if a question is an abstention case."""
    expected = question.get('expected', '')
    return expected and expected.strip().lower() == "i don't know"


# ----------------------------------------------------------------------
#  Metadata Builder (Phase 1)
# ----------------------------------------------------------------------

def build_texts_and_metadata(
    question,
    haystack_session_ids=None,
    questions_data=None,
    q_idx=0,
):
    """
    Extract texts and metadata.
    Now stores per‑turn metadata including turn_id, has_answer, turn_index, role.
    """
    texts = []
    metadatas = []
    answer_ids = set(question.get('answer_session_ids', []))
    question_id = question.get('question_id', '')

    session_ids = haystack_session_ids if haystack_session_ids is not None else []

    expected_text = None
    if questions_data:
        for q in questions_data:
            if q.get('question_id') == question_id:
                expected_text = q.get('expected')
                break

    if q_idx == 0:
        print(f"\n[DEBUG] question_id={question_id}, answer_ids={answer_ids}")
        print(f"[DEBUG] haystack_session_ids count: {len(session_ids)}")
        print(f"[DEBUG] expected_text='{expected_text[:50] if expected_text else 'None'}'")
        print(f"[DEBUG] Number of sessions: {len(question.get('haystack_sessions', []))}")
        print("-" * 60)

    for session_idx, session_data in enumerate(question.get('haystack_sessions', [])):
        session_id = None

        if session_idx < len(session_ids) and session_ids[session_idx]:
            session_id = session_ids[session_idx]

        if not session_id and isinstance(session_data, dict):
            session_id = session_data.get('id') or session_data.get('session_id')

        if not session_id and expected_text:
            if isinstance(session_data, dict):
                turns = session_data.get('turns', [])
            else:
                turns = session_data
            for turn in turns:
                content = turn.get('content', '') if isinstance(turn, dict) else turn
                if expected_text in content:
                    if answer_ids:
                        session_id = list(answer_ids)[0]
                    else:
                        session_id = f"answer_{question_id}"
                    break

        if not session_id:
            if session_idx < len(answer_ids) and list(answer_ids)[session_idx]:
                session_id = list(answer_ids)[session_idx]
            else:
                session_id = f"distractor_{question_id}_{session_idx}"

        if isinstance(session_data, dict):
            turns = session_data.get('turns', [])
        else:
            turns = session_data

        # Determine if this session contains the answer
        has_answer = session_id in answer_ids

        for turn_idx, turn in enumerate(turns):
            content = turn.get('content', '') if isinstance(turn, dict) else turn
            if not content or not content.strip():
                continue

            # Derive deterministic turn_id
            turn_id = f"{session_id}_{turn_idx}"
            role = turn.get('role', 'unknown') if isinstance(turn, dict) else 'unknown'

            texts.append(content)
            metadatas.append({
                "session_id": session_id,
                "turn_id": turn_id,
                "has_answer": has_answer,
                "turn_index": turn_idx,
                "role": role,
            })

    return texts, metadatas


# ----------------------------------------------------------------------
#  Evaluation (Phase 2)
# ----------------------------------------------------------------------

def evaluate_retrieval(ranked_results, gold_session_ids, gold_turn_ids, k_values=K_VALUES):
    """
    Compute session‑level and turn‑level metrics from the ranked candidate list.

    Returns:
        dict: {
            "session": { "recall_any": {k: ...}, "recall_all": {k: ...}, "ndcg_any": {k: ...} },
            "turn": { "recall_any": {k: ...}, "recall_all": {k: ...}, "ndcg_any": {k: ...} },
        }
    """
    gold_session_set = set(gold_session_ids)
    gold_turn_set = set(gold_turn_ids)

    # Build the ranked unique session list (preserving order)
    seen_sessions = set()
    ranked_sessions = []
    for r in ranked_results:
        meta = r.get("metadata", {})
        sid = meta.get("session_id")
        if sid and sid not in seen_sessions:
            seen_sessions.add(sid)
            ranked_sessions.append(sid)

    # Turn‑level: just use the ranked results order
    ranked_turns = [r.get("metadata", {}).get("turn_id") for r in ranked_results if r.get("metadata", {}).get("turn_id")]

    metrics = {"session": {}, "turn": {}}

    for mode, ranked_ids, gold_set in [
        ("session", ranked_sessions, gold_session_set),
        ("turn", ranked_turns, gold_turn_set),
    ]:
        recall_any = {}
        recall_all = {}
        ndcg_any = {}

        for k in k_values:
            top_k = ranked_ids[:k]
            found_any = any(g in gold_set for g in top_k)

            # recall_all: all gold IDs must be in the top‑k
            found_all = all(g in top_k for g in gold_set) if gold_set else True

            # NDCG@k: only for 'any' (binary relevance)
            dcg = 0.0
            idcg = 0.0
            for i, rid in enumerate(top_k, start=1):
                rel = 1.0 if rid in gold_set else 0.0
                dcg += rel / (i + 1.0)  # log2(i+1) simplified
            for i in range(1, min(k, len(gold_set)) + 1):
                idcg += 1.0 / (i + 1.0)
            ndcg = dcg / idcg if idcg > 0 else 0.0

            recall_any[k] = found_any
            recall_all[k] = found_all
            ndcg_any[k] = ndcg

        metrics[mode] = {
            "recall_any": recall_any,
            "recall_all": recall_all,
            "ndcg_any": ndcg_any,
        }

    return metrics


# ----------------------------------------------------------------------
#  Cache Manifest (Phase 7)
# ----------------------------------------------------------------------

def write_cache_manifest(cache_dir, question_id, dataset_hash, memory_count, vector_count):
    """Write a cache manifest for a per‑question cache."""
    manifest_path = cache_dir / f"{question_id}.manifest.json"
    manifest = {
        "question_id": question_id,
        "dataset_hash": dataset_hash,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dimension": settings.VECTOR_DIM,
        "memory_count": memory_count,
        "vector_count": vector_count,
        "created_at": datetime.utcnow().isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def validate_cache_manifest(cache_dir, question_id, dataset_hash):
    """Validate an existing cache manifest; return True if valid."""
    manifest_path = cache_dir / f"{question_id}.manifest.json"
    if not manifest_path.exists():
        return False
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        return manifest.get("dataset_hash") == dataset_hash
    except Exception:
        return False


# ----------------------------------------------------------------------
#  Check Retrieval (Legacy, kept for compatibility)
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
#  Clear / Rebuild / FAISS helpers
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

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
    parser.add_argument("--retrieval-mode", choices=["dense", "bm25", "fusion", "full"], default="fusion",
                        help="Ablation mode: dense, bm25, fusion, full")
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
    print(f"[LongMemEval] Retrieval mode: {args.retrieval_mode}")

    # Apply ablation mode
    if args.retrieval_mode == "dense":
        settings.WORKERS_TO_USE = ["faiss"]
        settings.RANKING_ENABLED = False
    elif args.retrieval_mode == "bm25":
        settings.WORKERS_TO_USE = ["bm25"]
        settings.RANKING_ENABLED = False
    elif args.retrieval_mode == "fusion":
        settings.WORKERS_TO_USE = ["fusion"]
        settings.RANKING_ENABLED = False
    elif args.retrieval_mode == "full":
        settings.WORKERS_TO_USE = ["fusion"]
        settings.RANKING_ENABLED = True

    settings.CONTEXT_TOKEN_BUDGET = 10000

    questions_data = load_questions_file()
    dataset_hash = compute_checksum(dataset_path)

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
    abstention_count = 0
    retrieval_evaluable_count = 0

    for q_idx, question in enumerate(questions):
        q_id = question.get('question_id', f'q_{q_idx}')
        question_text = question.get('question', '')
        answer_ids = question.get('answer_session_ids', [])

        # --- Abstention handling ---
        if is_abstention(question):
            abstention_count += 1
            # Still build a minimal record but skip retrieval metrics
            print(f"\n[Question {q_idx+1}/{len(questions)}] SKIP (abstention)")
            records.append({
                "query": question_text,
                "expected": question.get('expected', ''),
                "expected_ids": answer_ids,
                "expected_rank": None,
                "retrieved": False,
                "candidates": [],
                "runtime_ms": 0.0,
                "diagnostics": {},
                "abstention": True,
            })
            continue

        retrieval_evaluable_count += 1

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

        # ---- Cache validation ----
        db_cache_path = cache_dir / f"db_{q_id}.sqlite"
        faiss_cache_path = cache_dir / f"faiss_{q_id}.index"
        cache_valid = validate_cache_manifest(cache_dir, q_id, dataset_hash)
        db_cache_exists = db_cache_path.exists() and not args.rebuild_cache and cache_valid
        faiss_cache_exists = faiss_cache_path.exists() and not args.rebuild_cache and cache_valid

        skip_embedding_build = False
        skip_db_insert = False

        # ---- Load DB cache ----
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

        # ---- Load FAISS cache ----
        if faiss_cache_exists:
            print(f"\n    Loading cached FAISS for {q_id}...", end="", flush=True)
            try:
                controller.system.vector_store.load_from_file(str(faiss_cache_path))
                skip_embedding_build = True
                print(" loaded", end="", flush=True)
            except Exception as e:
                print(f" failed ({e}), rebuilding", end="", flush=True)
                skip_embedding_build = False

        # ---- Build FAISS if needed ----
        faiss_built = False
        if not args.skip_embedding and not faiss_cache_exists and skip_db_insert:
            print(f"\n    Building FAISS for {q_id}...", end="", flush=True)
            try:
                rows = controller.system.db.fetch_all()
                mem_ids = [row['id'] for row in rows]
                if len(mem_ids) == len(texts):
                    build_faiss_from_texts(controller, texts, mem_ids)
                    controller.system.vector_store.save_to_file(str(faiss_cache_path))
                    write_cache_manifest(cache_dir, q_id, dataset_hash, len(mem_ids), len(texts))
                    faiss_built = True
                    skip_embedding_build = True
                    print(" built and saved", end="", flush=True)
                else:
                    print(f" ID count mismatch: {len(mem_ids)} vs {len(texts)}, rebuilding DB+FAISS", end="", flush=True)
                    skip_db_insert = False
            except Exception as e:
                print(f" failed ({e}), will rebuild DB+FAISS", end="", flush=True)
                skip_db_insert = False

        # ---- Store ----
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
                        write_cache_manifest(cache_dir, q_id, dataset_hash, count, len(texts))
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

        raw_candidates = response.get("results", [])
        if raw_candidates:
            first_meta = raw_candidates[0].get("metadata", {})
            print(f"\n    [DEBUG] First candidate metadata: {first_meta}")
            print(f"    [DEBUG] Expected session IDs: {answer_ids}")
        else:
            print(f"\n    [DEBUG] No candidates returned!")

        # ---- Legacy check (kept for compatibility) ----
        found, rank = check_retrieval(response, answer_ids)
        if found:
            retrieval_hits += 1

        # ---- Official evaluation ----
        # Derive gold turn IDs from metadata (we stored has_answer in metadata)
        gold_turn_ids = set()
        for meta in metadatas:
            if meta.get('has_answer', False):
                turn_id = meta.get('turn_id')
                if turn_id:
                    gold_turn_ids.add(turn_id)

        # If we don't have turn IDs, fallback to session IDs
        if not gold_turn_ids:
            gold_turn_ids = set(answer_ids)

        metrics = evaluate_retrieval(raw_candidates, answer_ids, gold_turn_ids, K_VALUES)

        # ---- Build record ----
        record = {
            "query": question_text,
            "expected": question.get('expected', ''),
            "expected_ids": answer_ids,
            "expected_rank": rank,
            "retrieved": found,
            "candidates": raw_candidates,
            "runtime_ms": query_time * 1000,
            "diagnostics": response.get("diagnostics", {}),
            "metrics": metrics,
            "abstention": False,
        }
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

    # ---- Build output ----
    output = {
        "question_count": len(questions),
        "retrieval_evaluable": retrieval_evaluable_count,
        "abstentions_excluded": abstention_count,
        "records": records,
        "dataset_checksum": dataset_hash,
        "embedder": settings.EMBEDDING_MODEL,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "retrieval_mode": args.retrieval_mode,
        "settings": {
            key: str(value) for key, value in vars(settings).items()
            if not key.startswith("_") and not callable(value)
        },
        "command": " ".join(sys.argv),
    }

    # ---- Save results ----
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"longmemeval_{timestamp}.json"
        output_dir = Path("benchmark_output/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # ---- Also write a manifest ----
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "benchmark": "LongMemEval-S",
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_hash,
        },
        "run": {
            "started": datetime.now().isoformat(),
            "finished": datetime.now().isoformat(),
            "command": " ".join(sys.argv),
            "arguments": vars(args),
        },
        "system": {
            "embedder": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.VECTOR_DIM,
            "chat_model": settings.CHAT_MODEL,
            "vector_store": type(controller.system.vector_store).__name__,
            "database": "sqlite",
        },
        "retrieval": {
            "mode": args.retrieval_mode,
            "sources": getattr(settings, "WORKERS_TO_USE", []),
            "candidate_limit": settings.TOP_K,
        },
        "ranking": {
            "enabled": settings.RANKING_ENABLED,
            "mmr": settings.MMR_ENABLED,
        },
        "cache": {
            "enabled": True,
            "rebuild": args.rebuild_cache,
            "directory": str(cache_dir),
            "manifest": "per‑question",
        },
        "evaluation": {
            "k_values": list(K_VALUES),
            "abstentions_excluded": abstention_count,
            "retrieval_evaluable": retrieval_evaluable_count,
        },
        "settings_snapshot": {
            key: str(value) for key, value in vars(settings).items()
            if not key.startswith("_") and not callable(value)
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("[LongMemEval] Benchmark Complete")
    print("=" * 60)
    print(f"Total questions: {len(questions)}")
    print(f"Abstentions excluded: {abstention_count}")
    print(f"Retrieval evaluable: {retrieval_evaluable_count}")
    print(f"Total memories: {total_memories}")
    print(f"Retrieval hits (legacy any): {retrieval_hits}/{len(questions)} ({retrieval_hits/len(questions)*100:.2f}%)")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg time per question: {elapsed/len(questions):.2f}s" if len(questions) > 0 else "")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")
    print(f"Manifest saved to: {manifest_path}")

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
