#!/usr/bin/env python3
"""
LongMemEval-S adapter — PURE TEXT, using BatchLoader.insert_batch.
Matches the fast test_batch_load path exactly.
Only measures store time; no retrieval check.
"""

import json
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import info, debug
from shared.memory_interface import MemoryInterface
from benchmark.batch_loader import BatchLoader


def compute_checksum(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def build_raw_texts(question):
    """Extract ONLY the raw content, no role prefix."""
    texts = []
    for session_data in question.get('haystack_sessions', []):
        for turn in session_data:
            content = turn.get('content', '')
            if content and content.strip():
                texts.append(content)
    return texts


def clear_db_fast(controller):
    """Fast truncate clear (same as original)."""
    db = controller.system.db
    conn = db.DBConnection

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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="longmemeval_s_cleaned.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--no-parallel", action="store_true")
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

    # Setup
    memory = MemoryInterface()
    loader = BatchLoader(memory)
    controller = memory.controller

    total_haystacks = 0
    total_memories = 0
    start_time = time.perf_counter()

    for q_idx, question in enumerate(questions):
        question_text = question.get('question', '')

        print(f"\n[Question {q_idx+1}/{len(questions)}] {question_text[:60]}...")

        # ---- Extract raw texts (no metadata, no prefix) ----
        texts = build_raw_texts(question)
        haystack_mem_count = len(texts)
        total_memories += haystack_mem_count

        print(f"    Haystack: {haystack_mem_count} memories", end="", flush=True)

        # ---- Insert using BatchLoader.insert_batch (exactly like fast test) ----
        store_start = time.perf_counter()
        count = loader.insert_batch(
            texts,
            batch_size=args.batch_size,
            skip_embedding=args.skip_embedding,
            parallel_extract=not args.no_parallel,
            max_workers=args.workers,
        )
        store_time = time.perf_counter() - store_start

        print(f", inserted {count} in {store_time:.2f}s", end="", flush=True)

        # ---- Clear DB fast (no retrieval check) ----
        clear_db_fast(controller)
        print(", cleared")

        total_haystacks += 1

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 60)
    print("[LongMemEval] Pure-text insert test complete")
    print("=" * 60)
    print(f"Total questions: {len(questions)}")
    print(f"Total haystacks: {total_haystacks}")
    print(f"Total memories: {total_memories}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Avg time per haystack: {elapsed/total_haystacks:.2f}s" if total_haystacks > 0 else "")
    print("=" * 60)

    output_path = Path("benchmark_output/pure_text_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "questions": len(questions),
                "haystacks": total_haystacks,
                "memories": total_memories,
                "elapsed_seconds": elapsed,
                "avg_haystack_seconds": elapsed/total_haystacks if total_haystacks > 0 else 0,
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
