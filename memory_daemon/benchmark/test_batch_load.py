#!/usr/bin/env python3
"""
test_batch_load.py

Test script for batch loading benchmark memories.
Loads memories from benchmark_output/benchmark_memories.json
and inserts them into the memory system.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.memory_interface import MemoryInterface
from benchmark.batch_loader import BatchLoader
from core.logger import info, error


def main():
    parser = argparse.ArgumentParser(description="Load benchmark memories into the system")
    parser.add_argument(
        "--file",
        type=str,
        default="benchmark_output/benchmark_memories.json",
        help="Path to benchmark memories JSON file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of memories per batch"
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding (faster, no vector search)"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel extraction"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )

    args = parser.parse_args()

    info("=" * 60, category="benchmark")
    info("TEST BATCH LOAD", category="benchmark")
    info("=" * 60, category="benchmark")

    # Initialize
    memory = MemoryInterface()
    loader = BatchLoader(memory)

    # File path
    filepath = args.file

    if not Path(filepath).exists():
        error(f"File not found: {filepath}", category="benchmark")
        error("Run benchmark generator first to create test data.", category="benchmark")
        return

    # Load and insert
    info(f"Loading from: {filepath}", category="benchmark")
    info(f"Batch size: {args.batch_size}", category="benchmark")
    info(f"Skip embedding: {args.skip_embedding}", category="benchmark")
    info(f"Parallel extract: {not args.no_parallel}", category="benchmark")
    info(f"Workers: {args.workers}", category="benchmark")

    data = loader.load_file(filepath)
    if not data:
        error("No data loaded", category="benchmark")
        return

    info(f"Loaded {len(data)} items", category="benchmark")

    texts = loader.extract_texts(data)
    info(f"Extracted {len(texts)} texts", category="benchmark")

    if texts:
        count = loader.insert_batch(
            texts,
            batch_size=args.batch_size,
            skip_embedding=args.skip_embedding,
            parallel_extract=not args.no_parallel,
            max_workers=args.workers,
        )
        info(f"Inserted {count} memories", category="benchmark")
    else:
        error("No texts extracted", category="benchmark")


if __name__ == "__main__":
    main()
