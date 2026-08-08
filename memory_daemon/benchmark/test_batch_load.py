#!/usr/bin/env python3
"""
test_batch_load.py

Test script for batch loading benchmark memories.
Loads memories from benchmark_output/benchmark_memories.json
and inserts them into the memory system.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.memory_interface import MemoryInterface
from benchmark.batch_loader import BatchLoader
from core.logger import info, error


def main():
    """Load benchmark memories into the system."""
    info("=" * 60, category="benchmark")
    info("TEST BATCH LOAD", category="benchmark")
    info("=" * 60, category="benchmark")

    # Initialize
    memory = MemoryInterface()
    loader = BatchLoader(memory)

    # File path
    filepath = "benchmark_output/benchmark_memories.json"

    if not Path(filepath).exists():
        error(f"File not found: {filepath}", category="benchmark")
        error("Run benchmark generator first to create test data.", category="benchmark")
        return

    # Load and insert
    info(f"Loading from: {filepath}", category="benchmark")

    data = loader.load_file(filepath)
    if not data:
        error("No data loaded", category="benchmark")
        return

    info(f"Loaded {len(data)} items", category="benchmark")

    texts = loader.extract_texts(data)
    info(f"Extracted {len(texts)} texts", category="benchmark")

    if texts:
        count = loader.insert_batch(texts)
        info(f"Inserted {count} memories", category="benchmark")
    else:
        error("No texts extracted", category="benchmark")


if __name__ == "__main__":
    main()
