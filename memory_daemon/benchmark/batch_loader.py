"""
batch_loader.py

Loads memory datasets into the memory system.

Designed for:
- synthetic generators
- dataset imports
- benchmark preparation

Avoids thousands of HTTP calls.
"""

import json
import time
from pathlib import Path
from typing import List, Union, Dict, Any

from core.logger import debug, info, error


class BatchLoader:

    def __init__(self, memory_interface):
        self.memory = memory_interface
        info("[BatchLoader] Initialized", category="benchmark")

    # -----------------------------------------
    # LOAD FILE
    # -----------------------------------------

    def load_file(self, filepath: str) -> List[Any]:
        """
        Load data from a JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            List: Parsed JSON data
        """
        path = Path(filepath)

        if not path.exists():
            error(f"[BatchLoader] File not found: {filepath}", category="benchmark")
            return []

        try:
            with open(path, "r", encoding="utf8") as f:
                data = json.load(f)
            debug(f"[BatchLoader] Loaded {len(data) if isinstance(data, list) else '?'} items from {filepath}", category="benchmark")
            return data
        except json.JSONDecodeError as e:
            error(f"[BatchLoader] Invalid JSON in {filepath}: {e}", category="benchmark")
            return []
        except Exception as e:
            error(f"[BatchLoader] Error loading {filepath}: {e}", category="benchmark")
            return []

    # -----------------------------------------
    # NORMALIZE INPUT
    # -----------------------------------------

    def extract_texts(self, data: List[Any]) -> List[str]:
        """
        Extract text strings from various data formats.

        Supports:
        - List of strings
        - List of dicts with "text" or "memory" fields
        - List of dicts with "normalized_text" or "content" fields
        """
        if not data:
            return []

        texts = []

        for item in data:
            if isinstance(item, str):
                texts.append(item)

            elif isinstance(item, dict):
                # Try common field names
                text = (
                    item.get("text")
                    or item.get("normalized_text")
                    or item.get("memory")
                    or item.get("content")
                    or item.get("body")
                )

                if text:
                    texts.append(text)
                else:
                    # If no text field, try to serialize the whole item
                    # (skip large metadata-heavy items)
                    if len(item) <= 10:  # Only serialize small dicts
                        try:
                            serialized = json.dumps(item)
                            if len(serialized) < 1000:
                                texts.append(serialized)
                        except:
                            pass

            elif isinstance(item, list):
                # Recurse for nested lists
                texts.extend(self.extract_texts(item))

        debug(f"[BatchLoader] Extracted {len(texts)} texts from {len(data)} items", category="benchmark")
        return texts

    # -----------------------------------------
    # BATCH INSERT
    # -----------------------------------------

    def insert_batch(self, texts: List[str], batch_size: int = 100) -> int:
        """
        Insert texts into memory in batches.

        Args:
            texts: List of text strings
            batch_size: Number of texts per batch

        Returns:
            int: Number of texts stored
        """
        total = len(texts)

        if total == 0:
            info("[BatchLoader] No texts to insert", category="benchmark")
            return 0

        stored = 0
        start = time.perf_counter()

        info("=" * 60, category="benchmark")
        info("[BATCH LOAD START]", category="benchmark")
        info(f"Memories: {total}", category="benchmark")
        info("=" * 60, category="benchmark")

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]

            try:
                self.memory.store_many(batch)
                stored += len(batch)

                percent = (stored / total) * 100
                info(f"[Progress] {stored}/{total} ({percent:.1f}%)", category="benchmark")

            except Exception as e:
                error(f"[BatchLoader] Error inserting batch: {e}", category="benchmark")

        elapsed = time.perf_counter() - start

        info("=" * 60, category="benchmark")
        info("[BATCH COMPLETE]", category="benchmark")
        info(f"Stored: {stored}", category="benchmark")
        info(f"Runtime: {elapsed:.2f} seconds", category="benchmark")
        info("=" * 60, category="benchmark")

        return stored

    # -----------------------------------------
    # LOAD AND INSERT (combined)
    # -----------------------------------------

    def load_and_insert(self, filepath: str, batch_size: int = 100) -> int:
        """
        Load from file and insert into memory in one call.

        Args:
            filepath: Path to JSON file
            batch_size: Number of texts per batch

        Returns:
            int: Number of texts stored
        """
        data = self.load_file(filepath)
        if not data:
            return 0

        texts = self.extract_texts(data)
        if not texts:
            info("[BatchLoader] No texts extracted", category="benchmark")
            return 0

        return self.insert_batch(texts, batch_size)


# -----------------------------------------
# CLI TEST
# -----------------------------------------

if __name__ == "__main__":
    # Simple test
    from shared.memory_interface import MemoryInterface

    memory = MemoryInterface()
    loader = BatchLoader(memory)

    # Load and insert from benchmark file
    result = loader.load_and_insert("benchmark_output/benchmark_memories.json")
    print(f"Loaded {result} memories")
