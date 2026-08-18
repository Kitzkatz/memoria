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
from typing import List, Union, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.logger import debug, info, error


class BatchLoader:

    def __init__(self, memory_interface):
        self.memory = memory_interface
        info("[BatchLoader] Initialized", category="benchmark")

    # -----------------------------------------
    # LOAD FILE
    # -----------------------------------------

    def load_file(self, filepath: str) -> List[Any]:
        """Load data from a JSON file."""
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
        """Extract text strings from various data formats."""
        if not data:
            return []

        texts = []

        for item in data:
            if isinstance(item, str):
                texts.append(item)

            elif isinstance(item, dict):
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
                    if len(item) <= 10:
                        try:
                            serialized = json.dumps(item)
                            if len(serialized) < 1000:
                                texts.append(serialized)
                        except:
                            pass

            elif isinstance(item, list):
                texts.extend(self.extract_texts(item))

        debug(f"[BatchLoader] Extracted {len(texts)} texts from {len(data)} items", category="benchmark")
        return texts

    # -----------------------------------------
    # BATCH INSERT (for raw texts)
    # -----------------------------------------

    def insert_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
        skip_embedding: bool = False,
        parallel_extract: bool = True,
        max_workers: int = 4,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        skip_embedding_build: bool = False,
    ) -> int:
        """
        Insert texts into memory in batches.

        Args:
            texts: List of text strings
            batch_size: Number of texts per batch
            skip_embedding: If True, skip embedding (faster, no vector search)
            parallel_extract: If True, extract in parallel
            max_workers: Number of parallel workers
            metadatas: Optional list of metadata dicts corresponding to each text
            skip_embedding_build: If True, skip embedding computation and vector store ops (cached FAISS)

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
        info(f"Batch size: {batch_size}", category="benchmark")
        info(f"Skip embedding: {skip_embedding}", category="benchmark")
        info(f"Parallel extract: {parallel_extract}", category="benchmark")
        info(f"Metadatas provided: {metadatas is not None}", category="benchmark")
        info(f"Skip embedding build: {skip_embedding_build}", category="benchmark")
        info("=" * 60, category="benchmark")

        # Disable auto-store during batch load
        original_auto_store = getattr(self.memory.controller, "auto_store", False)
        if hasattr(self.memory.controller, "auto_store"):
            self.memory.controller.auto_store = False

        try:
            for i in range(0, total, batch_size):
                batch = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size] if metadatas else None

                try:
                    if parallel_extract and len(batch) > 10:
                        # Parallel extraction (preserves order)
                        records = self._extract_batch_parallel(batch, max_workers)
                    else:
                        # Sequential extraction
                        records = [self.memory.controller.system.extractor.extract(text) for text in batch]

                    # If metadata provided, merge into records
                    if batch_metadatas:
                        for rec, meta in zip(records, batch_metadatas):
                            if meta:
                                rec.metadata.update(meta)

                    # Store with extracted records
                    texts_to_store = [r.text for r in records]
                    if batch_metadatas:
                        metadata_to_store = [r.metadata for r in records]
                    else:
                        metadata_to_store = None

                    ids = self.memory.remember_many(
                        texts_to_store,
                        metadatas=metadata_to_store,
                        skip_embedding_build=skip_embedding_build
                    )
                    stored += len(ids)

                    percent = (stored / total) * 100
                    elapsed = time.perf_counter() - start
                    rate = stored / elapsed if elapsed > 0 else 0
                    info(f"[Progress] {stored}/{total} ({percent:.1f}%) @ {rate:.1f} mem/s", category="benchmark")

                except Exception as e:
                    error(f"[BatchLoader] Error inserting batch: {e}", category="benchmark")

        finally:
            # Restore auto-store
            if hasattr(self.memory.controller, "auto_store"):
                self.memory.controller.auto_store = original_auto_store

        elapsed = time.perf_counter() - start

        info("=" * 60, category="benchmark")
        info("[BATCH COMPLETE]", category="benchmark")
        info(f"Stored: {stored}", category="benchmark")
        info(f"Runtime: {elapsed:.2f} seconds", category="benchmark")
        info(f"Rate: {stored / elapsed:.1f} mem/s", category="benchmark")
        info("=" * 60, category="benchmark")

        return stored

    # -----------------------------------------
    # BATCH INSERT (for pre‑extracted records)
    # -----------------------------------------

    def insert_records(
        self,
        records: List,
        batch_size: int = 100,
        skip_embedding_build: bool = False,
    ) -> int:
        """Insert pre‑extracted MemoryRecord objects in batches.

        Args:
            records: List of MemoryRecord objects
            batch_size: Number of records per batch
            skip_embedding_build: If True, skip embedding computation and vector store ops (cached FAISS)

        Returns:
            int: Number of texts stored
        """
        total = len(records)
        if total == 0:
            return 0

        stored = 0
        start = time.perf_counter()

        info("=" * 60, category="benchmark")
        info("[BATCH LOAD START (RECORDS)]", category="benchmark")
        info(f"Memories: {total}", category="benchmark")
        info(f"Batch size: {batch_size}", category="benchmark")
        info(f"Skip embedding build: {skip_embedding_build}", category="benchmark")
        info("=" * 60, category="benchmark")

        original_auto_store = getattr(self.memory.controller, "auto_store", False)
        if hasattr(self.memory.controller, "auto_store"):
            self.memory.controller.auto_store = False

        try:
            for i in range(0, total, batch_size):
                batch = records[i:i+batch_size]
                texts = [r.text for r in batch]
                metadatas = [r.metadata for r in batch]
                ids = self.memory.controller.remember_many(
                    texts,
                    metadatas=metadatas,
                    skip_embedding_build=skip_embedding_build
                )
                stored += len(ids)

                percent = (stored / total) * 100
                elapsed = time.perf_counter() - start
                rate = stored / elapsed if elapsed > 0 else 0
                info(f"[Progress] {stored}/{total} ({percent:.1f}%) @ {rate:.1f} mem/s", category="benchmark")
        finally:
            if hasattr(self.memory.controller, "auto_store"):
                self.memory.controller.auto_store = original_auto_store

        elapsed = time.perf_counter() - start
        info("=" * 60, category="benchmark")
        info("[BATCH COMPLETE]", category="benchmark")
        info(f"Stored: {stored}", category="benchmark")
        info(f"Runtime: {elapsed:.2f} seconds", category="benchmark")
        info(f"Rate: {stored / elapsed:.1f} mem/s", category="benchmark")
        info("=" * 60, category="benchmark")
        return stored

    # -----------------------------------------
    # PARALLEL EXTRACTION (preserves order)
    # -----------------------------------------

    def _extract_batch_parallel(self, texts: List[str], max_workers: int = 4) -> List:
        """Extract memories in parallel, preserving input order."""
        extractor = self.memory.controller.system.extractor

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and map to indices to preserve order
            futures = {executor.submit(extractor.extract, text): idx for idx, text in enumerate(texts)}
            results = [None] * len(texts)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    error(f"[BatchLoader] Parallel extract error: {e}", category="benchmark")
                    # Fallback to sequential for the failed one
                    results[idx] = extractor.extract(texts[idx])
            return results

    # -----------------------------------------
    # LOAD AND INSERT (combined)
    # -----------------------------------------

    def load_and_insert(
        self,
        filepath: str,
        batch_size: int = 100,
        skip_embedding: bool = False,
        parallel_extract: bool = True,
        max_workers: int = 4,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        skip_embedding_build: bool = False,
    ) -> int:
        """
        Load from file and insert into memory in one call.

        Args:
            filepath: Path to JSON file
            batch_size: Number of texts per batch
            skip_embedding: If True, skip embedding
            parallel_extract: If True, extract in parallel
            max_workers: Number of parallel workers
            metadatas: Optional list of metadata dicts
            skip_embedding_build: If True, skip embedding computation (cached FAISS)

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

        return self.insert_batch(
            texts,
            batch_size=batch_size,
            skip_embedding=skip_embedding,
            parallel_extract=parallel_extract,
            max_workers=max_workers,
            metadatas=metadatas,
            skip_embedding_build=skip_embedding_build,
        )


# -----------------------------------------
# CLI TEST
# -----------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", default="benchmark_output/benchmark_memories.json")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    from shared.memory_interface import MemoryInterface

    memory = MemoryInterface()
    loader = BatchLoader(memory)

    result = loader.load_and_insert(
        args.file,
        batch_size=args.batch_size,
        skip_embedding=args.skip_embedding,
        parallel_extract=not args.no_parallel,
        max_workers=args.workers,
    )

    print(f"Loaded {result} memories")
