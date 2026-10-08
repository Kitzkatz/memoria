"""
Obsidian vault ingestion adapter for Memoria.

The adapter intentionally knows about Obsidian, not about FAISS/BM25/ranking.

It converts an Obsidian vault into Memoria-shaped records while preserving:

- source paths
- vault-relative structure
- frontmatter
- wiki-links
- graph metadata
- tasks
- template status
- parsed note content

The adapter does not perform storage, embedding, ranking, or retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

from .chunking import split_text
from .parser import ObsidianParser


class ObsidianAdapter:
    """Convert an Obsidian vault into Memoria ingestion records."""

    def __init__(
        self,
        vault_path: str | Path,
        *,
        include_templates: bool = True,
        max_chars: int | None = 512,
    ):
        """
        ``max_chars`` splits sections longer than the Embedder's default
        512-character limit into several records. Pass ``None`` to keep
        one record per section.
        """
        self.vault_path = Path(vault_path).expanduser().resolve()
        self.include_templates = include_templates

        if max_chars is not None and max_chars < 1:
            raise ValueError("max_chars must be a positive integer or None")

        self.max_chars = max_chars

        if not self.vault_path.exists():
            raise FileNotFoundError(
                f"Obsidian vault does not exist: {self.vault_path}"
            )

        if not self.vault_path.is_dir():
            raise ValueError(
                f"Obsidian vault path is not a directory: {self.vault_path}"
            )

        self.parser = ObsidianParser(self.vault_path)

    # --------------------------------------------------
    # DISCOVERY
    # --------------------------------------------------

    def iter_notes(self) -> Iterator[dict]:
        """
        Iterate over normalized Memoria records.

        Each non-empty Obsidian section becomes one record, or several
        when it is longer than ``max_chars``.
        """

        paths = self.parser.discover_notes()

        for path in paths:
            records = self.parser.parse(path)

            for record in records:
                self._validate_record(record)

                if (
                    record["metadata"].get("is_template", False)
                    and not self.include_templates
                ):
                    continue

                yield from self._split_record(record)

    def _split_record(self, record: dict) -> Iterator[dict]:
        """Yield the record, split into chunks if it is too long."""

        if self.max_chars is None or len(record["text"]) <= self.max_chars:
            yield record
            return

        chunks = split_text(record["text"], self.max_chars)
        tasks = record["metadata"].get("tasks", [])

        for index, chunk in enumerate(chunks):
            yield {
                "text": chunk,
                "metadata": {
                    **record["metadata"],
                    "tasks": [
                        task for task in tasks if task["text"] in chunk
                    ],
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                },
            }

    # --------------------------------------------------
    # INSERTION
    # --------------------------------------------------

    def load(
        self,
        insert: Callable[[str, dict], object],
    ) -> int:
        """
        Insert normalized notes using a caller-supplied function.

        This keeps the adapter independent from MemoryInterface,
        MemoryController, FAISS, BM25, or any particular storage layer.

        Parameters
        ----------
        insert:
            Callable receiving:

                insert(text, metadata)

        Returns
        -------
        int
            Number of records passed to the insertion function.
        """

        if not callable(insert):
            raise TypeError("insert must be callable")

        count = 0

        for record in self.iter_notes():
            insert(
                record["text"],
                record["metadata"],
            )
            count += 1

        return count

    # --------------------------------------------------
    # MATERIALIZED RECORDS
    # --------------------------------------------------

    def records(self) -> list[dict]:
        """Return the entire vault as normalized Memoria records."""
        return list(self.iter_notes())

    # --------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------

    def stats(self) -> dict:
        """
        Return lightweight vault ingestion statistics.

        This does not touch Memoria storage.
        """

        records = self.records()

        return {
            "vault_path": str(self.vault_path),
            "records": len(records),
            "templates": sum(
                1
                for record in records
                if record["metadata"].get("is_template", False)
            ),
            "linked_notes": sum(
                1
                for record in records
                if record["metadata"].get("links")
            ),
            "tasks": sum(
                len(record["metadata"].get("tasks", []))
                for record in records
            ),
        }

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    @staticmethod
    def _validate_record(record: dict) -> None:
        """Validate the minimal adapter-to-Memoria contract."""

        if not isinstance(record, dict):
            raise TypeError(
                f"Obsidian parser returned {type(record).__name__}, "
                "expected dict"
            )

        if "text" not in record:
            raise ValueError(
                "Obsidian record is missing required 'text' field"
            )

        if "metadata" not in record:
            raise ValueError(
                "Obsidian record is missing required 'metadata' field"
            )

        if not isinstance(record["text"], str):
            raise TypeError(
                "Obsidian record 'text' must be a string"
            )

        if not isinstance(record["metadata"], dict):
            raise TypeError(
                "Obsidian record 'metadata' must be a dict"
            )


    def load_into_memory(
        self,
        memory_interface,
        *,
        batch_size: int = 100,
        skip_embedding_build: bool = False,
    ) -> int:
        """Ingest this vault through Memoria's existing BatchLoader."""

        from benchmark.batch_loader import BatchLoader

        records = self.records()

        if not records:
            return 0

        texts = [record["text"] for record in records]
        metadatas = [record["metadata"] for record in records]

        loader = BatchLoader(memory_interface)

        return loader.insert_batch(
            texts,
            batch_size=batch_size,
            metadatas=metadatas,
            skip_embedding_build=skip_embedding_build,
        )
