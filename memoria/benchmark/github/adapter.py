from __future__ import annotations

from typing import Callable, Iterator

from .models import (
    DirectoryRecord,
    FileRecord,
    RepositoryRecord,
    SymbolRecord,
)
from .parser import GitHubParser


class GitHubAdapter:
    """Adapt a local Git repository into normalized Memoria records."""

    def __init__(
        self,
        repository_path: str,
        repository_name: str,
    ):
        self.repository_name = repository_name
        self.parser = GitHubParser(
            repository_path,
            repository_name=repository_name,
        )

    def iter_records(self) -> Iterator[dict]:
        """Yield all repository records in ingestion order."""

        yield self._normalize(self.parser.parse_repository())

        for record in self.parser.parse_directories():
            yield self._normalize(record)

        for record in self.parser.parse_files():
            yield self._normalize(record)

        for record in self.parser.parse_symbols():
            yield self._normalize(record)

    def records(self) -> list[dict]:
        """Return all normalized repository records."""

        return list(self.iter_records())

    def load(self, insert: Callable[[str, dict], object]) -> int:
        """Insert all records using the supplied Memoria insertion callable."""

        if not callable(insert):
            raise TypeError("insert must be callable")

        count = 0

        for record in self.iter_records():
            insert(
                record["text"],
                record["metadata"],
            )
            count += 1

        return count

    def repository(self) -> dict:
        """Return the normalized repository-level record."""

        return self._normalize(self.parser.parse_repository())

    def directories(self) -> list[dict]:
        """Return normalized directory records."""

        return [
            self._normalize(record)
            for record in self.parser.parse_directories()
        ]

    def files(self) -> list[dict]:
        """Return normalized file records."""

        return [
            self._normalize(record)
            for record in self.parser.parse_files()
        ]

    def symbols(self) -> list[dict]:
        """Return normalized symbol records."""

        return [
            self._normalize(record)
            for record in self.parser.parse_symbols()
        ]

    def stats(self) -> dict:
        """Return basic adapter statistics."""

        records = self.records()

        return {
            "repository": self.repository_name,
            "records": len(records),
            "repositories": sum(
                record["metadata"]["record_type"] == "repository"
                for record in records
            ),
            "directories": sum(
                record["metadata"]["record_type"] == "directory"
                for record in records
            ),
            "files": sum(
                record["metadata"]["record_type"] == "file"
                for record in records
            ),
            "symbols": sum(
                record["metadata"]["record_type"] == "symbol"
                for record in records
            ),
        }

    @staticmethod
    def _normalize(
        record: (
            RepositoryRecord
            | DirectoryRecord
            | FileRecord
            | SymbolRecord
        ),
    ) -> dict:
        """Convert a parser record into the normalized adapter shape."""

        return {
            "text": record.text,
            "metadata": dict(record.metadata),
        }
