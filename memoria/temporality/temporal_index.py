"""
Temporal index for storing and querying temporal metadata.

The temporal index is intentionally retrieval-focused:

    query constraint
        ↓
    temporal index
        ↓
    matching memory IDs

It does not rank memories semantically. Ranking belongs to the worker/ranker.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class TemporalIndex:
    """Index for temporal metadata of memories."""

    VERSION = 1

    def __init__(
        self,
        db,
        cache_path: str = "cache/temporal_index.json",
    ):
        self.db = db
        self.cache_path = Path(cache_path)

        # memory_id -> normalized temporal record
        self._index: Dict[int, Dict[str, Any]] = {}

        self._dirty = False

    # ================================================================
    # BUILD / PERSISTENCE
    # ================================================================

    def build(self) -> None:
        """
        Rebuild the temporal index from the database.

        Existing cached state is replaced completely. This keeps the
        index deterministic and prevents stale records from surviving
        database rebuilds.
        """
        new_index: Dict[int, Dict[str, Any]] = {}

        for row in self.db.fetch_all():
            memory_id = row.get("id")

            if memory_id is None:
                continue

            created_at = self._normalize_timestamp(
                row.get("created_at")
            )

            if created_at is None:
                continue

            updated_at = self._normalize_timestamp(
                row.get("updated_at")
            )

            new_index[int(memory_id)] = {
                "created_at": created_at,
                "updated_at": updated_at,
                "metadata": row.get("metadata") or {},
            }

        self._index = new_index
        self._dirty = True

    def save(self) -> None:
        """Persist the temporal index to disk."""
        if not self._dirty:
            return

        self.cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "version": self.VERSION,
            "memories": self._index,
        }

        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                indent=2,
                ensure_ascii=False,
            )

        self._dirty = False

    def load(self) -> bool:
        """
        Load the temporal index from disk.

        Returns True only when a valid index is loaded.
        """
        if not self.cache_path.exists():
            return False

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            # Support the new format.
            if isinstance(payload, dict) and "memories" in payload:
                version = payload.get("version", 1)

                if version != self.VERSION:
                    return False

                raw_index = payload["memories"]

            # Backward compatibility with the original flat format.
            elif isinstance(payload, dict):
                raw_index = payload

            else:
                return False

            rebuilt: Dict[int, Dict[str, Any]] = {}

            for raw_id, data in raw_index.items():
                try:
                    memory_id = int(raw_id)
                except (TypeError, ValueError):
                    continue

                if not isinstance(data, dict):
                    continue

                created_at = self._normalize_timestamp(
                    data.get("created_at")
                )

                if created_at is None:
                    continue

                rebuilt[memory_id] = {
                    "created_at": created_at,
                    "updated_at": self._normalize_timestamp(
                        data.get("updated_at")
                    ),
                    "metadata": data.get("metadata") or {},
                }

            self._index = rebuilt
            self._dirty = False

            return True

        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return False

    # ================================================================
    # LOOKUP
    # ================================================================

    def get_temporal_data(
        self,
        memory_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Return temporal metadata for a memory."""
        return self._index.get(memory_id)

    # ================================================================
    # TEMPORAL RETRIEVAL
    # ================================================================

    def search_by_timestamp(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[int]:
        """
        Return memory IDs whose created_at falls inside [start, end].

        Boundaries are inclusive.

        start=None means no lower bound.
        end=None means no upper bound.
        """
        start = self._normalize_datetime(start)
        end = self._normalize_datetime(end)

        if start and end and start > end:
            return []

        results = []

        for memory_id, data in self._index.items():
            dt = self._parse_timestamp(data.get("created_at"))

            if dt is None:
                continue

            if start is not None and dt < start:
                continue

            if end is not None and dt > end:
                continue

            results.append((dt, memory_id))

        # Deterministic chronological ordering.
        results.sort(key=lambda item: (item[0], item[1]))

        return [memory_id for _, memory_id in results]

    def search_by_updated_timestamp(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[int]:
        """
        Return memory IDs whose updated_at falls inside [start, end].
        """
        start = self._normalize_datetime(start)
        end = self._normalize_datetime(end)

        if start and end and start > end:
            return []

        results = []

        for memory_id, data in self._index.items():
            dt = self._parse_timestamp(data.get("updated_at"))

            if dt is None:
                continue

            if start is not None and dt < start:
                continue

            if end is not None and dt > end:
                continue

            results.append((dt, memory_id))

        results.sort(key=lambda item: (item[0], item[1]))

        return [memory_id for _, memory_id in results]

    def get_recent(
        self,
        limit: int = 100,
    ) -> List[int]:
        """
        Return the most recently created memories.

        Results are ordered newest → oldest.
        """
        if limit <= 0:
            return []

        results = []

        for memory_id, data in self._index.items():
            dt = self._parse_timestamp(data.get("created_at"))

            if dt is None:
                continue

            results.append((dt, memory_id))

        results.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        return [
            memory_id
            for _, memory_id in results[:limit]
        ]

    def get_recent_updated(
        self,
        limit: int = 100,
    ) -> List[int]:
        """
        Return memories with the most recent updated_at timestamp.
        """
        if limit <= 0:
            return []

        results = []

        for memory_id, data in self._index.items():
            dt = self._parse_timestamp(data.get("updated_at"))

            if dt is None:
                continue

            results.append((dt, memory_id))

        results.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        return [
            memory_id
            for _, memory_id in results[:limit]
        ]

    # ================================================================
    # DIRECT RANGE HELPERS
    # ================================================================

    def search_exact_date(
        self,
        date: datetime,
    ) -> List[int]:
        """
        Return memories created on the given calendar date.

        Time-of-day is ignored.
        """
        date = self._normalize_datetime(date)

        if date is None:
            return []

        start = datetime(
            date.year,
            date.month,
            date.day,
            tzinfo=date.tzinfo,
        )

        end = start.replace(
            hour=23,
            minute=59,
            second=59,
            microsecond=999999,
        )

        return self.search_by_timestamp(start, end)

    def search_year(
        self,
        year: int,
    ) -> List[int]:
        """Return memories created during a calendar year."""
        if year < 1:
            return []

        tz = timezone.utc

        start = datetime(
            year,
            1,
            1,
            tzinfo=tz,
        )

        end = datetime(
            year,
            12,
            31,
            23,
            59,
            59,
            999999,
            tzinfo=tz,
        )

        return self.search_by_timestamp(start, end)

    # ================================================================
    # INDEX MANAGEMENT
    # ================================================================

    def add(
        self,
        memory_id: int,
        created_at: Any,
        updated_at: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add or replace a memory in the temporal index.

        Returns False when created_at cannot be parsed.
        """
        normalized_created = self._normalize_timestamp(created_at)

        if normalized_created is None:
            return False

        self._index[int(memory_id)] = {
            "created_at": normalized_created,
            "updated_at": self._normalize_timestamp(updated_at),
            "metadata": metadata or {},
        }

        self._dirty = True

        return True

    def remove(self, memory_id: int) -> bool:
        """Remove a memory from the temporal index."""
        if memory_id not in self._index:
            return False

        del self._index[memory_id]
        self._dirty = True

        return True

    def clear(self) -> None:
        """Clear the entire temporal index."""
        self._index.clear()
        self._dirty = True

    def __len__(self) -> int:
        return len(self._index)

    # ================================================================
    # DATETIME NORMALIZATION
    # ================================================================

    @staticmethod
    def _parse_timestamp(
        value: Any,
    ) -> Optional[datetime]:
        """Parse a stored timestamp into a timezone-aware datetime."""
        if not value:
            return None

        if isinstance(value, datetime):
            return TemporalIndex._normalize_datetime(value)

        if not isinstance(value, str):
            return None

        try:
            return TemporalIndex._normalize_datetime(
                datetime.fromisoformat(value)
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_datetime(
        value: Optional[datetime],
    ) -> Optional[datetime]:
        """
        Normalize datetimes to UTC-aware values.

        Naive timestamps are interpreted as UTC rather than allowing
        naive/aware comparisons to explode at runtime.
        """
        if value is None:
            return None

        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)

    @staticmethod
    def _normalize_timestamp(
        value: Any,
    ) -> Optional[str]:
        """Normalize a timestamp and store it in canonical ISO format."""
        dt = TemporalIndex._parse_timestamp(value)

        if dt is None:
            return None

        return dt.isoformat()
