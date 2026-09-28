"""
Temporal index for standalone temporal retrieval.

The index is retrieval-focused:

    temporal constraints
        ↓
    temporal index
        ↓
    ordered memory IDs

It does not perform semantic retrieval or ranking-pipeline work.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .temporal_parser import TemporalConstraint


UTC = timezone.utc


class TemporalIndex:
    """In-memory temporal index over the canonical memories table."""

    def __init__(self, db=None):
        self.db = db

        # Primary created_at index.
        # Kept sorted chronologically for efficient range lookup.
        self._entries: List[Tuple[datetime, int]] = []
        self._by_id: Dict[int, datetime] = {}

        # Updated-at index. This is separate because updated_at may be
        # missing even when created_at exists.
        self._updated_entries: List[Tuple[datetime, int]] = []
        self._updated_by_id: Dict[int, datetime] = {}

        # Full temporal metadata for direct lookup.
        self._metadata: Dict[int, Dict[str, Any]] = {}

        self._built = False

    # ================================================================
    # BUILD
    # ================================================================

    def build(self, records=None) -> int:
        """
        Rebuild the temporal index.

        When records are omitted, the canonical database is used.

        Returns the number of indexed memories.
        """
        if records is None:
            if self.db is None:
                records = []
            else:
                records = self.db.fetch_all()

        entries: List[Tuple[datetime, int]] = []
        updated_entries: List[Tuple[datetime, int]] = []

        by_id: Dict[int, datetime] = {}
        updated_by_id: Dict[int, datetime] = {}
        metadata: Dict[int, Dict[str, Any]] = {}

        for record in records:
            try:
                memory_id = int(record["id"])
            except (KeyError, TypeError, ValueError):
                continue

            created_at = self._parse_timestamp(
                record.get("created_at")
            )

            if created_at is None:
                continue

            updated_at = self._parse_timestamp(
                record.get("updated_at")
            )

            entries.append((created_at, memory_id))
            by_id[memory_id] = created_at

            if updated_at is not None:
                updated_entries.append(
                    (updated_at, memory_id)
                )
                updated_by_id[memory_id] = updated_at

            metadata[memory_id] = {
                "created_at": created_at.isoformat(),
                "updated_at": (
                    updated_at.isoformat()
                    if updated_at is not None
                    else None
                ),
                "metadata": record.get("metadata") or {},
            }

        entries.sort(key=lambda item: (item[0], item[1]))
        updated_entries.sort(
            key=lambda item: (item[0], item[1])
        )

        self._entries = entries
        self._by_id = by_id

        self._updated_entries = updated_entries
        self._updated_by_id = updated_by_id

        self._metadata = metadata
        self._built = True

        return len(entries)

    # ================================================================
    # STATE
    # ================================================================

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def built(self) -> bool:
        return self._built

    def _ensure_built(self) -> None:
        if not self._built:
            self.build()

    # ================================================================
    # LOOKUP
    # ================================================================

    def get_temporal_data(
        self,
        memory_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Return normalized temporal metadata for a memory."""
        self._ensure_built()
        return self._metadata.get(int(memory_id))

    # ================================================================
    # CREATED-AT RETRIEVAL
    # ================================================================

    def search_by_timestamp(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[int]:
        """
        Return memory IDs whose created_at falls in [start, end).

        Results are newest first.

        start=None means no lower bound.
        end=None means no upper bound.
        """
        self._ensure_built()

        if not self._entries:
            return []

        if limit is not None and limit <= 0:
            return []

        start = self._normalize_datetime(start)
        end = self._normalize_datetime(end)

        if start is not None:
            left = bisect_left(
                self._entries,
                (start, -1),
            )
        else:
            left = 0

        if end is not None:
            right = bisect_left(
                self._entries,
                (end, -1),
            )
        else:
            right = len(self._entries)

        if left >= right:
            return []

        matches = reversed(self._entries[left:right])

        results: List[int] = []

        for _, memory_id in matches:
            results.append(memory_id)

            if (
                limit is not None
                and len(results) >= limit
            ):
                break

        return results

    def search_before(
        self,
        end: datetime,
        limit: int,
    ) -> List[Tuple[int, float]]:
        """Return memories created before end, newest first."""
        return [
            (memory_id, 0.0)
            for memory_id in self.search_by_timestamp(
                None,
                end,
                limit,
            )
        ]

    def search_after(
        self,
        start: datetime,
        limit: int,
    ) -> List[Tuple[int, float]]:
        """Return memories created at or after start, newest first."""
        return [
            (memory_id, 0.0)
            for memory_id in self.search_by_timestamp(
                start,
                None,
                limit,
            )
        ]

    def get_recent(
        self,
        limit: int = 100,
    ) -> List[int]:
        """Return newest-created memories first."""
        return self.search_by_timestamp(
            None,
            None,
            limit,
        )

    def search_most_recent(
        self,
        limit: int,
    ) -> List[Tuple[int, float]]:
        """Return newest-created memories as worker candidates."""
        return [
            (memory_id, 0.0)
            for memory_id in self.get_recent(limit)
        ]

    # ================================================================
    # UPDATED-AT RETRIEVAL
    # ================================================================

    def search_by_updated_timestamp(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[int]:
        """
        Return memory IDs whose updated_at falls in [start, end).

        Results are newest updated first.
        """
        self._ensure_built()

        if not self._updated_entries:
            return []

        if limit is not None and limit <= 0:
            return []

        start = self._normalize_datetime(start)
        end = self._normalize_datetime(end)

        if start is not None:
            left = bisect_left(
                self._updated_entries,
                (start, -1),
            )
        else:
            left = 0

        if end is not None:
            right = bisect_left(
                self._updated_entries,
                (end, -1),
            )
        else:
            right = len(self._updated_entries)

        if left >= right:
            return []

        matches = reversed(
            self._updated_entries[left:right]
        )

        results: List[int] = []

        for _, memory_id in matches:
            results.append(memory_id)

            if (
                limit is not None
                and len(results) >= limit
            ):
                break

        return results

    def get_recent_updated(
        self,
        limit: int = 100,
    ) -> List[int]:
        """Return memories with the newest updated_at timestamps."""
        return self.search_by_updated_timestamp(
            None,
            None,
            limit,
        )

    # ================================================================
    # DIRECT DATE HELPERS
    # ================================================================

    def search_exact_date(
        self,
        date: datetime,
        limit: Optional[int] = None,
    ) -> List[int]:
        """
        Return memories created on the given calendar date.

        The end boundary is exclusive.
        """
        date = self._normalize_datetime(date)

        if date is None:
            return []

        start = datetime(
            date.year,
            date.month,
            date.day,
            tzinfo=UTC,
        )

        # Avoid constructing 23:59:59.999999. The range is naturally
        # represented as [midnight, following midnight).
        from datetime import timedelta

        end = start + timedelta(days=1)

        return self.search_by_timestamp(
            start,
            end,
            limit,
        )

    def search_year(
        self,
        year: int,
        limit: Optional[int] = None,
    ) -> List[int]:
        """Return memories created during a calendar year."""
        if year < 1:
            return []

        start = datetime(
            year,
            1,
            1,
            tzinfo=UTC,
        )

        end = datetime(
            year + 1,
            1,
            1,
            tzinfo=UTC,
        )

        return self.search_by_timestamp(
            start,
            end,
            limit,
        )

    # ================================================================
    # CONSTRAINT RETRIEVAL
    # ================================================================

    def search_constraints(
        self,
        constraints: List[TemporalConstraint],
        limit: int,
    ) -> List[Tuple[int, float]]:
        """
        Resolve temporal constraints into an ordered candidate list.

        The returned score is an internal temporal-rank score. The
        second-stage RRF uses candidate position, not this numeric value.
        """
        self._ensure_built()

        if not constraints or limit <= 0:
            return []

        # First temporal position wins when multiple constraints match
        # the same memory. This preserves the behavior of the current
        # temporal branch while allowing richer constraints.
        rank: Dict[int, int] = {}

        for constraint in constraints:
            relation = constraint.relation

            if relation == "most_recent":
                candidates = self.search_most_recent(limit)

            elif relation == "before":
                if constraint.end is None:
                    continue

                candidates = self.search_before(
                    constraint.end,
                    limit,
                )

            elif relation == "after":
                if constraint.start is None:
                    continue

                candidates = self.search_after(
                    constraint.start,
                    limit,
                )

            elif relation == "between":
                memory_ids = self.search_by_timestamp(
                    constraint.start,
                    constraint.end,
                    limit,
                )

                candidates = [
                    (memory_id, 0.0)
                    for memory_id in memory_ids
                ]

            else:
                continue

            for position, (memory_id, _) in enumerate(
                candidates,
                start=1,
            ):
                previous = rank.get(memory_id)

                if previous is None or position < previous:
                    rank[memory_id] = position

        ordered = sorted(
            rank.items(),
            key=lambda item: (item[1], item[0]),
        )

        return [
            (memory_id, 1.0 / position)
            for memory_id, position in ordered[:limit]
        ]

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

        Rebuilds the sorted indexes after the mutation.
        """
        created = self._parse_timestamp(created_at)

        if created is None:
            return False

        memory_id = int(memory_id)
        updated = self._parse_timestamp(updated_at)

        self._by_id[memory_id] = created

        if updated is not None:
            self._updated_by_id[memory_id] = updated
        else:
            self._updated_by_id.pop(memory_id, None)

        self._metadata[memory_id] = {
            "created_at": created.isoformat(),
            "updated_at": (
                updated.isoformat()
                if updated is not None
                else None
            ),
            "metadata": metadata or {},
        }

        self._rebuild_sorted_entries()
        self._built = True

        return True

    def remove(
        self,
        memory_id: int,
    ) -> bool:
        """Remove a memory from the temporal index."""
        memory_id = int(memory_id)

        if memory_id not in self._by_id:
            return False

        self._by_id.pop(memory_id, None)
        self._updated_by_id.pop(memory_id, None)
        self._metadata.pop(memory_id, None)

        self._rebuild_sorted_entries()

        return True

    def clear(self) -> None:
        """Clear the entire temporal index."""
        self._entries.clear()
        self._by_id.clear()
        self._updated_entries.clear()
        self._updated_by_id.clear()
        self._metadata.clear()
        self._built = True

    def _rebuild_sorted_entries(self) -> None:
        self._entries = sorted(
            (
                (dt, memory_id)
                for memory_id, dt in self._by_id.items()
            ),
            key=lambda item: (item[0], item[1]),
        )

        self._updated_entries = sorted(
            (
                (dt, memory_id)
                for memory_id, dt in self._updated_by_id.items()
            ),
            key=lambda item: (item[0], item[1]),
        )

    # ================================================================
    # DATETIME NORMALIZATION
    # ================================================================

    @staticmethod
    def _parse_timestamp(
        value: Any,
    ) -> Optional[datetime]:
        """Parse a timestamp into a timezone-aware UTC datetime."""
        if value is None:
            return None

        if isinstance(value, datetime):
            dt = value
        else:
            value = str(value).strip()

            if not value:
                return None

            try:
                dt = datetime.fromisoformat(
                    value.replace("Z", "+00:00")
                )
            except ValueError:
                return None

        return TemporalIndex._normalize_datetime(dt)

    @staticmethod
    def _normalize_datetime(
        value: Optional[datetime],
    ) -> Optional[datetime]:
        """Normalize a datetime to timezone-aware UTC."""
        if value is None:
            return None

        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)

        return value.astimezone(UTC)
