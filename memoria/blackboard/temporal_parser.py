"""
Temporal parsing for standalone Memoria temporal retrieval.

The parser extracts temporal constraints that can be resolved directly
against memory.created_at.

It deliberately does not:
    - inspect memories
    - inspect sessions
    - perform semantic retrieval
    - rank semantic candidates
    - mutate CandidateRecord objects
    - perform event-relative resolution

The public parse() contract is intentionally compatible with the current
TemporalWorker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple


UTC = timezone.utc


@dataclass(frozen=True)
class TemporalConstraint:
    """A temporal constraint resolvable against memory timestamps."""

    relation: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    text: str = ""


class TemporalParser:
    """Parse resolvable temporal language into TemporalConstraint objects."""

    _ISO_DATE = re.compile(
        r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"
    )

    _YEAR = re.compile(
        r"\b((?:19|20)\d{2})\b"
    )

    _RELATIVE_DAY = re.compile(
        r"\b(today|yesterday|tomorrow)\b",
        re.IGNORECASE,
    )

    _MODIFIER = re.compile(
        r"\b(last|next|this|past|previous|coming)\s+"
        r"(\d+)?\s*"
        r"(day|days|week|weeks|month|months|year|years)\b",
        re.IGNORECASE,
    )

    _INTERVAL = re.compile(
        r"\b(\d+)\s+"
        r"(day|days|week|weeks|month|months|year|years)\s+"
        r"(ago|from\s+now)\b",
        re.IGNORECASE,
    )

    _BEFORE = re.compile(
        r"\b(?:before|prior to)\s+"
        r"((?:19|20)\d{2}(?:-\d{1,2}-\d{1,2})?)\b",
        re.IGNORECASE,
    )

    _AFTER = re.compile(
        r"\b(?:after|following)\s+"
        r"((?:19|20)\d{2}(?:-\d{1,2}-\d{1,2})?)\b",
        re.IGNORECASE,
    )

    _SINCE = re.compile(
        r"\bsince\s+"
        r"((?:19|20)\d{2}(?:-\d{1,2}-\d{1,2})?)\b",
        re.IGNORECASE,
    )

    _UNTIL = re.compile(
        r"\buntil\s+"
        r"((?:19|20)\d{2}(?:-\d{1,2}-\d{1,2})?)\b",
        re.IGNORECASE,
    )

    _BETWEEN = re.compile(
        r"\bbetween\s+"
        r"((?:19|20)\d{2}(?:-\d{1,2}-\d{1,2})?)"
        r"\s+and\s+"
        r"((?:19|20)\d{2}(?:-\d{1,2}-\d{1,2})?)\b",
        re.IGNORECASE,
    )

    _FROM_TO = re.compile(
        r"\bfrom\s+"
        r"((?:19|20)\d{2}(?:-\d{1,2}-\d{1,2})?)"
        r"\s+to\s+"
        r"((?:19|20)\d{2}(?:-\d{1,2}-\d{1,2})?)\b",
        re.IGNORECASE,
    )

    _RECENCY = re.compile(
        r"\b(?:"
        r"most\s+recent|"
        r"most\s+recently|"
        r"latest|"
        r"newest|"
        r"recent|"
        r"recently|"
        r"just\s+now|"
        r"just\s+mentioned|"
        r"just\s+said|"
        r"moments?\s+ago"
        r")\b",
        re.IGNORECASE,
    )

    _FIRST = re.compile(
        r"\bfirst\b",
        re.IGNORECASE,
    )

    _BARE_LAST = re.compile(
        r"\blast\s+(?:thing|time|event|job|place|person|one|visit|trip|purchase)\b",
        re.IGNORECASE,
    )

    _TEMPORAL_INTENT = re.compile(
        r"\b(?:"
        r"when|"
        r"what\s+(?:time|day|date|year)|"
        r"yesterday|today|tomorrow|"
        r"recent(?:ly)?|latest|newest|"
        r"before|after|since|until|during|"
        r"between|"
        r"from\s+.+?\s+to|"
        r"\d+\s+(?:day|days|week|weeks|month|months|year|years)"
        r"(?:\s+ago|\s+from\s+now)|"
        r"(?:19|20)\d{2}"
        r")\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _normalize_reference(
        reference_time: Any,
    ) -> Optional[datetime]:
        if reference_time is None:
            return None

        if isinstance(reference_time, datetime):
            dt = reference_time
        elif isinstance(reference_time, str):
            value = reference_time.strip()

            if not value:
                return None

            try:
                dt = datetime.fromisoformat(
                    value.replace("Z", "+00:00")
                )
            except ValueError:
                return None
        else:
            return None

        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)

        return dt.astimezone(UTC)

    @staticmethod
    def _parse_date(
        value: str,
    ) -> Optional[datetime]:
        value = value.strip()

        try:
            if re.fullmatch(r"(?:19|20)\d{2}", value):
                return datetime(
                    int(value),
                    1,
                    1,
                    tzinfo=UTC,
                )

            dt = datetime.fromisoformat(value)

            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)

            return dt.astimezone(UTC)

        except ValueError:
            return None

    @staticmethod
    def _unit_delta(
        amount: int,
        unit: str,
    ) -> timedelta:
        normalized = unit.lower()

        if normalized.startswith("day"):
            return timedelta(days=amount)

        if normalized.startswith("week"):
            return timedelta(weeks=amount)

        if normalized.startswith("month"):
            return timedelta(days=30 * amount)

        if normalized.startswith("year"):
            return timedelta(days=365 * amount)

        return timedelta(0)

    @staticmethod
    def _day_bounds(
        value: datetime,
    ) -> Tuple[datetime, datetime]:
        start = value.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

        return start, start + timedelta(days=1)

    @staticmethod
    def _year_bounds(
        year: int,
    ) -> Tuple[datetime, datetime]:
        start = datetime(
            year,
            1,
            1,
            tzinfo=UTC,
        )

        return start, datetime(
            year + 1,
            1,
            1,
            tzinfo=UTC,
        )

    @staticmethod
    def _span_overlaps(
        start: int,
        end: int,
        spans: List[Tuple[int, int]],
    ) -> bool:
        return any(
            start < other_end and end > other_start
            for other_start, other_end in spans
        )

    def _literal_range(
        self,
        value: str,
    ) -> Optional[Tuple[datetime, datetime]]:
        """Turn an ISO date or year into a half-open temporal range."""

        value = value.strip()

        if re.fullmatch(r"(?:19|20)\d{2}", value):
            return self._year_bounds(int(value))

        dt = self._parse_date(value)

        if dt is None:
            return None

        return self._day_bounds(dt)

    def _relative_day_range(
        self,
        word: str,
        reference: datetime,
    ) -> Tuple[datetime, datetime]:
        offsets = {
            "today": 0,
            "yesterday": -1,
            "tomorrow": 1,
        }

        target = reference + timedelta(
            days=offsets[word.lower()]
        )

        return self._day_bounds(target)

    def _modifier_range(
        self,
        modifier: str,
        amount: int,
        unit: str,
        reference: datetime,
    ) -> Tuple[datetime, datetime]:
        """Resolve last/next/this/past/previous/coming."""

        modifier = modifier.lower()
        delta = self._unit_delta(amount, unit)

        if modifier in {"last", "previous", "past"}:
            end = reference
            start = reference - delta
            return start, end

        if modifier in {"next", "coming"}:
            start = reference
            end = reference + delta
            return start, end

        # Preserve the old parser's semantics for "this".
        start = reference - delta
        end = reference + delta

        return start, end

    def parse(
        self,
        query: str,
        reference_time: Any = None,
    ) -> Dict[str, Any]:
        """
        Parse a query into the contract expected by TemporalWorker.

        Returns exactly the fields consumed by the current temporal worker:

            constraints
            has_temporal_constraint
            has_temporal_intent
            has_recency
            reference_time
        """

        text = query or ""
        reference = self._normalize_reference(
            reference_time
        )

        constraints: List[TemporalConstraint] = []

        if not text.strip():
            return {
                "constraints": [],
                "has_temporal_constraint": False,
                "has_temporal_intent": False,
                "has_recency": False,
                "reference_time": reference,
            }

        date_spans: List[Tuple[int, int]] = []

        # Temporal expressions that consume their own dates.
        # Dates inside these spans must not also become standalone
        # "between" constraints.
        consumed_date_spans: List[Tuple[int, int]] = []

        for pattern in (
            self._BEFORE,
            self._AFTER,
            self._SINCE,
            self._UNTIL,
            self._BETWEEN,
            self._FROM_TO,
        ):
            for match in pattern.finditer(text):
                consumed_date_spans.append(
                    (match.start(), match.end())
                )

        # ----------------------------------------------------------
        # 1. Explicit ISO dates
        # ----------------------------------------------------------

        for match in self._ISO_DATE.finditer(text):
            if self._span_overlaps(
                match.start(),
                match.end(),
                consumed_date_spans,
            ):
                continue

            value = match.group(0)
            temporal_range = self._literal_range(value)

            if temporal_range is None:
                continue

            start, end = temporal_range

            date_spans.append(
                (match.start(), match.end())
            )

            constraints.append(
                TemporalConstraint(
                    relation="between",
                    start=start,
                    end=end,
                    text=value,
                )
            )

        # ----------------------------------------------------------
        # 2. Explicit years
        # ----------------------------------------------------------

        for match in self._YEAR.finditer(text):
            if self._span_overlaps(
                match.start(),
                match.end(),
                date_spans,
            ):
                continue

            if self._span_overlaps(
                match.start(),
                match.end(),
                consumed_date_spans,
            ):
                continue

            year = int(match.group(1))
            start, end = self._year_bounds(year)

            constraints.append(
                TemporalConstraint(
                    relation="between",
                    start=start,
                    end=end,
                    text=match.group(0),
                )
            )

        # ----------------------------------------------------------
        # 3. today / yesterday / tomorrow
        # ----------------------------------------------------------

        if reference is not None:
            for match in self._RELATIVE_DAY.finditer(text):
                start, end = self._relative_day_range(
                    match.group(1),
                    reference,
                )

                constraints.append(
                    TemporalConstraint(
                        relation="between",
                        start=start,
                        end=end,
                        text=match.group(0),
                    )
                )

        # ----------------------------------------------------------
        # 4. last/next/this/past/previous/coming N units
        # ----------------------------------------------------------

        if reference is not None:
            for match in self._MODIFIER.finditer(text):
                modifier = match.group(1)
                amount = (
                    int(match.group(2))
                    if match.group(2)
                    else 1
                )
                unit = match.group(3)

                start, end = self._modifier_range(
                    modifier,
                    amount,
                    unit,
                    reference,
                )

                constraints.append(
                    TemporalConstraint(
                        relation="between",
                        start=start,
                        end=end,
                        text=match.group(0),
                    )
                )

        # ----------------------------------------------------------
        # 5. N units ago / N units from now
        # ----------------------------------------------------------

        if reference is not None:
            for match in self._INTERVAL.finditer(text):
                amount = int(match.group(1))
                unit = match.group(2)
                direction = match.group(3).lower()

                delta = self._unit_delta(
                    amount,
                    unit,
                )

                if direction == "ago":
                    target = reference - delta
                else:
                    target = reference + delta

                start, end = self._day_bounds(target)

                constraints.append(
                    TemporalConstraint(
                        relation="between",
                        start=start,
                        end=end,
                        text=match.group(0),
                    )
                )

        # ----------------------------------------------------------
        # 6. Explicit BEFORE
        # ----------------------------------------------------------

        for match in self._BEFORE.finditer(text):
            target = self._parse_date(match.group(1))

            if target is None:
                continue

            constraints.append(
                TemporalConstraint(
                    relation="before",
                    end=target,
                    text=match.group(0),
                )
            )

        # ----------------------------------------------------------
        # 7. Explicit AFTER
        # ----------------------------------------------------------

        for match in self._AFTER.finditer(text):
            target = self._parse_date(match.group(1))

            if target is None:
                continue

            constraints.append(
                TemporalConstraint(
                    relation="after",
                    start=target,
                    text=match.group(0),
                )
            )

        # ----------------------------------------------------------
        # 8. SINCE
        # ----------------------------------------------------------

        for match in self._SINCE.finditer(text):
            target = self._parse_date(match.group(1))

            if target is None:
                continue

            constraints.append(
                TemporalConstraint(
                    relation="after",
                    start=target,
                    text=match.group(0),
                )
            )

        # ----------------------------------------------------------
        # 9. UNTIL
        # ----------------------------------------------------------

        for match in self._UNTIL.finditer(text):
            target = self._parse_date(match.group(1))

            if target is None:
                continue

            constraints.append(
                TemporalConstraint(
                    relation="before",
                    end=target,
                    text=match.group(0),
                )
            )

        # ----------------------------------------------------------
        # 10. BETWEEN X AND Y
        # ----------------------------------------------------------

        for match in self._BETWEEN.finditer(text):
            start_range = self._literal_range(
                match.group(1)
            )
            end_range = self._literal_range(
                match.group(2)
            )

            if start_range is None or end_range is None:
                continue

            start = start_range[0]
            end = end_range[1]

            if end <= start:
                continue

            constraints.append(
                TemporalConstraint(
                    relation="between",
                    start=start,
                    end=end,
                    text=match.group(0),
                )
            )

        # ----------------------------------------------------------
        # 11. FROM X TO Y
        # ----------------------------------------------------------

        for match in self._FROM_TO.finditer(text):
            start_range = self._literal_range(
                match.group(1)
            )
            end_range = self._literal_range(
                match.group(2)
            )

            if start_range is None or end_range is None:
                continue

            start = start_range[0]
            end = end_range[1]

            if end <= start:
                continue

            constraints.append(
                TemporalConstraint(
                    relation="between",
                    start=start,
                    end=end,
                    text=match.group(0),
                )
            )

        # ----------------------------------------------------------
        # 12. Most recent / latest / newest / recent
        # ----------------------------------------------------------

        recency = bool(
            self._RECENCY.search(text)
        )

        if recency:
            constraints.append(
                TemporalConstraint(
                    relation="most_recent",
                    text=text,
                )
            )

        # ----------------------------------------------------------
        # 13. Temporal intent
        # ----------------------------------------------------------

        has_temporal_intent = bool(
            self._TEMPORAL_INTENT.search(text)
            or self._FIRST.search(text)
            or self._BARE_LAST.search(text)
        )

        # ----------------------------------------------------------
        # 14. Deduplicate constraints
        # ----------------------------------------------------------

        deduped: List[TemporalConstraint] = []
        seen = set()

        for constraint in constraints:
            key = (
                constraint.relation,
                constraint.start,
                constraint.end,
                constraint.text,
            )

            if key in seen:
                continue

            seen.add(key)
            deduped.append(constraint)

        constraints = deduped

        return {
            "constraints": constraints,
            "has_temporal_constraint": bool(
                constraints
            ),
            "has_temporal_intent": has_temporal_intent,
            "has_recency": recency,
            "reference_time": reference,
        }


def parse_temporal_query(
    query: str,
    reference_time: Any = None,
) -> Dict[str, Any]:
    """Compatibility helper for callers that want a one-shot parse."""

    return TemporalParser().parse(
        query,
        reference_time,
    )
