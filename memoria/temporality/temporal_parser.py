"""
Temporal parsing subsystem for Memoria.

The parser extracts temporal expressions and constraints from natural-language
queries without resolving them against retrieved memories.

Pipeline:

    query
      ↓
    TemporalParser
      ↓
    TemporalExpression[]
      ↓
    TemporalConstraint[]
      ↓
    TemporalPostProcessor
      ↓
    resolved temporal evidence

The parser is benchmark-agnostic.

It may operate against an optional reference time for deterministic
resolution of relative expressions, but it never inspects:

    * memories
    * sessions
    * turns
    * retrieval candidates
    * benchmark-specific structures

Event-relative expressions such as:

    "after I moved"
    "before I graduated"
    "since starting college"

remain unresolved and are handed to the post-processing layer.
"""

from __future__ import annotations
from core.logger import debug

import re

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Temporal types
# ---------------------------------------------------------------------------


class TemporalExpressionType(str, Enum):
    """Types of temporal expressions found in natural language."""

    DATE = "date"
    YEAR = "year"
    RELATIVE = "relative"
    DURATION = "duration"
    ORDERING = "ordering"
    RANGE = "range"
    RECENCY = "recency"
    EVENT_ANCHOR = "event_anchor"
    SESSION = "session"  # NEW: session-relative references


class TemporalRelation(str, Enum):
    """Relationships between temporal points, intervals, or events."""

    BEFORE = "before"
    AFTER = "after"
    SINCE = "since"
    UNTIL = "until"
    BETWEEN = "between"
    DURING = "during"
    MOST_RECENT = "most_recent"
    FIRST = "first"
    LAST = "last"
    SESSION = "session"  # NEW: session-relative relation


@dataclass
class TemporalExpression:
    """
    Raw temporal expression extracted from natural language.

    The expression may contain a resolved temporal value, but resolution of
    event-relative language is intentionally deferred.
    """

    expression_type: TemporalExpressionType
    text: str
    start: int
    end: int

    # Normalized/resolved value where possible.
    value: Any = None

    # Relative-expression metadata.
    modifier: Optional[str] = None
    unit: Optional[str] = None
    amount: Optional[int] = None

    # Event-relative anchor, e.g. "I moved to Chicago".
    event_anchor: Optional[str] = None

    # Session-relative metadata (NEW)
    session_reference: Optional[str] = None
    session_value: Optional[int] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalConstraint:
    """
    Semantic temporal constraint produced by the parser.

    Targets may be:

        datetime
        int year
        unresolved event text
        interval endpoints
        session number (NEW)

    The post-processor is responsible for resolving event-relative targets.
    """

    relation: TemporalRelation

    # Primary target.
    target: Any = None

    # Optional second endpoint.
    target_end: Any = None

    # True only when all required temporal values are resolved.
    resolved: bool = False

    # Original source text.
    text: Optional[str] = None

    # Character offsets in original query.
    start: Optional[int] = None
    end: Optional[int] = None

    # Additional semantic metadata.
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TemporalParser:
    """
    Extract temporal intent from natural-language queries.

    This class deliberately does NOT:

        * retrieve memories
        * inspect sessions
        * inspect turns
        * rank candidates
        * infer event dates
        * perform temporal reasoning

    It extracts and normalizes temporal intent only.
    """

    TIME_UNITS = {
        "day": timedelta(days=1),
        "days": timedelta(days=1),
        "week": timedelta(weeks=1),
        "weeks": timedelta(weeks=1),
        "month": timedelta(days=30),
        "months": timedelta(days=30),
        "year": timedelta(days=365),
        "years": timedelta(days=365),
    }

    # ------------------------------------------------------------------
    # Explicit temporal literals
    # ------------------------------------------------------------------

    DATE_PATTERN = re.compile(
        r"\b(\d{4})-(\d{2})-(\d{2})\b"
    )

    YEAR_PATTERN = re.compile(
        r"\b(19\d{2}|20\d{2}|21\d{2})\b"
    )

    # ------------------------------------------------------------------
    # Relative temporal expressions
    # ------------------------------------------------------------------

    RELATIVE_DAY_PATTERN = re.compile(
        r"\b(today|yesterday|tomorrow)\b",
        re.IGNORECASE,
    )

    MODIFIER_PATTERN = re.compile(
        r"\b(last|next|this|past|previous|coming)\s+"
        r"(\d+)?\s*"
        r"(day|days|week|weeks|month|months|year|years)\b",
        re.IGNORECASE,
    )

    INTERVAL_PATTERN = re.compile(
        r"\b(\d+)\s+"
        r"(day|days|week|weeks|month|months|year|years)\s+"
        r"(ago|from\s+now)\b",
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Relational temporal language
    # ------------------------------------------------------------------

    RELATION_PATTERN = re.compile(
        r"\b(before|after|since|until)\s+"
        r"(.+?)(?="
        r"\s+\b(?:before|after|since|until|between|from)\b"
        r"|[?.!,;]"
        r"|$"
        r")",
        re.IGNORECASE,
    )

    BETWEEN_PATTERN = re.compile(
        r"\bbetween\s+(.+?)\s+\band\b\s+(.+?)(?=[?.!,;]|$)",
        re.IGNORECASE,
    )

    RANGE_PATTERN = re.compile(
        r"\bfrom\s+(.+?)\s+\bto\b\s+(.+?)(?=[?.!,;]|$)",
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Ordering / recency
    # ------------------------------------------------------------------

    MOST_RECENT_PATTERN = re.compile(
        r"\b(?:most\s+recent|latest|newest)\b",
        re.IGNORECASE,
    )

    FIRST_PATTERN = re.compile(
        r"\bfirst\b",
        re.IGNORECASE,
    )

    LAST_PATTERN = re.compile(
        r"\blast\b",
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # NEW: Session-relative patterns
    # ------------------------------------------------------------------

    SESSION_PATTERNS = {
        # "session 3", "session #3", "session3"
        "session_number": re.compile(
            r"\b(?:session|conv|conversation)\s*#?\s*(\d+)\b",
            re.IGNORECASE,
        ),
        # "3 sessions ago"
        "sessions_ago": re.compile(
            r"(\d+)\s+sessions?\s+ago\b",
            re.IGNORECASE,
        ),
        # "previous session", "last session"
        "previous_session": re.compile(
            r"\b(?:previous|prior|last)\s+session\b",
            re.IGNORECASE,
        ),
        # "first session", "initial session"
        "first_session": re.compile(
            r"\b(?:first|initial)\s+session\b",
            re.IGNORECASE,
        ),
        # "earlier session"
        "earlier_session": re.compile(
            r"\b(?:earlier)\s+session\b",
            re.IGNORECASE,
        ),
        # "next session", "following session"
        "next_session": re.compile(
            r"\b(?:next|following)\s+session\b",
            re.IGNORECASE,
        ),
        # "session before" / "session after"
        "session_before": re.compile(
            r"\bsession\s+before\b",
            re.IGNORECASE,
        ),
        "session_after": re.compile(
            r"\bsession\s+after\b",
            re.IGNORECASE,
        ),
    }

    # ------------------------------------------------------------------
    # NEW: Conversational recency markers
    # ------------------------------------------------------------------

    CONVERSATIONAL_RECENCY_PATTERNS = {
        "very_recent": re.compile(
            r"\b(?:just\s+mentioned|just\s+said|just\s+now|moments?\s+ago)\b",
            re.IGNORECASE,
        ),
        "recent": re.compile(
            r"\b(?:a\s+little\s+while\s+ago|a\s+bit\s+ago|recently\s+(?:you|we|they)\s+(?:mentioned|said|talked|discussed))\b",
            re.IGNORECASE,
        ),
        "historical": re.compile(
            r"\b(?:back\s+then|remember\s+when|way\s+back|earlier\s+you\s+said|previously\s+you\s+said|earlier\s+we\s+discussed)\b",
            re.IGNORECASE,
        ),
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        reference_time: Optional[datetime] = None,
    ) -> None:
        """
        Configure a default temporal reference.

        The default is wall-clock time for normal interactive use.

        Benchmark callers should provide an explicit reference_time either
        here or to parse().
        """

        self.reference_time = (
            reference_time
            if reference_time is not None
            else datetime.now()
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def parse(
        self,
        query_text: str,
        reference_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Parse temporal intent from query text.

        Args:
            query_text:
                Natural-language query.

            reference_time:
                Optional per-call temporal reference.

                If omitted, the parser's configured reference_time is used.

                This is intentionally resolved per call so benchmark
                adapters can supply conversation-specific reference times
                without mutating parser state.

        Returns:
            {
                "expressions": List[TemporalExpression],
                "constraints": List[TemporalConstraint],
                "has_temporal_constraint": bool,
                "has_session_constraint": bool,  # NEW
                "has_conversational_recency": bool,  # NEW
            }
        """

        if not query_text or not query_text.strip():
            return {
                "expressions": [],
                "constraints": [],
                "has_temporal_constraint": False,
                "has_session_constraint": False,
                "has_conversational_recency": False,
            }

        ref_time = (
            reference_time
            if reference_time is not None
            else self.reference_time
        )

        expressions: List[TemporalExpression] = []
        constraints: List[TemporalConstraint] = []
        has_session_constraint = False
        has_conversational_recency = False

        # --------------------------------------------------------------
        # 1. Explicit ISO dates
        # --------------------------------------------------------------

        date_spans: List[tuple[int, int]] = []

        for match in self.DATE_PATTERN.finditer(query_text):
            year, month, day = map(int, match.groups())

            try:
                value = datetime(year, month, day)
            except ValueError:
                # Invalid dates are ignored rather than crashing retrieval.
                continue

            date_spans.append(
                (match.start(), match.end())
            )

            expressions.append(
                TemporalExpression(
                    expression_type=TemporalExpressionType.DATE,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    value=value,
                )
            )

            constraints.append(
                TemporalConstraint(
                    relation=TemporalRelation.DURING,
                    target=value,
                    resolved=True,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    metadata={
                        "granularity": "date",
                    },
                )
            )

        # --------------------------------------------------------------
        # 2. Explicit years
        # --------------------------------------------------------------

        for match in self.YEAR_PATTERN.finditer(query_text):
            # Do not treat the year component of an ISO date as an
            # independent temporal expression.
            if self._overlaps_any(
                match.start(),
                match.end(),
                date_spans,
            ):
                continue

            year = int(match.group(1))

            expressions.append(
                TemporalExpression(
                    expression_type=TemporalExpressionType.YEAR,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    value=year,
                )
            )

            constraints.append(
                TemporalConstraint(
                    relation=TemporalRelation.DURING,
                    target=year,
                    resolved=True,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    metadata={
                        "granularity": "year",
                    },
                )
            )

        # --------------------------------------------------------------
        # 3. today / yesterday / tomorrow
        # --------------------------------------------------------------

        relative_day_values = {
            "today": 0,
            "yesterday": -1,
            "tomorrow": 1,
        }

        for match in self.RELATIVE_DAY_PATTERN.finditer(query_text):
            word = match.group(1).lower()

            resolved = ref_time + timedelta(
                days=relative_day_values[word]
            )

            expressions.append(
                TemporalExpression(
                    expression_type=TemporalExpressionType.RELATIVE,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    value=resolved,
                    modifier=word,
                    metadata={
                        "reference_time": ref_time,
                        "granularity": "day",
                    },
                )
            )

            constraints.append(
                TemporalConstraint(
                    relation=TemporalRelation.DURING,
                    target=resolved,
                    resolved=True,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    metadata={
                        "modifier": word,
                        "reference_time": ref_time,
                        "granularity": "day",
                    },
                )
            )

        # --------------------------------------------------------------
        # 4. last/next/this/past/previous/coming N units
        # --------------------------------------------------------------

        for match in self.MODIFIER_PATTERN.finditer(query_text):
            modifier = match.group(1).lower()
            amount_text = match.group(2)
            unit = match.group(3).lower()

            amount = (
                int(amount_text)
                if amount_text
                else 1
            )

            delta = self._duration(amount, unit)

            if modifier in {
                "last",
                "previous",
                "past",
            }:
                start = ref_time - delta
                end = ref_time

                relation = TemporalRelation.BETWEEN

            elif modifier in {
                "next",
                "coming",
            }:
                start = ref_time
                end = ref_time + delta

                relation = TemporalRelation.BETWEEN

            else:
                # "this week/month/year".

                start = ref_time - delta
                end = ref_time + delta

                relation = TemporalRelation.BETWEEN

            resolved = (start, end)

            expressions.append(
                TemporalExpression(
                    expression_type=TemporalExpressionType.RELATIVE,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    value=resolved,
                    modifier=modifier,
                    unit=unit,
                    amount=amount,
                    metadata={
                        "reference_time": ref_time,
                    },
                )
            )

            constraints.append(
                TemporalConstraint(
                    relation=relation,
                    target=start,
                    target_end=end,
                    resolved=True,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    metadata={
                        "modifier": modifier,
                        "unit": unit,
                        "amount": amount,
                        "reference_time": ref_time,
                    },
                )
            )

        # --------------------------------------------------------------
        # 5. "3 days ago" / "2 weeks from now"
        # --------------------------------------------------------------

        for match in self.INTERVAL_PATTERN.finditer(query_text):
            amount = int(match.group(1))
            unit = match.group(2).lower()
            direction = match.group(3).lower()

            delta = self._duration(amount, unit)

            if direction == "ago":
                resolved = ref_time - delta
                relation = TemporalRelation.BEFORE
            else:
                resolved = ref_time + delta
                relation = TemporalRelation.AFTER

            expressions.append(
                TemporalExpression(
                    expression_type=TemporalExpressionType.DURATION,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    value=resolved,
                    unit=unit,
                    amount=amount,
                    metadata={
                        "direction": direction,
                        "reference_time": ref_time,
                    },
                )
            )

            constraints.append(
                TemporalConstraint(
                    relation=relation,
                    target=resolved,
                    resolved=True,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    metadata={
                        "amount": amount,
                        "unit": unit,
                        "direction": direction,
                        "reference_time": ref_time,
                    },
                )
            )

        # --------------------------------------------------------------
        # 6. BEFORE / AFTER / SINCE / UNTIL
        # --------------------------------------------------------------

        for match in self.RELATION_PATTERN.finditer(query_text):
            relation_text = match.group(1).lower()
            target_text = match.group(2).strip()

            relation = TemporalRelation(
                relation_text
            )

            target = self._parse_literal_temporal(
                target_text,
                reference_time=ref_time,
            )

            resolved = target is not None

            if not resolved:
                # Preserve the event description for the postprocessor.

                target = target_text

            expression_type = (
                TemporalExpressionType.DATE
                if resolved
                else TemporalExpressionType.EVENT_ANCHOR
            )

            expressions.append(
                TemporalExpression(
                    expression_type=expression_type,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    value=target,
                    event_anchor=(
                        None
                        if resolved
                        else target_text
                    ),
                    metadata={
                        "relation": relation_text,
                        "target_text": target_text,
                    },
                )
            )

            constraints.append(
                TemporalConstraint(
                    relation=relation,
                    target=target,
                    resolved=resolved,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    metadata={
                        "target_text": target_text,
                        "reference_time": ref_time,
                        "requires_resolution": not resolved,
                    },
                )
            )

        # --------------------------------------------------------------
        # 7. BETWEEN X AND Y
        # --------------------------------------------------------------

        for match in self.BETWEEN_PATTERN.finditer(query_text):
            start_text = match.group(1).strip()
            end_text = match.group(2).strip()

            start = self._parse_literal_temporal(
                start_text,
                reference_time=ref_time,
            )

            end = self._parse_literal_temporal(
                end_text,
                reference_time=ref_time,
            )

            start_resolved = start is not None
            end_resolved = end is not None

            expressions.append(
                TemporalExpression(
                    expression_type=TemporalExpressionType.RANGE,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    value=(start, end),
                    metadata={
                        "start_text": start_text,
                        "end_text": end_text,
                    },
                )
            )

            constraints.append(
                TemporalConstraint(
                    relation=TemporalRelation.BETWEEN,
                    target=(
                        start
                        if start_resolved
                        else start_text
                    ),
                    target_end=(
                        end
                        if end_resolved
                        else end_text
                    ),
                    resolved=(
                        start_resolved
                        and end_resolved
                    ),
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    metadata={
                        "start_text": start_text,
                        "end_text": end_text,
                        "requires_resolution": not (
                            start_resolved
                            and end_resolved
                        ),
                    },
                )
            )

        # --------------------------------------------------------------
        # 8. FROM X TO Y
        # --------------------------------------------------------------

        for match in self.RANGE_PATTERN.finditer(query_text):
            start_text = match.group(1).strip()
            end_text = match.group(2).strip()

            start = self._parse_literal_temporal(
                start_text,
                reference_time=ref_time,
            )

            end = self._parse_literal_temporal(
                end_text,
                reference_time=ref_time,
            )

            start_resolved = start is not None
            end_resolved = end is not None

            expressions.append(
                TemporalExpression(
                    expression_type=TemporalExpressionType.RANGE,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    value=(start, end),
                    metadata={
                        "start_text": start_text,
                        "end_text": end_text,
                        "syntax": "from_to",
                    },
                )
            )

            constraints.append(
                TemporalConstraint(
                    relation=TemporalRelation.BETWEEN,
                    target=(
                        start
                        if start_resolved
                        else start_text
                    ),
                    target_end=(
                        end
                        if end_resolved
                        else end_text
                    ),
                    resolved=(
                        start_resolved
                        and end_resolved
                    ),
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    metadata={
                        "start_text": start_text,
                        "end_text": end_text,
                        "syntax": "from_to",
                        "requires_resolution": not (
                            start_resolved
                            and end_resolved
                        ),
                    },
                )
            )

        # --------------------------------------------------------------
        # 9. Most recent / latest / newest
        # --------------------------------------------------------------

        match = self.MOST_RECENT_PATTERN.search(query_text)

        if match:
            expressions.append(
                TemporalExpression(
                    expression_type=TemporalExpressionType.RECENCY,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    modifier="most_recent",
                )
            )

            constraints.append(
                TemporalConstraint(
                    relation=TemporalRelation.MOST_RECENT,
                    resolved=False,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    metadata={
                        "requires_ordering": True,
                    },
                )
            )

        # --------------------------------------------------------------
        # 10. First
        # --------------------------------------------------------------

        match = self.FIRST_PATTERN.search(query_text)

        if match:
            expressions.append(
                TemporalExpression(
                    expression_type=TemporalExpressionType.ORDERING,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    modifier="first",
                )
            )

            constraints.append(
                TemporalConstraint(
                    relation=TemporalRelation.FIRST,
                    resolved=False,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    metadata={
                        "requires_ordering": True,
                    },
                )
            )

        # --------------------------------------------------------------
        # 11. Bare "last"
        # --------------------------------------------------------------

        if self._is_temporal_last(query_text):
            match = self.LAST_PATTERN.search(query_text)

            if match:
                expressions.append(
                    TemporalExpression(
                        expression_type=TemporalExpressionType.ORDERING,
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        modifier="last",
                    )
                )

                constraints.append(
                    TemporalConstraint(
                        relation=TemporalRelation.LAST,
                        resolved=False,
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        metadata={
                            "requires_ordering": True,
                        },
                    )
                )

        # --------------------------------------------------------------
        # 12. NEW: Session-relative patterns
        # --------------------------------------------------------------

        for pattern_name, pattern in self.SESSION_PATTERNS.items():
            for match in pattern.finditer(query_text):
                has_session_constraint = True

                if pattern_name == "session_number":
                    session_num = int(match.group(1))
                    session_ref = f"session_{session_num}"

                    expressions.append(
                        TemporalExpression(
                            expression_type=TemporalExpressionType.SESSION,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            session_reference="session_number",
                            session_value=session_num,
                            metadata={
                                "session_number": session_num,
                            },
                        )
                    )

                    constraints.append(
                        TemporalConstraint(
                            relation=TemporalRelation.SESSION,
                            target=session_num,
                            resolved=False,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            metadata={
                                "session_type": "session_number",
                                "session_value": session_num,
                                "requires_session_resolution": True,
                            },
                        )
                    )

                elif pattern_name == "sessions_ago":
                    delta = int(match.group(1))

                    expressions.append(
                        TemporalExpression(
                            expression_type=TemporalExpressionType.SESSION,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            session_reference="sessions_ago",
                            session_value=-delta,
                            metadata={
                                "sessions_ago": delta,
                            },
                        )
                    )

                    constraints.append(
                        TemporalConstraint(
                            relation=TemporalRelation.SESSION,
                            target=-delta,
                            resolved=False,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            metadata={
                                "session_type": "sessions_ago",
                                "session_delta": -delta,
                                "requires_session_resolution": True,
                            },
                        )
                    )

                elif pattern_name == "previous_session":
                    expressions.append(
                        TemporalExpression(
                            expression_type=TemporalExpressionType.SESSION,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            session_reference="previous_session",
                            session_value=-1,
                            metadata={
                                "session_type": "previous_session",
                            },
                        )
                    )

                    constraints.append(
                        TemporalConstraint(
                            relation=TemporalRelation.SESSION,
                            target=-1,
                            resolved=False,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            metadata={
                                "session_type": "previous_session",
                                "session_delta": -1,
                                "requires_session_resolution": True,
                            },
                        )
                    )

                elif pattern_name == "first_session":
                    expressions.append(
                        TemporalExpression(
                            expression_type=TemporalExpressionType.SESSION,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            session_reference="first_session",
                            session_value=1,
                            metadata={
                                "session_type": "first_session",
                            },
                        )
                    )

                    constraints.append(
                        TemporalConstraint(
                            relation=TemporalRelation.SESSION,
                            target=1,
                            resolved=False,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            metadata={
                                "session_type": "first_session",
                                "session_number": 1,
                                "requires_session_resolution": True,
                            },
                        )
                    )

                elif pattern_name == "earlier_session":
                    expressions.append(
                        TemporalExpression(
                            expression_type=TemporalExpressionType.SESSION,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            session_reference="earlier_session",
                            session_value=-1,
                            metadata={
                                "session_type": "earlier_session",
                            },
                        )
                    )

                    constraints.append(
                        TemporalConstraint(
                            relation=TemporalRelation.SESSION,
                            target=-1,
                            resolved=False,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            metadata={
                                "session_type": "earlier_session",
                                "session_delta": -1,
                                "requires_session_resolution": True,
                            },
                        )
                    )

                elif pattern_name == "next_session":
                    expressions.append(
                        TemporalExpression(
                            expression_type=TemporalExpressionType.SESSION,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            session_reference="next_session",
                            session_value=1,
                            metadata={
                                "session_type": "next_session",
                            },
                        )
                    )

                    constraints.append(
                        TemporalConstraint(
                            relation=TemporalRelation.SESSION,
                            target=1,
                            resolved=False,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            metadata={
                                "session_type": "next_session",
                                "session_delta": 1,
                                "requires_session_resolution": True,
                            },
                        )
                    )

                elif pattern_name == "session_before":
                    expressions.append(
                        TemporalExpression(
                            expression_type=TemporalExpressionType.SESSION,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            session_reference="session_before",
                            session_value=-1,
                            metadata={
                                "session_type": "session_before",
                            },
                        )
                    )

                    constraints.append(
                        TemporalConstraint(
                            relation=TemporalRelation.SESSION,
                            target=-1,
                            resolved=False,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            metadata={
                                "session_type": "session_before",
                                "session_delta": -1,
                                "requires_session_resolution": True,
                            },
                        )
                    )

                elif pattern_name == "session_after":
                    expressions.append(
                        TemporalExpression(
                            expression_type=TemporalExpressionType.SESSION,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            session_reference="session_after",
                            session_value=1,
                            metadata={
                                "session_type": "session_after",
                            },
                        )
                    )

                    constraints.append(
                        TemporalConstraint(
                            relation=TemporalRelation.SESSION,
                            target=1,
                            resolved=False,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            metadata={
                                "session_type": "session_after",
                                "session_delta": 1,
                                "requires_session_resolution": True,
                            },
                        )
                    )

        # --------------------------------------------------------------
        # 13. NEW: Conversational recency markers
        # --------------------------------------------------------------

        for recency_level, pattern in self.CONVERSATIONAL_RECENCY_PATTERNS.items():
            for match in pattern.finditer(query_text):
                has_conversational_recency = True

                expressions.append(
                    TemporalExpression(
                        expression_type=TemporalExpressionType.RECENCY,
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        modifier=recency_level,
                        metadata={
                            "recency_level": recency_level,
                            "conversational": True,
                        },
                    )
                )

                constraints.append(
                    TemporalConstraint(
                        relation=TemporalRelation.MOST_RECENT if recency_level == "very_recent" else TemporalRelation.DURING,
                        resolved=False,
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        metadata={
                            "recency_level": recency_level,
                            "conversational": True,
                            "requires_ordering": recency_level == "very_recent",
                        },
                    )
                )

        # --------------------------------------------------------------
        # Final ordering
        # --------------------------------------------------------------

        expressions.sort(
            key=lambda item: (
                item.start,
                item.end,
            )
        )

        constraints.sort(
            key=lambda item: (
                item.start
                if item.start is not None
                else -1,
                item.end
                if item.end is not None
                else -1,
            )
        )

        return {
            "expressions": expressions,
            "constraints": constraints,
            "has_temporal_constraint": bool(
                expressions or constraints
            ),
            "has_session_constraint": has_session_constraint,
            "has_conversational_recency": has_conversational_recency,
        }

    # ==================================================================
    # Literal resolution
    # ==================================================================

    def _parse_literal_temporal(
        self,
        text: str,
        reference_time: Optional[datetime] = None,
    ) -> Optional[Any]:
        """
        Resolve only self-contained temporal literals.

        Event descriptions are intentionally NOT resolved here.
        """

        text = text.strip()

        ref_time = (
            reference_time
            if reference_time is not None
            else self.reference_time
        )

        # --------------------------------------------------------------
        # ISO date
        # --------------------------------------------------------------

        match = self.DATE_PATTERN.fullmatch(text)

        if match:
            year, month, day = map(
                int,
                match.groups(),
            )

            try:
                return datetime(
                    year,
                    month,
                    day,
                )
            except ValueError:
                return None

        # --------------------------------------------------------------
        # Year
        # --------------------------------------------------------------

        match = self.YEAR_PATTERN.fullmatch(text)

        if match:
            return int(match.group(1))

        # --------------------------------------------------------------
        # Relative day
        # --------------------------------------------------------------

        relative_days = {
            "today": 0,
            "yesterday": -1,
            "tomorrow": 1,
        }

        lowered = text.lower()

        if lowered in relative_days:
            return (
                ref_time
                + timedelta(
                    days=relative_days[lowered]
                )
            )

        # --------------------------------------------------------------
        # X days ago / X weeks from now
        # --------------------------------------------------------------

        match = self.INTERVAL_PATTERN.fullmatch(text)

        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            direction = match.group(3).lower()

            delta = self._duration(
                amount,
                unit,
            )

            if direction == "ago":
                return ref_time - delta

            return ref_time + delta

        return None

    # ==================================================================
    # Helpers
    # ==================================================================

    def _duration(
        self,
        amount: int,
        unit: str,
    ) -> timedelta:
        """Convert a temporal quantity into a timedelta."""

        normalized = unit.lower()

        if normalized not in self.TIME_UNITS:
            raise ValueError(
                f"Unsupported temporal unit: {unit}"
            )

        return (
            self.TIME_UNITS[normalized]
            * amount
        )

    @staticmethod
    def _overlaps_any(
        start: int,
        end: int,
        spans: List[tuple[int, int]],
    ) -> bool:
        """Return True when a character span overlaps another span."""

        for other_start, other_end in spans:
            if start < other_end and end > other_start:
                return True

        return False

    def _is_temporal_last(
        self,
        query_text: str,
    ) -> bool:
        """
        Determine whether bare "last" is likely temporal/order language.

        Examples:

            "What was the last thing I bought?" -> True
            "What was my last job?"             -> True

        This remains intentionally conservative.
        """

        lowered = query_text.lower()

        if "most recent" in lowered:
            return False

        # Already handled as relative periods.
        if re.search(
            r"\blast\s+(?:"
            r"day|week|month|year|"
            r"\d+\s+(?:"
            r"day|days|week|weeks|month|months|year|years"
            r")"
            r")\b",
            lowered,
        ):
            return False

        return bool(
            re.search(
                r"\blast\s+(?:"
                r"thing|time|event|job|"
                r"place|person|one|visit|trip|purchase"
                r")\b",
                lowered,
            )
        )


# ---------------------------------------------------------------------------
# Session constraint resolver (NEW)
# ---------------------------------------------------------------------------



def resolve_session_constraints(
    constraints: List[TemporalConstraint],
    current_session: int,
    total_sessions: Optional[int] = None,
    all_sessions: Optional[List[int]] = None,
) -> List[TemporalConstraint]:
    """
    Resolve session-relative constraints against the session timeline.

    If all_sessions is provided, session-relative offsets are resolved
    by position within the sparse session timeline.

    IMPORTANT:
        `total_sessions` is a COUNT of sessions, not the maximum session
        ID. When `all_sessions` is provided, resolved targets must NOT be
        clamped using `total_sessions - 1`, because sparse session IDs can
        legitimately be greater than that value.

    The resolver never mutates the caller-owned TemporalConstraint
    objects. Every returned constraint is an independent copy, including
    its metadata dictionary.
    """

    resolved_constraints = []

    for constraint in constraints:
        # ------------------------------------------------------------
        # Clone the constraint before doing anything with it.
        #
        # In particular, metadata must be copied because resolution
        # writes resolver state into that dictionary.
        # ------------------------------------------------------------

        resolved_metadata = dict(
            constraint.metadata or {}
        )

        resolved_constraint = TemporalConstraint(
            relation=constraint.relation,
            target=constraint.target,
            target_end=constraint.target_end,
            resolved=constraint.resolved,
            text=constraint.text,
            start=constraint.start,
            end=constraint.end,
            metadata=resolved_metadata,
        )

        # ------------------------------------------------------------
        # Nothing to resolve.
        #
        # Still return the clone so callers always receive objects
        # owned by this resolver rather than a mixture of originals
        # and copies.
        # ------------------------------------------------------------

        if not resolved_metadata.get(
            "requires_session_resolution",
            False,
        ):
            resolved_constraints.append(
                resolved_constraint
            )
            continue

        session_type = resolved_metadata.get(
            "session_type"
        )

        # ------------------------------------------------------------
        # Explicit session number
        # ------------------------------------------------------------

        if session_type == "session_number":
            target = resolved_metadata.get(
                "session_value"
            )

            if target is not None:
                resolved_constraint.target = target
                resolved_constraint.resolved = True
                resolved_metadata["resolved_session"] = target

        # ------------------------------------------------------------
        # N sessions ago
        # ------------------------------------------------------------

        elif session_type == "sessions_ago":
            delta = resolved_metadata.get(
                "session_delta",
                0,
            )

            if (
                all_sessions
                and current_session in all_sessions
            ):
                current_idx = all_sessions.index(
                    current_session
                )

                target_idx = current_idx + delta

                debug(
                    f"[DEBUG] current_idx={current_idx}, "
                    f"target_idx={target_idx}"
                )

                if 0 <= target_idx < len(all_sessions):
                    target = all_sessions[target_idx]

                    debug(
                        f"[DEBUG] target (sparse)={target}"
                    )
                else:
                    # Preserve the existing arithmetic fallback
                    # when the requested relative position is outside
                    # the known sparse timeline.
                    target = current_session + delta

                    debug(
                        f"[DEBUG] target (fallback)={target}"
                    )

            else:
                target = current_session + delta

                debug(
                    f"[DEBUG] target (arithmetic)={target}"
                )

            # --------------------------------------------------------
            # IMPORTANT:
            #
            # When all_sessions exists, `target` is an actual session
            # ID from that sparse timeline. DO NOT clamp it against
            # total_sessions - 1.
            #
            # Example:
            #     total_sessions = 27
            #     target session = 45
            #
            # 45 is valid because 27 is the number of sessions, not
            # the highest session ID.
            # --------------------------------------------------------

            if not all_sessions:
                if total_sessions is not None:
                    target = max(
                        0,
                        min(
                            target,
                            total_sessions - 1,
                        ),
                    )

            resolved_constraint.target = target
            resolved_constraint.resolved = True

            resolved_metadata["resolved_session"] = target
            resolved_metadata["session_delta_applied"] = delta

        # ------------------------------------------------------------
        # Previous / earlier session
        # ------------------------------------------------------------

        elif session_type in [
            "previous_session",
            "earlier_session",
            "session_before",
        ]:
            if (
                all_sessions
                and current_session in all_sessions
            ):
                current_idx = all_sessions.index(
                    current_session
                )

                target_idx = current_idx - 1

                if target_idx >= 0:
                    target = all_sessions[target_idx]
                else:
                    target = current_session - 1

            else:
                target = current_session - 1

            # Only apply count-based clamping when we're using the
            # arithmetic representation rather than sparse IDs.
            if not all_sessions and total_sessions is not None:
                target = max(0, target)

            resolved_constraint.target = target
            resolved_constraint.resolved = True
            resolved_metadata["resolved_session"] = target

        # ------------------------------------------------------------
        # Next session
        # ------------------------------------------------------------

        elif session_type in [
            "next_session",
            "session_after",
        ]:
            if (
                all_sessions
                and current_session in all_sessions
            ):
                current_idx = all_sessions.index(
                    current_session
                )

                target_idx = current_idx + 1

                if target_idx < len(all_sessions):
                    target = all_sessions[target_idx]
                else:
                    target = current_session + 1

            else:
                target = current_session + 1

            # Again, sparse session IDs are not bounded by
            # total_sessions - 1.
            if not all_sessions and total_sessions is not None:
                target = min(
                    target,
                    total_sessions - 1,
                )

            resolved_constraint.target = target
            resolved_constraint.resolved = True
            resolved_metadata["resolved_session"] = target

        # ------------------------------------------------------------
        # First session
        # ------------------------------------------------------------

        elif session_type == "first_session":
            target = (
                all_sessions[0]
                if all_sessions
                else 0
            )

            resolved_constraint.target = target
            resolved_constraint.resolved = True
            resolved_metadata["resolved_session"] = target

        # ------------------------------------------------------------
        # Append only the independent copy.
        # ------------------------------------------------------------

        resolved_constraints.append(
            resolved_constraint
        )

    return resolved_constraints



# ---------------------------------------------------------------------------
# Compatibility helper
# ---------------------------------------------------------------------------


def parse_temporal_query(
    query_text: str,
    reference_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Convenience entry point for callers that don't need to instantiate
    TemporalParser manually.
    """

    return TemporalParser(
        reference_time=reference_time,
    ).parse(
        query_text,
        reference_time=reference_time,
    )


def parse_with_session_context(
    query_text: str,
    current_session: int,
    total_sessions: Optional[int] = None,
    reference_time: Optional[datetime] = None,
    all_sessions: Optional[List[int]] = None,  # <-- ADD THIS
) -> Dict[str, Any]:
    """
    Parse temporal constraints and resolve session references.
    """
    parser = TemporalParser(reference_time=reference_time)
    result = parser.parse(query_text, reference_time=reference_time)

    if result.get("has_session_constraint", False):
        result["constraints"] = resolve_session_constraints(
            result["constraints"],
            current_session,
            total_sessions,
            all_sessions,  # <-- PASS THIS
        )

    result["has_resolved_constraints"] = any(
        c.resolved for c in result["constraints"]
    )

    return result
