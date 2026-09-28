"""Standalone temporal retrieval worker for Memoria.

Temporal retrieval is intentionally independent of semantic retrieval.

Pipeline:

    query
        -> TemporalParser
        -> TemporalIndex candidate retrieval
        -> temporal relevance scoring
        -> ordered temporal candidates

This worker does NOT:
    - perform FAISS retrieval
    - perform BM25 retrieval
    - perform semantic retrieval
    - perform session resolution
    - mutate CandidateRecord objects
    - invoke FusionWorker

The query handler is responsible for combining this ranked result with
the existing FAISS+BM25 fusion result.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from cache.config import settings
from core.logger import debug

from .temporal_index import TemporalIndex
from .temporal_parser import TemporalConstraint, TemporalParser
from .workers import Worker, _get_shard_config


class TemporalWorker(Worker):
    """Standalone temporal retrieval worker."""

    def __init__(self, temporal_index: TemporalIndex):
        self.temporal_index = temporal_index
        self.parser = TemporalParser()

    # ================================================================
    # MAIN ENTRY POINT
    # ================================================================

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()

        query = str(
            payload.get("query")
            or payload.get("query_text")
            or ""
        ).strip()

        limit = int(
            payload.get(
                "top_k",
                payload.get(
                    "limit",
                    getattr(settings, "TOP_K", 50),
                ),
            )
            or 50
        )
        limit = max(0, limit)

        shard_id, num_shards = _get_shard_config(payload)

        if not query or limit <= 0:
            return self._empty_result(
                query=query,
                elapsed_ms=self._elapsed_ms(start_time),
                shard_id=shard_id,
                num_shards=num_shards,
            )

        reference_time = self._get_reference_time(payload)

        parsed = self._parser_for(reference_time).parse(
            query,
            reference_time=reference_time,
        )

        constraints = parsed.get("constraints", [])
        has_temporal_intent = bool(
            parsed.get("has_temporal_intent", False)
        )

        # ------------------------------------------------------------
        # No temporal intent at all.
        # ------------------------------------------------------------

        if not has_temporal_intent and not constraints:
            return self._empty_result(
                query=query,
                elapsed_ms=self._elapsed_ms(start_time),
                shard_id=shard_id,
                num_shards=num_shards,
                diagnostics={
                    "has_temporal_intent": False,
                    "constraint_count": 0,
                },
            )

        # ------------------------------------------------------------
        # Temporal language exists, but it could not be resolved into
        # an actual temporal constraint.
        #
        # Example:
        #     "when did I meet Melanie?"
        #
        # Do NOT manufacture semantic retrieval here.
        # The normal FAISS/BM25 branch handles the semantic question.
        # ------------------------------------------------------------

        if not constraints:
            return self._empty_result(
                query=query,
                elapsed_ms=self._elapsed_ms(start_time),
                shard_id=shard_id,
                num_shards=num_shards,
                diagnostics={
                    "has_temporal_intent": has_temporal_intent,
                    "constraint_count": 0,
                    "unresolved_temporal_intent": True,
                },
            )

        # ------------------------------------------------------------
        # Temporal candidate discovery.
        #
        # Temporal retrieval is global over the temporal index.
        # It is intentionally NOT sharded like FAISS/BM25 retrieval.
        # ------------------------------------------------------------

        source_limit = limit

        candidates = self._retrieve_candidates(
            constraints,
            source_limit,
        )

        if not candidates:
            return {
                "source": "temporal",
                "active": True,
                "candidates": [],
                "count": 0,
                "diagnostics": self._diagnostics(
                    parsed,
                    constraints,
                    reference_time,
                    elapsed_ms=self._elapsed_ms(start_time),
                    shard_id=shard_id,
                    num_shards=num_shards,
                    source_limit=source_limit,
                    candidate_count=0,
                    scored_count=0,
                ),
            }

        # ------------------------------------------------------------
        # Temporal relevance scoring.
        # ------------------------------------------------------------

        scored = self._score_temporal(
            candidates,
            constraints,
            reference_time,
        )

        # Temporal results are global and must not pass through the
        # generic semantic shard filter.

        # Deterministic ordering.
        scored.sort(
            key=lambda item: (-item[1], item[0])
        )

        scored = scored[:limit]

        elapsed_ms = self._elapsed_ms(start_time)

        debug(
            f"[TemporalWorker]: "
            f"temporal={elapsed_ms:.2f}ms, "
            f"constraints={len(constraints)}, "
            f"source_limit={source_limit}, "
            f"retrieved={len(candidates)}, "
            f"scored={len(scored)}"
        )

        return {
            "source": "temporal",
            "active": True,
            "candidates": scored,
            "count": len(scored),
            "diagnostics": self._diagnostics(
                parsed,
                constraints,
                reference_time,
                elapsed_ms=elapsed_ms,
                shard_id=shard_id,
                num_shards=num_shards,
                source_limit=source_limit,
                candidate_count=len(candidates),
                scored_count=len(scored),
            ),
        }

    # ================================================================
    # TEMPORAL CANDIDATE DISCOVERY
    # ================================================================

    def _retrieve_candidates(
        self,
        constraints: List[TemporalConstraint],
        limit: int,
    ) -> List[Tuple[int, float]]:
        """Retrieve candidate IDs from the temporal index."""

        return self.temporal_index.search_constraints(
            constraints,
            limit,
        )

    # ================================================================
    # TEMPORAL SCORING
    # ================================================================

    def _score_temporal(
        self,
        candidates: List[Tuple[int, float]],
        constraints: List[TemporalConstraint],
        reference_time: Optional[datetime],
    ) -> List[Tuple[int, float]]:
        """Score candidates using temporal relevance only.

        The TemporalIndex supplies ordering for pure ordering constraints
        such as first/oldest and most_recent/latest/newest. That ordering
        is preserved here instead of being destroyed by a memory-ID
        tiebreaker.
        """

        scored: List[Tuple[int, float, int]] = []

        ordering_relations = {
            "first",
            "oldest",
            "last",
            "newest",
            "most_recent",
            "latest",
        }

        has_ordering_constraint = any(
            str(getattr(constraint, "relation", "")).lower()
            in ordering_relations
            for constraint in constraints
        )

        for position, (memory_id, _) in enumerate(
            candidates,
            start=1,
        ):
            memory = self.temporal_index.get_temporal_data(
                memory_id
            )

            if not memory:
                continue

            created_at = memory.get("created_at")

            if not created_at:
                continue

            dt = self._parse_datetime(created_at)

            if dt is None:
                continue

            score, matched = self._calculate_score(
                dt,
                constraints,
                reference_time,
            )

            # Pure ordering constraints rely on the order returned by
            # TemporalIndex. Give those candidates a deterministic
            # decreasing score so that later sorting preserves that
            # ordering.
            if score <= 0.0 and has_ordering_constraint:
                score = 1.0 / position

            if score <= 0.0:
                continue

            scored.append(
                (
                    int(memory_id),
                    score,
                    position,
                )
            )

            if matched:
                debug(
                    f"[Temporal] memory={memory_id} "
                    f"score={score:.4f} "
                    f"matched={matched}"
                )

        # Score remains primary. Original temporal-index position is
        # the secondary key, so equal temporal scores preserve the
        # index's intended ordering.
        scored.sort(
            key=lambda item: (
                -item[1],
                item[2],
                item[0],
            )
        )

        return [
            (memory_id, score)
            for memory_id, score, _ in scored
        ]

    # ================================================================
    # SCORE CALCULATION
    # ================================================================

    def _calculate_score(
        self,
        dt: datetime,
        constraints: List[TemporalConstraint],
        reference_time: Optional[datetime],
    ) -> Tuple[float, List[str]]:
        """Calculate temporal relevance for one memory."""

        score = 0.0
        matched: List[str] = []

        dt = self._normalize_datetime(dt)

        if reference_time is not None:
            reference_time = self._normalize_datetime(
                reference_time
            )

        exact_boost = self._setting(
            "TEMPORAL_EXACT_MATCH_BOOST",
            1.0,
        )

        adjacent_boost = self._setting(
            "TEMPORAL_ADJACENT_BOOST",
            0.5,
        )

        recency_scale = self._setting(
            "TEMPORAL_RECENCY_SCALE",
            14.0,
        )

        conv_boost = self._setting(
            "TEMPORAL_CONVERSATIONAL_BOOST",
            0.5,
        )

        if recency_scale <= 0:
            recency_scale = 14.0

        for constraint in constraints:
            relation = str(
                getattr(constraint, "relation", "")
            ).lower()

            start = getattr(
                constraint,
                "start",
                None,
            )

            end = getattr(
                constraint,
                "end",
                None,
            )

            # --------------------------------------------------------
            # BETWEEN / explicit ranges
            # --------------------------------------------------------

            if relation in {
                "between",
                "during",
            }:
                if (
                    isinstance(start, datetime)
                    and isinstance(end, datetime)
                ):
                    start = self._normalize_datetime(start)
                    end = self._normalize_datetime(end)

                    if start <= dt < end:
                        score += self._interval_score(
                            dt,
                            start,
                            end,
                        )
                        matched.append(relation)

                continue

            # --------------------------------------------------------
            # BEFORE / UNTIL
            # --------------------------------------------------------

            if relation in {
                "before",
                "until",
            }:
                boundary = end or start

                if isinstance(boundary, datetime):
                    boundary = self._normalize_datetime(
                        boundary
                    )

                    if dt <= boundary:
                        score += self._boundary_score(
                            boundary,
                            dt,
                        )
                        matched.append(relation)

                continue

            # --------------------------------------------------------
            # AFTER / SINCE
            # --------------------------------------------------------

            if relation in {
                "after",
                "since",
            }:
                boundary = start or end

                if isinstance(boundary, datetime):
                    boundary = self._normalize_datetime(
                        boundary
                    )

                    if dt >= boundary:
                        score += self._boundary_score(
                            dt,
                            boundary,
                        )
                        matched.append(relation)

                continue

            # --------------------------------------------------------
            # MOST RECENT / LATEST
            # --------------------------------------------------------

            if relation in {
                "most_recent",
                "latest",
            }:
                if reference_time is not None:
                    age_days = max(
                        0.0,
                        (
                            reference_time - dt
                        ).total_seconds() / 86400.0,
                    )

                    recency = math.exp(
                        -age_days / recency_scale
                    )

                    score += recency * conv_boost
                    matched.append("most_recent")

                continue

            # --------------------------------------------------------
            # FIRST / OLDEST
            # --------------------------------------------------------

            if relation in {
                "first",
                "oldest",
            }:
                # Candidate discovery supplies oldest-first ordering.
                # _score_temporal() preserves that ordering.
                continue

            # --------------------------------------------------------
            # LAST / NEWEST
            # --------------------------------------------------------

            if relation in {
                "last",
                "newest",
            }:
                if reference_time is not None:
                    age_days = max(
                        0.0,
                        (
                            reference_time - dt
                        ).total_seconds() / 86400.0,
                    )

                    recency = math.exp(
                        -age_days / recency_scale
                    )

                    score += recency * conv_boost
                    matched.append("last")

                continue

            # --------------------------------------------------------
            # Conversational recency metadata.
            #
            # This is backward-compatible with the richer old parser.
            # The current parser does not need to provide it yet.
            # --------------------------------------------------------

            metadata = getattr(
                constraint,
                "metadata",
                None,
            )

            if isinstance(metadata, dict):
                if metadata.get("conversational"):
                    recency_level = metadata.get(
                        "recency_level"
                    )

                    if reference_time is not None:
                        age_days = max(
                            0.0,
                            (
                                reference_time - dt
                            ).total_seconds() / 86400.0,
                        )

                        if (
                            recency_level == "very_recent"
                            and age_days <= 2
                        ):
                            score += conv_boost
                            matched.append(
                                "very_recent"
                            )

                        elif (
                            recency_level == "recent"
                            and age_days <= 7
                        ):
                            score += conv_boost * 0.6
                            matched.append(
                                "recent"
                            )

                        elif (
                            recency_level == "historical"
                            and age_days > 30
                        ):
                            score += conv_boost * 0.6
                            matched.append(
                                "historical"
                            )

        return (
            min(1.0, score),
            matched,
        )

    # ================================================================
    # SCORING HELPERS
    # ================================================================

    @staticmethod
    def _boundary_score(
        dt: datetime,
        target: datetime,
    ) -> float:
        """Score proximity to a before/after boundary."""

        days = abs(
            (
                dt - target
            ).total_seconds()
        ) / 86400.0

        return 0.5 * max(
            0.0,
            1.0 - days / 30.0,
        )

    @staticmethod
    def _interval_score(
        dt: datetime,
        start: datetime,
        end: datetime,
    ) -> float:
        """Score position within a temporal interval."""

        duration = max(
            1.0,
            (
                end - start
            ).total_seconds() / 86400.0,
        )

        distance_from_start = (
            dt - start
        ).total_seconds() / 86400.0

        normalized = (
            distance_from_start / duration
        )

        center_distance = abs(
            normalized - 0.5
        )

        return max(
            0.0,
            1.0 - center_distance,
        )

    # ================================================================
    # REFERENCE TIME
    # ================================================================

    def _get_reference_time(
        self,
        payload: Dict[str, Any],
    ) -> Optional[datetime]:
        """Read the temporal anchor supplied by the caller.

        There is intentionally NO datetime.now() fallback.
        """

        value = (
            payload.get("reference_time")
            or payload.get("query_time")
            or payload.get("conversation_time")
        )

        if value is None:
            return None

        if isinstance(value, datetime):
            return self._normalize_datetime(value)

        if isinstance(value, str):
            try:
                return self._normalize_datetime(
                    datetime.fromisoformat(
                        value.replace("Z", "+00:00")
                    )
                )
            except ValueError:
                debug(
                    "[TemporalWorker] "
                    f"Invalid reference_time={value!r}"
                )

        return None

    def _parser_for(
        self,
        reference_time: Optional[datetime],
    ) -> TemporalParser:
        """Return the existing parser instance.

        The parser receives reference_time through parse(), not through
        its constructor.
        """

        return self.parser

    # ================================================================
    # DATETIME HELPERS
    # ================================================================

    @staticmethod
    def _normalize_datetime(
        value: datetime,
    ) -> datetime:
        if value.tzinfo is None:
            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )

    @classmethod
    def _parse_datetime(
        cls,
        value: Any,
    ) -> Optional[datetime]:

        if isinstance(value, datetime):
            return cls._normalize_datetime(value)

        if not isinstance(value, str):
            return None

        try:
            dt = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except ValueError:
            return None

        return cls._normalize_datetime(dt)

    # ================================================================
    # SETTINGS / DIAGNOSTICS
    # ================================================================

    @staticmethod
    def _setting(
        name: str,
        default: float,
    ) -> float:
        value = getattr(
            settings,
            name,
            default,
        )

        try:
            value = float(value)
        except (TypeError, ValueError):
            return default

        if not math.isfinite(value):
            return default

        return value

    @staticmethod
    def _elapsed_ms(
        start_time: float,
    ) -> float:
        return (
            time.perf_counter() - start_time
        ) * 1000.0

    def _diagnostics(
        self,
        parsed: Dict[str, Any],
        constraints: List[TemporalConstraint],
        reference_time: Optional[datetime],
        *,
        elapsed_ms: float,
        shard_id: int,
        num_shards: int,
        source_limit: int,
        candidate_count: int,
        scored_count: int,
    ) -> Dict[str, Any]:

        return {
            "expressions": parsed.get(
                "expressions",
                [],
            ),
            "has_temporal_intent": bool(
                parsed.get(
                    "has_temporal_intent",
                    False,
                )
            ),
            "has_temporal_constraint": bool(
                constraints
            ),
            "constraint_count": len(
                constraints
            ),
            "constraints": [
                {
                    "relation": getattr(
                        constraint,
                        "relation",
                        "",
                    ),
                    "text": getattr(
                        constraint,
                        "text",
                        "",
                    ),
                    "start": (
                        constraint.start.isoformat()
                        if getattr(
                            constraint,
                            "start",
                            None,
                        )
                        else None
                    ),
                    "end": (
                        constraint.end.isoformat()
                        if getattr(
                            constraint,
                            "end",
                            None,
                        )
                        else None
                    ),
                }
                for constraint in constraints
            ],
            "reference_time": (
                reference_time.isoformat()
                if reference_time
                else None
            ),
            "source_limit": source_limit,
            "candidate_count": candidate_count,
            "scored_count": scored_count,
            "elapsed_ms": elapsed_ms,
            "shard_id": shard_id,
            "num_shards": num_shards,
        }

    @staticmethod
    def _empty_result(
        *,
        query: str,
        elapsed_ms: float,
        shard_id: int,
        num_shards: int,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        data = {
            "query": query,
            "candidate_count": 0,
            "scored_count": 0,
            "elapsed_ms": elapsed_ms,
            "shard_id": shard_id,
            "num_shards": num_shards,
        }

        if diagnostics:
            data.update(diagnostics)

        return {
            "source": "temporal",
            "active": False,
            "candidates": [],
            "count": 0,
            "diagnostics": data,
        }
