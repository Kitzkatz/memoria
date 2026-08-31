"""
Temporal Worker — standalone retrieval using temporal constraints.

Primary path:
    temporal query -> standalone retrieval -> temporal-scored candidates

The worker performs pure temporal retrieval based on session context
and temporal constraints. It does NOT mix with semantic retrieval.
"""

import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from cache.config import settings
from core.logger import debug
from temporality.temporal_parser import (
    TemporalParser,
    TemporalConstraint,
    TemporalRelation,
)
from temporality.temporal_parser import resolve_session_constraints
from .workers import Worker
from .workers import _get_shard_config, _shard_filter


class TemporalWorker(Worker):
    """
    Temporal standalone retrieval worker.

    Responsibilities:
        - Parse temporal intent from the query.
        - Retrieve memories based on session/temporal constraints.
        - Score candidates by temporal relevance.
        - Return candidates with temporal scores.

    It does NOT:
        - perform semantic retrieval (FAISS)
        - perform lexical retrieval (BM25)
        - perform graph retrieval
        - mix with other retrieval sources
    """

    def __init__(self, db, temporal_index=None, enable_diagnostics=None):
        self.db = db
        self.temporal_index = temporal_index
        self.parser = TemporalParser()
        self.enable_diagnostics = enable_diagnostics if enable_diagnostics is not None else settings.DEBUG

    # ================================================================
    # SESSION CONTEXT HELPERS
    # ================================================================

    def _get_temporal_context(self) -> Dict[str, Any]:
        """
        Read temporal context from system/db if available.
        Returns dict with current_session, total_sessions, reference_time.
        """
        context = {
            "current_session": 0,
            "total_sessions": None,
            "reference_time": None,
        }

        # Try to get from db (which may have reference to system)
        if hasattr(self.db, '_temporal_context'):
            ctx = self.db._temporal_context
            context["current_session"] = ctx.get("current_session", 0)
            context["total_sessions"] = ctx.get("total_sessions", None)
            context["reference_time"] = ctx.get("reference_time", None)
            if self.enable_diagnostics:
                debug(f"[Temporal] Read context from db: current={context['current_session']}, total={context['total_sessions']}")
        
        # Try to get from system if db doesn't have it
        elif hasattr(self.db, 'system') and hasattr(self.db.system, '_temporal_context'):
            ctx = self.db.system._temporal_context
            context["current_session"] = ctx.get("current_session", 0)
            context["total_sessions"] = ctx.get("total_sessions", None)
            context["reference_time"] = ctx.get("reference_time", None)
            if self.enable_diagnostics:
                debug(f"[Temporal] Read context from system: current={context['current_session']}, total={context['total_sessions']}")

        return context

    def _set_temporal_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract temporal context from payload and store for reuse.
        Returns updated context dict.
        """
        context = {
            "current_session": payload.get("current_session", 0),
            "total_sessions": payload.get("total_sessions", None),
            "reference_time": self._get_reference_time(payload),
        }
        
        # Store on db for other methods to access
        if hasattr(self.db, '_temporal_context'):
            self.db._temporal_context = context
        
        return context

    def _get_all_sessions(self) -> List[int]:
        """
        Get sorted list of all session indices from the database.
        Used for sparse session resolution.
        """
        session_set = set()
        rows = self.db.fetch_all()
        debug(f"[DEBUG] _get_all_sessions: fetched {len(rows)} rows")  # <-- DEBUG
        for row in rows:
            metadata = row.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    import json
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            session_idx = metadata.get("session_idx")
            if session_idx is not None:
                session_set.add(session_idx)
        result = sorted(session_set)
        debug(f"[DEBUG] _get_all_sessions: found {len(result)} sessions: {result}")  # <-- DEBUG
        return result

    # ================================================================
    # STANDALONE RETRIEVAL
    # ================================================================

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standalone retrieval entry point for scheduler.
        
        Returns:
            {
                "source": "temporal",
                "candidates": [(memory_id, score), ...],
                "count": int,
                "diagnostics": {...}
            }
        """
        start = time.perf_counter()

        query_text = payload.get("query_text", "")
        limit = payload.get("limit", settings.TOP_K)

        shard_id, num_shards = _get_shard_config(payload)

        if not query_text:
            return {
                "source": "temporal",
                "candidates": [],
                "count": 0,
                "error": "No query text provided",
            }

        # ---- Extract and store temporal context ----
        context = self._set_temporal_context(payload)
        reference_time = context["reference_time"]
        current_session = context["current_session"]
        total_sessions = context["total_sessions"]

        debug(f"[DEBUG] process: current_session={current_session}, total_sessions={total_sessions}")  # <-- DEBUG

        debug(f"[Temporal] Processing: '{query_text[:60]}...' session={current_session}/{total_sessions or '?'}")

        # ---- Parse query ----
        parser = self._parser_for(reference_time)
        parsed = parser.parse(query_text)
        constraints = parsed.get("constraints", [])

        # ---- Get all sessions for sparse resolution ----
        all_sessions = self._get_all_sessions()
        debug(f"[DEBUG] process: all_sessions={all_sessions}")  # <-- DEBUG

        # ---- Resolve session constraints with sparse session support ----
        if parsed.get("has_session_constraint", False):
            debug(f"[DEBUG] process: resolving session constraints")  # <-- DEBUG
            debug(f"[DEBUG] Before resolve: constraints[0].target = {constraints[0].target}")
            constraints = resolve_session_constraints(
                constraints,
                current_session,
                total_sessions,
                all_sessions,  # <-- Pass sparse session list
            )
            debug(f"[DEBUG] After resolve: constraints[0].target = {constraints[0].target}")
            if self.enable_diagnostics:
                for c in constraints:
                    if c.metadata.get("requires_session_resolution", False):
                        debug(f"[Temporal] Resolved: {c.metadata.get('session_type')} -> {c.target}")

        # ---- No constraints -> return empty ----
        if not constraints:
            debug(f"[Temporal] No temporal constraints found for: '{query_text[:60]}...'")
            return {
                "source": "temporal",
                "candidates": [],
                "count": 0,
                "diagnostics": {
                    "expressions": parsed.get("expressions", []),
                    "constraints": [],
                    "has_temporal_constraint": False,
                    "has_session_constraint": parsed.get("has_session_constraint", False),
                    "reference_time": (
                        reference_time.isoformat()
                        if reference_time
                        else None
                    ),
                },
            }

        # ---- Retrieve candidates ----
        candidates = self._retrieve_temporal_candidates(
            constraints,
            limit,
        )

        # ---- Score candidates ----
        scored = self._score_temporal(
            candidates,
            constraints,
            reference_time,
            current_session,
        )

        # ---- Apply shard filter ----
        scored = _shard_filter(
            scored,
            lambda c: c[0],
            shard_id,
            num_shards,
        )

        scored = scored[:limit]

        elapsed_ms = (time.perf_counter() - start) * 1000

        debug(
            f"[TemporalWorker] "
            f"(shard {shard_id}/{num_shards}): "
            f"temporal={elapsed_ms:.2f}ms, "
            f"constraints={len(constraints)}, "
            f"candidates={len(scored)}"
        )

        return {
            "source": "temporal",
            "candidates": scored,
            "count": len(scored),
            "diagnostics": {
                "expressions": parsed.get("expressions", []),
                "constraints": constraints,
                "reference_time": (
                    reference_time.isoformat()
                    if reference_time
                    else None
                ),
                "has_session_constraint": parsed.get("has_session_constraint", False),
                "scored_count": len(scored),
                "shard_id": shard_id,
                "num_shards": num_shards,
            },
        }

    # ================================================================
    # REFERENCE TIME
    # ================================================================

    def _get_reference_time(
        self,
        payload: Dict[str, Any],
    ) -> datetime:
        """
        Obtain the temporal anchor for relative expressions.
        """

        value = (
            payload.get("reference_time")
            or payload.get("query_time")
            or payload.get("conversation_time")
        )

        if value is None:
            return datetime.now(timezone.utc)

        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                debug(
                    f"TemporalWorker: invalid reference_time={value!r}"
                )

        return datetime.now(timezone.utc)

    def _parser_for(
        self,
        reference_time: Optional[datetime],
    ) -> TemporalParser:
        """
        Create a parser anchored to the supplied reference time.
        """

        return TemporalParser(
            reference_time=reference_time,
        )

    # ================================================================
    # TEMPORAL CANDIDATE DISCOVERY
    # ================================================================

    def _retrieve_temporal_candidates(
        self,
        constraints: List[Any],
        limit: int,
    ) -> List[Tuple[int, float]]:

        if self.temporal_index:
            memory_ids = self._search_index(
                constraints,
                limit,
            )
        else:
            memory_ids = self._search_db(
                constraints,
                limit,
            )

        return [
            (memory_id, 0.0)
            for memory_id in memory_ids
        ]

    def _search_index(
        self,
        constraints: List[Any],
        limit: int,
    ) -> List[int]:
        """
        Retrieve memory IDs using resolved temporal constraints.
        """

        results = set()

        for constraint in constraints:
            relation = constraint.relation
            target = constraint.target
            target_end = constraint.target_end

            if not constraint.resolved:
                continue

            # ---- SESSION CONSTRAINTS ----
            if relation == TemporalRelation.SESSION:
                if target is not None:
                    # Try to get session memories from temporal_index
                    if self.temporal_index and hasattr(self.temporal_index, 'search_by_session'):
                        session_results = self.temporal_index.search_by_session(target)
                        results.update(session_results)
                    else:
                        # Fallback: query DB directly by session_idx metadata
                        rows = self.db.fetch_all()
                        for row in rows:
                            metadata = row.get("metadata", {})
                            if isinstance(metadata, str):
                                try:
                                    import json
                                    metadata = json.loads(metadata)
                                except:
                                    metadata = {}
                            if metadata.get("session_idx") == target:
                                results.add(row["id"])
                            # Also check for session_id directly in row
                            elif row.get("session_id") == target:
                                results.add(row["id"])
                continue

            # ---- DATE/TIME CONSTRAINTS ----
            if relation == TemporalRelation.BETWEEN:
                if (
                    isinstance(target, datetime)
                    and isinstance(target_end, datetime)
                ):
                    results.update(
                        self.temporal_index.search_by_timestamp(
                            target,
                            target_end,
                        )
                    )

            elif relation == TemporalRelation.AFTER:
                if isinstance(target, datetime):
                    results.update(
                        self.temporal_index.search_by_timestamp(
                            start=target,
                            end=None,
                        )
                    )

            elif relation == TemporalRelation.BEFORE:
                if isinstance(target, datetime):
                    results.update(
                        self.temporal_index.search_by_timestamp(
                            start=None,
                            end=target,
                        )
                    )

            elif relation == TemporalRelation.DURING:
                if isinstance(target, datetime):
                    results.update(
                        self.temporal_index.search_by_timestamp(
                            target,
                            target,
                        )
                    )
                elif isinstance(target, int):
                    # Year-based lookup
                    if hasattr(self.temporal_index, 'search_by_year'):
                        results.update(
                            self.temporal_index.search_by_year(target)
                        )

            elif relation == TemporalRelation.MOST_RECENT:
                results.update(
                    self.temporal_index.get_recent(limit)
                )

        # If no results found, try DB fallback for session constraints
        if not results:
            for constraint in constraints:
                if constraint.relation == TemporalRelation.SESSION and constraint.target is not None:
                    rows = self.db.fetch_all()
                    for row in rows:
                        metadata = row.get("metadata", {})
                        if isinstance(metadata, str):
                            try:
                                import json
                                metadata = json.loads(metadata)
                            except:
                                metadata = {}
                        session_idx = metadata.get("session_idx")
                        if session_idx == constraint.target:
                            results.add(row["id"])
                            debug(f"[Temporal] Found session {constraint.target} memory: {row['id']}")

        return list(results)[:limit]

    def _search_db(
        self,
        constraints: List[Any],
        limit: int,
    ) -> List[int]:
        """
        Conservative database fallback.
        """

        rows = self.db.fetch_all()
        results = []

        for row in rows:
            if row.get("created_at"):
                results.append(row["id"])

        return results[:limit]

    # ================================================================
    # TEMPORAL SCORING
    # ================================================================

    def _score_temporal(
        self,
        candidates: List[Tuple[int, float]],
        constraints: List[Any],
        reference_time: datetime,
        current_session: int = 0,
    ) -> List[Tuple[int, float]]:
        debug(f"[DEBUG] _score_temporal: constraints[0].target = {constraints[0].target if constraints else 'empty'}")

        scored = []

        for memory_id, _ in candidates:

            memory = (
                self.temporal_index.get_temporal_data(memory_id)
                if self.temporal_index
                else self.db.get_memory(memory_id)
            )

            if not memory:
                continue

            created_at = memory.get("created_at")

            if not created_at:
                continue

            dt = self._parse_datetime(created_at)

            if dt is None:
                continue

            # Get session_id from memory
            memory_session = None
            # Try multiple possible locations
            if "session_id" in memory:
                memory_session = memory.get("session_id")
            elif "session_idx" in memory:
                memory_session = memory.get("session_idx")
            elif "metadata" in memory and memory["metadata"]:
                if isinstance(memory["metadata"], dict):
                    memory_session = memory["metadata"].get("session_idx")

            score, matched = self._calculate_score(
                dt,
                constraints,
                reference_time,
                current_session,
                memory_session,
            )

            if score > 0.0:
                scored.append(
                    (
                        memory_id,
                        score,
                    )
                )

        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return scored

    # ================================================================
    # SCORE CALCULATION
    # ================================================================

    def _calculate_score(
        self,
        dt: datetime,
        constraints: List[Any],
        reference_time: datetime,
        current_session: int = 0,
        memory_session: Optional[int] = None,
    ) -> Tuple[float, List[str]]:

        score = 0.0
        matched = []

        dt = self._normalize_datetime(dt)
        reference_time = self._normalize_datetime(reference_time)

        # Load configurable weights from settings with safe fallbacks
        # If settings don't have these attributes, use hardcoded defaults
        exact_boost = getattr(settings, "TEMPORAL_EXACT_MATCH_BOOST", 1.0)
        adjacent_boost = getattr(settings, "TEMPORAL_ADJACENT_BOOST", 0.5)
        recency_scale = getattr(settings, "TEMPORAL_RECENCY_SCALE", 14.0)
        conv_boost = getattr(settings, "TEMPORAL_CONVERSATIONAL_BOOST", 0.5)

        # Ensure values are valid (not None, not 0 for critical ones)
        if exact_boost is None:
            exact_boost = 1.0
        if adjacent_boost is None:
            adjacent_boost = 0.5
        if recency_scale is None or recency_scale <= 0:
            recency_scale = 14.0
        if conv_boost is None:
            conv_boost = 0.5

        for constraint in constraints:

            if not constraint.resolved:
                continue

            relation = constraint.relation
            target = constraint.target
            target_end = constraint.target_end

            # --------------------------------------------------------
            # SESSION CONSTRAINTS (HIGHEST PRIORITY)
            # --------------------------------------------------------

            if relation == TemporalRelation.SESSION:
                if memory_session is not None and target is not None:
                    if memory_session == target:
                        score += exact_boost
                        matched.append("session_match")
                    elif abs(memory_session - target) <= 1:
                        score += adjacent_boost
                        matched.append("session_adjacent")
                continue

            # --------------------------------------------------------
            # DURING / exact date
            # --------------------------------------------------------

            if relation == TemporalRelation.DURING:

                if isinstance(target, datetime):
                    target = self._normalize_datetime(target)

                    if dt.date() == target.date():
                        score += exact_boost
                        matched.append("during")

                elif isinstance(target, int):
                    if dt.year == target:
                        score += 0.5
                        matched.append("year")

            # --------------------------------------------------------
            # AFTER
            # --------------------------------------------------------

            elif relation == TemporalRelation.AFTER:

                if isinstance(target, datetime):
                    target = self._normalize_datetime(target)

                    if dt >= target:
                        score += self._boundary_score(
                            dt,
                            target,
                        )
                        matched.append("after")

            # --------------------------------------------------------
            # BEFORE
            # --------------------------------------------------------

            elif relation == TemporalRelation.BEFORE:

                if isinstance(target, datetime):
                    target = self._normalize_datetime(target)

                    if dt <= target:
                        score += self._boundary_score(
                            target,
                            dt,
                        )
                        matched.append("before")

            # --------------------------------------------------------
            # BETWEEN
            # --------------------------------------------------------

            elif relation == TemporalRelation.BETWEEN:

                if (
                    isinstance(target, datetime)
                    and isinstance(target_end, datetime)
                ):
                    start = self._normalize_datetime(target)
                    end = self._normalize_datetime(target_end)

                    if start <= dt <= end:
                        score += self._interval_score(
                            dt,
                            start,
                            end,
                        )
                        matched.append("between")

            # --------------------------------------------------------
            # MOST RECENT
            # --------------------------------------------------------

            elif relation == TemporalRelation.MOST_RECENT:

                age_days = max(
                    0.0,
                    (reference_time - dt).total_seconds() / 86400.0,
                )

                recency = math.exp(-age_days / recency_scale)
                score += recency * conv_boost
                matched.append("most_recent")

            # --------------------------------------------------------
            # CONVERSATIONAL RECENCY
            # --------------------------------------------------------

            if constraint.metadata.get("conversational"):
                recency_level = constraint.metadata.get("recency_level")
                age_days = max(
                    0.0,
                    (reference_time - dt).total_seconds() / 86400.0,
                )

                if recency_level == "very_recent" and age_days <= 2:
                    score += conv_boost
                    matched.append("very_recent")
                elif recency_level == "recent" and age_days <= 7:
                    score += conv_boost * 0.6
                    matched.append("recent")
                elif recency_level == "historical" and age_days > 30:
                    score += conv_boost * 0.6
                    matched.append("historical")

            # --------------------------------------------------------
            # FIRST / LAST - ordering constraints (not scored here)
            # --------------------------------------------------------

            elif relation in {TemporalRelation.FIRST, TemporalRelation.LAST}:
                continue

        return (
            min(1.0, score),
            matched,
        )

    @staticmethod
    def _boundary_score(
        dt: datetime,
        target: datetime,
    ) -> float:

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

    @staticmethod
    def _normalize_datetime(
        value: datetime,
    ) -> datetime:
        """
        Normalize naive/aware datetimes so comparisons do not explode.
        """

        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)

    @staticmethod
    def _parse_datetime(
        value: Any,
    ) -> Optional[datetime]:

        if isinstance(value, datetime):
            return value

        if not isinstance(value, str):
            return None

        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    # ================================================================
    # POST-PROCESSOR (Legacy - kept for compatibility)
    # ================================================================

    def score_candidates(
        self,
        candidates: List[Any],
        query_text: str,
        tokens: Optional[List[str]] = None,
        reference_time: Optional[datetime] = None,
        current_session: int = 0,
        total_sessions: Optional[int] = None,
    ) -> List[Any]:
        """
        Attach temporal evidence to existing CandidateRecord objects.
        Kept for backward compatibility with post-processing path.
        """

        if not candidates:
            return candidates

        # ---- READ SESSION CONTEXT ----
        if current_session == 0 and total_sessions is None:
            context = self._get_temporal_context()
            current_session = context.get("current_session", 0)
            total_sessions = context.get("total_sessions", None)
            if reference_time is None:
                reference_time = context.get("reference_time", None)

        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        parser = self._parser_for(reference_time)
        parsed = parser.parse(query_text)
        constraints = parsed.get("constraints", [])

        # ---- Get all sessions for sparse resolution ----
        all_sessions = self._get_all_sessions()
        debug(f"[scoring] score_candidates: all_sessions={all_sessions}")  # <-- DEBUG

        # Resolve session constraints if present
        if parsed.get("has_session_constraint", False):
            debug(f"[DEBUG] score_candidates: resolving session constraints")  # <-- DEBUG
            constraints = resolve_session_constraints(
                constraints,
                current_session,
                total_sessions,
                all_sessions,  # <-- Pass sparse session list
            )
            debug(f"[DEBUG] After resolver, constraints[0].target = {constraints[0].target}")
            if self.enable_diagnostics:
                for c in constraints:
                    if c.metadata.get("requires_session_resolution", False):
                        debug(f"[Temporal] Resolved: {c.metadata.get('session_type')} -> {c.target}")

        if not constraints:
            for candidate in candidates:
                candidate.temporal_score = 0.0
                candidate.temporal_matches = []
            return candidates

        for candidate in candidates:

            memory = getattr(
                candidate,
                "memory",
                None,
            )

            if memory is None:
                candidate.temporal_score = 0.0
                candidate.temporal_matches = []
                continue

            created_at = getattr(
                memory,
                "created_at",
                None,
            )

            if not created_at:
                candidate.temporal_score = 0.0
                candidate.temporal_matches = []
                continue

            dt = self._parse_datetime(created_at)

            if dt is None:
                candidate.temporal_score = 0.0
                candidate.temporal_matches = []
                continue

            # Get session_id from memory
            memory_session = getattr(memory, "session_id", None)
            if memory_session is None:
                memory_session = getattr(memory, "session_idx", None)
            if memory_session is None and hasattr(memory, "metadata"):
                memory_session = memory.metadata.get("session_idx")

            score, matched = self._calculate_score(
                dt,
                constraints,
                reference_time,
                current_session,
                memory_session,
            )

            candidate.temporal_score = score
            candidate.temporal_matches = matched

        return candidates
