from datetime import datetime, timezone
from collections import defaultdict
import math
import time as time_module
from cache.config import settings
from routing.matrix import ROUTING_MATRIX
from ranking.signal_router import SignalRouter
from ranking.signal_registry import get_registry
from core.logger import debug


class MemoryRanker:

    def __init__(self, tfidf_ranker=None, feedback_loop=None, numpy_graph=None, enable_diagnostics=None):
        self.tfidf_ranker = tfidf_ranker
        self.feedback_loop = feedback_loop
        self.numpy_graph = numpy_graph

        if enable_diagnostics is None:
            self.enable_diagnostics = getattr(settings, "RANKER_DIAGNOSTICS", settings.DEBUG)
        else:
            self.enable_diagnostics = enable_diagnostics

        # ---- Signal Registry Integration ----
        self.registry = get_registry()
        self.signal_router = SignalRouter(self.registry)

        # Fallback: keep default weights from routing matrix
        general_config = ROUTING_MATRIX.get("general", {})
        self._default_weights = general_config.get("signals", {}).copy()

        self._timing_accumulator = defaultdict(float)
        self._candidate_count = 0

    # ---------------------------------
    # Recency (PER-TYPE: different decay rates)
    # ---------------------------------

    def recency_score(self, created_at, memory_type="general", decay_days=30):
        """
        Per-type recency scoring:
        - Episodic: Fast decay (7 days) — events matter most when recent
        - Procedural: Moderate decay (30 days) — skills fade but persist
        - Semantic: Slow decay (365 days) — facts are permanent
        - Code: Minimal decay (730 days) — code doesn't age
        - Science: Moderate decay (90 days) — science evolves
        - General: Default decay (30 days)
        """
        try:
            created = datetime.fromisoformat(created_at)
            now = datetime.now(timezone.utc)
            age = max(0, (now - created).days)

            # Episodic: fast decay (7 days)
            if memory_type == "episodic":
                return math.exp(-age / 7)
            
            # Code: minimal decay (730 days)
            if memory_type == "code":
                return math.exp(-age / 730)
            
            # Procedural: moderate decay (30 days)
            if memory_type == "procedural":
                return math.exp(-age / 30)
            
            # Science: moderate decay (90 days)
            if memory_type == "science":
                return math.exp(-age / 90)
            
            # Semantic: slow decay (365 days)
            if memory_type == "semantic":
                return math.exp(-age / 365)

            # Default
            return math.exp(-age / decay_days)
        except Exception:
            return 0.5

    # ---------------------------------
    # Token similarity (PER-TYPE: query frequency OR union ratio)
    # ---------------------------------

    def token_overlap(self, query_tokens, memory_tokens, memory_type="general"):
        """
        Per-type token scoring:
        - Episodic, Procedural, Code: Query frequency (exact matches)
        - Semantic, Science, General: Union ratio (conceptual overlap)
        """
        if not query_tokens or not memory_tokens:
            return 0.0

        q = set(query_tokens)
        m = set(memory_tokens)

        # Query frequency for exact-match types
        if memory_type in ["episodic", "procedural", "code"]:
            matches = sum(1 for token in q if token in m)
            return matches / max(len(q), 1)

        # Union ratio for conceptual types
        else:  # semantic, science, general
            intersection = len(q & m)
            union = len(q | m)
            return intersection / max(union, 1)

    # ---------------------------------
    # Entity overlap (PER-TYPE: different matching strategies)
    # ---------------------------------

    def entity_overlap(self, query_entities, memory_entities, memory_type="general"):
        """
        Per-type entity scoring:
        - Code: Exact symbol matching (functions, classes, imports)
        - Science: Conceptual entity matching (formulas, theories)
        - Others: Standard overlap ratio
        """
        if not query_entities or not memory_entities:
            return 0.0

        q = {str(e).lower() for e in query_entities}
        m = {str(e).lower() for e in memory_entities}

        # Code: exact symbol matching (functions, classes, imports)
        if memory_type == "code":
            matches = sum(1 for e in q if e in m)
            return matches / max(len(q), 1)

        # Science: conceptual entity matching
        if memory_type == "science":
            overlap = len(q & m)
            return overlap / max(len(q), 1)

        # Default: overlap ratio
        return len(q & m) / max(len(q), 1)

    # ---------------------------------
    # Semantic score
    # ---------------------------------

    def semantic_score(self, distance):
        return 1.0 / (1.0 + float(distance))

    def tfidf_score(self, query_tokens, memory_tokens):
        if not hasattr(self, 'tfidf_ranker') or not self.tfidf_ranker:
            return 0.0
        return self.tfidf_ranker.document_score(query_tokens, memory_tokens)

    def graph_distance_score(self, query_entities, memory_entities):
        if not query_entities or not memory_entities or not self.numpy_graph:
            return 0.0

        best_distance = float('inf')
        for q_entity in query_entities:
            for m_entity in memory_entities:
                dist = self.numpy_graph.shortest_path(q_entity, m_entity)
                if dist is not None and dist < best_distance:
                    best_distance = dist
                    if best_distance == 1:
                        break
            if best_distance == 1:
                break

        if best_distance == float('inf') or best_distance is None:
            return 0.0
        return 1.0 / (best_distance + 1e-6)

    def get_weights_for_type(self, query):
        """
        Get weights from the signal registry based on memory type.
        Falls back to routing matrix if registry is not available.
        """
        # First try: get memory type from query
        memory_type = query.metadata.get("memory_type_hint", "general")
        
        # Try registry
        try:
            weights = self.signal_router.get_active_signals(memory_type)
            if weights:
                return weights
        except Exception:
            pass
        
        # Fallback: routing matrix
        routing_signals = query.metadata.get("routing_signals")
        if routing_signals and isinstance(routing_signals, dict):
            merged = self._default_weights.copy()
            for key, value in routing_signals.items():
                if key in merged:
                    merged[key] = value
            return merged
        
        return self._default_weights

    # ---------------------------------
    # Compute one candidate score
    # ---------------------------------

    def compute_score(self, candidate, query):
        t0 = time_module.perf_counter()

        weights = self.get_weights_for_type(query)

        # Get memory type for per-type scoring
        memory_type = candidate.memory.memory_type or "general"

        # Skip signals that don't help this type
        # Check weights to see if signals are active
        skip_tfidf = memory_type in ["episodic"] or weights.get("tfidf", 0.0) <= 0.001
        skip_graph = memory_type in ["semantic", "general"] or weights.get("graph_distance", 0.0) <= 0.001

        t_semantic = time_module.perf_counter()
        semantic = self.semantic_score(candidate.distance)
        t_importance = time_module.perf_counter()
        importance = max(0.0, min(float(candidate.memory.importance), 1.0))
        t_recency = time_module.perf_counter()
        recency = self.recency_score(candidate.memory.created_at, memory_type)
        t_token = time_module.perf_counter()
        token = self.token_overlap(query.tokens, candidate.memory.tokens, memory_type)
        t_entity = time_module.perf_counter()
        entity = self.entity_overlap(query.entities, candidate.memory.entities, memory_type)
        t_tfidf = time_module.perf_counter()
        tfidf = self.tfidf_score(query.tokens, candidate.memory.tokens) if not skip_tfidf else 0.0
        t_graph = time_module.perf_counter()
        graph_dist = self.graph_distance_score(query.entities, candidate.memory.entities) if not skip_graph else 0.0
        t_subject = time_module.perf_counter()

        subject = 0.0
        attribute = 0.0
        if candidate.memory.metadata:
            mem_subject = candidate.memory.metadata.get("subject")
            mem_attribute = candidate.memory.metadata.get("attribute")
            query_subject = query.metadata.get("subject")
            query_attribute = query.metadata.get("attribute")

            if mem_subject and query_subject:
                if str(mem_subject).lower() == str(query_subject).lower():
                    subject = 1.0
            if mem_attribute and query_attribute:
                if str(mem_attribute) == str(query_attribute):
                    attribute = 1.0
        t_feedback = time_module.perf_counter()

        auto_feedback = 0.0
        if self.feedback_loop and weights.get("feedback", 0.0) > 0:
            auto_feedback = self.feedback_loop.get_boost(candidate.memory.id)
        combined_feedback = max(-1.0, min(auto_feedback, 1.0))
        t_score = time_module.perf_counter()

        # ============================================================
        # FIX: Removed hardcoded fallback defaults, replaced with 0.0.
        # Now any signal missing from weights dict has zero effect.
        # ============================================================
        score = (
            semantic * weights.get("semantic", 0.0)
            + importance * weights.get("importance", 0.0)
            + recency * weights.get("recency", 0.0)
            + token * weights.get("token", 0.0)
            + combined_feedback * weights.get("feedback", 0.0)
            + entity * weights.get("entity", 0.0)
            + subject * weights.get("subject", 0.0)
            + attribute * weights.get("attribute", 0.0)
            + tfidf * weights.get("tfidf", 0.0)
            + graph_dist * weights.get("graph_distance", 0.0)
        )
        t_end = time_module.perf_counter()

        candidate.semantic_score = semantic
        candidate.importance_score = importance
        candidate.recency_score = recency
        candidate.token_score = token
        candidate.base_score = score

        # Timing accumulation (in memory only, no file I/O)
        self._timing_accumulator["semantic"] += (t_importance - t_semantic) * 1000
        self._timing_accumulator["importance"] += (t_recency - t_importance) * 1000
        self._timing_accumulator["recency"] += (t_token - t_recency) * 1000
        self._timing_accumulator["token"] += (t_entity - t_token) * 1000
        self._timing_accumulator["entity"] += (t_tfidf - t_entity) * 1000
        self._timing_accumulator["tfidf"] += (t_graph - t_tfidf) * 1000
        self._timing_accumulator["graph_distance"] += (t_subject - t_graph) * 1000
        self._timing_accumulator["subject_attribute"] += (t_feedback - t_subject) * 1000
        self._timing_accumulator["feedback"] += (t_score - t_feedback) * 1000
        self._timing_accumulator["score_sum"] += (t_end - t_score) * 1000
        self._timing_accumulator["total"] += (t_end - t0) * 1000
        self._candidate_count += 1

        # ✅ ALWAYS store ranker signals (needed for benchmark analysis)
        candidate.diagnostics["ranker"] = {
            "semantic": semantic,
            "importance": importance,
            "recency": recency,
            "token": token,
            "feedback": combined_feedback,
            "entity": entity,
            "subject": subject,
            "attribute": attribute,
            "graph_distance": graph_dist,
            "tfidf": tfidf,
            "weights": weights,
            "score": round(score, 4),
        }

        # Extra diagnostics if enabled
        if self.enable_diagnostics:
            candidate.diagnostics["tfidf_score"] = tfidf
            candidate.diagnostics["graph_distance"] = graph_dist
            candidate.diagnostics["feedback_auto"] = auto_feedback
            candidate.diagnostics["feedback_combined"] = combined_feedback

        return candidate

    # ---------------------------------
    # Pipeline entry
    # ---------------------------------

    def rank(self, candidates, query):
        # Reset timing accumulator
        self._timing_accumulator.clear()
        self._candidate_count = 0

        start_total = time_module.perf_counter()

        updated = []
        for candidate in candidates:
            updated.append(self.compute_score(candidate, query))

        elapsed_total = (time_module.perf_counter() - start_total) * 1000

        updated.sort(key=lambda x: x.base_score, reverse=True)

        # ✅ Timing breakdown is kept in memory (no file I/O)
        # Access via self._timing_accumulator if needed for diagnostics
        if self._candidate_count > 0 and self.enable_diagnostics:
            debug(f"[Ranker] {self._candidate_count} candidates in {elapsed_total:.2f}ms", category="ranking")

        return updated

    # ---------------------------------
    # Get timing breakdown (for diagnostics)
    # ---------------------------------

    def get_timing_breakdown(self) -> dict:
        """Return the timing breakdown for the last rank operation."""
        if self._candidate_count == 0:
            return {}

        return {
            "candidate_count": self._candidate_count,
            "breakdown": {
                key: {
                    "total_ms": round(value, 3),
                    "avg_ms": round(value / self._candidate_count, 3)
                }
                for key, value in self._timing_accumulator.items()
            }
        }
