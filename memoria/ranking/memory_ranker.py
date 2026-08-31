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

        # ---- Per-type signal configuration (moved from hardcoded logic) ----
        self._type_config = self._build_type_config()

    def _build_type_config(self) -> dict:
        """
        Build per-type configuration from registry.
        This replaces hardcoded if/else chains with data-driven config.
        """
        config = {
            "general": {
                "recency_decay": 30,
                "token_strategy": "union_ratio",
                "entity_strategy": "overlap_ratio",
                "skip_tfidf": False,
                "skip_graph": False,
            },
            "episodic": {
                "recency_decay": 7,
                "token_strategy": "query_frequency",
                "entity_strategy": "overlap_ratio",
                "skip_tfidf": True,
                "skip_graph": False,
            },
            "procedural": {
                "recency_decay": 30,
                "token_strategy": "query_frequency",
                "entity_strategy": "overlap_ratio",
                "skip_tfidf": False,
                "skip_graph": False,
            },
            "semantic": {
                "recency_decay": 365,
                "token_strategy": "union_ratio",
                "entity_strategy": "overlap_ratio",
                "skip_tfidf": False,
                "skip_graph": True,
            },
            "code": {
                "recency_decay": 730,
                "token_strategy": "query_frequency",
                "entity_strategy": "exact_symbol",
                "skip_tfidf": False,
                "skip_graph": False,
            },
            "science": {
                "recency_decay": 90,
                "token_strategy": "union_ratio",
                "entity_strategy": "conceptual",
                "skip_tfidf": False,
                "skip_graph": False,
            },
        }

        # Override with registry values if available
        try:
            for memory_type in self.registry.list_memory_types():
                if memory_type not in config:
                    config[memory_type] = {
                        "recency_decay": 30,
                        "token_strategy": "union_ratio",
                        "entity_strategy": "overlap_ratio",
                        "skip_tfidf": False,
                        "skip_graph": False,
                    }
        except Exception:
            pass  # Registry not fully initialized yet

        return config

    def _get_type_config(self, memory_type: str) -> dict:
        """Get configuration for a memory type with fallback to general."""
        return self._type_config.get(memory_type, self._type_config["general"])

    # ---------------------------------
    # Recency (now data-driven)
    # ---------------------------------

    def recency_score(self, created_at, memory_type="general"):
        """
        Per-type recency scoring using decay days from config.
        """
        try:
            created = datetime.fromisoformat(created_at)
            now = datetime.now(timezone.utc)
            age = max(0, (now - created).days)

            decay_days = self._get_type_config(memory_type).get("recency_decay", 30)
            return math.exp(-age / decay_days)
        except Exception:
            return 0.5

    # ---------------------------------
    # Token similarity (now data-driven)
    # ---------------------------------

    def token_overlap(self, query_tokens, memory_tokens, memory_type="general"):
        """
        Per-type token scoring using strategy from config.
        """
        if not query_tokens or not memory_tokens:
            return 0.0

        q = set(query_tokens)
        m = set(memory_tokens)

        strategy = self._get_type_config(memory_type).get("token_strategy", "union_ratio")

        if strategy == "query_frequency":
            # Exact matches: count query tokens present in memory
            matches = sum(1 for token in q if token in m)
            return matches / max(len(q), 1)
        else:  # union_ratio (default)
            # Conceptual overlap: intersection over union
            intersection = len(q & m)
            union = len(q | m)
            return intersection / max(union, 1)

    # ---------------------------------
    # Entity overlap (now data-driven)
    # ---------------------------------

    def entity_overlap(self, query_entities, memory_entities, memory_type="general"):
        """
        Per-type entity scoring using strategy from config.
        """
        if not query_entities or not memory_entities:
            return 0.0

        q = {str(e).lower() for e in query_entities}
        m = {str(e).lower() for e in memory_entities}

        strategy = self._get_type_config(memory_type).get("entity_strategy", "overlap_ratio")

        if strategy == "exact_symbol":
            # Code: exact symbol matching
            matches = sum(1 for e in q if e in m)
            return matches / max(len(q), 1)

        elif strategy == "conceptual":
            # Science: conceptual entity matching
            overlap = len(q & m)
            return overlap / max(len(q), 1)

        else:  # overlap_ratio (default)
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
        Tracks source for diagnostics.
        """
        memory_type = query.metadata.get("memory_type_hint", "general")
        source = "default"

        # Try registry
        try:
            weights = self.signal_router.get_active_signals(memory_type)
            if weights:
                source = "registry"
                return weights, source
        except Exception:
            pass

        # Fallback: routing matrix via query metadata
        routing_signals = query.metadata.get("routing_signals")
        if routing_signals and isinstance(routing_signals, dict):
            merged = self._default_weights.copy()
            for key, value in routing_signals.items():
                if key in merged:
                    merged[key] = value
            source = "routing_metadata"
            return merged, source

        # Final fallback: default weights
        return self._default_weights.copy(), source

    # ---------------------------------
    # Compute one candidate score
    # ---------------------------------

    def compute_score(self, candidate, query):
        t0 = time_module.perf_counter()

        weights, weights_source = self.get_weights_for_type(query)

        # Get memory type for per-type scoring
        memory_type = candidate.memory.memory_type or "general"

        # Get type config for skip flags
        type_config = self._get_type_config(memory_type)
        skip_tfidf = type_config.get("skip_tfidf", False) or weights.get("tfidf", 0.0) <= 0.001
        skip_graph = type_config.get("skip_graph", False) or weights.get("graph_distance", 0.0) <= 0.001

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
        t_feedback = time_module.perf_counter()

        auto_feedback = 0.0
        if self.feedback_loop and weights.get("feedback", 0.0) > 0:
            auto_feedback = self.feedback_loop.get_boost(candidate.memory.id)
        combined_feedback = max(-1.0, min(auto_feedback, 1.0))
        t_score = time_module.perf_counter()

        # ============================================================
        # REMOVED: subject and attribute from MemoryRanker.
        # These are now owned exclusively by AttributeBooster.
        # ============================================================
        score = (
            semantic * weights.get("semantic", 0.0)
            + importance * weights.get("importance", 0.0)
            + recency * weights.get("recency", 0.0)
            + token * weights.get("token", 0.0)
            + combined_feedback * weights.get("feedback", 0.0)
            + entity * weights.get("entity", 0.0)
            + tfidf * weights.get("tfidf", 0.0)
            + graph_dist * weights.get("graph_distance", 0.0)
        )
        t_end = time_module.perf_counter()

        candidate.semantic_score = semantic
        candidate.importance_score = importance
        candidate.recency_score = recency
        candidate.token_score = token
        candidate.base_score = score

        # Timing accumulation
        self._timing_accumulator["semantic"] += (t_importance - t_semantic) * 1000
        self._timing_accumulator["importance"] += (t_recency - t_importance) * 1000
        self._timing_accumulator["recency"] += (t_token - t_recency) * 1000
        self._timing_accumulator["token"] += (t_entity - t_token) * 1000
        self._timing_accumulator["entity"] += (t_tfidf - t_entity) * 1000
        self._timing_accumulator["tfidf"] += (t_graph - t_tfidf) * 1000
        self._timing_accumulator["graph_distance"] += (t_feedback - t_graph) * 1000
        self._timing_accumulator["feedback"] += (t_score - t_feedback) * 1000
        self._timing_accumulator["score_sum"] += (t_end - t_score) * 1000
        self._timing_accumulator["total"] += (t_end - t0) * 1000
        self._candidate_count += 1

        # Diagnostics with weights_source
        candidate.diagnostics["ranker"] = {
            "semantic": semantic,
            "importance": importance,
            "recency": recency,
            "token": token,
            "feedback": combined_feedback,
            "entity": entity,
            "graph_distance": graph_dist,
            "tfidf": tfidf,
            "weights": weights,
            "weights_source": weights_source,
            "memory_type": memory_type,
            "type_config": type_config,
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
        self._timing_accumulator.clear()
        self._candidate_count = 0

        start_total = time_module.perf_counter()

        updated = []
        for candidate in candidates:
            updated.append(self.compute_score(candidate, query))

        elapsed_total = (time_module.perf_counter() - start_total) * 1000

        updated.sort(key=lambda x: x.base_score, reverse=True)

        if self._candidate_count > 0 and self.enable_diagnostics:
            debug(f"[Ranker] {self._candidate_count} candidates in {elapsed_total:.2f}ms", category="ranking")

        return updated

    # ---------------------------------
    # Get timing breakdown
    # ---------------------------------

    def get_timing_breakdown(self) -> dict:
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
