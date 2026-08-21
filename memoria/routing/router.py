"""
Router — Uses the routing matrix to route queries to the right pools,
workers, and signals.
"""

from routing.matrix import ROUTING_MATRIX, compute_detection_score, get_detection_weights
from cache.config import settings
from core.logger import debug


# Type priority order (explicit, not implicit dictionary order)
TYPE_PRIORITY = [
    "semantic",
    "episodic",
    "procedural",
    "code",
    "science",
    "general"
]


class Router:
    def __init__(self, matrix=None, plugin_manager=None):
        self.matrix = matrix or ROUTING_MATRIX
        self.default_type = "general"
        self.type_priority = TYPE_PRIORITY
        self.plugin_manager = plugin_manager  # <-- NEW
        # Cache for route lookups
        self._route_cache = {}
        self._signal_cache = {}

    def route(self, memory_type: str) -> dict:
        """
        Return the routing configuration for a memory type.
        Cached for performance.
        """
        # ---- Plugin hook: pre-routing ----
        if self.plugin_manager:
            try:
                self.plugin_manager.memoria_routing_pre(memory_type)
            except Exception as e:
                debug(f"[Plugin] router_pre error: {e}")

        # Check cache first
        if memory_type in self._route_cache:
            route = self._route_cache[memory_type]
        else:
            # Get config, fallback to general if not found
            if memory_type not in self.matrix:
                debug(f"Unknown type '{memory_type}', falling back to '{self.default_type}'")
                config = self.matrix[self.default_type]
            else:
                config = self.matrix[memory_type]

            # Cache and return
            self._route_cache[memory_type] = config
            route = config

        # ---- Plugin hook: post-routing ----
        if self.plugin_manager:
            try:
                self.plugin_manager.memoria_routing_post(route)
            except Exception as e:
                debug(f"[Plugin] router_post error: {e}")

        return route

    def get_signals(self, memory_type: str) -> dict:
        """Return the signal weights for a memory type. Cached."""
        if memory_type in self._signal_cache:
            return self._signal_cache[memory_type]

        signals = self.route(memory_type)["signals"]
        self._signal_cache[memory_type] = signals
        return signals

    def get_pool(self, memory_type: str) -> str:
        """Return the primary pool for a memory type."""
        return self.route(memory_type)["pool"]

    def get_workers(self, memory_type: str) -> list:
        """Return the workers to use for a memory type."""
        return self.route(memory_type)["workers"]

    def get_graph_depth(self, memory_type: str) -> int:
        """Return the graph depth for a memory type."""
        return self.route(memory_type)["graph_depth"]

    def get_fallback_pools(self, memory_type: str) -> list:
        """Return the fallback pools for a memory type."""
        return self.route(memory_type).get("fallback_pools", [])

    def get_entity_required(self, memory_type: str) -> bool:
        """Return whether entities are required for a memory type."""
        return self.route(memory_type).get("entity_required", False)

    def get_attribute_required(self, memory_type: str) -> bool:
        """Return whether attributes are required for a memory type."""
        return self.route(memory_type).get("attribute_required", False)

    def get_temporal_weight(self, memory_type: str) -> float:
        """Return the temporal weight for a memory type."""
        return self.route(memory_type).get("temporal_weight", 0.0)

    def get_description(self, memory_type: str) -> str:
        """Return the description of a memory type."""
        return self.route(memory_type).get("description", "")

    def get_detection_hints(self, memory_type: str) -> dict:
        """Return the detection hints for a memory type."""
        return self.route(memory_type).get("detection", {"keywords": [], "exclude": [], "min_confidence": 0.0})

    def list_types(self) -> list:
        """Return all available memory types."""
        return list(self.matrix.keys())

    def is_valid_type(self, memory_type: str) -> bool:
        """Check if a memory type is valid."""
        return memory_type in self.matrix

    def get_matching_type(self, query: str, entities: list = None, attributes: list = None) -> tuple:
        """
        Determine which memory type matches a query based on detection hints.
        Returns (type_name, confidence).

        Uses the matrix's compute_detection_score() for consistent scoring.
        Tie-breaking is explicit via TYPE_PRIORITY.
        """
        best_type = self.default_type
        best_score = float("-inf")
        best_confidence = 0.0

        for type_name in self.type_priority:
            if type_name not in self.matrix:
                continue

            config = self.matrix[type_name]
            detection = config.get("detection", {})
            min_conf = detection.get("min_confidence", 0.0)

            # Use the matrix's helper for consistent scoring
            score, confidence, matched_keywords, matched_excludes = compute_detection_score(
                type_name, query, entities, attributes
            )

            # Check if this type is viable
            if confidence >= min_conf and confidence > best_confidence:
                best_confidence = confidence
                best_type = type_name
                best_score = score

            # If we have a very high confidence match, break early
            if confidence >= 0.9:
                break

        debug(
            f"Query '{query[:50]}...' matched type '{best_type}' "
            f"with confidence {best_confidence:.3f}, score {best_score:.2f}",
            category="router"
        )

        return best_type, best_confidence

    def get_matching_type_detailed(self, query: str, entities: list = None, attributes: list = None) -> dict:
        """
        Get detailed routing information including per-type scores.
        Useful for debugging and diagnostics.
        """
        results = {}

        for type_name in self.type_priority:
            if type_name not in self.matrix:
                continue

            config = self.matrix[type_name]
            detection = config.get("detection", {})
            min_conf = detection.get("min_confidence", 0.0)

            score, confidence, matched_keywords, matched_excludes = compute_detection_score(
                type_name, query, entities, attributes
            )

            results[type_name] = {
                "score": score,
                "confidence": confidence,
                "min_confidence": min_conf,
                "passed": confidence >= min_conf,
                "matched_keywords": matched_keywords,
                "matched_excludes": matched_excludes,
                "entity_required": config.get("entity_required", False),
                "attribute_required": config.get("attribute_required", False),
            }

        # Find the best match
        best_type, best_confidence = self.get_matching_type(query, entities, attributes)
        results["best"] = best_type
        results["best_confidence"] = best_confidence

        return results

    def clear_cache(self):
        """Clear the route cache (useful if matrix changes at runtime)."""
        self._route_cache.clear()
        self._signal_cache.clear()
