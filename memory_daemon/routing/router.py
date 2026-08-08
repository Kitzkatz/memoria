"""
Router — Uses the routing matrix to route queries to the right pools,
workers, and signals.
"""

from routing.matrix import ROUTING_MATRIX
from cache.config import settings
from core.logger import debug


class Router:
    def __init__(self, matrix=None):
        self.matrix = matrix or ROUTING_MATRIX
        self.default_type = "general"
        # Cache for route lookups
        self._route_cache = {}
        self._signal_cache = {}

    def route(self, memory_type: str) -> dict:
        """
        Return the routing configuration for a memory type.
        Cached for performance.
        """
        # Check cache first
        if memory_type in self._route_cache:
            return self._route_cache[memory_type]
        
        # Get config, fallback to general if not found
        if memory_type not in self.matrix:
            debug(f"Unknown type '{memory_type}', falling back to '{self.default_type}'")
            config = self.matrix[self.default_type]
        else:
            config = self.matrix[memory_type]
        
        # Cache and return
        self._route_cache[memory_type] = config
        return config

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

    def get_matching_type(self, query: str, entities: list = None) -> tuple:
        """
        Determine which memory type matches a query based on detection hints.
        Returns (type_name, confidence).
        """
        best_type = self.default_type
        best_confidence = 0.0
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for type_name, config in self.matrix.items():
            detection = config.get("detection", {})
            keywords = detection.get("keywords", [])
            exclude = detection.get("exclude", [])
            min_conf = detection.get("min_confidence", 0.0)
            
            # Count keyword matches
            keyword_matches = sum(1 for kw in keywords if kw in query_lower)
            
            # Count exclude matches (negative)
            exclude_matches = sum(1 for ex in exclude if ex in query_lower)
            
            # Simple confidence: keyword density
            if len(keywords) > 0:
                confidence = keyword_matches / len(keywords)
            else:
                confidence = 0.0
            
            # Penalize for exclude matches
            if exclude_matches > 0:
                confidence *= 0.5  # 50% penalty if any exclude matched
            
            # If entities provided, boost confidence for entity-heavy types
            if entities and detection.get("entity_required", False):
                confidence *= 1.2  # 20% boost if type wants entities
            
            if confidence > best_confidence and confidence >= min_conf:
                best_confidence = confidence
                best_type = type_name
        
        debug(f"Query '{query[:50]}...' matched type '{best_type}' with confidence {best_confidence:.3f}")
        
        return best_type, best_confidence

    def clear_cache(self):
        """Clear the route cache (useful if matrix changes at runtime)."""
        self._route_cache.clear()
        self._signal_cache.clear()
