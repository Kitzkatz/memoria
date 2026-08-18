"""
Signal Router — Uses the registry to determine which signals to compute.

This sits between the query router and the ranker.
"""

from typing import Dict, List, Set, Optional
from ranking.signal_registry import get_registry, SignalName, SignalWeight, MemoryType


class SignalRouter:
    """
    Routes signals based on memory type and runtime toggles.

    - Reads from the registry
    - Applies runtime overrides
    - Returns active signals for ranking
    """

    def __init__(self, registry=None):
        self.registry = registry or get_registry()
        self._active_cache = {}

    def get_active_signals(self, memory_type: MemoryType = "general") -> Dict[SignalName, SignalWeight]:
        """Get active signals with weights for a memory type."""
        cache_key = memory_type
        if cache_key in self._active_cache:
            return self._active_cache[cache_key]

        active = self.registry.get_weights(memory_type)
        self._active_cache[cache_key] = active
        return active

    def get_signal_names(self, memory_type: MemoryType = "general") -> List[SignalName]:
        """Get list of active signal names."""
        return list(self.get_active_signals(memory_type).keys())

    def get_signal_weights(self, memory_type: MemoryType = "general") -> List[float]:
        """Get list of active signal weights."""
        return list(self.get_active_signals(memory_type).values())

    def get_signal_info(self, memory_type: MemoryType = "general") -> List[Dict]:
        """Get full info for active signals."""
        active = self.get_active_signals(memory_type)
        return [
            {
                "name": name,
                "weight": weight,
                "cost": self.registry.get_cost(name),
                "description": self.registry.get_description(name),
                "category": self.registry.get_category(name),
            }
            for name, weight in active.items()
        ]

    def is_signal_active(self, name: SignalName, memory_type: MemoryType = "general") -> bool:
        """Check if a specific signal is active."""
        return name in self.get_active_signals(memory_type)

    def clear_cache(self) -> None:
        """Clear the active cache."""
        self._active_cache.clear()
