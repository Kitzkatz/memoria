"""
Signal Toggles — Runtime overrides for signal toggles.

This allows enabling/disabling signals without touching the registry.
"""

from typing import Dict, Optional, Set
from ranking.signal_registry import get_registry, SignalName, MemoryType


class SignalToggles:
    """
    Runtime toggles for signals.

    - Override global toggles
    - Override per-type toggles
    - Can be set via config or API
    """

    def __init__(self, registry=None):
        self.registry = registry or get_registry()
        self._global_overrides: Dict[SignalName, bool] = {}
        self._per_type_overrides: Dict[MemoryType, Dict[SignalName, bool]] = {}

    def enable(self, name: SignalName) -> None:
        """Enable a signal globally (override)."""
        self._global_overrides[name] = True

    def disable(self, name: SignalName) -> None:
        """Disable a signal globally (override)."""
        self._global_overrides[name] = False

    def enable_for_type(self, name: SignalName, memory_type: MemoryType) -> None:
        """Enable a signal for a specific memory type."""
        if memory_type not in self._per_type_overrides:
            self._per_type_overrides[memory_type] = {}
        self._per_type_overrides[memory_type][name] = True

    def disable_for_type(self, name: SignalName, memory_type: MemoryType) -> None:
        """Disable a signal for a specific memory type."""
        if memory_type not in self._per_type_overrides:
            self._per_type_overrides[memory_type] = {}
        self._per_type_overrides[memory_type][name] = False

    def is_enabled(self, name: SignalName, memory_type: MemoryType = "general") -> bool:
        """Check if a signal is enabled (considering overrides)."""
        # Check per-type override first
        if memory_type in self._per_type_overrides:
            if name in self._per_type_overrides[memory_type]:
                return self._per_type_overrides[memory_type][name]

        # Check global override
        if name in self._global_overrides:
            return self._global_overrides[name]

        # Check registry
        return self.registry.is_enabled_for_type(name, memory_type)

    def reset(self) -> None:
        """Reset all overrides."""
        self._global_overrides.clear()
        self._per_type_overrides.clear()

    def get_overrides(self) -> Dict:
        """Get all current overrides."""
        return {
            "global": self._global_overrides,
            "per_type": self._per_type_overrides,
        }
