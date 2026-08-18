"""
Signal Registry — Loads from declarative JSON file.

This is the loader for the signal registry. The actual data lives in
signal_registry.json. This file provides the API for accessing signals.

The JSON file is the single source of truth.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from cache.config import settings

# Type aliases
SignalName = str
SignalWeight = float
SignalCost = str  # "low", "medium", "high"
MemoryType = str


# Default path to the registry JSON file
DEFAULT_REGISTRY_PATH = Path(__file__).parent / "signal_registry.json"


class SignalRegistry:
    """
    Registry for all ranking signals. Loads from declarative JSON.

    Provides:
    - Signal definitions (weight, cost, description, category)
    - Per-type weights
    - Global toggles
    - Validation
    - Export/import
    """

    def __init__(self, registry_path: Optional[str] = None):
        self._path = Path(registry_path or DEFAULT_REGISTRY_PATH)
        self._weights = {}
        self._per_type = {}
        self._costs = {}
        self._descriptions = {}
        self._categories = {}
        self._toggles = {}
        self._custom_signals = {}
        self._compute_fns = {}
        self._version = "1.0"

        # Load from JSON
        self._load()

    # -------------------------
    # Loading
    # -------------------------

    def _load(self) -> None:
        """Load the registry from the JSON file."""
        if not self._path.exists():
            raise FileNotFoundError(
                f"Signal registry not found at {self._path}. "
                "Please ensure ranking/signal_registry.json exists."
            )

        with open(self._path, "r") as f:
            data = json.load(f)

        self._version = data.get("version", "1.0")
        signals = data.get("signals", {})

        for name, config in signals.items():
            self._weights[name] = config.get("default_weight", 0.10)
            self._per_type[name] = config.get("per_type", {})
            self._costs[name] = config.get("cost", "low")
            self._descriptions[name] = config.get("description", f"Signal: {name}")
            self._categories[name] = config.get("category", "ranking")
            self._toggles[name] = config.get("enabled", True)

    def reload(self) -> None:
        """Reload the registry from the JSON file."""
        self._weights.clear()
        self._per_type.clear()
        self._costs.clear()
        self._descriptions.clear()
        self._categories.clear()
        self._toggles.clear()
        self._load()

    # -------------------------
    # Registration (for custom signals)
    # -------------------------

    def register(
        self,
        name: str,
        compute_fn: Optional[Callable] = None,
        default_weight: Optional[float] = None,
        per_type: Optional[Dict[str, float]] = None,
        cost: SignalCost = "low",
        description: str = "",
        category: str = "ranking",
        enabled: bool = True,
    ) -> None:
        """Register a new signal at runtime (custom)."""
        if name in self._weights and name not in self._custom_signals:
            raise ValueError(f"Signal '{name}' already exists in the registry")

        self._weights[name] = default_weight if default_weight is not None else 0.10
        self._per_type[name] = per_type or {}
        self._costs[name] = cost
        self._descriptions[name] = description or f"Custom signal: {name}"
        self._categories[name] = category
        self._toggles[name] = enabled
        self._custom_signals[name] = True

        if compute_fn:
            self._compute_fns[name] = compute_fn

    def unregister(self, name: str) -> None:
        """Remove a custom signal."""
        if name in self._custom_signals:
            del self._weights[name]
            del self._per_type[name]
            del self._costs[name]
            del self._descriptions[name]
            del self._categories[name]
            del self._toggles[name]
            del self._custom_signals[name]
            if name in self._compute_fns:
                del self._compute_fns[name]

    def save(self, path: Optional[str] = None) -> None:
        """Save the current registry to JSON (including runtime changes)."""
        save_path = Path(path or self._path)

        data = {
            "version": self._version,
            "description": "Signal registry for Memory Daemon",
            "signals": {}
        }

        for name in self._weights:
            # Skip custom signals if they're not meant to be saved
            if name in self._custom_signals:
                continue

            data["signals"][name] = {
                "enabled": self._toggles.get(name, True),
                "default_weight": self._weights[name],
                "cost": self._costs.get(name, "low"),
                "category": self._categories.get(name, "ranking"),
                "description": self._descriptions.get(name, f"Signal: {name}"),
                "per_type": self._per_type.get(name, {}),
            }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    # -------------------------
    # Access
    # -------------------------

    def get_weight(self, name: SignalName, memory_type: MemoryType = "general") -> SignalWeight:
        """Get the weight for a signal and memory type."""
        if name not in self._weights:
            return 0.0
        return self._per_type.get(name, {}).get(memory_type, self._weights[name])

    def get_weights(self, memory_type: MemoryType = "general") -> Dict[SignalName, SignalWeight]:
        """Get all weights for a memory type."""
        return {
            name: self.get_weight(name, memory_type)
            for name in self._weights
            if self._toggles.get(name, True)
            and self.get_weight(name, memory_type) > 0.001
        }

    def get_active(self, memory_type: MemoryType = "general") -> Dict[SignalName, SignalWeight]:
        """Get active signals (enabled and weight > 0)."""
        return self.get_weights(memory_type)

    def get_cost(self, name: SignalName) -> SignalCost:
        """Get the cost of a signal."""
        return self._costs.get(name, "unknown")

    def get_description(self, name: SignalName) -> str:
        """Get the description of a signal."""
        return self._descriptions.get(name, "No description")

    def get_category(self, name: SignalName) -> str:
        """Get the category of a signal."""
        return self._categories.get(name, "unknown")

    def is_enabled(self, name: SignalName) -> bool:
        """Check if a signal is globally enabled."""
        return self._toggles.get(name, True)

    def is_enabled_for_type(self, name: SignalName, memory_type: MemoryType = "general") -> bool:
        """Check if a signal is enabled for a specific memory type."""
        if not self._toggles.get(name, True):
            return False
        return True

    def get_all(self) -> Dict[SignalName, Dict]:
        """Get all signal metadata."""
        return {
            name: {
                "weight": self._weights[name],
                "per_type": self._per_type.get(name, {}),
                "cost": self._costs.get(name, "unknown"),
                "description": self._descriptions.get(name, ""),
                "category": self._categories.get(name, "unknown"),
                "enabled": self._toggles.get(name, True),
                "custom": name in self._custom_signals,
            }
            for name in self._weights
        }

    def get_compute_fn(self, name: SignalName) -> Optional[Callable]:
        """Get the compute function for a signal."""
        return self._compute_fns.get(name)

    # -------------------------
    # Toggles
    # -------------------------

    def enable(self, name: SignalName) -> None:
        """Enable a signal globally."""
        if name in self._toggles:
            self._toggles[name] = True

    def disable(self, name: SignalName) -> None:
        """Disable a signal globally."""
        if name in self._toggles:
            self._toggles[name] = False

    def toggle(self, name: SignalName, enabled: bool) -> None:
        """Set global toggle for a signal."""
        if name in self._toggles:
            self._toggles[name] = enabled

    # -------------------------
    # Weights
    # -------------------------

    def set_weight(self, name: SignalName, weight: SignalWeight) -> None:
        """Set the default weight for a signal."""
        if name in self._weights:
            self._weights[name] = max(0.0, min(1.0, weight))

    def set_per_type_weight(self, name: SignalName, memory_type: MemoryType, weight: SignalWeight) -> None:
        """Set the per-type weight for a signal."""
        if name not in self._per_type:
            self._per_type[name] = {}
        self._per_type[name][memory_type] = max(0.0, min(1.0, weight))

    def normalize_weights(self, memory_type: MemoryType = "general") -> None:
        """Normalize weights to sum to 1.0 for a memory type."""
        weights = self.get_weights(memory_type)
        total = sum(weights.values())
        if total > 0:
            for name in weights:
                self._per_type[name][memory_type] = weights[name] / total

    # -------------------------
    # Validation
    # -------------------------

    def validate(self) -> bool:
        """Validate the registry."""
        for name, weight in self._weights.items():
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"Signal '{name}' weight {weight} out of range")
        return True

    # -------------------------
    # Export/Import
    # -------------------------

    def to_dict(self) -> Dict:
        """Export the registry to a dict."""
        return {
            "version": self._version,
            "weights": self._weights,
            "per_type": self._per_type,
            "costs": self._costs,
            "descriptions": self._descriptions,
            "categories": self._categories,
            "toggles": self._toggles,
            "custom_signals": list(self._custom_signals.keys()),
        }

    def from_dict(self, data: Dict) -> None:
        """Import the registry from a dict."""
        self._weights = data.get("weights", {})
        self._per_type = data.get("per_type", {})
        self._costs = data.get("costs", {})
        self._descriptions = data.get("descriptions", {})
        self._categories = data.get("categories", {})
        self._toggles = data.get("toggles", {})
        self._custom_signals = {name: True for name in data.get("custom_signals", [])}

    def load(self, path: Optional[str] = None) -> None:
        """Load the registry from a JSON file."""
        load_path = Path(path or self._path)
        with open(load_path, "r") as f:
            data = json.load(f)
        self.from_dict(data)


# -------------------------
# Global instance
# -------------------------

_registry = None

def get_registry() -> SignalRegistry:
    """Get the global signal registry instance."""
    global _registry
    if _registry is None:
        _registry = SignalRegistry()
    return _registry

def reset_registry() -> None:
    """Reset the global registry to defaults."""
    global _registry
    _registry = SignalRegistry()
