"""
Core components for the memory system.

Exposes the plugin manager and hook utilities for external use.
"""

from .plugin_manager import MemoriaPluginManager, hookspec, hookimpl
from . import hooks

__all__ = [
    "MemoriaPluginManager",
    "hookspec",
    "hookimpl",
    "hooks",
]
