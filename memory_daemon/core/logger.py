"""
Logger — Memory Daemon logging with category support.

Categories are used to enable/disable specific debug sections:
- retrieval, ranking, graph, ingestion, memory, system, routing, blackboard
- benchmark, db, cache, core, interface, api, maintenance, watcher
"""

import time
from cache.config import settings


# Global debug flag
DEBUG = getattr(settings, "DEBUG", True)

# Category-level debug control
_DEBUG_CATEGORIES: dict = {
    "retrieval": DEBUG,
    "ranking": DEBUG,
    "graph": DEBUG,
    "ingestion": DEBUG,
    "memory": DEBUG,
    "system": DEBUG,
    "routing": DEBUG,
    "blackboard": DEBUG,
    "benchmark": DEBUG,
    "db": DEBUG,
    "cache": DEBUG,
    "core": DEBUG,
    "interface": DEBUG,
    "api": DEBUG,
    "maintenance": DEBUG,
    "watcher": DEBUG,
    "general": DEBUG,
}


def debug(*args, category: str = None):
    """
    Print debug message if debugging is enabled.

    Args:
        *args: Message to print
        category: Optional category name to enable/disable specific debug sections
    """
    if not DEBUG:
        return

    if category is not None:
        if not _DEBUG_CATEGORIES.get(category, _DEBUG_CATEGORIES.get("general", True)):
            return
        prefix = f"[{category.upper()}]"
    else:
        prefix = "[DEBUG]"

    print(prefix, *args)


def info(*args, category: str = None):
    """Print info message. Category is accepted but not filtered."""
    if category:
        print(f"[INFO][{category.upper()}]", *args)
    else:
        print("[INFO]", *args)


def warn(*args, category: str = None):
    """Print warning message. Category is accepted but not filtered."""
    if category:
        print(f"[WARN][{category.upper()}]", *args)
    else:
        print("[WARN]", *args)


def error(*args, category: str = None):
    """Print error message. Category is accepted but not filtered."""
    if category:
        print(f"[ERROR][{category.upper()}]", *args)
    else:
        print("[ERROR]", *args)


def enable_category(category: str):
    """Enable debug output for a specific category."""
    if category in _DEBUG_CATEGORIES:
        _DEBUG_CATEGORIES[category] = True


def disable_category(category: str):
    """Disable debug output for a specific category."""
    if category in _DEBUG_CATEGORIES:
        _DEBUG_CATEGORIES[category] = False


def enable_all_categories():
    """Enable debug for all categories."""
    for cat in _DEBUG_CATEGORIES:
        _DEBUG_CATEGORIES[cat] = True


def disable_all_categories():
    """Disable debug for all categories."""
    for cat in _DEBUG_CATEGORIES:
        _DEBUG_CATEGORIES[cat] = False


def get_debug_categories():
    """Return set of enabled categories."""
    return {cat for cat, enabled in _DEBUG_CATEGORIES.items() if enabled}


class Timer:
    def __init__(self, label: str, category: str = None):
        self.label = label
        self.category = category

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start
        debug(f"{self.label}: {elapsed:.6f}s", category=self.category)
