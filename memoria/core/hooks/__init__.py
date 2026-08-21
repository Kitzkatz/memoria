"""
Hook specifications for all Memoria subsystems.

Each module defines a set of hooks that plugins can implement.
"""

from . import (
    retrieval,
    ranking,
    storage,
    ingestion,
    query,
    scheduler,
    routing,
    feedback,
    evaluation,
    lifecycle,
)

__all__ = [
    "retrieval",
    "ranking",
    "storage",
    "ingestion",
    "query",
    "scheduler",
    "routing",
    "feedback",
    "evaluation",
    "lifecycle",
]
