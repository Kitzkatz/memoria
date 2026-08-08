from fastapi import APIRouter, HTTPException
from typing import Optional

from memory.memory_controller import MemoryController
from tools.diagnostics import Diagnostics
from core.logger import debug, info, error

router = APIRouter(
    prefix="/debug",
    tags=["Debug"]
)

mc = MemoryController()


# -----------------------------------------
# STATS
# -----------------------------------------

@router.get("/stats")
def stats():
    """Get system statistics."""
    try:
        db = mc.system.db
        vector = mc.system.vector_store

        return {
            "db_rows": db.count(),
            "vector_rows": vector.count(),
            "embedding_cache": mc.system.embedding_cache.count(),
        }
    except Exception as e:
        error(f"[API] Stats error: {e}", category="api")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------
# LATEST
# -----------------------------------------

@router.get("/latest")
def latest(limit: int = 10):
    """Get the latest memories."""
    try:
        db = mc.system.db
        return {
            "latest": db.latest(limit)
        }
    except Exception as e:
        error(f"[API] Latest error: {e}", category="api")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------
# HEALTH
# -----------------------------------------

@router.get("/health")
def health():
    """Full health check."""
    try:
        # Diagnostics.full() expects a manager object
        # We need to pass the system or adapt Diagnostics
        return Diagnostics.full(mc.system)
    except Exception as e:
        error(f"[API] Health error: {e}", category="api")
        return {
            "status": "degraded",
            "error": str(e)
        }


# -----------------------------------------
# PROBE
# -----------------------------------------

@router.get("/probe")
def probe():
    """Detailed system probe."""
    try:
        system = mc.system
        db = system.db
        vector = system.vector_store

        db_count = db.count()
        vec_count = vector.count()

        return {
            "db_rows": db_count,
            "vector_rows": vec_count,
            "sync": db_count == vec_count,
            "sample": db.latest(5),
            "health": Diagnostics.full(system)
        }
    except Exception as e:
        error(f"[API] Probe error: {e}", category="api")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------
# CACHE
# -----------------------------------------

@router.get("/cache")
def cache_stats():
    """Get embedding cache statistics."""
    try:
        cache = mc.system.embedding_cache
        return {
            "cache_size": cache.count(),
            "cache_path": str(cache.cache_path),
            "max_size": cache._max_size,
        }
    except Exception as e:
        error(f"[API] Cache stats error: {e}", category="api")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------
# GRAPH
# -----------------------------------------

@router.get("/graph")
def graph_stats():
    """Get graph statistics."""
    try:
        graph = mc.system.numpy_graph
        return graph.stats()
    except Exception as e:
        error(f"[API] Graph stats error: {e}", category="api")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------
# ROUTER
# -----------------------------------------

@router.get("/router")
def router_info():
    """Get router configuration."""
    try:
        router = mc.system.router
        return {
            "types": router.list_types(),
            "default": router.default_type,
        }
    except Exception as e:
        error(f"[API] Router info error: {e}", category="api")
        raise HTTPException(status_code=500, detail=str(e))
