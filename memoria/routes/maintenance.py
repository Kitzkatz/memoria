from fastapi import APIRouter, HTTPException
import time

from memory.memory_controller import MemoryController
from core.logger import debug, info, error

router = APIRouter(
    prefix="/maintenance",
    tags=["Maintenance"]
)

mc = MemoryController()


# -----------------------------------------
# REBUILD INDEX
# -----------------------------------------

@router.post("/rebuild_index")
def rebuild_index():
    """Rebuild the FAISS index from the database."""
    try:
        start = time.perf_counter()

        system = mc.system
        db = system.db
        vector = system.vector_store
        embedder = system.embedder

        debug("[MAINTENANCE] Starting index rebuild...", category="maintenance")

        # Full reset first (prevents ghost vectors)
        vector.reset()

        rows = db.fetch_all()
        rebuilt = 0

        for row in rows:
            text = row.get("normalized_text") or row.get("text", "")
            if not text:
                continue

            embedding = embedder.embed(text)
            if embedding:
                vector.add(row["id"], embedding)
                rebuilt += 1

        vector.save()

        elapsed = time.perf_counter() - start

        info(f"[MAINTENANCE] Rebuilt {rebuilt} vectors in {elapsed:.2f}s", category="maintenance")

        return {
            "status": "complete",
            "rebuilt": rebuilt,
            "db_rows": db.count(),
            "vector_rows": vector.count(),
            "synced": rebuilt == db.count() == vector.count(),
            "seconds": round(elapsed, 3)
        }

    except Exception as e:
        error(f"[MAINTENANCE] Rebuild error: {e}", category="maintenance")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------
# VERIFY
# -----------------------------------------

@router.get("/verify")
def verify():
    """Verify database and vector store are in sync."""
    try:
        system = mc.system
        db = system.db
        vector = system.vector_store

        db_count = db.count()
        vec_count = vector.count()

        return {
            "db_rows": db_count,
            "vector_rows": vec_count,
            "synced": db_count == vec_count
        }

    except Exception as e:
        error(f"[MAINTENANCE] Verify error: {e}", category="maintenance")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------
# RESET INDEX
# -----------------------------------------

@router.post("/reset_index")
def reset_index():
    """Reset the FAISS index (removes all vectors)."""
    try:
        system = mc.system
        vector = system.vector_store

        vector.reset()
        vector.save()

        info("[MAINTENANCE] Index reset", category="maintenance")

        return {
            "status": "reset",
            "vectors": vector.count()
        }

    except Exception as e:
        error(f"[MAINTENANCE] Reset error: {e}", category="maintenance")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------
# FORCE SAVE
# -----------------------------------------

@router.post("/save")
def save():
    """Force save the vector index."""
    try:
        system = mc.system
        vector = system.vector_store

        vector.save()

        debug("[MAINTENANCE] Force save complete", category="maintenance")

        return {
            "status": "saved",
            "vectors": vector.count()
        }

    except Exception as e:
        error(f"[MAINTENANCE] Save error: {e}", category="maintenance")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------
# CLEANUP
# -----------------------------------------

@router.post("/cleanup")
def cleanup(dry_run: bool = True):
    """Run maintenance cleanup (prune old entries, etc.)."""
    try:
        system = mc.system

        # Run pruner if available
        if hasattr(system, 'pruner'):
            result = system.pruner.prune_now(dry_run=dry_run)
            return {
                "status": "complete",
                "dry_run": dry_run,
                "pruned": result.get("pruned", 0),
                "total": result.get("total", 0),
            }
        else:
            return {
                "status": "skipped",
                "message": "Pruner not available"
            }

    except Exception as e:
        error(f"[MAINTENANCE] Cleanup error: {e}", category="maintenance")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------
# CONSOLIDATE
# -----------------------------------------

@router.post("/consolidate")
def consolidate(threshold: float = 0.85, dry_run: bool = True):
    """Run consolidation on duplicate memories."""
    try:
        system = mc.system

        if hasattr(system, 'consolidator'):
            result = system.consolidator.run(
                threshold=threshold,
                dry_run=dry_run
            )
            return {
                "status": "complete",
                "dry_run": dry_run,
                "threshold": threshold,
                "result": result
            }
        else:
            return {
                "status": "skipped",
                "message": "Consolidator not available"
            }

    except Exception as e:
        error(f"[MAINTENANCE] Consolidate error: {e}", category="maintenance")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------
# REBUILD CACHE
# -----------------------------------------

@router.post("/rebuild_cache")
def rebuild_cache():
    """Rebuild the embedding cache."""
    try:
        start = time.perf_counter()

        system = mc.system
        db = system.db
        vector = system.vector_store
        cache = system.embedding_cache

        cache.rebuild(db, vector)

        elapsed = time.perf_counter() - start

        return {
            "status": "complete",
            "cache_size": cache.count(),
            "seconds": round(elapsed, 3)
        }

    except Exception as e:
        error(f"[MAINTENANCE] Rebuild cache error: {e}", category="maintenance")
        raise HTTPException(status_code=500, detail=str(e))
