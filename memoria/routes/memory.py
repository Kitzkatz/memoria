from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from memory.memory_controller import MemoryController
from core.logger import debug, info, error

router = APIRouter(
    prefix="/memory",
    tags=["Memory"]
)

mc = MemoryController()


# --------------------------------------------------
# Models
# --------------------------------------------------

class MemoryInput(BaseModel):
    text: str


class BatchInput(BaseModel):
    texts: List[str]


class QueryInput(BaseModel):
    query: str
    limit: Optional[int] = None


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@router.post("/store")
def store(inp: MemoryInput):
    """Store a single memory."""
    try:
        mem_id = mc.remember(inp.text)
        debug(f"[API] Store: id={mem_id}, text={inp.text[:50]}...", category="api")
        return {
            "status": "stored",
            "id": mem_id
        }
    except Exception as e:
        error(f"[API] Store error: {e}", category="api")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
def query(inp: MemoryInput):
    """Query the memory system."""
    try:
        response = mc.recall(inp.text)
        debug(f"[API] Query: {inp.text[:50]}..., {len(response.get('results', []))} results", category="api")
        return {
            "query": inp.text,
            "count": len(response.get("results", [])),
            "results": response.get("results", []),
            "diagnostics": response.get("diagnostics", {})
        }
    except Exception as e:
        error(f"[API] Query error: {e}", category="api")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch_store")
def batch_store(inp: BatchInput):
    """Store multiple memories in batch."""
    try:
        if not inp.texts:
            return {"stored": 0, "ids": []}

        debug(f"[API] Batch store: {len(inp.texts)} texts", category="api")
        ids = mc.remember_many(inp.texts)

        # Force save to disk
        mc.system.vector_store.save()

        return {
            "stored": len(ids),
            "ids": ids
        }
    except Exception as e:
        error(f"[API] Batch store error: {e}", category="api")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reflect")
def reflect():
    """Run reflection on the memory system."""
    try:
        debug("[API] Reflect called", category="api")
        return {
            "reflection": mc.reflect()
        }
    except Exception as e:
        error(f"[API] Reflect error: {e}", category="api")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test_store")
def test_store(inp: MemoryInput):
    """Test endpoint for storing a memory."""
    try:
        mem_id = mc.remember(inp.text)
        debug(f"[API] Test store: id={mem_id}", category="api")
        return {
            "id": mem_id,
            "status": "stored"
        }
    except Exception as e:
        error(f"[API] Test store error: {e}", category="api")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/store_many")
def store_many(inp: BatchInput):
    """Alias for batch_store."""
    return batch_store(inp)


@router.get("/stats")
def stats():
    """Get memory statistics."""
    try:
        db = mc.system.db
        return {
            "memory_count": db.count(),
            "goals": len(mc.list_goals()),
        }
    except Exception as e:
        error(f"[API] Stats error: {e}", category="api")
        raise HTTPException(status_code=500, detail=str(e))
