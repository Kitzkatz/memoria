from core.logger import debug
from fastapi import APIRouter
import time

from memory.memory_controller import MemoryController

router = APIRouter(
    prefix="/maintenance",
    tags=["Maintenance"]
)

mc = MemoryController()


# -----------------------------------------
# REBUILD INDEX (FIXED)
# -----------------------------------------

@router.post("/rebuild_index")
def rebuild_index():

    start = time.perf_counter()

    manager = mc.manager
    db = manager.db
    vector = manager.vector_store
    embedder = manager.embedder

    debug("\n[REBUILD] Starting rebuild...")

    # 🔥 CRITICAL: full reset first (prevents ghost vectors)
    vector.reset()

    rows = db.fetch_all()

    rebuilt = 0

    for row in rows:

        text = row["normalized_text"] or row["text"]

        embedding = embedder.embed(text)

        vector.add(
            row["id"],
            embedding
        )

        rebuilt += 1

    vector.save()

    elapsed = time.perf_counter() - start

    return {
        "status": "complete",
        "rebuilt": rebuilt,
        "db_rows": db.count(),
        "vector_rows": vector.count(),
        "synced": rebuilt == db.count() == vector.count(),
        "seconds": round(elapsed, 3)
    }


# -----------------------------------------
# VERIFY
# -----------------------------------------

@router.get("/verify")
def verify():

    db = mc.manager.db
    vector = mc.manager.vector_store

    return {
        "db_rows": db.count(),
        "vector_rows": vector.count(),
        "synced": db.count() == vector.count()
    }


# -----------------------------------------
# RESET INDEX
# -----------------------------------------

@router.post("/reset_index")
def reset():

    vector = mc.manager.vector_store

    vector.reset()
    vector.save()

    return {
        "status": "reset",
        "vectors": vector.count()
    }


# -----------------------------------------
# FORCE SAVE
# -----------------------------------------

@router.post("/save")
def save():

    vector = mc.manager.vector_store
    vector.save()

    return {
        "status": "saved",
        "vectors": vector.count()
    }



##from fastapi import APIRouter
##import time
##
##from memory.memory_controller import MemoryController
##
##router = APIRouter(
##
##    prefix="/maintenance",
##
##    tags=["Maintenance"]
##
##)
##
##mc = MemoryController()
##
##
### -----------------------------------------
### Rebuild FAISS
### -----------------------------------------
##
##@router.post("/rebuild_index")
##def rebuild_index():
##
##    start = time.perf_counter()
##
##    manager = mc.manager
##
##    db = manager.db
##
##    vector = manager.vector_store
##
##    embedder = manager.embedder
##
##    debug(id(vs))
##
##    debug("\n[REBUILD] Starting rebuild...")
##
##    vector.reset()
##
##    rebuilt = 0
##
##    rows = db.fetch_all()
##
##    for row in rows:
##
##        text = row["normalized_text"] or row["text"]
##
##        embedding = embedder.embed(text)
##
##        vector.add(
##
##            row["id"],
##
##            embedding
##
##        )
##
##        rebuilt += 1
##
##    vector.save()
##
##    elapsed = time.perf_counter() - start
##
##    synced = (
##
##        rebuilt == db.count()
##
##        and
##
##        rebuilt == vector.count()
##
##    )
##
##    return {
##
##        "status": "complete",
##
##        "rebuilt": rebuilt,
##
##        "db_rows": db.count(),
##
##        "vector_rows": vector.count(),
##
##        "synced": synced,
##
##        "seconds": round(elapsed,3)
##
##    }
##
##
### -----------------------------------------
### Verify
### -----------------------------------------
##
##@router.get("/verify")
##def verify():
##
##    db = mc.manager.db
##
##    vector = mc.manager.vector_store
##
##    return {
##
##        "database": db.count(),
##
##        "vectors": vector.count(),
##
##        "synced": db.count()==vector.count()
##
##    }
##
##
### -----------------------------------------
### Reset Index
### -----------------------------------------
##
##@router.post("/reset_index")
##def reset():
##
##    vector = mc.manager.vector_store
##
##    vector.reset()
##
##    vector.save()
##
##    return {
##
##        "status":"reset",
##
##        "vectors":vector.count()
##
##    }
##
##
### -----------------------------------------
### Force Save
### -----------------------------------------
##
##@router.post("/save")
##def save():
##
##    mc.manager.vector_store.save()
##
##    return {
##
##        "status":"saved",
##
##        "vectors":mc.manager.vector_store.count()
##
##    }
