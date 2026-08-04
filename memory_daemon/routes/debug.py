from fastapi import APIRouter

from memory.memory_controller import MemoryController
from tools.diagnostics import Diagnostics

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

    db = mc.manager.db
    vector = mc.manager.vector_store

    return {
        "db_rows": db.count(),
        "vector_rows": vector.count()
    }


# -----------------------------------------
# LATEST
# -----------------------------------------

@router.get("/latest")
def latest(limit: int = 10):

    return {
        "latest": mc.manager.db.latest(limit)
    }


# -----------------------------------------
# HEALTH
# -----------------------------------------

@router.get("/health")
def health():

    return Diagnostics.full(mc.manager)


# -----------------------------------------
# PROBE (FIXED)
# -----------------------------------------

@router.get("/probe")
def probe():

    manager = mc.manager
    db = manager.db
    vector = manager.vector_store

    db_count = db.count()
    vec_count = vector.count()

    return {
        "db_rows": db_count,
        "vector_rows": vec_count,
        "sync": db_count == vec_count,
        "sample": db.latest(5),
        "health": Diagnostics.full(manager)
    }



##from fastapi import APIRouter
##
##from memory.memory_controller import MemoryController
##from tools.diagnostics import Diagnostics
##
##router = APIRouter(
##
##    prefix="/debug",
##
##    tags=["Debug"]
##
##)
##
##mc = MemoryController()
##
##
##@router.get("/stats")
##
##def stats():
##
##    return {
##
##        "db_rows": mc.manager.db.count(),
##
##        "vectors": mc.manager.vector_store.count()
##
##    }
##
##
##@router.get("/latest")
##
##def latest(limit: int = 10):
##
##    return {
##
##        "latest": mc.manager.db.latest(limit)
##
##    }
##
##
##@router.get("/health")
##
##def health():
##
##    return Diagnostics.full(
##
##        mc.manager
##
##    )
##
##
##@router.get("/probe")
##
##def probe():
##
##    manager = mc.manager
##
##    db = manager.db
##
##    vector = manager.vector_store
##
##    return {
##
##        "db_rows": db.count(),
##
##        "vector_rows": vector.count(),
##
##        "sync": db.count() == vector.count(),
##
##        "sample": db.latest(5),
##
##        "health": Diagnostics.full(manager)
##
##    }
