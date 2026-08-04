from fastapi import APIRouter
from pydantic import BaseModel
import time

from memory.memory_controller import MemoryController

router = APIRouter(
    prefix="/benchmark",
    tags=["Benchmark"]
)

mc = MemoryController()


# ----------------------------------------------------
# Models
# ----------------------------------------------------

class BenchmarkItem(BaseModel):
    query: str
    expected: str


# ----------------------------------------------------
# Accuracy Benchmark
# ----------------------------------------------------

@router.post("/run")
def benchmark(payload: list[BenchmarkItem]):

    start = time.perf_counter()

    total = len(payload)

    if total == 0:
        return {
            "accuracy": 0,
            "correct": 0,
            "total": 0,
            "failed": 0,
            "runtime": 0.0,
            "sample_failures": []
        }

    correct = 0
    failures = []

    for item in payload:

        results = mc.recall(item.query)

        joined = " ".join(
            r.get("text", "")
            for r in results
        ).lower()

        if item.expected.lower() in joined:

            correct += 1

        else:

            failures.append({

                "query": item.query,

                "expected": item.expected,

                "returned": [

                    r.get("text", "")

                    for r in results[:3]

                ]

            })

    elapsed = time.perf_counter() - start

    return {

        "accuracy": round((correct / total) * 100, 2),

        "correct": correct,

        "total": total,

        "failed": len(failures),

        "runtime": round(elapsed, 3),

        "sample_failures": failures[:20]

    }


# ----------------------------------------------------
# Query Speed Benchmark
# ----------------------------------------------------

@router.post("/speed")
def query_speed(payload: list[str]):

    start = time.perf_counter()

    for query in payload:

        mc.recall(query)

    elapsed = time.perf_counter() - start

    qps = 0

    if elapsed > 0:

        qps = round(len(payload) / elapsed, 2)

    return {

        "queries": len(payload),

        "seconds": round(elapsed, 3),

        "qps": qps

    }


# ----------------------------------------------------
# Store Speed Benchmark
# ----------------------------------------------------

@router.post("/store_speed")
def store_speed(payload: list[str]):

    start = time.perf_counter()

    stored = 0

    for memory in payload:

        mc.remember(memory)

        stored += 1

    mc.manager.vector_store.save()

    elapsed = time.perf_counter() - start

    stores_per_second = 0

    if elapsed > 0:

        stores_per_second = round(stored / elapsed, 2)

    return {

        "stored": stored,

        "seconds": round(elapsed, 3),

        "stores_per_second": stores_per_second,

        "db_rows": mc.manager.db.count(),

        "vector_rows": mc.manager.vector_store.count(),

        "synced": mc.manager.db.count() == mc.manager.vector_store.count()

    }



##from fastapi import APIRouter
##from pydantic import BaseModel
##import time
##
##from memory.memory_controller import MemoryController
##
##router = APIRouter(
##
##    prefix="/benchmark",
##
##    tags=["Benchmark"]
##
##)
##
##mc = MemoryController()
##
##
##class BenchmarkItem(BaseModel):
##
##    query:str
##
##    expected:str
##
##
### ---------------------------------------
### Accuracy
### ---------------------------------------
##
##@router.post("/run")
##def benchmark(payload:list[BenchmarkItem]):
##
##    start=time.perf_counter()
##
##    total=len(payload)
##
##    correct=0
##
##    failures=[]
##
##    for item in payload:
##
##        results=mc.recall(item.query)
##
##        joined=" ".join(
##
##            r["text"]
##
##            for r in results
##
##        ).lower()
##
##        if item.expected.lower() in joined:
##
##            correct+=1
##
##        else:
##
##            failures.append({
##
##                "query":item.query,
##
##                "expected":item.expected,
##
##                "returned":[
##
##                    r["text"]
##
##                    for r in results[:3]
##
##                ]
##
##            })
##
##    elapsed=time.perf_counter()-start
##
##    return{
##
##        "accuracy":
##
##            round(
##
##                correct/total*100,
##
##                2
##
##            ),
##
##        "correct":correct,
##
##        "total":total,
##
##        "failed":len(failures),
##
##        "runtime":round(elapsed,3),
##
##        "sample_failures":failures[:20]
##
##    }
##
##
### ---------------------------------------
### Query Speed
### ---------------------------------------
##
##@router.post("/speed")
##def query_speed(payload:list[str]):
##
##    start=time.perf_counter()
##
##    for q in payload:
##
##        mc.recall(q)
##
##    elapsed=time.perf_counter()-start
##
##    return{
##
##        "queries":len(payload),
##
##        "seconds":round(elapsed,3),
##
##        "qps":round(
##
##            len(payload)/elapsed,
##
##            2
##
##        )
##
##    }
##
##
### ---------------------------------------
### Store Speed
### ---------------------------------------
##
##@router.post("/store_speed")
##def store_speed(payload:list[str]):
##
##    start=time.perf_counter()
##
##    for memory in payload:
##
##        mc.remember(memory)
##
##    mc.manager.vector_store.save()
##
##    elapsed=time.perf_counter()-start
##
##    return{
##
##        "stored":len(payload),
##
##        "seconds":round(elapsed,3),
##
##        "stores_per_second":
##
##            round(
##
##                len(payload)/elapsed,
##
##                2
##
##            )
##
##    }
