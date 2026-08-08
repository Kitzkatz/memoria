from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import time
from typing import List, Optional

from memory.memory_controller import MemoryController
from core.logger import debug, info, error

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


class SpeedBenchmarkPayload(BaseModel):
    queries: List[str]


class StoreBenchmarkPayload(BaseModel):
    texts: List[str]


# ----------------------------------------------------
# Accuracy Benchmark
# ----------------------------------------------------

@router.post("/run")
def benchmark(payload: List[BenchmarkItem]):
    """Run accuracy benchmark with expected results."""
    try:
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
            results_list = results.get("results", [])

            joined = " ".join(
                r.get("text", "")
                for r in results_list
            ).lower()

            if item.expected.lower() in joined:
                correct += 1
            else:
                failures.append({
                    "query": item.query,
                    "expected": item.expected,
                    "returned": [
                        r.get("text", "")
                        for r in results_list[:3]
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

    except Exception as e:
        error(f"[BENCHMARK] Run error: {e}", category="benchmark")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------
# Query Speed Benchmark
# ----------------------------------------------------

@router.post("/speed")
def query_speed(payload: SpeedBenchmarkPayload):
    """Benchmark query speed."""
    try:
        queries = payload.queries
        if not queries:
            return {
                "queries": 0,
                "seconds": 0.0,
                "qps": 0.0
            }

        start = time.perf_counter()

        for query in queries:
            mc.recall(query)

        elapsed = time.perf_counter() - start

        qps = round(len(queries) / elapsed, 2) if elapsed > 0 else 0.0

        return {
            "queries": len(queries),
            "seconds": round(elapsed, 3),
            "qps": qps
        }

    except Exception as e:
        error(f"[BENCHMARK] Speed error: {e}", category="benchmark")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------
# Store Speed Benchmark
# ----------------------------------------------------

@router.post("/store_speed")
def store_speed(payload: StoreBenchmarkPayload):
    """Benchmark store speed."""
    try:
        texts = payload.texts
        if not texts:
            return {
                "stored": 0,
                "seconds": 0.0,
                "stores_per_second": 0.0,
                "db_rows": 0,
                "vector_rows": 0,
                "synced": True
            }

        start = time.perf_counter()

        # Use batch storage for efficiency
        ids = mc.remember_many(texts)

        # Force save
        mc.system.vector_store.save()

        elapsed = time.perf_counter() - start

        db = mc.system.db
        vector = mc.system.vector_store

        stores_per_second = round(len(texts) / elapsed, 2) if elapsed > 0 else 0.0

        return {
            "stored": len(ids),
            "seconds": round(elapsed, 3),
            "stores_per_second": stores_per_second,
            "db_rows": db.count(),
            "vector_rows": vector.count(),
            "synced": db.count() == vector.count()
        }

    except Exception as e:
        error(f"[BENCHMARK] Store speed error: {e}", category="benchmark")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------
# Full Benchmark Suite
# ----------------------------------------------------

@router.post("/full")
def full_benchmark(
    accuracy_payload: Optional[List[BenchmarkItem]] = None,
    speed_payload: Optional[SpeedBenchmarkPayload] = None,
    store_payload: Optional[StoreBenchmarkPayload] = None
):
    """Run all benchmarks in one call."""
    try:
        results = {}

        # Accuracy
        if accuracy_payload:
            results["accuracy"] = benchmark(accuracy_payload)

        # Speed
        if speed_payload:
            results["speed"] = query_speed(speed_payload)

        # Store speed
        if store_payload:
            results["store_speed"] = store_speed(store_payload)

        return results

    except Exception as e:
        error(f"[BENCHMARK] Full error: {e}", category="benchmark")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------
# Quick Benchmark (minimal)
# ----------------------------------------------------

@router.post("/quick")
def quick_benchmark():
    """Quick benchmark with a small set of queries."""
    try:
        # Small test set
        test_queries = [
            "what is the meaning of life",
            "who is the president",
            "what is 2+2",
        ]

        start = time.perf_counter()

        for query in test_queries:
            mc.recall(query)

        elapsed = time.perf_counter() - start

        return {
            "queries": len(test_queries),
            "seconds": round(elapsed, 3),
            "qps": round(len(test_queries) / elapsed, 2) if elapsed > 0 else 0.0,
            "note": "Quick benchmark with 3 sample queries"
        }

    except Exception as e:
        error(f"[BENCHMARK] Quick error: {e}", category="benchmark")
        raise HTTPException(status_code=500, detail=str(e))
