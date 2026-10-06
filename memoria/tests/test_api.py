"""
Memoria HTTP API tests.

These tests exercise the mounted FastAPI application through TestClient.

The route implementations are tested against small fakes so the suite
validates the HTTP/API contract without depending on the real memory engine,
FAISS, embeddings, LLMs, filesystem state, or benchmark data.

No bare pytest/Python assertions are used. Failures go through require()
so the error messages identify the API contract that failed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import app
import routes.benchmark as benchmark_routes
import routes.chat as chat_routes
import routes.debug as debug_routes
import routes.maintenance as maintenance_routes
import routes.memory as memory_routes


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def require(condition, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_status(response, expected: int, endpoint: str) -> None:
    require(
        response.status_code == expected,
        f"{endpoint}: expected HTTP {expected}, "
        f"got {response.status_code}; body={response.text}",
    )


def require_json(response, expected: dict, endpoint: str) -> None:
    require_status(response, 200, endpoint)

    actual = response.json()

    require(
        actual == expected,
        f"{endpoint}: unexpected JSON.\n"
        f"Expected: {expected!r}\n"
        f"Actual:   {actual!r}",
    )


# ---------------------------------------------------------------------------
# Fake system components
# ---------------------------------------------------------------------------

class FakeVectorStore:
    def __init__(self):
        self.save_calls = 0
        self.reset_calls = 0
        self.add_calls = []
        self._count = 3

    def save(self):
        self.save_calls += 1

    def reset(self):
        self.reset_calls += 1
        self._count = 0

    def count(self):
        return self._count

    def add(self, mem_id, embedding):
        self.add_calls.append((mem_id, embedding))
        self._count += 1


class FakeDB:
    def __init__(self):
        self._rows = [
            {
                "id": 1,
                "text": "first memory",
                "normalized_text": "first memory",
            },
            {
                "id": 2,
                "text": "second memory",
                "normalized_text": "second memory",
            },
            {
                "id": 3,
                "text": "third memory",
                "normalized_text": "third memory",
            },
        ]

    def count(self):
        return len(self._rows)

    def fetch_all(self):
        return list(self._rows)

    def latest(self, limit):
        return self._rows[-limit:]

    def set_rows(self, rows):
        self._rows = rows


class FakeEmbedder:
    def embed(self, text):
        return [0.1, 0.2, 0.3]


class FakeCache:
    def __init__(self):
        self.rebuild_calls = 0
        self.cache_path = "/fake/cache"
        self._max_size = 100

    def count(self):
        return 7

    def rebuild(self, db, vector):
        self.rebuild_calls += 1


class FakeGraph:
    def stats(self):
        return {
            "nodes": 3,
            "edges": 2,
        }


class FakeRouter:
    default_type = "general"

    def list_types(self):
        return ["general", "task", "goal"]


class FakePruner:
    def __init__(self):
        self.calls = []

    def prune_now(self, dry_run=True):
        self.calls.append(dry_run)
        return {
            "pruned": 2,
            "total": 10,
        }


class FakeConsolidator:
    def __init__(self):
        self.calls = []

    def run(self, threshold=0.85, dry_run=True):
        self.calls.append((threshold, dry_run))
        return {
            "merged": 2,
            "examined": 10,
        }


class FakeSystem:
    def __init__(self):
        self.db = FakeDB()
        self.vector_store = FakeVectorStore()
        self.embedder = FakeEmbedder()
        self.embedding_cache = FakeCache()
        self.numpy_graph = FakeGraph()
        self.router = FakeRouter()
        self.pruner = FakePruner()
        self.consolidator = FakeConsolidator()


class FakeController:
    def __init__(self):
        self.system = FakeSystem()
        self.chat_history = ["previous chat"]

        self.remember_calls = []
        self.remember_many_calls = []
        self.recall_calls = []
        self.reflect_calls = 0
        self.chat_calls = []
        self.raw_chat_calls = []

    def remember(self, text):
        self.remember_calls.append(text)
        return 101

    def remember_many(self, texts):
        self.remember_many_calls.append(list(texts))
        return list(range(201, 201 + len(texts)))

    def recall(self, query):
        self.recall_calls.append(query)
        return {
            "results": [
                {
                    "id": 1,
                    "text": "alpha memory",
                },
                {
                    "id": 2,
                    "text": "beta memory",
                },
            ],
            "diagnostics": {
                "worker": "fake",
            },
        }

    def reflect(self):
        self.reflect_calls += 1
        return "reflection result"

    def stats(self):
        return {
            "memories": 3,
            "vectors": 3,
        }

    def chat(self, text, top_n=None):
        self.chat_calls.append((text, top_n))
        return "chat response"

    def raw_chat(self, text):
        self.raw_chat_calls.append(text)
        return "raw response"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_controller(monkeypatch):
    controller = FakeController()

    # Important:
    # The route modules use:
    #
    #     from memory.controller_instance import controller as mc
    #
    # Therefore patch the already-bound mc names inside each route module.
    monkeypatch.setattr(memory_routes, "mc", controller)
    monkeypatch.setattr(maintenance_routes, "mc", controller)
    monkeypatch.setattr(debug_routes, "mc", controller)
    monkeypatch.setattr(chat_routes, "mc", controller)
    monkeypatch.setattr(benchmark_routes, "mc", controller)

    # Debug/probe/health call Diagnostics.full() directly.
    fake_diagnostics = Mock()
    fake_diagnostics.full.return_value = {
        "status": "healthy",
        "checks": 3,
    }

    monkeypatch.setattr(
        debug_routes,
        "Diagnostics",
        fake_diagnostics,
    )

    return controller


@pytest.fixture
def client(fake_controller):
    with TestClient(app.app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Application / routing
# ---------------------------------------------------------------------------

def test_app_root(client):
    response = client.get("/")

    require_status(response, 200, "GET /")
    require(
        "text/html" in response.headers.get("content-type", ""),
        f"GET /: expected HTML response, got "
        f"{response.headers.get('content-type')!r}",
    )


def test_health(client):
    response = client.get("/health")

    require_json(
        response,
        {
            "status": "ok",
            "version": "1.0",
            "service": "Memoria",
        },
        "GET /health",
    )


def test_openapi_contains_all_api_paths(client):
    response = client.get("/openapi.json")

    require_status(response, 200, "GET /openapi.json")

    schema = response.json()
    paths = schema.get("paths", {})

    expected = {
        "/",
        "/health",

        "/memory/store",
        "/memory/query",
        "/memory/batch_store",
        "/memory/reflect",
        "/memory/test_store",
        "/memory/store_many",
        "/memory/stats",

        "/maintenance/rebuild_index",
        "/maintenance/verify",
        "/maintenance/reset_index",
        "/maintenance/save",
        "/maintenance/cleanup",
        "/maintenance/consolidate",
        "/maintenance/rebuild_cache",

        "/debug/stats",
        "/debug/latest",
        "/debug/health",
        "/debug/probe",
        "/debug/cache",
        "/debug/graph",
        "/debug/router",

        "/chat/",
        "/chat/history",
        "/chat/raw",

        "/benchmark/run",
        "/benchmark/speed",
        "/benchmark/store_speed",
        "/benchmark/full",
        "/benchmark/quick",
    }

    missing = sorted(expected - set(paths))

    require(
        not missing,
        f"OpenAPI is missing expected paths: {missing}",
    )


def test_docs_available(client):
    response = client.get("/docs")

    require_status(response, 200, "GET /docs")


# ---------------------------------------------------------------------------
# Memory API
# ---------------------------------------------------------------------------

def test_memory_store(client, fake_controller):
    response = client.post(
        "/memory/store",
        json={"text": "hello memory"},
    )

    require_json(
        response,
        {
            "status": "stored",
            "id": 101,
        },
        "POST /memory/store",
    )

    require(
        fake_controller.remember_calls == ["hello memory"],
        f"POST /memory/store: remember calls were "
        f"{fake_controller.remember_calls!r}",
    )


def test_memory_store_validation(client):
    response = client.post(
        "/memory/store",
        json={},
    )

    require_status(response, 422, "POST /memory/store missing text")


def test_memory_query(client, fake_controller):
    response = client.post(
        "/memory/query",
        json={"text": "find alpha"},
    )

    require_status(response, 200, "POST /memory/query")

    body = response.json()

    require(
        body.get("query") == "find alpha",
        f"POST /memory/query: unexpected query field: {body!r}",
    )

    require(
        body.get("count") == 2,
        f"POST /memory/query: unexpected count: {body!r}",
    )

    require(
        len(body.get("results", [])) == 2,
        f"POST /memory/query: unexpected results: {body!r}",
    )

    require(
        body.get("diagnostics") == {"worker": "fake"},
        f"POST /memory/query: unexpected diagnostics: {body!r}",
    )

    require(
        fake_controller.recall_calls == ["find alpha"],
        f"POST /memory/query: recall calls were "
        f"{fake_controller.recall_calls!r}",
    )


def test_memory_query_validation(client):
    response = client.post(
        "/memory/query",
        json={"query": "this is QueryInput shape"},
    )

    # The current endpoint accepts MemoryInput, not QueryInput.
    require_status(
        response,
        422,
        "POST /memory/query with query-only payload",
    )


def test_memory_batch_store(client, fake_controller):
    response = client.post(
        "/memory/batch_store",
        json={
            "texts": [
                "one",
                "two",
                "three",
            ]
        },
    )

    require_json(
        response,
        {
            "stored": 3,
            "ids": [201, 202, 203],
        },
        "POST /memory/batch_store",
    )

    require(
        fake_controller.remember_many_calls == [
            ["one", "two", "three"]
        ],
        f"POST /memory/batch_store: remember_many calls were "
        f"{fake_controller.remember_many_calls!r}",
    )

    require(
        fake_controller.system.vector_store.save_calls == 1,
        "POST /memory/batch_store: vector store was not saved exactly once",
    )


def test_memory_batch_store_empty(client, fake_controller):
    response = client.post(
        "/memory/batch_store",
        json={"texts": []},
    )

    require_json(
        response,
        {
            "stored": 0,
            "ids": [],
        },
        "POST /memory/batch_store empty",
    )

    require(
        fake_controller.remember_many_calls == [],
        "POST /memory/batch_store empty: remember_many should not be called",
    )


def test_memory_store_many_alias(client, fake_controller):
    response = client.post(
        "/memory/store_many",
        json={
            "texts": [
                "one",
                "two",
            ]
        },
    )

    require_json(
        response,
        {
            "stored": 2,
            "ids": [201, 202],
        },
        "POST /memory/store_many",
    )

    require(
        fake_controller.remember_many_calls == [["one", "two"]],
        f"POST /memory/store_many: remember_many calls were "
        f"{fake_controller.remember_many_calls!r}",
    )


def test_memory_reflect(client, fake_controller):
    response = client.post("/memory/reflect")

    require_json(
        response,
        {
            "reflection": "reflection result",
        },
        "POST /memory/reflect",
    )

    require(
        fake_controller.reflect_calls == 1,
        "POST /memory/reflect: reflect was not called exactly once",
    )


def test_memory_test_store(client, fake_controller):
    response = client.post(
        "/memory/test_store",
        json={"text": "test memory"},
    )

    require_json(
        response,
        {
            "id": 101,
            "status": "stored",
        },
        "POST /memory/test_store",
    )

    require(
        fake_controller.remember_calls == ["test memory"],
        f"POST /memory/test_store: remember calls were "
        f"{fake_controller.remember_calls!r}",
    )


def test_memory_stats_500_translation(
    client,
    fake_controller,
    monkeypatch,
):
    def explode(*args, **kwargs):
        raise RuntimeError("forced API failure")

    monkeypatch.setattr(
        fake_controller,
        "stats",
        explode,
    )

    response = client.get("/memory/stats")

    require_status(
        response,
        500,
        "GET /memory/stats forced failure",
    )

    body = response.json()

    require(
        body.get("detail") == "forced API failure",
        f"GET /memory/stats forced failure: unexpected body {body!r}",
    )

# ---------------------------------------------------------------------------
# Maintenance API
# ---------------------------------------------------------------------------

def test_maintenance_verify(client):
    response = client.get("/maintenance/verify")

    require_json(
        response,
        {
            "db_rows": 3,
            "vector_rows": 3,
            "synced": True,
        },
        "GET /maintenance/verify",
    )


def test_maintenance_save(client, fake_controller):
    response = client.post("/maintenance/save")

    require_json(
        response,
        {
            "status": "saved",
            "vectors": 3,
        },
        "POST /maintenance/save",
    )

    require(
        fake_controller.system.vector_store.save_calls == 1,
        "POST /maintenance/save: vector store was not saved exactly once",
    )


def test_maintenance_reset_index(client, fake_controller):
    response = client.post("/maintenance/reset_index")

    require_json(
        response,
        {
            "status": "reset",
            "vectors": 0,
        },
        "POST /maintenance/reset_index",
    )

    require(
        fake_controller.system.vector_store.reset_calls == 1,
        "POST /maintenance/reset_index: reset was not called exactly once",
    )

    require(
        fake_controller.system.vector_store.save_calls == 1,
        "POST /maintenance/reset_index: save was not called exactly once",
    )


def test_maintenance_rebuild_index(client, fake_controller):
    response = client.post("/maintenance/rebuild_index")

    require_status(
        response,
        200,
        "POST /maintenance/rebuild_index",
    )

    body = response.json()

    require(
        body.get("status") == "complete",
        f"POST /maintenance/rebuild_index: unexpected status: {body!r}",
    )

    require(
        body.get("rebuilt") == 3,
        f"POST /maintenance/rebuild_index: expected 3 rebuilt rows: {body!r}",
    )

    require(
        body.get("db_rows") == 3,
        f"POST /maintenance/rebuild_index: unexpected db_rows: {body!r}",
    )

    require(
        body.get("vector_rows") == 3,
        f"POST /maintenance/rebuild_index: unexpected vector_rows: {body!r}",
    )

    require(
        body.get("synced") is True,
        f"POST /maintenance/rebuild_index: expected synced=True: {body!r}",
    )

    require(
        body.get("seconds", -1) >= 0,
        f"POST /maintenance/rebuild_index: invalid seconds: {body!r}",
    )

    require(
        fake_controller.system.vector_store.reset_calls == 1,
        "POST /maintenance/rebuild_index: reset was not called",
    )

    require(
        fake_controller.system.vector_store.save_calls == 1,
        "POST /maintenance/rebuild_index: save was not called",
    )

    require(
        len(fake_controller.system.vector_store.add_calls) == 3,
        "POST /maintenance/rebuild_index: expected three vector additions",
    )


def test_maintenance_cleanup_dry_run(client, fake_controller):
    response = client.post(
        "/maintenance/cleanup",
        params={"dry_run": "true"},
    )

    require_json(
        response,
        {
            "status": "complete",
            "dry_run": True,
            "pruned": 2,
            "total": 10,
        },
        "POST /maintenance/cleanup",
    )

    require(
        fake_controller.system.pruner.calls == [True],
        f"POST /maintenance/cleanup: pruner calls were "
        f"{fake_controller.system.pruner.calls!r}",
    )


def test_maintenance_cleanup_live(client, fake_controller):
    response = client.post(
        "/maintenance/cleanup",
        params={"dry_run": "false"},
    )

    require_status(response, 200, "POST /maintenance/cleanup live")

    body = response.json()

    require(
        body.get("dry_run") is False,
        f"POST /maintenance/cleanup live: unexpected body {body!r}",
    )

    require(
        fake_controller.system.pruner.calls == [False],
        f"POST /maintenance/cleanup live: pruner calls were "
        f"{fake_controller.system.pruner.calls!r}",
    )


def test_maintenance_consolidate(client, fake_controller):
    response = client.post(
        "/maintenance/consolidate",
        params={
            "threshold": "0.9",
            "dry_run": "true",
        },
    )

    require_json(
        response,
        {
            "status": "complete",
            "dry_run": True,
            "threshold": 0.9,
            "result": {
                "merged": 2,
                "examined": 10,
            },
        },
        "POST /maintenance/consolidate",
    )

    require(
        fake_controller.system.consolidator.calls == [(0.9, True)],
        f"POST /maintenance/consolidate: consolidator calls were "
        f"{fake_controller.system.consolidator.calls!r}",
    )


def test_maintenance_rebuild_cache(client, fake_controller):
    response = client.post("/maintenance/rebuild_cache")

    require_status(
        response,
        200,
        "POST /maintenance/rebuild_cache",
    )

    body = response.json()

    require(
        body.get("status") == "complete",
        f"POST /maintenance/rebuild_cache: unexpected status: {body!r}",
    )

    require(
        body.get("cache_size") == 7,
        f"POST /maintenance/rebuild_cache: unexpected cache size: {body!r}",
    )

    require(
        body.get("seconds", -1) >= 0,
        f"POST /maintenance/rebuild_cache: invalid seconds: {body!r}",
    )

    require(
        fake_controller.system.embedding_cache.rebuild_calls == 1,
        "POST /maintenance/rebuild_cache: cache.rebuild was not called",
    )


# ---------------------------------------------------------------------------
# Debug API
# ---------------------------------------------------------------------------

def test_debug_stats(client):
    response = client.get("/debug/stats")

    require_json(
        response,
        {
            "db_rows": 3,
            "vector_rows": 3,
            "embedding_cache": 7,
        },
        "GET /debug/stats",
    )


def test_debug_latest_default_limit(client):
    response = client.get("/debug/latest")

    require_status(response, 200, "GET /debug/latest")

    body = response.json()

    require(
        body.get("latest") == fake_latest_expected(),
        f"GET /debug/latest: unexpected body: {body!r}",
    )


def fake_latest_expected():
    return [
        {
            "id": 1,
            "text": "first memory",
            "normalized_text": "first memory",
        },
        {
            "id": 2,
            "text": "second memory",
            "normalized_text": "second memory",
        },
        {
            "id": 3,
            "text": "third memory",
            "normalized_text": "third memory",
        },
    ]


def test_debug_latest_custom_limit(client):
    response = client.get(
        "/debug/latest",
        params={"limit": 2},
    )

    require_status(response, 200, "GET /debug/latest?limit=2")

    body = response.json()

    require(
        len(body.get("latest", [])) == 2,
        f"GET /debug/latest?limit=2: unexpected body: {body!r}",
    )


def test_debug_health(client):
    response = client.get("/debug/health")

    require_json(
        response,
        {
            "status": "healthy",
            "checks": 3,
        },
        "GET /debug/health",
    )


def test_debug_probe(client):
    response = client.get("/debug/probe")

    require_status(response, 200, "GET /debug/probe")

    body = response.json()

    require(
        body.get("db_rows") == 3,
        f"GET /debug/probe: unexpected db_rows: {body!r}",
    )

    require(
        body.get("vector_rows") == 3,
        f"GET /debug/probe: unexpected vector_rows: {body!r}",
    )

    require(
        body.get("sync") is True,
        f"GET /debug/probe: unexpected sync: {body!r}",
    )

    require(
        body.get("health") == {
            "status": "healthy",
            "checks": 3,
        },
        f"GET /debug/probe: unexpected health: {body!r}",
    )


def test_debug_cache(client):
    response = client.get("/debug/cache")

    require_json(
        response,
        {
            "cache_size": 7,
            "cache_path": "/fake/cache",
            "max_size": 100,
        },
        "GET /debug/cache",
    )


def test_debug_graph(client):
    response = client.get("/debug/graph")

    require_json(
        response,
        {
            "nodes": 3,
            "edges": 2,
        },
        "GET /debug/graph",
    )


def test_debug_router(client):
    response = client.get("/debug/router")

    require_json(
        response,
        {
            "types": ["general", "task", "goal"],
            "default": "general",
        },
        "GET /debug/router",
    )


# ---------------------------------------------------------------------------
# Chat API
# ---------------------------------------------------------------------------

def test_chat(client, fake_controller):
    response = client.post(
        "/chat/",
        json={
            "text": "hello",
            "top_n": 5,
        },
    )

    require_json(
        response,
        {
            "input": "hello",
            "response": "chat response",
            "status": "ok",
        },
        "POST /chat/",
    )

    require(
        fake_controller.chat_calls == [("hello", 5)],
        f"POST /chat/: chat calls were "
        f"{fake_controller.chat_calls!r}",
    )


def test_chat_without_top_n(client, fake_controller):
    response = client.post(
        "/chat/",
        json={
            "text": "hello",
        },
    )

    require_status(response, 200, "POST /chat/ without top_n")

    require(
        fake_controller.chat_calls == [("hello", None)],
        f"POST /chat/ without top_n: chat calls were "
        f"{fake_controller.chat_calls!r}",
    )


def test_chat_empty_prompt(client):
    response = client.post(
        "/chat/",
        json={"text": "   "},
    )

    require_json(
        response,
        {
            "input": "   ",
            "response": "Empty prompt provided.",
            "status": "error",
        },
        "POST /chat/ empty",
    )


def test_chat_validation(client):
    response = client.post(
        "/chat/",
        json={},
    )

    require_status(response, 422, "POST /chat/ missing text")


def test_chat_history(client):
    response = client.get("/chat/history")

    require_json(
        response,
        {
            "history": ["previous chat"],
        },
        "GET /chat/history",
    )


def test_chat_raw(client, fake_controller):
    response = client.post(
        "/chat/raw",
        json={"text": "raw prompt"},
    )

    require_json(
        response,
        {
            "input": "raw prompt",
            "response": "raw response",
            "status": "ok",
        },
        "POST /chat/raw",
    )

    require(
        fake_controller.raw_chat_calls == ["raw prompt"],
        f"POST /chat/raw: raw_chat calls were "
        f"{fake_controller.raw_chat_calls!r}",
    )


def test_chat_raw_empty_prompt(client):
    response = client.post(
        "/chat/raw",
        json={"text": ""},
    )

    require_json(
        response,
        {
            "input": "",
            "response": "Empty prompt provided.",
            "status": "error",
        },
        "POST /chat/raw empty",
    )


# ---------------------------------------------------------------------------
# Benchmark API
# ---------------------------------------------------------------------------

def test_benchmark_run_empty(client):
    response = client.post(
        "/benchmark/run",
        json=[],
    )

    require_json(
        response,
        {
            "accuracy": 0,
            "correct": 0,
            "total": 0,
            "failed": 0,
            "runtime": 0.0,
            "sample_failures": [],
        },
        "POST /benchmark/run empty",
    )


def test_benchmark_run(client, fake_controller):
    response = client.post(
        "/benchmark/run",
        json=[
            {
                "query": "alpha",
                "expected": "alpha memory",
            },
            {
                "query": "missing",
                "expected": "not present",
            },
        ],
    )

    require_status(response, 200, "POST /benchmark/run")

    body = response.json()

    require(
        body.get("total") == 2,
        f"POST /benchmark/run: unexpected total: {body!r}",
    )

    require(
        body.get("correct") == 1,
        f"POST /benchmark/run: unexpected correct count: {body!r}",
    )

    require(
        body.get("failed") == 1,
        f"POST /benchmark/run: unexpected failed count: {body!r}",
    )

    require(
        body.get("accuracy") == 50.0,
        f"POST /benchmark/run: unexpected accuracy: {body!r}",
    )

    require(
        body.get("runtime", -1) >= 0,
        f"POST /benchmark/run: invalid runtime: {body!r}",
    )

    require(
        len(body.get("sample_failures", [])) == 1,
        f"POST /benchmark/run: unexpected failures: {body!r}",
    )

    require(
        fake_controller.recall_calls == ["alpha", "missing"],
        f"POST /benchmark/run: recall calls were "
        f"{fake_controller.recall_calls!r}",
    )


def test_benchmark_speed_empty(client):
    response = client.post(
        "/benchmark/speed",
        json={"queries": []},
    )

    require_json(
        response,
        {
            "queries": 0,
            "seconds": 0.0,
            "qps": 0.0,
        },
        "POST /benchmark/speed empty",
    )


def test_benchmark_speed(client, fake_controller):
    response = client.post(
        "/benchmark/speed",
        json={
            "queries": [
                "one",
                "two",
            ]
        },
    )

    require_status(response, 200, "POST /benchmark/speed")

    body = response.json()

    require(
        body.get("queries") == 2,
        f"POST /benchmark/speed: unexpected query count: {body!r}",
    )

    require(
        body.get("seconds", -1) >= 0,
        f"POST /benchmark/speed: invalid seconds: {body!r}",
    )

    require(
        body.get("qps", -1) >= 0,
        f"POST /benchmark/speed: invalid qps: {body!r}",
    )

    require(
        fake_controller.recall_calls == ["one", "two"],
        f"POST /benchmark/speed: recall calls were "
        f"{fake_controller.recall_calls!r}",
    )


def test_benchmark_store_speed_empty(client):
    response = client.post(
        "/benchmark/store_speed",
        json={"texts": []},
    )

    require_json(
        response,
        {
            "stored": 0,
            "seconds": 0.0,
            "stores_per_second": 0.0,
            "db_rows": 0,
            "vector_rows": 0,
            "synced": True,
        },
        "POST /benchmark/store_speed empty",
    )


def test_benchmark_store_speed(client, fake_controller):
    response = client.post(
        "/benchmark/store_speed",
        json={
            "texts": [
                "one",
                "two",
            ]
        },
    )

    require_status(
        response,
        200,
        "POST /benchmark/store_speed",
    )

    body = response.json()

    require(
        body.get("stored") == 2,
        f"POST /benchmark/store_speed: unexpected stored count: {body!r}",
    )

    require(
        body.get("seconds", -1) >= 0,
        f"POST /benchmark/store_speed: invalid seconds: {body!r}",
    )

    require(
        body.get("stores_per_second", -1) >= 0,
        f"POST /benchmark/store_speed: invalid stores_per_second: {body!r}",
    )

    require(
        body.get("db_rows") == 3,
        f"POST /benchmark/store_speed: unexpected db_rows: {body!r}",
    )

    require(
        body.get("vector_rows") == 3,
        f"POST /benchmark/store_speed: unexpected vector_rows: {body!r}",
    )

    require(
        body.get("synced") is True,
        f"POST /benchmark/store_speed: unexpected sync state: {body!r}",
    )

    require(
        fake_controller.remember_many_calls == [["one", "two"]],
        f"POST /benchmark/store_speed: remember_many calls were "
        f"{fake_controller.remember_many_calls!r}",
    )

    require(
        fake_controller.system.vector_store.save_calls == 1,
        "POST /benchmark/store_speed: vector store was not saved",
    )


def test_benchmark_full_empty_body(client):
    response = client.post(
        "/benchmark/full",
        json={},
    )

    require_status(
        response,
        200,
        "POST /benchmark/full empty",
    )

    require(
        response.json() == {},
        f"POST /benchmark/full empty: unexpected body "
        f"{response.json()!r}",
    )


def test_benchmark_quick(client, fake_controller):
    response = client.post("/benchmark/quick")

    require_status(response, 200, "POST /benchmark/quick")

    body = response.json()

    require(
        body.get("queries") == 3,
        f"POST /benchmark/quick: expected three queries: {body!r}",
    )

    require(
        body.get("seconds", -1) >= 0,
        f"POST /benchmark/quick: invalid seconds: {body!r}",
    )

    require(
        body.get("qps", -1) >= 0,
        f"POST /benchmark/quick: invalid qps: {body!r}",
    )

    require(
        body.get("note") == "Quick benchmark with 3 sample queries",
        f"POST /benchmark/quick: unexpected note: {body!r}",
    )

    require(
        fake_controller.recall_calls == [
            "what is the meaning of life",
            "who is the president",
            "what is 2+2",
        ],
        f"POST /benchmark/quick: recall calls were "
        f"{fake_controller.recall_calls!r}",
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("/memory/store", {"text": "boom"}),
        ("/memory/query", {"text": "boom"}),
        ("/memory/batch_store", {"texts": ["boom"]}),
        ("/memory/test_store", {"text": "boom"}),
        ("/memory/reflect", None),
        ("/chat/", {"text": "boom"}),
        ("/chat/raw", {"text": "boom"}),
    ],
)
def test_memory_and_chat_500_translation(
    client,
    fake_controller,
    monkeypatch,
    endpoint,
    payload,
):
    def explode(*args, **kwargs):
        raise RuntimeError("forced API failure")

    if endpoint == "/memory/store":
        monkeypatch.setattr(fake_controller, "remember", explode)
    elif endpoint == "/memory/query":
        monkeypatch.setattr(fake_controller, "recall", explode)
    elif endpoint == "/memory/batch_store":
        monkeypatch.setattr(fake_controller, "remember_many", explode)
    elif endpoint == "/memory/test_store":
        monkeypatch.setattr(fake_controller, "remember", explode)
    elif endpoint == "/memory/reflect":
        monkeypatch.setattr(fake_controller, "reflect", explode)
    elif endpoint == "/chat/":
        monkeypatch.setattr(fake_controller, "chat", explode)
    elif endpoint == "/chat/raw":
        monkeypatch.setattr(fake_controller, "raw_chat", explode)

    if payload is None:
        response = client.post(endpoint)
    else:
        response = client.post(endpoint, json=payload)

    require_status(
        response,
        500,
        f"POST {endpoint} forced failure",
    )

    body = response.json()

    require(
        body.get("detail") == "forced API failure",
        f"POST {endpoint} forced failure: unexpected body {body!r}",
    )


def test_memory_stats_500_translation(
    client,
    fake_controller,
    monkeypatch,
):
    def explode(*args, **kwargs):
        raise RuntimeError("forced API failure")

    monkeypatch.setattr(
        fake_controller,
        "stats",
        explode,
    )

    response = client.get("/memory/stats")

    require_status(
        response,
        500,
        "GET /memory/stats forced failure",
    )

    body = response.json()

    require(
        body.get("detail") == "forced API failure",
        f"GET /memory/stats forced failure: unexpected body {body!r}",
    )
# ---------------------------------------------------------------------------
# HTTP method / schema validation sanity
# ---------------------------------------------------------------------------

def test_unknown_endpoint_returns_404(client):
    response = client.get("/definitely-not-a-real-endpoint")

    require_status(
        response,
        404,
        "GET unknown endpoint",
    )


def test_memory_store_rejects_wrong_field_type(client):
    response = client.post(
        "/memory/store",
        json={"text": 123},
    )

    require_status(
        response,
        422,
        "POST /memory/store wrong text type",
    )


def test_batch_store_rejects_wrong_field_type(client):
    response = client.post(
        "/memory/batch_store",
        json={"texts": "not-a-list"},
    )

    require_status(
        response,
        422,
        "POST /memory/batch_store wrong texts type",
    )


def test_chat_rejects_wrong_top_n_type(client):
    response = client.post(
        "/chat/",
        json={
            "text": "hello",
            "top_n": "not-an-int",
        },
    )

    require_status(
        response,
        422,
        "POST /chat/ wrong top_n type",
    )


def test_cleanup_rejects_invalid_boolean(client):
    response = client.post(
        "/maintenance/cleanup",
        params={"dry_run": "not-a-bool"},
    )

    require_status(
        response,
        422,
        "POST /maintenance/cleanup invalid dry_run",
    )


def test_consolidate_rejects_invalid_threshold(client):
    response = client.post(
        "/maintenance/consolidate",
        params={"threshold": "not-a-number"},
    )

    require_status(
        response,
        422,
        "POST /maintenance/consolidate invalid threshold",
    )


def test_latest_rejects_invalid_limit(client):
    response = client.get(
        "/debug/latest",
        params={"limit": "not-an-int"},
    )

    require_status(
        response,
        422,
        "GET /debug/latest invalid limit",
    )
