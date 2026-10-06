import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import gui


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def require(condition, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_status(response, expected: int, context: str) -> None:
    require(
        response.status_code == expected,
        f"{context}: expected HTTP {expected}, "
        f"got {response.status_code}; body={response.text}",
    )


def require_json(response, context: str):
    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError(
            f"{context}: response was not valid JSON: {response.text}"
        ) from exc


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeDB:
    def __init__(self):
        self.count_value = 12

    def count(self):
        return self.count_value


class FakeSystem:
    def __init__(self):
        self.db = FakeDB()


class FakeController:
    def __init__(self):
        self.system = FakeSystem()


class FakeMemoryInterface:
    def __init__(self):
        self.controller = FakeController()

        self.recall_calls = []
        self.remember_calls = []
        self.remember_many_calls = []
        self.chat_calls = []
        self.ingest_code_calls = []
        self.ingest_pdf_calls = []
        self.set_goal_calls = []
        self.list_goals_calls = []

        self.goals = [
            {"id": "goal-1", "goal": "Test goal", "progress": "started"},
        ]

    def recall(self, query):
        self.recall_calls.append(query)
        return {
            "results": [
                {"id": "memory-1", "text": query},
            ],
            "count": 1,
        }

    def remember(self, text):
        self.remember_calls.append(text)
        return "memory-1"

    def remember_many(self, texts):
        self.remember_many_calls.append(texts)
        return [f"memory-{index}" for index in range(1, len(texts) + 1)]

    def chat(self, prompt, auto_store=None):
        self.chat_calls.append(
            {
                "prompt": prompt,
                "auto_store": auto_store,
            }
        )
        return f"response to: {prompt}"

    def ingest_code(self, directory, max_files=1000):
        self.ingest_code_calls.append(
            {
                "directory": directory,
                "max_files": max_files,
            }
        )
        return {
            "files_processed": 3,
            "symbols_ingested": 7,
        }

    def ingest_pdf(self, filepath, max_pages=100):
        self.ingest_pdf_calls.append(
            {
                "filepath": filepath,
                "max_pages": max_pages,
            }
        )
        return max_pages

    def set_goal(self, goal, progress):
        self.set_goal_calls.append(
            {
                "goal": goal,
                "progress": progress,
            }
        )
        return "goal-1"

    def list_goals(self, status=None):
        self.list_goals_calls.append(status)

        if status is None:
            return list(self.goals)

        return [
            goal
            for goal in self.goals
            if goal.get("progress") == status
        ]


class FakeSignalRegistry:
    def __init__(self):
        self.enabled = {
            "recency": True,
            "semantic": True,
            "keyword": False,
        }

    def is_enabled(self, name):
        return self.enabled.get(name, False)

    def enable(self, name):
        self.enabled[name] = True

    def disable(self, name):
        self.enabled[name] = False

    def get_cost(self, name):
        return 1.0

    def is_enabled_for_type(self, name, memory_type):
        return self.enabled.get(name, False)

    def get_description(self, name):
        return f"description for {name}"

    def get_category(self, name):
        return "test"


class FakeSignalRouter:
    def __init__(self, registry):
        self.registry = registry
        self.clear_cache_calls = 0

    def get_active_signals(self, memory_type):
        return {
            name: 1.0
            for name, enabled in self.registry.enabled.items()
            if enabled
        }

    def clear_cache(self):
        self.clear_cache_calls += 1


class FakeHistory:
    
    def __init__(self):
        self.entries = [
            {
                "query": "first query",
                "timestamp": 1000.0,
                "results": [],
            },
            {
                "query": "second query",
                "timestamp": 1001.0,
                "results": [],
            },
        ]

    def get_recent(self, n=10):
        return self.entries[-n:]

    def get_frequent(self, n=10):
        return ["second query", "first query"][:n]

    def get_context(self):
        return [entry["query"] for entry in self.entries[-5:]]

    def get_previous_query(self):
        if len(self.entries) >= 2:
            return self.entries[-2]["query"]
        return None

    
class FakeSettings:
    AUTO_STORE_MEMORIES = True
    AUTO_STORE_THRESHOLD = 0.75
    AUTO_STORE_MAX_PER_SESSION = 10
    AUTO_STORE_TYPES = ["general"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_memory():
    memory = FakeMemoryInterface()
    gui.memory = memory
    return memory


@pytest.fixture
def fake_history(monkeypatch):
    history = FakeHistory()

    monkeypatch.setattr(gui, "HAS_QUERY_HISTORY", True)
    monkeypatch.setattr(gui, "query_history", history)

    return history


@pytest.fixture
def fake_signals(monkeypatch):
    registry = FakeSignalRegistry()
    router = FakeSignalRouter(registry)

    monkeypatch.setattr(gui, "HAS_SIGNAL_REGISTRY", True)
    monkeypatch.setattr(gui, "get_registry", lambda: registry)
    monkeypatch.setattr(
        gui,
        "SignalRouter",
        lambda registry: router,
    )

    return registry, router


@pytest.fixture
def fake_settings(monkeypatch):
    monkeypatch.setattr(gui, "settings", FakeSettings(), raising=False)
    return gui.settings


@pytest.fixture
def client(fake_memory):
    with TestClient(gui.app) as test_client:
        gui.memory = fake_memory
        yield test_client

# ---------------------------------------------------------------------------
# Application / startup
# ---------------------------------------------------------------------------

def test_gui_startup_initializes_memory(monkeypatch):
    created = []

    class StartupMemory:
        def __init__(self):
            created.append(self)

    monkeypatch.setattr(gui, "MemoryInterface", StartupMemory)

    with TestClient(gui.app):
        require(
            gui.memory is created[0],
            "GUI startup did not initialize MemoryInterface",
        )

def test_gui_startup_initializes_resources(monkeypatch):
    created_memory = []
    created_history = []

    class StartupMemory:
        def __init__(self):
            created_memory.append(self)

    class StartupHistory:
        def __init__(self):
            created_history.append(self)

    monkeypatch.setattr(gui, "MemoryInterface", StartupMemory)
    monkeypatch.setattr(gui, "QueryHistory", StartupHistory)
    monkeypatch.setattr(gui, "HAS_QUERY_HISTORY", True)

    with TestClient(gui.app):
        require(
            gui.memory is created_memory[0],
            "GUI startup did not initialize MemoryInterface",
        )
        require(
            gui.query_history is created_history[0],
            "GUI startup did not initialize QueryHistory",
        )

        
def test_gui_root(client, monkeypatch, tmp_path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()

    html_path = template_dir / "index.html"
    html_path.write_text(
        "<html><body>Memoria GUI</body></html>",
        encoding="utf-8",
    )

    monkeypatch.setattr(gui, "TEMPLATES_DIR", template_dir)

    response = client.get("/")

    require_status(response, 200, "GET /")
    require(
        "Memoria GUI" in response.text,
        "GET / did not return GUI HTML",
    )


def test_gui_root_missing_template(client, monkeypatch, tmp_path):
    monkeypatch.setattr(gui, "TEMPLATES_DIR", tmp_path)

    response = client.get("/")

    require_status(response, 200, "GET / missing template")
    require(
        "GUI template not found" in response.text,
        "GET / missing template did not return fallback HTML",
    )


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def test_gui_query(client, fake_memory):
    response = client.post(
        "/query",
        json={"query": "hello memory"},
    )

    require_status(response, 200, "POST /query")
    body = require_json(response, "POST /query")

    require(
        body["count"] == 1,
        f"POST /query unexpected result: {body!r}",
    )
    require(
        fake_memory.recall_calls == ["hello memory"],
        f"POST /query did not call recall correctly: "
        f"{fake_memory.recall_calls!r}",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": None},
    ],
)
def test_gui_query_validation(client, payload):
    response = client.post("/query", json=payload)

    require_status(response, 400, "POST /query validation")


def test_gui_store(client, fake_memory):
    response = client.post(
        "/store",
        json={"text": "store this"},
    )

    require_status(response, 200, "POST /store")
    body = require_json(response, "POST /store")

    require(
        body["id"] == "memory-1",
        f"POST /store unexpected body: {body!r}",
    )
    require(
        fake_memory.remember_calls == ["store this"],
        f"POST /store did not call remember: "
        f"{fake_memory.remember_calls!r}",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": None},
    ],
)
def test_gui_store_validation(client, payload):
    response = client.post("/store", json=payload)

    require_status(response, 400, "POST /store validation")


def test_gui_store_many(client, fake_memory):
    texts = ["one", "two", "three"]

    response = client.post(
        "/store_many",
        json={"texts": texts},
    )

    require_status(response, 200, "POST /store_many")
    body = require_json(response, "POST /store_many")

    require(
        body["ids"] == ["memory-1", "memory-2", "memory-3"],
        f"POST /store_many unexpected body: {body!r}",
    )
    require(
        fake_memory.remember_many_calls == [texts],
        f"POST /store_many did not call remember_many: "
        f"{fake_memory.remember_many_calls!r}",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"texts": []},
        {"texts": None},
        {"texts": "not a list"},
    ],
)
def test_gui_store_many_validation(client, payload):
    response = client.post("/store_many", json=payload)

    require_status(response, 400, "POST /store_many validation")


def test_gui_chat(client, fake_memory):
    response = client.post(
        "/chat",
        json={
            "prompt": "hello",
            "auto_store": True,
        },
    )

    require_status(response, 200, "POST /chat")
    body = require_json(response, "POST /chat")

    require(
        body["response"] == "response to: hello",
        f"POST /chat unexpected body: {body!r}",
    )
    require(
        fake_memory.chat_calls[-1]["auto_store"] is True,
        f"POST /chat did not pass auto_store: "
        f"{fake_memory.chat_calls!r}",
    )


def test_gui_chat_without_auto_store(client, fake_memory):
    response = client.post(
        "/chat",
        json={"prompt": "hello"},
    )

    require_status(response, 200, "POST /chat without auto_store")
    require(
        fake_memory.chat_calls[-1]["auto_store"] is None,
        "POST /chat without auto_store did not pass None",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompt": ""},
        {"prompt": None},
    ],
)
def test_gui_chat_validation(client, payload):
    response = client.post("/chat", json=payload)

    require_status(response, 400, "POST /chat validation")


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def test_gui_ingest_code(client, fake_memory, tmp_path):
    response = client.post(
        "/ingest_code",
        json={
            "directory": str(tmp_path),
            "max_files": 25,
        },
    )

    require_status(response, 200, "POST /ingest_code")
    body = require_json(response, "POST /ingest_code")

    require(
        body["files_processed"] == 3,
        f"POST /ingest_code unexpected body: {body!r}",
    )
    require(
        fake_memory.ingest_code_calls[-1]["max_files"] == 25,
        "POST /ingest_code did not pass max_files",
    )


def test_gui_ingest_code_missing_directory(client):
    response = client.post(
        "/ingest_code",
        json={},
    )

    require_status(response, 400, "POST /ingest_code missing directory")


def test_gui_ingest_code_nonexistent_directory(client, tmp_path):
    missing = tmp_path / "does-not-exist"

    response = client.post(
        "/ingest_code",
        json={"directory": str(missing)},
    )

    require_status(
        response,
        400,
        "POST /ingest_code nonexistent directory",
    )


def test_gui_ingest_pdf(client, fake_memory, tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-test")

    response = client.post(
        "/ingest_pdf",
        json={
            "filepath": str(pdf),
            "max_pages": 7,
        },
    )

    require_status(response, 200, "POST /ingest_pdf")
    body = require_json(response, "POST /ingest_pdf")

    require(
        "7 pages" in body["message"],
        f"POST /ingest_pdf unexpected body: {body!r}",
    )
    require(
        fake_memory.ingest_pdf_calls[-1]["max_pages"] == 7,
        "POST /ingest_pdf did not pass max_pages",
    )


def test_gui_ingest_pdf_missing_filepath(client):
    response = client.post("/ingest_pdf", json={})

    require_status(response, 400, "POST /ingest_pdf missing filepath")


def test_gui_ingest_pdf_nonexistent_file(client, tmp_path):
    missing = tmp_path / "missing.pdf"

    response = client.post(
        "/ingest_pdf",
        json={"filepath": str(missing)},
    )

    require_status(
        response,
        400,
        "POST /ingest_pdf nonexistent file",
    )


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------

def test_gui_set_goal(client, fake_memory):
    response = client.post(
        "/set_goal",
        json={
            "goal": "Finish Memoria",
            "progress": "started",
        },
    )

    require_status(response, 200, "POST /set_goal")
    body = require_json(response, "POST /set_goal")

    require(
        body["id"] == "goal-1",
        f"POST /set_goal unexpected body: {body!r}",
    )


def test_gui_set_goal_default_progress(client, fake_memory):
    response = client.post(
        "/set_goal",
        json={"goal": "Finish testing"},
    )

    require_status(response, 200, "POST /set_goal default progress")
    require(
        fake_memory.set_goal_calls[-1]["progress"] == "started",
        "POST /set_goal did not use default progress",
    )


def test_gui_set_goal_missing_goal(client):
    response = client.post("/set_goal", json={})

    require_status(response, 400, "POST /set_goal missing goal")


def test_gui_list_goals(client, fake_memory):
    response = client.get("/list_goals")

    require_status(response, 200, "GET /list_goals")
    body = require_json(response, "GET /list_goals")

    require(
        len(body["goals"]) == 1,
        f"GET /list_goals unexpected body: {body!r}",
    )


def test_gui_list_goals_status_filter(client, fake_memory):
    response = client.get(
        "/list_goals",
        params={"status": "started"},
    )

    require_status(response, 200, "GET /list_goals status")
    require(
        fake_memory.list_goals_calls[-1] == "started",
        "GET /list_goals did not pass status",
    )


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def test_gui_signals_unavailable(client, monkeypatch):
    monkeypatch.setattr(gui, "HAS_SIGNAL_REGISTRY", False)

    response = client.get("/signals")

    require_status(response, 501, "GET /signals unavailable")


def test_gui_signals(client, fake_signals):
    response = client.get(
        "/signals",
        params={"memory_type": "general"},
    )

    require_status(response, 200, "GET /signals")
    body = require_json(response, "GET /signals")

    require(
        body["memory_type"] == "general",
        f"GET /signals unexpected memory_type: {body!r}",
    )
    require(
        body["total_weight"] == 2.0,
        f"GET /signals unexpected weight: {body!r}",
    )


def test_gui_signals_custom_memory_type(client, fake_signals):
    response = client.get(
        "/signals",
        params={"memory_type": "code"},
    )

    require_status(response, 200, "GET /signals custom type")
    body = require_json(response, "GET /signals custom type")

    require(
        body["memory_type"] == "code",
        "GET /signals did not preserve memory_type",
    )


def test_gui_toggle_signal_enable(client, fake_signals):
    registry, router = fake_signals
    registry.enabled["semantic"] = False

    response = client.post(
        "/signals/toggle",
        json={
            "name": "semantic",
            "enable": True,
        },
    )

    require_status(response, 200, "POST /signals/toggle enable")
    body = require_json(response, "POST /signals/toggle enable")

    require(
        body["enabled"] is True,
        f"Signal enable returned unexpected body: {body!r}",
    )
    require(
        registry.is_enabled("semantic") is True,
        "Signal was not enabled",
    )
    require(
        router.clear_cache_calls == 1,
        "Signal router cache was not cleared",
    )


def test_gui_toggle_signal_disable(client, fake_signals):
    registry, router = fake_signals

    response = client.post(
        "/signals/toggle",
        json={
            "name": "semantic",
            "enable": False,
        },
    )

    require_status(response, 200, "POST /signals/toggle disable")
    require(
        registry.is_enabled("semantic") is False,
        "Signal was not disabled",
    )


def test_gui_toggle_signal_implicit_toggle(client, fake_signals):
    registry, _ = fake_signals

    registry.enabled["semantic"] = True

    response = client.post(
        "/signals/toggle",
        json={"name": "semantic"},
    )

    require_status(response, 200, "POST /signals/toggle implicit")
    body = require_json(response, "POST /signals/toggle implicit")

    require(
        body["enabled"] is False,
        f"Implicit signal toggle unexpected body: {body!r}",
    )


def test_gui_toggle_signal_missing_name(client, fake_signals):
    response = client.post(
        "/signals/toggle",
        json={},
    )

    require_status(response, 400, "POST /signals/toggle missing name")


def test_gui_toggle_signal_unavailable(client, monkeypatch):
    monkeypatch.setattr(gui, "HAS_SIGNAL_REGISTRY", False)

    response = client.post(
        "/signals/toggle",
        json={"name": "semantic"},
    )

    require_status(response, 501, "POST /signals/toggle unavailable")


# ---------------------------------------------------------------------------
# Query history
# ---------------------------------------------------------------------------

def test_gui_history_unavailable(client, monkeypatch):
    monkeypatch.setattr(gui, "HAS_QUERY_HISTORY", False)

    response = client.get("/history")

    require_status(response, 501, "GET /history unavailable")


def test_gui_history_recent(client, fake_history):
    response = client.get(
        "/history",
        params={
            "mode": "recent",
            "limit": 1,
        },
    )

    require_status(response, 200, "GET /history recent")
    body = require_json(response, "GET /history recent")

    require(
        body["mode"] == "recent",
        f"GET /history recent unexpected mode: {body!r}",
    )
    require(
        body["count"] == 1,
        f"GET /history recent unexpected count: {body!r}",
    )
    require(
        body["entries"][0]["query"] == "second query",
        f"GET /history recent unexpected entries: {body!r}",
    )


def test_gui_history_frequent(client, fake_history):
    response = client.get(
        "/history",
        params={
            "mode": "frequent",
            "limit": 2,
        },
    )

    require_status(response, 200, "GET /history frequent")
    body = require_json(response, "GET /history frequent")

    require(
        body["mode"] == "frequent",
        f"GET /history frequent unexpected mode: {body!r}",
    )
    require(
        body["queries"] == ["second query", "first query"],
        f"GET /history frequent unexpected queries: {body!r}",
    )


def test_gui_history_context(client, fake_history):
    response = client.get(
        "/history",
        params={"mode": "context"},
    )

    require_status(response, 200, "GET /history context")
    body = require_json(response, "GET /history context")

    require(
        body["mode"] == "context",
        f"GET /history context unexpected mode: {body!r}",
    )
    require(
        body["queries"] == ["first query", "second query"],
        f"GET /history context unexpected queries: {body!r}",
    )


def test_gui_history_previous(client, fake_history):
    response = client.get(
        "/history",
        params={"mode": "previous"},
    )

    require_status(response, 200, "GET /history previous")
    body = require_json(response, "GET /history previous")

    require(
        body["mode"] == "previous",
        f"GET /history previous unexpected mode: {body!r}",
    )
    require(
        body["query"] == "first query",
        f"GET /history previous unexpected query: {body!r}",
    )

def test_gui_history_invalid_mode(client, fake_history):
    response = client.get(
        "/history",
        params={"mode": "search"},
    )

    require_status(response, 400, "GET /history invalid mode")
    body = require_json(response, "GET /history invalid mode")

    require(
        "Unknown history mode" in body["error"],
        f"GET /history invalid mode unexpected body: {body!r}",
    )


def test_gui_history_invalid_limit(client, fake_history):
    response = client.get(
        "/history",
        params={
            "mode": "recent",
            "limit": 0,
        },
    )

    require_status(response, 400, "GET /history invalid limit")
    body = require_json(response, "GET /history invalid limit")

    require(
        body["error"] == "Limit must be greater than 0",
        f"GET /history invalid limit unexpected body: {body!r}",
    )

    
# ---------------------------------------------------------------------------
# Auto-store settings
# ---------------------------------------------------------------------------

def test_gui_auto_store_get(client, monkeypatch):
    settings = FakeSettings()
    monkeypatch.setattr(
        "cache.config.settings",
        settings,
    )

    response = client.get("/settings/auto-store")

    require_status(response, 200, "GET /settings/auto-store")
    body = require_json(response, "GET /settings/auto-store")

    require(
        body["auto_store"] is True,
        f"Unexpected auto-store setting: {body!r}",
    )
    require(
        body["threshold"] == 0.75,
        f"Unexpected threshold: {body!r}",
    )


def test_gui_auto_store_update(client, monkeypatch):
    settings = FakeSettings()
    monkeypatch.setattr(
        "cache.config.settings",
        settings,
    )

    response = client.post(
        "/settings/auto-store",
        json={
            "auto_store": False,
            "threshold": 0.9,
            "max_per_session": 25,
            "types": ["general", "code"],
        },
    )

    require_status(response, 200, "POST /settings/auto-store")
    body = require_json(response, "POST /settings/auto-store")

    require(
        settings.AUTO_STORE_MEMORIES is False,
        "Auto-store boolean was not updated",
    )
    require(
        settings.AUTO_STORE_THRESHOLD == 0.9,
        "Auto-store threshold was not updated",
    )
    require(
        settings.AUTO_STORE_MAX_PER_SESSION == 25,
        "Auto-store max/session was not updated",
    )
    require(
        settings.AUTO_STORE_TYPES == ["general", "code"],
        "Auto-store types were not updated",
    )
    require(
        set(body["updated"]) == {
            "auto_store",
            "threshold",
            "max_per_session",
            "types",
        },
        f"Unexpected updated fields: {body!r}",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"threshold": -0.1},
        {"threshold": 1.1},
        {"max_per_session": 0},
        {"max_per_session": -1},
        {"types": "not-a-list"},
    ],
)
def test_gui_auto_store_update_validation(
    client,
    monkeypatch,
    payload,
):
    settings = FakeSettings()

    monkeypatch.setattr(
        "cache.config.settings",
        settings,
    )

    response = client.post(
        "/settings/auto-store",
        json=payload,
    )

    require_status(
        response,
        400,
        f"POST /settings/auto-store invalid payload {payload!r}",
    )


def test_gui_auto_store_toggle_explicit(client, monkeypatch):
    settings = FakeSettings()

    monkeypatch.setattr(
        "cache.config.settings",
        settings,
    )

    response = client.post(
        "/settings/auto-store/toggle",
        json={"enable": False},
    )

    require_status(
        response,
        200,
        "POST /settings/auto-store/toggle explicit",
    )

    require(
        settings.AUTO_STORE_MEMORIES is False,
        "Explicit auto-store disable failed",
    )


def test_gui_auto_store_toggle_implicit(client, monkeypatch):
    settings = FakeSettings()
    settings.AUTO_STORE_MEMORIES = True

    monkeypatch.setattr(
        "cache.config.settings",
        settings,
    )

    response = client.post(
        "/settings/auto-store/toggle",
        json={},
    )

    require_status(
        response,
        200,
        "POST /settings/auto-store/toggle implicit",
    )

    require(
        settings.AUTO_STORE_MEMORIES is False,
        "Implicit auto-store toggle failed",
    )


# ---------------------------------------------------------------------------
# Health / stats
# ---------------------------------------------------------------------------

def test_gui_health(client, fake_memory, monkeypatch):
    fake_memory.controller.system.db.count_value = 42

    monkeypatch.setattr(gui, "HAS_SIGNAL_REGISTRY", True)
    monkeypatch.setattr(gui, "HAS_QUERY_HISTORY", True)

    response = client.get("/health")

    require_status(response, 200, "GET /health")
    body = require_json(response, "GET /health")

    require(
        body["status"] == "ok",
        f"Unexpected GUI health status: {body!r}",
    )
    require(
        body["memory_count"] == 42,
        f"Unexpected GUI memory count: {body!r}",
    )


def test_gui_health_degraded(client, fake_memory):
    def explode():
        raise RuntimeError("forced health failure")

    fake_memory.controller.system.db.count = explode

    response = client.get("/health")

    require_status(response, 200, "GET /health degraded")

    body = require_json(response, "GET /health degraded")

    require(
        body["status"] == "degraded",
        f"Expected degraded health: {body!r}",
    )
    require(
        body["error"] == "forced health failure",
        f"Unexpected degraded health error: {body!r}",
    )


def test_gui_stats(client, fake_memory, monkeypatch):
    fake_memory.controller.system.db.count_value = 33

    settings = FakeSettings()

    monkeypatch.setattr(
        "cache.config.settings",
        settings,
    )

    response = client.get("/stats")

    require_status(response, 200, "GET /stats")
    body = require_json(response, "GET /stats")

    require(
        body["memory_count"] == 33,
        f"Unexpected GUI stats memory count: {body!r}",
    )
    require(
        body["goals"] == 1,
        f"Unexpected GUI stats goals: {body!r}",
    )
    
    require(
        body["auto_store"]["enabled"] is True,
        f"Unexpected auto-store stats: {body!r}",
    )


# ---------------------------------------------------------------------------
# Forced endpoint failures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "endpoint,payload,method_name",
    [
        ("/query", {"query": "boom"}, "recall"),
        ("/store", {"text": "boom"}, "remember"),
        ("/store_many", {"texts": ["boom"]}, "remember_many"),
        ("/chat", {"prompt": "boom"}, "chat"),
        ("/ingest_code", None, "ingest_code"),
        ("/ingest_pdf", None, "ingest_pdf"),
        ("/set_goal", {"goal": "boom"}, "set_goal"),
        ("/list_goals", None, "list_goals"),
    ],
)
def test_gui_core_500_translation(
    client,
    fake_memory,
    monkeypatch,
    tmp_path,
    endpoint,
    payload,
    method_name,
):
    def explode(*args, **kwargs):
        raise RuntimeError("forced GUI failure")

    monkeypatch.setattr(fake_memory, method_name, explode)

    if endpoint == "/ingest_code":
        payload = {"directory": str(tmp_path)}

    if endpoint == "/ingest_pdf":
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-test")
        payload = {"filepath": str(pdf)}

    if payload is None:
        response = client.get(endpoint)
    elif endpoint == "/list_goals":
        response = client.get(endpoint)
    else:
        response = client.post(endpoint, json=payload)

    require_status(
        response,
        500,
        f"{endpoint} forced failure",
    )

    body = require_json(response, f"{endpoint} forced failure")

    require(
        body.get("error") == "forced GUI failure",
        f"{endpoint} unexpected error body: {body!r}",
    )


def test_gui_stats_500_translation(client, fake_memory):
    def explode():
        raise RuntimeError("forced GUI stats failure")

    fake_memory.controller.system.db.count = explode

    response = client.get("/stats")

    require_status(response, 500, "GET /stats forced failure")

    body = require_json(response, "GET /stats forced failure")

    require(
        body.get("error") == "forced GUI stats failure",
        f"GET /stats unexpected error body: {body!r}",
    )


# ---------------------------------------------------------------------------
# Route inventory
# ---------------------------------------------------------------------------

def test_gui_expected_routes():
    expected = {
        ("GET", "/"),
        ("POST", "/query"),
        ("POST", "/store"),
        ("POST", "/store_many"),
        ("POST", "/chat"),
        ("POST", "/ingest_code"),
        ("POST", "/ingest_pdf"),
        ("POST", "/set_goal"),
        ("GET", "/list_goals"),
        ("GET", "/signals"),
        ("POST", "/signals/toggle"),
        ("GET", "/history"),
        ("GET", "/settings/auto-store"),
        ("POST", "/settings/auto-store"),
        ("POST", "/settings/auto-store/toggle"),
        ("GET", "/health"),
        ("GET", "/stats"),
    }

    actual = {
        (route.methods.copy().pop(), route.path)
        for route in gui.app.routes
        if route.path != "/openapi.json"
        and route.path != "/docs"
        and route.path != "/docs/oauth2-redirect"
        and route.path != "/redoc"
        and hasattr(route, "methods")
        and route.methods
    }

    require(
        expected == actual,
        f"GUI route inventory mismatch:\n"
        f"Expected: {sorted(expected)!r}\n"
        f"Actual: {sorted(actual)!r}",
    )


def test_gui_unknown_endpoint(client):
    response = client.get("/definitely-not-a-real-gui-endpoint")

    require_status(
        response,
        404,
        "GUI unknown endpoint",
    )
