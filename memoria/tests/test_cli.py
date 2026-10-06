"""
CLI regression tests for Memoria.

These tests exercise command parsing, dispatch, formatting, and CLI-specific
behavior without rebuilding the full memory stack for every test.

No assertions are used. Failures raise RuntimeError with diagnostic messages.
"""

import json
import sys
from types import SimpleNamespace

import pytest

import cli


# ---------------------------------------------------------------------------
# Fake backend
# ---------------------------------------------------------------------------

class FakeDB:
    def __init__(self):
        self.memories = [
            {"id": 1, "text": "first memory"},
            {"id": 2, "text": "second memory"},
        ]

    def count(self):
        return len(self.memories)

    def fetch_all(self):
        return list(self.memories)

    def integrity_check(self):
        return "OK"

    def sanity_check(self):
        return {
            "db_count": len(self.memories),
            "columns": ["id", "text"],
        }


class FakeGraph:
    def __init__(self, found=True):
        self.found = found
        self.last_depth = None

    def find_entity(self, entity):
        return {"name": entity} if self.found else None

    def neighbors(self, entity, depth=1):
        self.last_depth = depth
        return [
            {
                "relation": "related_to",
                "target": "other",
                "source": "test",
            }
        ]


class FakeSystem:
    def __init__(self, graph_found=True):
        self.db = FakeDB()
        self.graph_search = FakeGraph(graph_found)


class FakeController:
    def __init__(self, graph_found=True):
        self.system = FakeSystem(graph_found)


class FakeMemoryInterface:
    last_instance = None

    def __init__(self):
        self.controller = FakeController()
        self.calls = []
        FakeMemoryInterface.last_instance = self

    def remember(self, text):
        self.calls.append(("remember", text))
        return "mem-1"

    def remember_many(self, texts):
        self.calls.append(("remember_many", texts))
        return [f"mem-{i}" for i in range(len(texts))]

    def recall(self, query):
        self.calls.append(("recall", query))
        return {
            "results": [
                {
                    "id": "1",
                    "rank": 1,
                    "final_score": 0.95,
                    "text": "result",
                },
                {
                    "id": "2",
                    "rank": 2,
                    "final_score": 0.85,
                    "text": "second result",
                },
            ]
        }

    def set_goal(self, goal, progress):
        self.calls.append(("set_goal", goal, progress))
        return 7

    def update_goal(self, goal_id, progress=None, status=None):
        self.calls.append(("update_goal", goal_id, progress, status))

    def list_goals(self, status=None):
        self.calls.append(("list_goals", status))
        return [
            SimpleNamespace(
                id=7,
                goal="Test goal",
                progress="started",
                status=status or "active",
            )
        ]

    def chat(self, prompt, auto_store=None):
        self.calls.append(("chat", prompt, auto_store))
        return f"response:{prompt}"


def install_fake_interface(monkeypatch, interface=None):
    if interface is None:
        interface = FakeMemoryInterface()

    monkeypatch.setattr(cli, "MemoryInterface", lambda: interface)
    return interface


def run_cli(monkeypatch, *argv, interface=None):
    install_fake_interface(monkeypatch, interface)
    monkeypatch.setattr(cli.sys, "argv", ["memory", *argv])
    cli.main()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def last_call():
    interface = FakeMemoryInterface.last_instance
    require(interface is not None, "MemoryInterface was not instantiated")
    require(interface.calls, "No MemoryInterface calls were recorded")
    return interface.calls[-1]


# ---------------------------------------------------------------------------
# Parser / top-level commands
# ---------------------------------------------------------------------------

def test_cli_version(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "argv", ["memory", "--version"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    require(exc.value.code == 0, "Version command did not exit successfully")

    out = capsys.readouterr().out
    require("Memoria v1.0" in out, "Version output missing")


def test_cli_help(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "argv", ["memory", "--help"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    require(exc.value.code == 0, "Help command did not exit successfully")

    out = capsys.readouterr().out

    commands = (
        "store",
        "recall",
        "store-many",
        "set-goal",
        "update-goal",
        "list-goals",
        "chat",
        "info",
        "graph",
        "doctor",
        "serve",
        "export",
        "import",
        "config",
        "auto-store",
    )

    for command in commands:
        require(command in out, f"Missing command from help: {command}")


def test_cli_invalid_command(monkeypatch):
    monkeypatch.setattr(cli.sys, "argv", ["memory", "does-not-exist"])

    with pytest.raises(SystemExit):
        cli.main()


def test_cli_missing_command(monkeypatch):
    monkeypatch.setattr(cli.sys, "argv", ["memory"])

    with pytest.raises(SystemExit):
        cli.main()


def test_cli_store_requires_text(monkeypatch):
    monkeypatch.setattr(cli.sys, "argv", ["memory", "store"])

    with pytest.raises(SystemExit):
        cli.main()


def test_cli_recall_requires_query(monkeypatch):
    monkeypatch.setattr(cli.sys, "argv", ["memory", "recall"])

    with pytest.raises(SystemExit):
        cli.main()


def test_cli_recall_rejects_invalid_format(monkeypatch):
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["memory", "recall", "test", "--format", "invalid"],
    )

    with pytest.raises(SystemExit):
        cli.main()


# ---------------------------------------------------------------------------
# Store / recall
# ---------------------------------------------------------------------------

def test_cli_store(monkeypatch, capsys):
    run_cli(monkeypatch, "store", "hello")

    require(
        last_call() == ("remember", "hello"),
        "Store did not pass text to MemoryInterface",
    )

    out = capsys.readouterr().out
    require(
        "Stored memory with ID: mem-1" in out,
        "Store result was not reported",
    )


def test_cli_recall_table(monkeypatch, capsys):
    run_cli(monkeypatch, "recall", "hello", "--limit", "1")

    require(
        last_call() == ("recall", "hello"),
        "Recall did not pass query to MemoryInterface",
    )

    out = capsys.readouterr().out

    require(
        "Found 2 results, showing first 1" in out,
        "Recall result count missing",
    )
    require("result" in out, "Recall result missing")


def test_cli_recall_raw(monkeypatch, capsys):
    run_cli(monkeypatch, "recall", "hello", "--format", "raw")

    out = capsys.readouterr().out

    require("results" in out, "Raw recall output missing")
    require("result" in out, "Raw recall result missing")


def test_cli_recall_json(monkeypatch, capsys):
    run_cli(monkeypatch, "recall", "hello", "--format", "json")

    out = capsys.readouterr().out

    data = json.loads(out)

    require(isinstance(data, dict), "JSON recall did not produce an object")
    require("results" in data, "JSON recall results missing")


# ---------------------------------------------------------------------------
# Store-many
# ---------------------------------------------------------------------------

def test_cli_store_many(monkeypatch, tmp_path, capsys):
    source = tmp_path / "memories.json"
    source.write_text(
        json.dumps(["one", "two", "three"]),
        encoding="utf8",
    )

    run_cli(monkeypatch, "store-many", str(source))

    require(
        last_call() == (
            "remember_many",
            ["one", "two", "three"],
        ),
        "store-many did not pass complete list",
    )

    out = capsys.readouterr().out
    require("Stored 3 memories" in out, "Store-many count missing")


def test_cli_store_many_rejects_non_list(monkeypatch, tmp_path, capsys):
    source = tmp_path / "memories.json"
    source.write_text(
        json.dumps({"text": "invalid"}),
        encoding="utf8",
    )

    run_cli(monkeypatch, "store-many", str(source))

    out = capsys.readouterr().out

    require(
        "JSON file must contain a list of strings" in out,
        "Invalid store-many structure was not rejected",
    )


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------

def test_cli_set_goal(monkeypatch, capsys):
    run_cli(monkeypatch, "set-goal", "test goal")

    require(
        last_call() == ("set_goal", "test goal", "started"),
        "set-goal did not use default progress",
    )

    out = capsys.readouterr().out
    require("Goal set with ID: 7" in out, "Goal ID missing")


def test_cli_set_goal_custom_progress(monkeypatch):
    run_cli(
        monkeypatch,
        "set-goal",
        "test goal",
        "--progress",
        "halfway",
    )

    require(
        last_call() == ("set_goal", "test goal", "halfway"),
        "Custom goal progress was not propagated",
    )


def test_cli_update_goal_requires_change(monkeypatch, capsys):
    run_cli(monkeypatch, "update-goal", "7")

    out = capsys.readouterr().out

    require(
        "At least one of --progress or --status is required" in out,
        "Missing update-goal validation",
    )


def test_cli_update_goal_progress(monkeypatch, capsys):
    run_cli(
        monkeypatch,
        "update-goal",
        "7",
        "--progress",
        "halfway",
    )

    require(
        last_call() == (
            "update_goal",
            7,
            "halfway",
            None,
        ),
        "Goal progress was not propagated",
    )

    out = capsys.readouterr().out
    require("Goal 7 updated." in out, "Goal update was not reported")


def test_cli_update_goal_status(monkeypatch):
    run_cli(
        monkeypatch,
        "update-goal",
        "7",
        "--status",
        "completed",
    )

    require(
        last_call() == (
            "update_goal",
            7,
            None,
            "completed",
        ),
        "Goal status was not propagated",
    )


def test_cli_list_goals(monkeypatch, capsys):
    run_cli(monkeypatch, "list-goals")

    out = capsys.readouterr().out

    for value in (
        "ID",
        "Goal",
        "Progress",
        "Status",
        "Test goal",
    ):
        require(value in out, f"Missing goal output: {value}")


def test_cli_list_goals_status(monkeypatch, capsys):
    run_cli(monkeypatch, "list-goals", "--status", "active")

    require(
        last_call() == ("list_goals", "active"),
        "Goal status filter was not propagated",
    )

    out = capsys.readouterr().out
    require("active" in out, "Goal status missing")


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def test_cli_chat_one_shot_config_default(monkeypatch, capsys):
    original = cli.settings.AUTO_STORE_MEMORIES

    try:
        cli.settings.AUTO_STORE_MEMORIES = True

        run_cli(monkeypatch, "chat", "hello")

        require(
            last_call() == ("chat", "hello", None),
            "Config-default chat mode was wrong",
        )

        out = capsys.readouterr().out
        require("response:hello" in out, "Chat response missing")
    finally:
        cli.settings.AUTO_STORE_MEMORIES = original


def test_cli_chat_auto_store_override(monkeypatch, capsys):
    run_cli(monkeypatch, "chat", "hello", "--auto-store")

    require(
        last_call() == ("chat", "hello", True),
        "Auto-store override was not True",
    )

    out = capsys.readouterr().out
    require("response:hello" in out, "Chat response missing")


def test_cli_chat_no_auto_store_override(monkeypatch, capsys):
    run_cli(monkeypatch, "chat", "hello", "--no-auto-store")

    require(
        last_call() == ("chat", "hello", False),
        "No-auto-store override was not False",
    )

    out = capsys.readouterr().out
    require("response:hello" in out, "Chat response missing")


def test_cli_chat_interactive_config_mode(monkeypatch, capsys):
    original = cli.settings.AUTO_STORE_MEMORIES

    try:
        cli.settings.AUTO_STORE_MEMORIES = True

        inputs = iter(["hello", "exit"])

        install_fake_interface(monkeypatch)
        monkeypatch.setattr(
            "builtins.input",
            lambda prompt="": next(inputs),
        )
        monkeypatch.setattr(
            cli.sys,
            "argv",
            ["memory", "chat"],
        )

        cli.main()

        interface = FakeMemoryInterface.last_instance

        require(
            ("chat", "hello", None) in interface.calls,
            "Interactive chat did not use config mode",
        )

        out = capsys.readouterr().out
        require(
            "Entering interactive chat mode" in out,
            "Interactive mode did not start",
        )
        require("Goodbye." in out, "Interactive mode did not exit")
    finally:
        cli.settings.AUTO_STORE_MEMORIES = original


def test_cli_chat_interactive_override(monkeypatch, capsys):
    inputs = iter(["hello", "quit"])

    install_fake_interface(monkeypatch)

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": next(inputs),
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["memory", "chat", "--no-auto-store"],
    )

    cli.main()

    interface = FakeMemoryInterface.last_instance

    require(
        ("chat", "hello", False) in interface.calls,
        "Interactive auto-store override was not propagated",
    )

    out = capsys.readouterr().out
    require(
        "Auto-store: disabled" in out,
        "Interactive override status missing",
    )


def test_cli_chat_interactive_keyboard_interrupt(monkeypatch, capsys):
    def interrupt(_prompt=""):
        raise KeyboardInterrupt

    install_fake_interface(monkeypatch)

    monkeypatch.setattr(
        "builtins.input",
        interrupt,
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["memory", "chat"],
    )

    cli.main()

    out = capsys.readouterr().out

    require(
        "Goodbye." in out,
        "Interactive KeyboardInterrupt did not exit cleanly",
    )


def test_cli_chat_interactive_recovers_from_error(monkeypatch, capsys):
    class ErrorInterface(FakeMemoryInterface):
        def chat(self, prompt, auto_store=None):
            raise RuntimeError("chat failure")

    monkeypatch.setattr(cli, "MemoryInterface", ErrorInterface)

    inputs = iter(["hello", "exit"])

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": next(inputs),
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["memory", "chat"],
    )

    cli.main()

    out = capsys.readouterr().out

    require(
        "Error: chat failure" in out,
        "Interactive chat error was not reported",
    )
    require(
        "Goodbye." in out,
        "Interactive chat did not recover after error",
    )


# ---------------------------------------------------------------------------
# Info / graph / doctor
# ---------------------------------------------------------------------------

def test_cli_info(monkeypatch, capsys):
    run_cli(monkeypatch, "info")

    out = capsys.readouterr().out

    for value in (
        "Memoria v1.0",
        "Database:",
        "Total memories: 2",
        "Embedding model:",
        "LLM URL:",
        "Top K:",
        "Debug mode:",
        "CLI output format:",
    ):
        require(value in out, f"Missing info output: {value}")


def test_cli_graph(monkeypatch, capsys):
    run_cli(monkeypatch, "graph", "alice", "--depth", "3")

    out = capsys.readouterr().out

    require(
        "Neighbors of 'alice' (depth 3):" in out,
        "Graph depth/header missing",
    )
    require("related_to" in out, "Graph relation missing")
    require("other" in out, "Graph target missing")


def test_cli_graph_entity_not_found(monkeypatch, capsys):
    interface = FakeMemoryInterface()
    interface.controller = FakeController(graph_found=False)

    run_cli(
        monkeypatch,
        "graph",
        "missing",
        interface=interface,
    )

    out = capsys.readouterr().out

    require(
        "Entity 'missing' not found." in out,
        "Missing graph entity path failed",
    )


def test_cli_doctor_success(monkeypatch, capsys):
    run_cli(monkeypatch, "doctor")

    out = capsys.readouterr().out

    for value in (
        "Running integrity checks...",
        "Integrity check: OK",
        "DB count: 2",
        "Columns:",
    ):
        require(value in out, f"Missing doctor output: {value}")


def test_cli_doctor_handles_integrity_error(monkeypatch, capsys):
    class BrokenDB(FakeDB):
        def integrity_check(self):
            raise RuntimeError("integrity failure")

    class BrokenInterface(FakeMemoryInterface):
        def __init__(self):
            self.controller = SimpleNamespace(
                system=SimpleNamespace(
                    db=BrokenDB(),
                    graph_search=FakeGraph(),
                )
            )
            self.calls = []
            FakeMemoryInterface.last_instance = self

    monkeypatch.setattr(cli, "MemoryInterface", BrokenInterface)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["memory", "doctor"],
    )

    cli.main()

    out = capsys.readouterr().out

    require(
        "Integrity check error: integrity failure" in out,
        "Integrity error was not handled",
    )
    require(
        "DB count: 2" in out,
        "Doctor did not continue after integrity failure",
    )


def test_cli_doctor_handles_sanity_error(monkeypatch, capsys):
    class BrokenDB(FakeDB):
        def sanity_check(self):
            raise RuntimeError("sanity failure")

    class BrokenInterface(FakeMemoryInterface):
        def __init__(self):
            self.controller = SimpleNamespace(
                system=SimpleNamespace(
                    db=BrokenDB(),
                    graph_search=FakeGraph(),
                )
            )
            self.calls = []
            FakeMemoryInterface.last_instance = self

    monkeypatch.setattr(cli, "MemoryInterface", BrokenInterface)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["memory", "doctor"],
    )

    cli.main()

    out = capsys.readouterr().out

    require(
        "Sanity check error: sanity failure" in out,
        "Sanity error was not handled",
    )


# ---------------------------------------------------------------------------
# Export / import
# ---------------------------------------------------------------------------

def test_cli_export(monkeypatch, tmp_path, capsys):
    target = tmp_path / "export.json"

    run_cli(monkeypatch, "export", str(target))

    require(target.exists(), "Export file was not created")

    data = json.loads(target.read_text(encoding="utf8"))

    require(isinstance(data, list), "Export did not produce a list")
    require(len(data) == 2, "Unexpected export count")

    out = capsys.readouterr().out

    require(
        "Exported 2 memories" in out,
        "Export result was not reported",
    )


def test_cli_import_text(monkeypatch, tmp_path, capsys):
    source = tmp_path / "import.json"
    source.write_text(
        json.dumps(
            [
                {"id": 1, "text": "alpha"},
                {"id": 2, "normalized_text": "beta"},
            ]
        ),
        encoding="utf8",
    )

    run_cli(monkeypatch, "import", str(source))

    # Import accepts both canonical text and normalized_text records.
    require(
        last_call() == (
            "remember_many",
            ["alpha", "beta"],
        ),
        "Import did not extract text and normalized_text records",
    )

    out = capsys.readouterr().out

    require(
        "Imported 2 memories" in out,
        "Import result was not reported",
    )


def test_cli_import_normalized_text(monkeypatch, tmp_path, capsys):
    source = tmp_path / "import.json"
    source.write_text(
        json.dumps(
            [{"normalized_text": "normalized memory"}]
        ),
        encoding="utf8",
    )

    run_cli(monkeypatch, "import", str(source))

    # This is intentionally expected to expose the current CLI bug.
    # The current fallback expression supports normalized_text, but the
    # surrounding filter requires item["text"] to exist.
    require(
        last_call() == (
            "remember_many",
            ["normalized memory"],
        ),
        "normalized_text import failed",
    )

    out = capsys.readouterr().out

    require(
        "Imported 1 memories" in out,
        "normalized_text import count was not reported",
    )


def test_cli_import_rejects_non_list(monkeypatch, tmp_path, capsys):
    source = tmp_path / "import.json"
    source.write_text(
        json.dumps({"text": "bad"}),
        encoding="utf8",
    )

    run_cli(monkeypatch, "import", str(source))

    out = capsys.readouterr().out

    require(
        "import file must contain a list" in out,
        "Invalid import structure was not rejected",
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_cli_config(monkeypatch, capsys):
    run_cli(monkeypatch, "config")

    out = capsys.readouterr().out

    require(
        "Current configuration:" in out,
        "Config header missing",
    )


# ---------------------------------------------------------------------------
# Serve
# ---------------------------------------------------------------------------

def test_cli_serve(monkeypatch, capsys):
    calls = []

    class FakeUvicorn:
        @staticmethod
        def run(*args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(cli, "uvicorn", FakeUvicorn)
    monkeypatch.setattr(cli, "HAS_UVICORN", True)

    run_cli(
        monkeypatch,
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "9123",
        "--reload",
    )

    require(
        len(calls) == 1,
        "uvicorn.run was not called",
    )

    args, kwargs = calls[0]

    require(
        args == ("app:app",),
        "Unexpected uvicorn application",
    )
    require(
        kwargs["host"] == "127.0.0.1",
        "Host was not propagated",
    )
    require(
        kwargs["port"] == 9123,
        "Port was not propagated",
    )
    require(
        kwargs["reload"] is True,
        "Reload flag was not propagated",
    )

    out = capsys.readouterr().out

    # serve intentionally emits its startup message through info().
    require(
        "Starting server on 127.0.0.1:9123" in out,
        "Server startup message missing",
    )


def test_cli_serve_without_uvicorn(monkeypatch, capsys):
    monkeypatch.setattr(cli, "HAS_UVICORN", False)

    run_cli(monkeypatch, "serve")

    out = capsys.readouterr().out

    require(
        "uvicorn is not installed" in out,
        "Missing-uvicorn branch failed",
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def test_cli_benchmark(monkeypatch, capsys):
    calls = []

    class FakeRunner:
        def __init__(self):
            calls.append(("init",))

        def run(self, limit=None):
            calls.append(("run", limit))
            return "benchmark-output.json"

    monkeypatch.setattr(cli, "HAS_BENCHMARK", True)
    monkeypatch.setattr(cli, "BenchmarkRunner", FakeRunner)

    run_cli(
        monkeypatch,
        "benchmark",
        "--limit",
        "17",
    )

    require(
        ("init",) in calls,
        "BenchmarkRunner was not constructed",
    )
    require(
        ("run", 17) in calls,
        "Benchmark limit was not propagated",
    )

    out = capsys.readouterr().out

    require(
        "benchmark-output.json" in out,
        "Benchmark output path missing",
    )





# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

class FakeRegistry:
    def __init__(self):
        self.calls = []
        self.enabled = {}

    def save(self, path):
        self.calls.append(("save", path))

    def load(self, path):
        self.calls.append(("load", path))

    def reload(self):
        self.calls.append(("reload",))

    def enable(self, name):
        self.calls.append(("enable", name))
        self.enabled[name] = True

    def disable(self, name):
        self.calls.append(("disable", name))
        self.enabled[name] = False

    def is_enabled(self, name):
        self.calls.append(("is_enabled", name))
        return self.enabled.get(name, False)

    def get_cost(self, name):
        return 1.0


class FakeRouter:
    last_instance = None

    def __init__(self, registry):
        self.registry = registry
        self.calls = []
        FakeRouter.last_instance = self

    def get_active_signals(self, memory_type):
        self.calls.append(
            ("get_active_signals", memory_type)
        )
        return {
            "bm25": 1.0,
            "semantic": 0.5,
        }

    def clear_cache(self):
        self.calls.append(("clear_cache",))


def install_fake_signals(monkeypatch):
    registry = FakeRegistry()

    monkeypatch.setattr(
        cli,
        "HAS_SIGNAL_REGISTRY",
        True,
    )
    monkeypatch.setattr(
        cli,
        "get_registry",
        lambda: registry,
    )
    monkeypatch.setattr(
        cli,
        "SignalRouter",
        FakeRouter,
    )

    return registry


def test_cli_signals_list(monkeypatch, capsys):
    install_fake_signals(monkeypatch)

    run_cli(monkeypatch, "signals", "--list")

    out = capsys.readouterr().out

    require(
        "Signals for type: general" in out,
        "Signal type header missing",
    )
    require("bm25" in out, "BM25 signal missing")
    require("semantic" in out, "Semantic signal missing")


def test_cli_signals_toggle_enable(monkeypatch, capsys):
    registry = install_fake_signals(monkeypatch)

    run_cli(
        monkeypatch,
        "signals",
        "--toggle",
        "bm25",
        "--enable",
    )

    require(
        ("enable", "bm25") in registry.calls,
        "Signal enable was not called",
    )

    router = FakeRouter.last_instance

    require(
        ("clear_cache",) in router.calls,
        "Signal router cache was not cleared",
    )

    out = capsys.readouterr().out

    require(
        "Enabled signal: bm25" in out,
        "Signal enable was not reported",
    )


def test_cli_signals_toggle_disable(monkeypatch, capsys):
    registry = install_fake_signals(monkeypatch)

    run_cli(
        monkeypatch,
        "signals",
        "--toggle",
        "bm25",
        "--disable",
    )

    require(
        ("disable", "bm25") in registry.calls,
        "Signal disable was not called",
    )

    router = FakeRouter.last_instance

    require(
        ("clear_cache",) in router.calls,
        "Signal router cache was not cleared",
    )

    out = capsys.readouterr().out

    require(
        "Disabled signal: bm25" in out,
        "Signal disable was not reported",
    )


def test_cli_signals_toggle_without_direction(monkeypatch, capsys):
    registry = install_fake_signals(monkeypatch)

    run_cli(
        monkeypatch,
        "signals",
        "--toggle",
        "bm25",
    )

    require(
        ("is_enabled", "bm25") in registry.calls,
        "Signal state was not checked",
    )
    require(
        ("enable", "bm25") in registry.calls,
        "Disabled signal was not toggled on",
    )

    router = FakeRouter.last_instance

    require(
        ("clear_cache",) in router.calls,
        "Signal router cache was not cleared",
    )

    out = capsys.readouterr().out

    require(
        "Enabled signal: bm25" in out,
        "Toggle result was not reported",
    )


def test_cli_signals_rejects_conflicting_flags(monkeypatch, capsys):
    install_fake_signals(monkeypatch)

    run_cli(
        monkeypatch,
        "signals",
        "--toggle",
        "bm25",
        "--enable",
        "--disable",
    )

    out = capsys.readouterr().out

    require(
        "Cannot use both --enable and --disable" in out,
        "Conflicting signal flags were not rejected",
    )


def test_cli_signals_export(monkeypatch, tmp_path, capsys):
    registry = install_fake_signals(monkeypatch)
    target = tmp_path / "signals.json"

    run_cli(
        monkeypatch,
        "signals",
        "--export",
        str(target),
    )

    require(
        ("save", str(target)) in registry.calls,
        "Signal export was not called",
    )

    out = capsys.readouterr().out

    require(
        "Exported registry" in out,
        "Signal export was not reported",
    )


def test_cli_signals_import(monkeypatch, tmp_path, capsys):
    registry = install_fake_signals(monkeypatch)
    source = tmp_path / "signals.json"
    source.write_text("{}", encoding="utf8")

    run_cli(
        monkeypatch,
        "signals",
        "--import",
        str(source),
    )

    require(
        ("load", str(source)) in registry.calls,
        "Signal import was not called",
    )

    router = FakeRouter.last_instance

    require(
        ("clear_cache",) in router.calls,
        "Signal router cache was not cleared after import",
    )

    out = capsys.readouterr().out

    require(
        "Imported registry" in out,
        "Signal import was not reported",
    )


def test_cli_signals_reset(monkeypatch, capsys):
    registry = install_fake_signals(monkeypatch)

    run_cli(
        monkeypatch,
        "signals",
        "--reset",
    )

    require(
        ("reload",) in registry.calls,
        "Signal registry was not reloaded",
    )

    router = FakeRouter.last_instance

    require(
        ("clear_cache",) in router.calls,
        "Signal router cache was not cleared after reset",
    )

    out = capsys.readouterr().out

    require(
        "Registry reset to defaults" in out,
        "Signal reset was not reported",
    )





# ---------------------------------------------------------------------------
# Query history
# ---------------------------------------------------------------------------

class FakeQueryHistory:
    def __init__(self):
        self.entries = [
            {
                "timestamp": 0,
                "results": [{"id": "1"}],
                "query": "first query",
            },
            {
                "timestamp": 0,
                "results": [
                    {"id": "2"},
                    {"id": "3"},
                ],
                "query": "second query",
            },
        ]

    def get_recent(self, n=10):
        return self.entries[-n:]

    def get_frequent(self, n=10):
        return [
            "second query",
            "first query",
        ][:n]


def test_cli_query_history_recent(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "HAS_QUERY_HISTORY",
        True,
    )
    monkeypatch.setattr(
        cli,
        "QueryHistory",
        FakeQueryHistory,
    )

    run_cli(
        monkeypatch,
        "query-history",
        "--limit",
        "1",
    )

    out = capsys.readouterr().out

    require(
        "Timestamp" in out,
        "Query-history header missing",
    )
    require(
        "second query" in out,
        "Recent query missing",
    )
    require(
        "first query" not in out,
        "Query-history limit was not respected",
    )


def test_cli_query_history_frequent(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "HAS_QUERY_HISTORY",
        True,
    )
    monkeypatch.setattr(
        cli,
        "QueryHistory",
        FakeQueryHistory,
    )

    run_cli(
        monkeypatch,
        "query-history",
        "--frequent",
        "--limit",
        "1",
    )

    out = capsys.readouterr().out

    require(
        "Most frequent queries (top 1)" in out,
        "Frequent-query header missing",
    )
    require(
        "second query" in out,
        "Frequent query missing",
    )


# ---------------------------------------------------------------------------
# Auto-store
# ---------------------------------------------------------------------------

def test_cli_auto_store_status(monkeypatch, capsys):
    run_cli(
        monkeypatch,
        "auto-store",
        "--status",
    )

    out = capsys.readouterr().out

    for value in (
        "Auto-store:",
        "Threshold:",
        "Max per session:",
        "Types:",
    ):
        require(
            value in out,
            f"Missing auto-store status field: {value}",
        )


def test_cli_auto_store_enable(monkeypatch, capsys):
    original = cli.settings.AUTO_STORE_MEMORIES

    try:
        run_cli(
            monkeypatch,
            "auto-store",
            "--enable",
        )

        require(
            cli.settings.AUTO_STORE_MEMORIES is True,
            "Auto-store was not enabled",
        )

        out = capsys.readouterr().out

        require(
            "Auto-store enabled" in out,
            "Enable result missing",
        )
    finally:
        cli.settings.AUTO_STORE_MEMORIES = original


def test_cli_auto_store_disable(monkeypatch, capsys):
    original = cli.settings.AUTO_STORE_MEMORIES

    try:
        run_cli(
            monkeypatch,
            "auto-store",
            "--disable",
        )

        require(
            cli.settings.AUTO_STORE_MEMORIES is False,
            "Auto-store was not disabled",
        )

        out = capsys.readouterr().out

        require(
            "Auto-store disabled" in out,
            "Disable result missing",
        )
    finally:
        cli.settings.AUTO_STORE_MEMORIES = original


def test_cli_auto_store_threshold(monkeypatch, capsys):
    original = cli.settings.AUTO_STORE_THRESHOLD

    try:
        run_cli(
            monkeypatch,
            "auto-store",
            "--threshold",
            "0.75",
        )

        require(
            cli.settings.AUTO_STORE_THRESHOLD == 0.75,
            "Threshold was not updated",
        )

        out = capsys.readouterr().out

        require(
            "Auto-store threshold set to 0.75" in out,
            "Threshold result missing",
        )
    finally:
        cli.settings.AUTO_STORE_THRESHOLD = original


def test_cli_auto_store_rejects_bad_threshold(monkeypatch, capsys):
    original = cli.settings.AUTO_STORE_THRESHOLD

    try:
        run_cli(
            monkeypatch,
            "auto-store",
            "--threshold",
            "1.5",
        )

        require(
            cli.settings.AUTO_STORE_THRESHOLD == original,
            "Invalid threshold changed settings",
        )

        out = capsys.readouterr().out

        require(
            "Threshold must be between 0.0 and 1.0" in out,
            "Invalid threshold was not rejected",
        )
    finally:
        cli.settings.AUTO_STORE_THRESHOLD = original


def test_cli_auto_store_max(monkeypatch, capsys):
    original = cli.settings.AUTO_STORE_MAX_PER_SESSION

    try:
        run_cli(
            monkeypatch,
            "auto-store",
            "--max",
            "5",
        )

        require(
            cli.settings.AUTO_STORE_MAX_PER_SESSION == 5,
            "Auto-store max was not updated",
        )

        out = capsys.readouterr().out

        require(
            "Auto-store max per session set to 5" in out,
            "Max result missing",
        )
    finally:
        cli.settings.AUTO_STORE_MAX_PER_SESSION = original


def test_cli_auto_store_rejects_bad_max(monkeypatch, capsys):
    original = cli.settings.AUTO_STORE_MAX_PER_SESSION

    try:
        run_cli(
            monkeypatch,
            "auto-store",
            "--max",
            "0",
        )

        require(
            cli.settings.AUTO_STORE_MAX_PER_SESSION == original,
            "Invalid max changed settings",
        )

        out = capsys.readouterr().out

        require(
            "Max must be > 0" in out,
            "Invalid max was not rejected",
        )
    finally:
        cli.settings.AUTO_STORE_MAX_PER_SESSION = original


def test_cli_auto_store_types(monkeypatch, capsys):
    original = cli.settings.AUTO_STORE_TYPES

    try:
        run_cli(
            monkeypatch,
            "auto-store",
            "--types",
            "general,code,project",
        )

        require(
            cli.settings.AUTO_STORE_TYPES
            == ["general", "code", "project"],
            "Auto-store types were not updated",
        )

        out = capsys.readouterr().out

        require(
            "Auto-store types set to: general, code, project" in out,
            "Types result missing",
        )
    finally:
        cli.settings.AUTO_STORE_TYPES = original


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def test_print_table_empty(capsys):
    cli.print_table([], 5)

    out = capsys.readouterr().out

    require(
        out.strip() == "No results found.",
        "Empty table output was incorrect",
    )


def test_print_table_with_scores(capsys):
    cli.print_table(
        [
            {
                "rank": 1,
                "final_score": 0.9876,
                "text": "hello",
            }
        ],
        1,
    )

    out = capsys.readouterr().out

    for value in (
        "Rank",
        "Score",
        "0.9876",
        "hello",
    ):
        require(
            value in out,
            f"Missing table output: {value}",
        )


def test_print_table_without_scores(capsys):
    cli.print_table(
        [
            {
                "rank": 1,
                "final_score": 0.9876,
                "text": "hello",
            }
        ],
        1,
        show_scores=False,
    )

    out = capsys.readouterr().out

    require(
        "Score" not in out,
        "Score column should not be present",
    )
    require(
        "hello" in out,
        "Table text missing",
    )


def test_print_query_history_empty(capsys):
    cli.print_query_history([])

    out = capsys.readouterr().out

    require(
        out.strip() == "No history entries found.",
        "Empty query-history output was incorrect",
    )


def test_print_goals_empty(capsys):
    cli.print_goals([])

    out = capsys.readouterr().out

    require(
        out.strip() == "No goals found.",
        "Empty goals output was incorrect",
    )


def test_print_signals_empty(capsys):
    cli.print_signals({})

    out = capsys.readouterr().out

    require(
        out.strip() == "No signals found.",
        "Empty signals output was incorrect",
    )


def test_print_signals_without_registry(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "HAS_SIGNAL_REGISTRY",
        False,
    )

    cli.print_signals(
        {"bm25": 1.0},
        "general",
    )

    out = capsys.readouterr().out

    require(
        "bm25" in out,
        "Signal name missing",
    )
    require(
        "unknown" in out,
        "Fallback signal cost missing",
    )


def test_print_query_diff_error(capsys):
    cli.print_query_diff(
        {"error": "comparison failed"}
    )

    out = capsys.readouterr().out

    require(
        "Error: comparison failed" in out,
        "Query diff error was not printed",
    )


def test_print_query_diff_full(capsys):
    cli.print_query_diff(
        {
            "entry1": {
                "id": 1,
                "query": "first query",
            },
            "entry2": {
                "id": 2,
                "query": "second query",
            },
            "common_results": 3,
            "only_in_first": 1,
            "only_in_second": 2,
            "score_changes": [
                {
                    "text": "memory text",
                    "delta": 0.2,
                    "score_old": 0.5,
                    "score_new": 0.7,
                }
            ],
        }
    )

    out = capsys.readouterr().out

    for value in (
        "Comparing 1 vs 2",
        "Common results: 3",
        "Only in first: 1",
        "Only in second: 2",
        "memory text",
    ):
        require(
            value in out,
            f"Missing query-diff output: {value}",
        )


def test_print_auto_store_status(monkeypatch, capsys):
    monkeypatch.setattr(
        cli.settings,
        "AUTO_STORE_MEMORIES",
        True,
    )
    monkeypatch.setattr(
        cli.settings,
        "AUTO_STORE_THRESHOLD",
        0.75,
    )
    monkeypatch.setattr(
        cli.settings,
        "AUTO_STORE_MAX_PER_SESSION",
        5,
    )
    monkeypatch.setattr(
        cli.settings,
        "AUTO_STORE_TYPES",
        ["fact", "preference"],
    )

    cli.print_auto_store_status()

    out = capsys.readouterr().out

    for value in (
        "Auto-store: enabled",
        "Threshold: 0.75",
        "Max per session: 5",
        "fact",
        "preference",
    ):
        require(
            value in out,
            f"Missing auto-store status output: {value}",
        )


# ---------------------------------------------------------------------------
# Global exception handling
# ---------------------------------------------------------------------------

def test_cli_global_exception_path(monkeypatch, capsys):
    class ExplodingInterface:
        def __init__(self):
            raise RuntimeError("interface exploded")

    monkeypatch.setattr(
        cli,
        "MemoryInterface",
        ExplodingInterface,
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["memory", "info"],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    require(
        exc.value.code == 1,
        "CLI exception path did not exit with code 1",
    )

    captured = capsys.readouterr()

    require(
        "Error: interface exploded" in captured.err,
        "CLI initialization error was not reported",
    )


def test_cli_keyboard_interrupt_path(monkeypatch, capsys):
    class InterruptingInterface:
        def __init__(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        cli,
        "MemoryInterface",
        InterruptingInterface,
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["memory", "info"],
    )

    try:
        cli.main()
    except SystemExit as exc:
        require(
            exc.code == 1,
            "CLI did not exit with status 1 after KeyboardInterrupt",
        )
    except KeyboardInterrupt as exc:
        raise RuntimeError(
            "CLI leaked KeyboardInterrupt instead of handling it"
        ) from exc

    out = capsys.readouterr().out

    require(
        "Interrupted" in out,
        "CLI did not report KeyboardInterrupt",
    )
