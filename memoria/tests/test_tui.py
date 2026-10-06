import json
import sys
from types import SimpleNamespace

import pytest

import tui


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class FakeDB:
    def __init__(self):
        self.calls = []

    def count(self):
        self.calls.append(("count",))
        return 42

    def integrity_check(self):
        self.calls.append(("integrity_check",))
        return "OK"

    def sanity_check(self):
        self.calls.append(("sanity_check",))
        return {
            "db_count": 42,
            "columns": ["id", "text", "timestamp"],
        }

    def fetch_all(self):
        self.calls.append(("fetch_all",))
        return [
            {"id": 1, "text": "memory one"},
            {"id": 2, "text": "memory two"},
        ]


class FakeGraph:
    def __init__(self):
        self.calls = []

    def find_entity(self, name):
        self.calls.append(("find_entity", name))
        if name == "missing":
            return None
        return {"name": name}

    def neighbors(self, name, depth=1):
        self.calls.append(("neighbors", name, depth))
        return [
            {
                "relation": "knows",
                "target": "Bob",
                "source": "test",
            },
            {
                "relation": "uses",
                "target": "Python",
                "source": "test",
            },
        ]


class FakeSystem:
    def __init__(self):
        self.db = FakeDB()
        self.graph_search = FakeGraph()


class FakeController:
    def __init__(self):
        self.system = FakeSystem()


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
        self.calls.append(
            ("update_goal", goal_id, progress, status)
        )

    def list_goals(self, status=None):
        self.calls.append(("list_goals", status))
        return [
            {
                "id": 7,
                "goal": "Test goal",
                "progress": "started",
                "status": status or "active",
            }
        ]

    def chat(self, prompt, auto_store=None):
        self.calls.append(("chat", prompt, auto_store))
        return f"response:{prompt}"


class FakeRegistry:
    last_instance = None

    def __init__(self):
        self.calls = []
        self.enabled = {
            "semantic": True,
            "lexical": False,
        }
        FakeRegistry.last_instance = self

    def get_cost(self, name):
        self.calls.append(("get_cost", name))
        return 1.5

    def is_enabled(self, name):
        self.calls.append(("is_enabled", name))
        return self.enabled.get(name, False)

    def enable(self, name):
        self.calls.append(("enable", name))
        self.enabled[name] = True

    def disable(self, name):
        self.calls.append(("disable", name))
        self.enabled[name] = False

    def reload(self):
        self.calls.append(("reload",))


class FakeRouter:
    last_instance = None

    def __init__(self, registry):
        self.registry = registry
        self.calls = []
        FakeRouter.last_instance = self

    def get_active_signals(self, memory_type):
        self.calls.append(("get_active_signals", memory_type))
        return {
            "semantic": 0.8,
            "lexical": 0.2,
        }

    def clear_cache(self):
        self.calls.append(("clear_cache",))


class FakeHistory:
    last_instance = None

    def __init__(self):
        self.calls = []
        FakeHistory.last_instance = self

        self.entries = [
            {
                "query": "hello world",
                "timestamp": 1760000400.0,
                "results": [
                    {
                        "id": "1",
                        "rank": 1,
                        "text": "result one",
                    },
                    {
                        "id": "2",
                        "rank": 2,
                        "text": "result two",
                    },
                ],
            },
            {
                "query": "what happened",
                "timestamp": 1760000460.0,
                "results": [
                    {
                        "id": "3",
                        "rank": 1,
                        "text": "temporal result",
                    }
                ],
            },
        ]

    def get_recent(self, n=10):
        self.calls.append(("get_recent", n))
        return self.entries[-n:]

    def get_frequent(self, n=10):
        self.calls.append(("get_frequent", n))
        return ["what happened", "hello world"][:n]

    def get_context(self):
        self.calls.append(("get_context",))
        return ["hello world", "what happened"]

    def get_previous_query(self):
        self.calls.append(("get_previous_query",))
        return "hello world"

def install_fakes(monkeypatch):
    monkeypatch.setattr(
        tui,
        "MemoryInterface",
        FakeMemoryInterface,
    )

    if tui.HAS_SIGNAL_REGISTRY:
        monkeypatch.setattr(
            tui,
            "get_registry",
            lambda: FakeRegistry(),
        )
        monkeypatch.setattr(
            tui,
            "SignalRouter",
            FakeRouter,
        )

    if tui.HAS_QUERY_HISTORY:
        monkeypatch.setattr(
            tui,
            "QueryHistory",
            FakeHistory,
        )


def make_shell(monkeypatch):
    install_fakes(monkeypatch)
    return tui.MemoryShell()


def last_memory_call():
    require(
        FakeMemoryInterface.last_instance is not None,
        "MemoryInterface was not constructed",
    )
    require(
        FakeMemoryInterface.last_instance.calls,
        "MemoryInterface received no calls",
    )
    return FakeMemoryInterface.last_instance.calls[-1]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def test_format_table_empty():
    require(
        tui.format_table([], 10) == "No results found.",
        "Empty table output incorrect",
    )


def test_format_table_with_scores():
    results = [
        {
            "rank": 1,
            "final_score": 0.95,
            "text": "hello",
        }
    ]

    output = tui.format_table(results, 10)

    require("Rank" in output, "Rank header missing")
    require("Score" in output, "Score header missing")
    require("0.9500" in output, "Score formatting incorrect")
    require("hello" in output, "Result text missing")


def test_format_table_without_scores():
    results = [
        {
            "rank": 1,
            "final_score": 0.95,
            "text": "hello",
        }
    ]

    output = tui.format_table(results, 10, show_scores=False)

    require("Rank" in output, "Rank header missing")
    require("Score" not in output, "Score column should be hidden")
    require("hello" in output, "Result text missing")


def test_format_table_limit():
    results = [
        {
            "rank": 1,
            "final_score": 0.95,
            "text": "first",
        },
        {
            "rank": 2,
            "final_score": 0.85,
            "text": "second",
        },
    ]

    output = tui.format_table(results, 1)

    require("first" in output, "First result missing")
    require("second" not in output, "Limit was not applied")


def test_format_table_width():
    results = [
        {
            "rank": 1,
            "final_score": 0.95,
            "text": "abcdefghijklmnopqrstuvwxyz",
        }
    ]

    output = tui.format_table(results, 1, width=25)

    require(
        len(output.splitlines()[-1]) <= 25,
        "Table width was not respected",
    )


def test_format_goals_empty():
    require(
        tui.format_goals([]) == "No goals found.",
        "Empty goal output incorrect",
    )


def test_format_goals():
    goals = [
        {
            "id": 7,
            "goal": "Test goal",
            "progress": "started",
            "status": "active",
        }
    ]

    output = tui.format_goals(goals)

    require("Test goal" in output, "Goal text missing")
    require("started" in output, "Goal progress missing")
    require("active" in output, "Goal status missing")


def test_format_signals_empty(monkeypatch):
    install_fakes(monkeypatch)

    require(
        tui.format_signals({}) == "No signals found.",
        "Empty signal output incorrect",
    )


def test_format_signals(monkeypatch):
    install_fakes(monkeypatch)

    output = tui.format_signals(
        {
            "semantic": 0.8,
            "lexical": 0.2,
        }
    )

    require("semantic" in output, "Signal missing")
    require("0.8000" in output, "Signal weight missing")
    require("1.5" in output, "Signal cost missing")


def test_format_query_history_empty():
    require(
        tui.format_query_history([]) == "No history entries found.",
        "Empty history output incorrect",
    )



def test_format_query_diff_error():
    output = tui.format_query_diff(
        {"error": "bad comparison"}
    )

    require(
        output == "Error: bad comparison",
        "Diff error formatting incorrect",
    )


def test_format_query_diff():
    diff = {
        "entry1": {
            "id": "one",
            "query": "first query",
        },
        "entry2": {
            "id": "two",
            "query": "second query",
        },
        "common_results": 1,
        "only_in_first": 2,
        "only_in_second": 3,
        "score_changes": [
            {
                "text": "changed",
                "delta": 0.1,
                "score_old": 0.8,
                "score_new": 0.9,
            }
        ],
    }

    output = tui.format_query_diff(diff)

    require("one" in output, "First ID missing")
    require("two" in output, "Second ID missing")
    require("Common results: 1" in output, "Common result count missing")
    require("changed" in output, "Score change missing")


def test_format_auto_store_status(monkeypatch):
    old_enabled = tui.settings.AUTO_STORE_MEMORIES
    old_threshold = tui.settings.AUTO_STORE_THRESHOLD
    old_max = tui.settings.AUTO_STORE_MAX_PER_SESSION
    old_types = list(tui.settings.AUTO_STORE_TYPES)

    try:
        tui.settings.AUTO_STORE_MEMORIES = True
        tui.settings.AUTO_STORE_THRESHOLD = 0.7
        tui.settings.AUTO_STORE_MAX_PER_SESSION = 5
        tui.settings.AUTO_STORE_TYPES = ["general", "fact"]

        output = tui.format_auto_store_status()

        require("enabled" in output, "Auto-store status missing")
        require("0.7" in output, "Threshold missing")
        require("5" in output, "Max session count missing")
        require("general, fact" in output, "Types missing")
    finally:
        tui.settings.AUTO_STORE_MEMORIES = old_enabled
        tui.settings.AUTO_STORE_THRESHOLD = old_threshold
        tui.settings.AUTO_STORE_MAX_PER_SESSION = old_max
        tui.settings.AUTO_STORE_TYPES = old_types


# ---------------------------------------------------------------------------
# Shell initialization and basic command behavior
# ---------------------------------------------------------------------------


def test_shell_initialization(monkeypatch):
    shell = make_shell(monkeypatch)

    require(shell.mem is FakeMemoryInterface.last_instance,
            "Shell did not create MemoryInterface")
    require(shell.default_limit == tui.settings.CLI_DEFAULT_LIMIT,
            "Default limit incorrect")
    require(shell.show_scores == tui.settings.CLI_SHOW_SCORES,
            "Score display setting incorrect")
    require(shell.table_width == tui.settings.CLI_TABLE_WIDTH,
            "Table width setting incorrect")
    require(shell.in_chat_mode is False,
            "Shell started in chat mode")
    require(shell.chat_history == [],
            "Chat history not initialized")
    require(shell.auto_store_override is None,
            "Auto-store override not initialized")


def test_shell_initializes_signal_components(monkeypatch):
    if not tui.HAS_SIGNAL_REGISTRY:
        pytest.skip("Signal registry unavailable")

    shell = make_shell(monkeypatch)

    require(shell.registry is FakeRegistry.last_instance,
            "Signal registry not initialized")
    require(shell.signal_router is FakeRouter.last_instance,
            "Signal router not initialized")


def test_shell_initializes_history(monkeypatch):
    if not tui.HAS_QUERY_HISTORY:
        pytest.skip("Query history unavailable")

    shell = make_shell(monkeypatch)

    require(shell.history is FakeHistory.last_instance,
            "Query history not initialized")


def test_store(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("store hello world")

    require(
        last_memory_call() == ("remember", "hello world"),
        "Store did not pass text correctly",
    )
    require(
        "Stored ID: mem-1" in capsys.readouterr().out,
        "Store result missing",
    )


def test_store_missing_argument(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("store")

    require(
        "Usage: store <text>" in capsys.readouterr().out,
        "Store usage message missing",
    )


def test_store_error(monkeypatch, capsys):
    class BrokenInterface(FakeMemoryInterface):
        def remember(self, text):
            raise RuntimeError("store failed")

    monkeypatch.setattr(tui, "MemoryInterface", BrokenInterface)

    shell = tui.MemoryShell()
    shell.onecmd("store hello")

    require(
        "Error: store failed" in capsys.readouterr().out,
        "Store error was not reported",
    )


def test_recall_default_limit(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("recall hello")

    require(
        last_memory_call() == ("recall", "hello"),
        "Recall query incorrect",
    )

    output = capsys.readouterr().out

    require("Found 2 results" in output, "Recall count missing")
    require("result" in output, "Recall result missing")


def test_recall_explicit_limit(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("recall hello 1")

    require(
        last_memory_call() == ("recall", "hello"),
        "Recall query with limit incorrect",
    )

    output = capsys.readouterr().out

    require(
        "showing first 1" in output,
        "Recall limit output missing",
    )
    require("result" in output, "Recall result missing")
    require("second result" not in output,
            "Recall limit did not truncate table")


def test_recall_empty_query(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("recall")

    require(
        "Usage: recall <query> [limit]" in capsys.readouterr().out,
        "Recall usage message missing",
    )


def test_recall_numeric_query_edge(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("recall 123")

    require(
        "Usage: recall <query> [limit]" in capsys.readouterr().out,
        "Numeric-only recall should be rejected",
    )


def test_recall_json(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("recall_json hello")

    output = capsys.readouterr().out

    require('"results"' in output, "Recall JSON output missing")
    require('"result"' in output, "Recall JSON result missing")


def test_recall_json_missing_argument(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("recall_json")

    require(
        "Usage: recall_json <query>" in capsys.readouterr().out,
        "Recall JSON usage missing",
    )


def test_store_many(monkeypatch, tmp_path, capsys):
    source = tmp_path / "store.json"
    source.write_text(
        json.dumps(["alpha", "beta"]),
        encoding="utf8",
    )

    shell = make_shell(monkeypatch)
    shell.onecmd(f"store_many {source}")

    require(
        last_memory_call() == (
            "remember_many",
            ["alpha", "beta"],
        ),
        "Store_many data incorrect",
    )

    require(
        "Stored 2 memories" in capsys.readouterr().out,
        "Store_many result missing",
    )


def test_store_many_missing_argument(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("store_many")

    require(
        "Usage: store-many <file>" in capsys.readouterr().out,
        "Store-many usage missing",
    )


def test_store_many_missing_file(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("store_many /does/not/exist.json")

    require(
        "not found" in capsys.readouterr().out,
        "Missing store_many file was not reported",
    )


def test_store_many_invalid_json(monkeypatch, tmp_path, capsys):
    source = tmp_path / "bad.json"
    source.write_text("{bad", encoding="utf8")

    shell = make_shell(monkeypatch)
    shell.onecmd(f"store_many {source}")

    require(
        "Invalid JSON" in capsys.readouterr().out,
        "Invalid store_many JSON was not reported",
    )


def test_store_many_requires_list(monkeypatch, tmp_path, capsys):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"text": "no"}), encoding="utf8")

    shell = make_shell(monkeypatch)
    shell.onecmd(f"store_many {source}")

    require(
        "JSON must contain a list" in capsys.readouterr().out,
        "Store_many did not reject non-list JSON",
    )


def test_set_goal(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("set_goal build started")

    require(
        last_memory_call() == (
            "set_goal",
            "build",
            "started",
        ),
        "Set_goal arguments incorrect",
    )

    require(
        "Goal ID: 7" in capsys.readouterr().out,
        "Goal ID missing",
    )


def test_set_goal_default_progress(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("set_goal build")

    require(
        last_memory_call() == (
            "set_goal",
            "build",
            "started",
        ),
        "Default goal progress incorrect",
    )


def test_set_goal_missing(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("set_goal")

    require(
        "Usage: set-goal <goal> [progress]" in capsys.readouterr().out,
        "Set-goal usage missing",
    )


def test_update_goal(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd(
        "update_goal 7 --progress halfway --status active"
    )

    require(
        last_memory_call() == (
            "update_goal",
            7,
            "halfway",
            "active",
        ),
        "Update_goal arguments incorrect",
    )

    require(
        "Goal 7 updated." in capsys.readouterr().out,
        "Goal update result missing",
    )


def test_update_goal_progress_only(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("update_goal 7 --progress halfway")

    require(
        last_memory_call() == (
            "update_goal",
            7,
            "halfway",
            None,
        ),
        "Progress-only update incorrect",
    )


def test_update_goal_status_only(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("update_goal 7 --status done")

    require(
        last_memory_call() == (
            "update_goal",
            7,
            None,
            "done",
        ),
        "Status-only update incorrect",
    )


def test_update_goal_requires_change(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("update_goal 7")

    require(
        "At least one" in capsys.readouterr().out,
        "Update_goal accepted no changes",
    )


def test_update_goal_invalid_id(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("update_goal nope --status done")

    require(
        "Goal ID must be a number" in capsys.readouterr().out,
        "Invalid goal ID not reported",
    )


def test_list_goals(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("list_goals")

    output = capsys.readouterr().out

    require("Test goal" in output, "Goal missing")
    require("active" in output, "Goal status missing")


def test_list_goals_status(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("list_goals --status completed")

    require(
        last_memory_call() == (
            "list_goals",
            "completed",
        ),
        "Goal status filter incorrect",
    )


def test_stats(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("stats")

    require(
        "Total memories: 42" in capsys.readouterr().out,
        "Stats count missing",
    )


def test_info(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("info")

    output = capsys.readouterr().out

    require("Memoria v1.0" in output, "Version missing")
    require("Database:" in output, "Database missing")
    require("Total memories: 42" in output, "Memory count missing")
    require("Embedding model:" in output, "Embedding model missing")
    require("LLM URL:" in output, "LLM URL missing")
    require("Top K:" in output, "Top K missing")
    require("Debug mode:" in output, "Debug mode missing")


def test_graph(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("graph Alice 2")

    graph = shell.mem.controller.system.graph_search

    require(
        ("find_entity", "Alice") in graph.calls,
        "Graph entity lookup missing",
    )

    require(
        ("neighbors", "Alice", 2) in graph.calls,
        "Graph depth incorrect",
    )

    output = capsys.readouterr().out

    require("Neighbors of 'Alice' (depth 2)" in output,
            "Graph header missing")
    require("Bob" in output, "Graph neighbor missing")


def test_graph_default_depth(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("graph Alice")

    graph = shell.mem.controller.system.graph_search

    require(
        ("neighbors", "Alice", 1) in graph.calls,
        "Graph default depth incorrect",
    )


def test_graph_missing_argument(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("graph")

    require(
        "Usage: graph <entity> [depth]" in capsys.readouterr().out,
        "Graph usage missing",
    )


def test_graph_missing_entity(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("graph missing")

    require(
        "not found" in capsys.readouterr().out,
        "Missing graph entity not reported",
    )


def test_graph_invalid_depth(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("graph Alice nope")

    output = capsys.readouterr().out

    require(
        "Error:" in output,
        "Invalid graph depth was not handled",
    )


def test_doctor(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("doctor")

    output = capsys.readouterr().out

    require("Running integrity checks" in output,
            "Doctor did not start")
    require("Integrity check: OK" in output,
            "Integrity result missing")
    require("DB count: 42" in output,
            "Doctor DB count missing")
    require("Columns:" in output,
            "Doctor columns missing")


def test_export(monkeypatch, tmp_path, capsys):
    target = tmp_path / "export.json"

    shell = make_shell(monkeypatch)
    shell.onecmd(f"export {target}")

    data = json.loads(target.read_text(encoding="utf8"))

    require(
        len(data) == 2,
        "Exported memory count incorrect",
    )
    require(
        data[0]["text"] == "memory one",
        "Exported memory data incorrect",
    )

    require(
        "Exported 2 memories" in capsys.readouterr().out,
        "Export result missing",
    )


def test_export_missing_argument(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("export")

    require(
        "Usage: export <file>" in capsys.readouterr().out,
        "Export usage missing",
    )


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def test_import_text_and_normalized_text(monkeypatch, tmp_path, capsys):
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

    shell = make_shell(monkeypatch)
    shell.onecmd(f"import {source}")

    require(
        last_memory_call() == (
            "remember_many",
            ["alpha", "beta"],
        ),
        "Import did not extract text and normalized_text",
    )

    require(
        "Imported 2 memories" in capsys.readouterr().out,
        "Import result missing",
    )


def test_import_missing_argument(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("import")

    require(
        "Usage: import <file>" in capsys.readouterr().out,
        "Import usage missing",
    )


def test_import_missing_file(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("import /does/not/exist.json")

    require(
        "not found" in capsys.readouterr().out,
        "Missing import file not reported",
    )


def test_import_invalid_json(monkeypatch, tmp_path, capsys):
    source = tmp_path / "bad.json"
    source.write_text("{bad", encoding="utf8")

    shell = make_shell(monkeypatch)
    shell.onecmd(f"import {source}")

    require(
        "Invalid JSON" in capsys.readouterr().out,
        "Invalid import JSON not reported",
    )


def test_import_requires_list(monkeypatch, tmp_path, capsys):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"text": "bad"}), encoding="utf8")

    shell = make_shell(monkeypatch)
    shell.onecmd(f"import {source}")

    require(
        "JSON must contain a list" in capsys.readouterr().out,
        "Import accepted non-list JSON",
    )


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


def test_signals(monkeypatch, capsys):
    if not tui.HAS_SIGNAL_REGISTRY:
        pytest.skip("Signal registry unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("signals general")

    require(
        ("get_active_signals", "general")
        in shell.signal_router.calls,
        "Signal lookup incorrect",
    )

    output = capsys.readouterr().out

    require("Signals for type: general" in output,
            "Signal header missing")
    require("semantic" in output,
            "Signal missing")


def test_signals_default_type(monkeypatch, capsys):
    if not tui.HAS_SIGNAL_REGISTRY:
        pytest.skip("Signal registry unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("signals")

    require(
        ("get_active_signals", "general")
        in shell.signal_router.calls,
        "Default signal type incorrect",
    )


def test_signal_toggle_enable(monkeypatch, capsys):
    if not tui.HAS_SIGNAL_REGISTRY:
        pytest.skip("Signal registry unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("signal_toggle lexical")

    registry = shell.registry

    require(
        ("enable", "lexical") in registry.calls,
        "Toggle did not enable disabled signal",
    )
    require(
        ("clear_cache",) in shell.signal_router.calls,
        "Toggle did not clear router cache",
    )


def test_signal_toggle_disable(monkeypatch, capsys):
    if not tui.HAS_SIGNAL_REGISTRY:
        pytest.skip("Signal registry unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("signal_toggle semantic")

    registry = shell.registry

    require(
        ("disable", "semantic") in registry.calls,
        "Toggle did not disable enabled signal",
    )


def test_signal_toggle_missing_argument(monkeypatch, capsys):
    if not tui.HAS_SIGNAL_REGISTRY:
        pytest.skip("Signal registry unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("signal_toggle")

    require(
        "Usage: signal_toggle" in capsys.readouterr().out,
        "Signal toggle usage missing",
    )


def test_signal_enable(monkeypatch, capsys):
    if not tui.HAS_SIGNAL_REGISTRY:
        pytest.skip("Signal registry unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("signal_enable semantic")

    require(
        ("enable", "semantic") in shell.registry.calls,
        "Signal enable missing",
    )
    require(
        ("clear_cache",) in shell.signal_router.calls,
        "Signal enable did not clear cache",
    )


def test_signal_disable(monkeypatch, capsys):
    if not tui.HAS_SIGNAL_REGISTRY:
        pytest.skip("Signal registry unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("signal_disable semantic")

    require(
        ("disable", "semantic") in shell.registry.calls,
        "Signal disable missing",
    )


def test_signal_reset(monkeypatch, capsys):
    if not tui.HAS_SIGNAL_REGISTRY:
        pytest.skip("Signal registry unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("signal_reset")

    require(
        ("reload",) in shell.registry.calls,
        "Signal reset did not reload registry",
    )
    require(
        ("clear_cache",) in shell.signal_router.calls,
        "Signal reset did not clear cache",
    )


def test_signal_enable_missing_argument(monkeypatch, capsys):
    if not tui.HAS_SIGNAL_REGISTRY:
        pytest.skip("Signal registry unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("signal_enable")

    require(
        "Usage: signal_enable" in capsys.readouterr().out,
        "Signal enable usage missing",
    )


def test_signal_disable_missing_argument(monkeypatch, capsys):
    if not tui.HAS_SIGNAL_REGISTRY:
        pytest.skip("Signal registry unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("signal_disable")

    require(
        "Usage: signal_disable" in capsys.readouterr().out,
        "Signal disable usage missing",
    )


def test_signal_reset_without_registry(monkeypatch, capsys):
    monkeypatch.setattr(tui, "HAS_SIGNAL_REGISTRY", False)

    shell = make_shell(monkeypatch)

    shell.onecmd("signal_reset")

    require(
        "Signal registry not available" in capsys.readouterr().out,
        "Missing signal registry branch failed",
    )


# ---------------------------------------------------------------------------
# Query history
# ---------------------------------------------------------------------------


def test_history_recent(monkeypatch, capsys):
    if not tui.HAS_QUERY_HISTORY:
        pytest.skip("Query history unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("history")

    require(
        ("get_recent", 10) in shell.history.calls,
        "History recent lookup incorrect",
    )

    output = capsys.readouterr().out

    require(
        "hello world" in output,
        "Recent history missing",
    )


def test_history_recent_explicit_limit(monkeypatch, capsys):
    if not tui.HAS_QUERY_HISTORY:
        pytest.skip("Query history unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("history recent 1")

    require(
        ("get_recent", 1) in shell.history.calls,
        "History recent limit incorrect",
    )


def test_history_recent_invalid_limit(monkeypatch, capsys):
    if not tui.HAS_QUERY_HISTORY:
        pytest.skip("Query history unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("history recent nope")

    require(
        "limit must be a number" in capsys.readouterr().out,
        "Invalid history limit not reported",
    )


def test_history_recent_nonpositive_limit(monkeypatch, capsys):
    if not tui.HAS_QUERY_HISTORY:
        pytest.skip("Query history unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("history recent 0")

    require(
        "limit must be greater than 0" in capsys.readouterr().out,
        "Nonpositive history limit not reported",
    )


def test_history_frequent(monkeypatch, capsys):
    if not tui.HAS_QUERY_HISTORY:
        pytest.skip("Query history unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("history frequent")

    require(
        ("get_frequent", 10) in shell.history.calls,
        "History frequent lookup incorrect",
    )

    output = capsys.readouterr().out

    require(
        "what happened" in output,
        "Frequent query missing",
    )


def test_history_frequent_explicit_limit(monkeypatch, capsys):
    if not tui.HAS_QUERY_HISTORY:
        pytest.skip("Query history unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("history frequent 1")

    require(
        ("get_frequent", 1) in shell.history.calls,
        "History frequent limit incorrect",
    )


def test_history_frequent_invalid_limit(monkeypatch, capsys):
    if not tui.HAS_QUERY_HISTORY:
        pytest.skip("Query history unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("history frequent nope")

    require(
        "limit must be a number" in capsys.readouterr().out,
        "Invalid frequent limit not reported",
    )


def test_history_context(monkeypatch, capsys):
    if not tui.HAS_QUERY_HISTORY:
        pytest.skip("Query history unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("history context")

    require(
        ("get_context",) in shell.history.calls,
        "History context lookup incorrect",
    )

    output = capsys.readouterr().out

    require(
        "hello world" in output,
        "History context missing",
    )


def test_history_previous(monkeypatch, capsys):
    if not tui.HAS_QUERY_HISTORY:
        pytest.skip("Query history unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("history previous")

    require(
        ("get_previous_query",) in shell.history.calls,
        "Previous query lookup incorrect",
    )

    require(
        "hello world" in capsys.readouterr().out,
        "Previous query missing",
    )


def test_history_unknown_subcommand(monkeypatch, capsys):
    if not tui.HAS_QUERY_HISTORY:
        pytest.skip("Query history unavailable")

    shell = make_shell(monkeypatch)

    shell.onecmd("history nope")

    output = capsys.readouterr().out

    require(
        "Unknown history subcommand" in output,
        "Unknown history command not reported",
    )

    require(
        "frequent" in output,
        "History command list missing",
    )


def test_history_missing_dependency(monkeypatch, capsys):
    monkeypatch.setattr(
        tui,
        "HAS_QUERY_HISTORY",
        False,
    )

    shell = make_shell(monkeypatch)

    shell.onecmd("history")

    require(
        "Query history not available" in capsys.readouterr().out,
        "Missing query history branch failed",
    )
# ---------------------------------------------------------------------------
# Auto-store
# ---------------------------------------------------------------------------


def test_autostore_status(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("autostore status")

    require(
        "Auto-store:" in capsys.readouterr().out,
        "Auto-store status missing",
    )


def test_autostore_on(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    old = tui.settings.AUTO_STORE_MEMORIES

    try:
        shell.onecmd("autostore on")

        require(
            tui.settings.AUTO_STORE_MEMORIES is True,
            "Auto-store was not enabled",
        )
    finally:
        tui.settings.AUTO_STORE_MEMORIES = old

    require(
        "enabled" in capsys.readouterr().out,
        "Auto-store enable output missing",
    )


def test_autostore_off(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    old = tui.settings.AUTO_STORE_MEMORIES

    try:
        shell.onecmd("autostore off")

        require(
            tui.settings.AUTO_STORE_MEMORIES is False,
            "Auto-store was not disabled",
        )
    finally:
        tui.settings.AUTO_STORE_MEMORIES = old

    require(
        "disabled" in capsys.readouterr().out,
        "Auto-store disable output missing",
    )


def test_autostore_threshold_valid(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    old = tui.settings.AUTO_STORE_THRESHOLD

    try:
        shell.onecmd("autostore threshold 0.75")

        require(
            tui.settings.AUTO_STORE_THRESHOLD == 0.75,
            "Auto-store threshold not changed",
        )
    finally:
        tui.settings.AUTO_STORE_THRESHOLD = old

    require(
        "0.75" in capsys.readouterr().out,
        "Threshold output missing",
    )


def test_autostore_threshold_invalid_number(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("autostore threshold nope")

    require(
        "must be a number" in capsys.readouterr().out,
        "Invalid threshold not reported",
    )


def test_autostore_threshold_out_of_range(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("autostore threshold 2")

    require(
        "between 0.0 and 1.0" in capsys.readouterr().out,
        "Out-of-range threshold not reported",
    )


def test_autostore_max_valid(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    old = tui.settings.AUTO_STORE_MAX_PER_SESSION

    try:
        shell.onecmd("autostore max 12")

        require(
            tui.settings.AUTO_STORE_MAX_PER_SESSION == 12,
            "Auto-store max not changed",
        )
    finally:
        tui.settings.AUTO_STORE_MAX_PER_SESSION = old

    require(
        "12" in capsys.readouterr().out,
        "Auto-store max output missing",
    )


def test_autostore_max_invalid_number(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("autostore max nope")

    require(
        "must be a number" in capsys.readouterr().out,
        "Invalid max not reported",
    )


def test_autostore_max_nonpositive(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("autostore max 0")

    require(
        "must be > 0" in capsys.readouterr().out,
        "Nonpositive max not reported",
    )


def test_autostore_types(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    old = list(tui.settings.AUTO_STORE_TYPES)

    try:
        shell.onecmd("autostore types fact,preference,task")

        require(
            tui.settings.AUTO_STORE_TYPES
            == ["fact", "preference", "task"],
            "Auto-store types incorrect",
        )
    finally:
        tui.settings.AUTO_STORE_TYPES = old

    require(
        "fact, preference, task" in capsys.readouterr().out,
        "Auto-store types output missing",
    )


def test_autostore_unknown(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("autostore nope")

    require(
        "Unknown autostore subcommand" in capsys.readouterr().out,
        "Unknown autostore command not reported",
    )


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------


def test_enter_chat_mode(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.enter_chat_mode()

    require(shell.in_chat_mode is True,
            "Chat mode was not entered")
    require(shell.prompt == shell.chat_prompt,
            "Chat prompt not changed")

    output = capsys.readouterr().out

    require("Entering chat mode" in output,
            "Chat entry message missing")


def test_enter_chat_mode_with_initial_prompt(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.enter_chat_mode("hello")

    require(
        last_memory_call() == (
            "chat",
            "hello",
            None,
        ),
        "Initial chat prompt not sent",
    )

    require(
        shell.chat_history == [
            {
                "user": "hello",
                "assistant": "response:hello",
            }
        ],
        "Initial chat history not stored",
    )


def test_exit_chat_mode(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.enter_chat_mode()
    shell.exit_chat_mode()

    require(shell.in_chat_mode is False,
            "Chat mode did not exit")
    require(shell.prompt == "Memory> ",
            "Main prompt not restored")

    require(
        "Exited chat mode" in capsys.readouterr().out,
        "Chat exit output missing",
    )


def test_do_chat_without_argument(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("chat")

    require(shell.in_chat_mode is True,
            "Chat command did not enter chat mode")


def test_do_chat_with_argument(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("chat hello")

    require(
        last_memory_call() == (
            "chat",
            "hello",
            None,
        ),
        "Chat command did not send initial prompt",
    )


def test_back_outside_chat(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("back")

    require(
        "Not in chat mode" in capsys.readouterr().out,
        "Back outside chat not handled",
    )


def test_chat_history_command(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.enter_chat_mode()
    shell.onecmd("hello")
    shell.onecmd(".history")

    output = capsys.readouterr().out

    require("User: hello" in output,
            "Chat history user entry missing")
    require("Assistant: response:hello" in output,
            "Chat history assistant entry missing")


def test_chat_clear(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.enter_chat_mode("hello")
    shell.onecmd(".clear")

    require(
        shell.chat_history == [],
        "Chat history was not cleared",
    )

    require(
        "Chat history cleared" in capsys.readouterr().out,
        "Chat clear output missing",
    )


def test_chat_help(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.enter_chat_mode()
    shell.onecmd(".help")

    output = capsys.readouterr().out

    require(".back" in output, "Chat help missing .back")
    require(".history" in output, "Chat help missing .history")
    require(".auto-on" in output, "Chat help missing .auto-on")


def test_chat_info(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.enter_chat_mode("hello")
    shell.onecmd(".info")

    require(
        "Chat history: 1 messages." in capsys.readouterr().out,
        "Chat info output incorrect",
    )


def test_chat_auto_on(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.enter_chat_mode()
    shell.onecmd(".auto-on")

    require(
        shell.auto_store_override is True,
        "Chat auto-on did not set override",
    )


def test_chat_auto_off(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.enter_chat_mode()
    shell.onecmd(".auto-off")

    require(
        shell.auto_store_override is False,
        "Chat auto-off did not set override",
    )


def test_chat_auto_status(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.enter_chat_mode()
    shell.onecmd(".auto-status")

    output = capsys.readouterr().out

    require("Auto-store:" in output,
            "Chat auto-status missing status")
    require("Threshold:" in output,
            "Chat auto-status missing threshold")
    require("Max per session:" in output,
            "Chat auto-status missing max")


def test_chat_override_passed_to_memory(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.enter_chat_mode()
    shell.onecmd(".auto-on")
    shell.onecmd("hello")

    require(
        last_memory_call() == (
            "chat",
            "hello",
            True,
        ),
        "Chat auto-store override was not passed",
    )


def test_chat_empty_response(monkeypatch, capsys):
    class EmptyChatInterface(FakeMemoryInterface):
        def chat(self, prompt, auto_store=None):
            self.calls.append(
                ("chat", prompt, auto_store)
            )
            return ""

    monkeypatch.setattr(
        tui,
        "MemoryInterface",
        EmptyChatInterface,
    )

    shell = tui.MemoryShell()
    shell.onecmd("chat hello")

    require(
        "No response from assistant" in capsys.readouterr().out,
        "Empty chat response not handled",
    )


def test_chat_error(monkeypatch, capsys):
    class BrokenChatInterface(FakeMemoryInterface):
        def chat(self, prompt, auto_store=None):
            raise RuntimeError("chat failed")

    monkeypatch.setattr(
        tui,
        "MemoryInterface",
        BrokenChatInterface,
    )

    shell = tui.MemoryShell()
    shell.onecmd("chat hello")

    require(
        "[Error: chat failed]" in capsys.readouterr().out,
        "Chat error not handled",
    )


def test_chat_back_and_exit(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.enter_chat_mode()

    result = shell.onecmd(".back")

    require(result is None,
            "Chat .back should continue shell")
    require(shell.in_chat_mode is False,
            ".back did not exit chat mode")

    shell.enter_chat_mode()

    result = shell.onecmd(".exit")

    require(result is None,
            "Chat .exit should return to shell")
    require(shell.in_chat_mode is False,
            ".exit did not exit chat mode")


def test_quit(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    result = shell.onecmd("quit")

    require(result is True,
            "quit did not stop shell")

    require(
        "Goodbye." in capsys.readouterr().out,
        "Quit output missing",
    )


def test_q_alias(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    result = shell.onecmd("q")

    require(result is True,
            "q did not stop shell")


def test_quit_from_chat(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.enter_chat_mode()
    result = shell.onecmd("quit")

    require(result is True,
            "quit from chat did not stop shell")
    require(shell.in_chat_mode is False,
            "quit from chat did not exit chat mode")


def test_h_alias(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("h")

    output = capsys.readouterr().out

    require(
        "store" in output.lower(),
        "h alias did not invoke help",
    )


def test_unknown_command_main_mode(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("definitely_unknown")

    require(
        "Unknown command" in capsys.readouterr().out,
        "Unknown command was not reported",
    )


def test_unknown_command_chat_mode_becomes_message(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.enter_chat_mode()
    shell.onecmd("ordinary message")

    require(
        last_memory_call() == (
            "chat",
            "ordinary message",
            None,
        ),
        "Chat message was not routed to chat",
    )


def test_emptyline(monkeypatch):
    shell = make_shell(monkeypatch)

    require(
        shell.emptyline() is None,
        "emptyline should do nothing",
    )


# ---------------------------------------------------------------------------
# Error handling inside commands
# ---------------------------------------------------------------------------


def test_recall_error(monkeypatch, capsys):
    class BrokenInterface(FakeMemoryInterface):
        def recall(self, query):
            raise RuntimeError("recall failed")

    monkeypatch.setattr(tui, "MemoryInterface", BrokenInterface)

    shell = tui.MemoryShell()
    shell.onecmd("recall hello")

    require(
        "Error: recall failed" in capsys.readouterr().out,
        "Recall error not handled",
    )


def test_set_goal_error(monkeypatch, capsys):
    class BrokenInterface(FakeMemoryInterface):
        def set_goal(self, goal, progress):
            raise RuntimeError("goal failed")

    monkeypatch.setattr(tui, "MemoryInterface", BrokenInterface)

    shell = tui.MemoryShell()
    shell.onecmd("set_goal test")

    require(
        "Error: goal failed" in capsys.readouterr().out,
        "Set_goal error not handled",
    )


def test_update_goal_error(monkeypatch, capsys):
    class BrokenInterface(FakeMemoryInterface):
        def update_goal(self, goal_id, progress=None, status=None):
            raise RuntimeError("update failed")

    monkeypatch.setattr(tui, "MemoryInterface", BrokenInterface)

    shell = tui.MemoryShell()
    shell.onecmd("update_goal 7 --status done")

    require(
        "Error: update failed" in capsys.readouterr().out,
        "Update_goal error not handled",
    )


# ---------------------------------------------------------------------------
# Cmd framework / help
# ---------------------------------------------------------------------------


def test_help_lists_tui_commands(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("help")

    output = capsys.readouterr().out

    for command in (
        "store",
        "recall",
        "store_many",
        "set_goal",
        "update_goal",
        "list_goals",
        "stats",
        "info",
        "graph",
        "doctor",
        "export",
        "import",
        "signals",
        "signal_toggle",
        "signal_enable",
        "signal_disable",
        "signal_reset",
        "history",
        "autostore",
        "quit",
    ):
        require(
            command in output,
            f"Help output missing {command}",
        )


def test_help_specific_command(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("help store")

    output = capsys.readouterr().out

    require(
        "store" in output.lower(),
        "Specific command help missing",
    )


def test_question_mark_help(monkeypatch, capsys):
    shell = make_shell(monkeypatch)

    shell.onecmd("? store")

    output = capsys.readouterr().out

    require(
        "store" in output.lower(),
        "Question-mark help did not dispatch",
    )


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_runs_cmdloop(monkeypatch):
    calls = []

    class FakeShell:
        def __init__(self):
            calls.append(("init",))

        def cmdloop(self):
            calls.append(("cmdloop",))

    monkeypatch.setattr(tui, "MemoryShell", FakeShell)

    tui.main()

    require(
        calls == [("init",), ("cmdloop",)],
        "main did not construct and run MemoryShell",
    )


def test_main_handles_keyboard_interrupt(monkeypatch, capsys):
    class InterruptingShell:
        def __init__(self):
            pass

        def cmdloop(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        tui,
        "MemoryShell",
        InterruptingShell,
    )

    try:
        tui.main()
    except SystemExit as exc:
        require(
            exc.code == 0,
            "KeyboardInterrupt did not exit cleanly",
        )
    except KeyboardInterrupt as exc:
        raise RuntimeError(
            "TUI leaked KeyboardInterrupt from cmdloop"
        ) from exc

    require(
        "Goodbye." in capsys.readouterr().out,
        "KeyboardInterrupt goodbye message missing",
    )


def test_main_handles_constructor_exception(monkeypatch, capsys):
    class BrokenShell:
        def __init__(self):
            raise RuntimeError("shell exploded")

    monkeypatch.setattr(tui, "MemoryShell", BrokenShell)

    try:
        tui.main()
    except RuntimeError as exc:
        raise RuntimeError(
            "TUI leaked constructor exception"
        ) from exc
    except SystemExit as exc:
        require(
            exc.code != 0,
            "Constructor exception did not produce failure exit",
        )
    else:
        raise RuntimeError(
            "TUI did not report constructor failure"
        )


# ---------------------------------------------------------------------------
# Dependency-missing branches
# ---------------------------------------------------------------------------


def test_signals_missing_dependency(monkeypatch, capsys):
    monkeypatch.setattr(
        tui,
        "HAS_SIGNAL_REGISTRY",
        False,
    )

    shell = make_shell(monkeypatch)

    shell.onecmd("signals")

    require(
        "Signal registry not available" in capsys.readouterr().out,
        "Missing signal registry branch failed",
    )


def test_signal_toggle_missing_dependency(monkeypatch, capsys):
    monkeypatch.setattr(
        tui,
        "HAS_SIGNAL_REGISTRY",
        False,
    )

    shell = make_shell(monkeypatch)

    shell.onecmd("signal_toggle semantic")

    require(
        "Signal registry not available" in capsys.readouterr().out,
        "Signal toggle dependency branch failed",
    )


def test_history_missing_dependency(monkeypatch, capsys):
    monkeypatch.setattr(
        tui,
        "HAS_QUERY_HISTORY",
        False,
    )

    shell = make_shell(monkeypatch)

    shell.onecmd("history")

    require(
        "Query history not available" in capsys.readouterr().out,
        "Missing query history branch failed",
    )
