import asyncio
import inspect

import pytest

import memoria_mcp.server as mcp_server


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class FakeController:
    def __init__(self):
        self.calls = []

    def recall(self, query):
        self.calls.append(("recall", query))
        return {
            "results": [{"id": 1, "text": "found memory"}],
            "diagnostics": {"source": "fake"},
        }

    def remember(self, text, metadata=None):
        self.calls.append(("remember", text, metadata))
        return 101

    def remember_many(self, texts, metadatas=None):
        self.calls.append(("remember_many", texts, metadatas))
        return [201 + index for index in range(len(texts))]

    def fetch(self, mem_id):
        self.calls.append(("fetch", mem_id))
        return {
            "id": mem_id,
            "text": "fetched memory",
        }

    def update(self, mem_id, **fields):
        self.calls.append(("update", mem_id, fields))

    def delete(self, mem_id):
        self.calls.append(("delete", mem_id))


@pytest.fixture
def controller(monkeypatch):
    fake = FakeController()
    monkeypatch.setattr(mcp_server, "_controller", fake)
    return fake


@pytest.fixture(autouse=True)
def reset_controller():
    original = mcp_server._controller
    mcp_server._controller = None

    yield

    mcp_server._controller = original


def run_async(coro):
    return asyncio.run(coro)


# --------------------------------------------------
# Server registration
# --------------------------------------------------


def test_mcp_server_name():
    require(mcp_server.mcp.name == "Memoria", "MCP server name is incorrect")


def test_mcp_registers_expected_tools():
    tools = run_async(mcp_server.mcp.list_tools())
    names = {tool.name for tool in tools}

    expected = {
        "memory_search",
        "memory_store",
        "memory_store_many",
        "memory_fetch",
        "memory_update",
        "memory_delete",
    }

    require(names == expected, f"Unexpected MCP tools: {sorted(names)}")


def test_mcp_tool_count():
    tools = run_async(mcp_server.mcp.list_tools())
    require(len(tools) == 6, f"Expected 6 MCP tools, got {len(tools)}")


@pytest.mark.parametrize(
    "tool_name",
    [
        "memory_search",
        "memory_store",
        "memory_store_many",
        "memory_fetch",
        "memory_update",
        "memory_delete",
    ],
)
def test_mcp_tool_has_description(tool_name):
    tools = run_async(mcp_server.mcp.list_tools())
    tool = next((item for item in tools if item.name == tool_name), None)

    require(tool is not None, f"Tool not registered: {tool_name}")
    require(
        bool(tool.description),
        f"Tool has no description: {tool_name}",
    )


# --------------------------------------------------
# Lazy controller
# --------------------------------------------------


def test_controller_is_created_lazily(monkeypatch):
    created = []

    class StartupController:
        def __init__(self):
            created.append(self)

    monkeypatch.setattr(mcp_server, "MemoryController", StartupController)
    mcp_server._controller = None

    require(len(created) == 0, "Controller was created before first use")

    controller = mcp_server._get_controller()

    require(len(created) == 1, "Controller was not created lazily")
    require(controller is created[0], "Wrong controller returned")


def test_controller_is_reused(monkeypatch):
    created = []

    class StartupController:
        def __init__(self):
            created.append(self)

    monkeypatch.setattr(mcp_server, "MemoryController", StartupController)
    mcp_server._controller = None

    first = mcp_server._get_controller()
    second = mcp_server._get_controller()

    require(len(created) == 1, f"Expected one controller, got {len(created)}")
    require(first is second, "Controller was recreated instead of reused")


# --------------------------------------------------
# Direct tool behavior
# --------------------------------------------------


def test_memory_search(controller):
    result = mcp_server.memory_search("what did I build?")

    require(
        result == {
            "results": [{"id": 1, "text": "found memory"}],
            "diagnostics": {"source": "fake"},
        },
        f"Unexpected search result: {result}",
    )

    require(
        controller.calls == [("recall", "what did I build?")],
        f"Unexpected controller calls: {controller.calls}",
    )


def test_memory_store(controller):
    result = mcp_server.memory_store(
        "test memory",
        metadata={"source": "test"},
    )

    require(
        result == {
            "id": 101,
            "text": "test memory",
        },
        f"Unexpected store result: {result}",
    )

    require(
        controller.calls == [
            ("remember", "test memory", {"source": "test"})
        ],
        f"Unexpected controller calls: {controller.calls}",
    )


def test_memory_store_without_metadata(controller):
    result = mcp_server.memory_store("plain memory")

    require(result["id"] == 101, f"Unexpected ID: {result}")
    require(result["text"] == "plain memory", f"Unexpected text: {result}")

    require(
        controller.calls == [("remember", "plain memory", None)],
        f"Unexpected controller calls: {controller.calls}",
    )


def test_memory_store_many(controller):
    texts = ["first", "second", "third"]
    metadatas = [
        {"source": "a"},
        {"source": "b"},
        {"source": "c"},
    ]

    result = mcp_server.memory_store_many(texts, metadatas)

    require(
        result == {
            "ids": [201, 202, 203],
            "count": 3,
        },
        f"Unexpected batch result: {result}",
    )

    require(
        controller.calls == [
            ("remember_many", texts, metadatas)
        ],
        f"Unexpected controller calls: {controller.calls}",
    )


def test_memory_store_many_without_metadata(controller):
    result = mcp_server.memory_store_many(["one", "two"])

    require(
        result == {
            "ids": [201, 202],
            "count": 2,
        },
        f"Unexpected batch result: {result}",
    )

    require(
        controller.calls == [
            ("remember_many", ["one", "two"], None)
        ],
        f"Unexpected controller calls: {controller.calls}",
    )


def test_memory_store_many_empty(controller):
    result = mcp_server.memory_store_many([])

    require(
        result == {
            "ids": [],
            "count": 0,
        },
        f"Unexpected empty batch result: {result}",
    )

    require(
        controller.calls == [
            ("remember_many", [], None)
        ],
        f"Unexpected controller calls: {controller.calls}",
    )


def test_memory_fetch(controller):
    result = mcp_server.memory_fetch(42)

    require(
        result == {
            "id": 42,
            "text": "fetched memory",
        },
        f"Unexpected fetch result: {result}",
    )

    require(
        controller.calls == [("fetch", 42)],
        f"Unexpected controller calls: {controller.calls}",
    )


def test_memory_fetch_missing(controller):
    controller.fetch = lambda mem_id: None

    result = mcp_server.memory_fetch(999)

    require(result is None, f"Expected None, got {result}")


# --------------------------------------------------
# Update behavior
# --------------------------------------------------


def test_memory_update_without_fields(controller):
    result = mcp_server.memory_update(42)

    require(
        result == {
            "id": 42,
            "updated": False,
            "reason": "No fields supplied",
        },
        f"Unexpected no-op update result: {result}",
    )

    require(
        controller.calls == [],
        f"Controller was called for no-op update: {controller.calls}",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", "new text"),
        ("normalized_text", "normalized"),
        ("tokens", ["new", "tokens"]),
        ("token_count", 2),
        ("importance", 0.75),
        ("memory_type", "fact"),
        ("metadata", {"source": "test"}),
        ("entities", ["Alice"]),
        ("relationships", [{"type": "knows"}]),
        ("last_accessed", "2026-10-05T12:00:00"),
    ],
)
def test_memory_update_single_field(controller, field, value):
    result = mcp_server.memory_update(
        42,
        **{field: value},
    )

    require(
        result == {
            "id": 42,
            "updated": True,
            "fields": [field],
        },
        f"Unexpected update result for {field}: {result}",
    )

    require(
        controller.calls == [
            ("update", 42, {field: value})
        ],
        f"Unexpected controller call for {field}: {controller.calls}",
    )


def test_memory_update_multiple_fields(controller):
    result = mcp_server.memory_update(
        42,
        text="updated",
        importance=0.8,
        memory_type="fact",
        metadata={"source": "test"},
    )

    require(
        result == {
            "id": 42,
            "updated": True,
            "fields": [
                "text",
                "importance",
                "memory_type",
                "metadata",
            ],
        },
        f"Unexpected update result: {result}",
    )

    require(
        controller.calls == [
            (
                "update",
                42,
                {
                    "text": "updated",
                    "importance": 0.8,
                    "memory_type": "fact",
                    "metadata": {"source": "test"},
                },
            )
        ],
        f"Unexpected controller call: {controller.calls}",
    )


def test_memory_update_all_fields(controller):
    fields = {
        "text": "text",
        "normalized_text": "normalized",
        "tokens": ["a", "b"],
        "token_count": 2,
        "importance": 0.5,
        "memory_type": "fact",
        "metadata": {"source": "test"},
        "entities": ["entity"],
        "relationships": ["relationship"],
        "last_accessed": "2026-10-05T12:00:00",
    }

    result = mcp_server.memory_update(42, **fields)

    require(result["id"] == 42, f"Unexpected ID: {result}")
    require(result["updated"] is True, f"Update did not succeed: {result}")
    require(
        result["fields"] == list(fields.keys()),
        f"Unexpected fields: {result}",
    )

    require(
        controller.calls == [("update", 42, fields)],
        f"Unexpected controller call: {controller.calls}",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", ""),
        ("normalized_text", ""),
        ("tokens", []),
        ("token_count", 0),
        ("importance", 0.0),
        ("memory_type", ""),
        ("metadata", {}),
        ("entities", []),
        ("relationships", []),
        ("last_accessed", ""),
    ],
)
def test_memory_update_preserves_falsy_values(controller, field, value):
    result = mcp_server.memory_update(
        42,
        **{field: value},
    )

    require(
        result["updated"] is True,
        f"Falsy value was discarded for {field}: {result}",
    )

    require(
        controller.calls == [
            ("update", 42, {field: value})
        ],
        f"Falsy value was not forwarded for {field}: {controller.calls}",
    )


# --------------------------------------------------
# Delete
# --------------------------------------------------


def test_memory_delete(controller):
    result = mcp_server.memory_delete(42)

    require(
        result == {
            "id": 42,
            "deleted": True,
        },
        f"Unexpected delete result: {result}",
    )

    require(
        controller.calls == [("delete", 42)],
        f"Unexpected controller calls: {controller.calls}",
    )


# --------------------------------------------------
# MCP tool invocation through the server
# --------------------------------------------------


@pytest.mark.parametrize(
    "tool_name,arguments",
    [
        ("memory_search", {"query": "find this"}),
        ("memory_store", {"text": "store this"}),
        ("memory_store_many", {"texts": ["one", "two"]}),
        ("memory_fetch", {"mem_id": 7}),
        ("memory_update", {"mem_id": 7, "text": "updated"}),
        ("memory_delete", {"mem_id": 7}),
    ],
)
def test_registered_tools_are_callable(tool_name, arguments, controller):
    async def invoke():
        return await mcp_server.mcp.call_tool(tool_name, arguments)

    result = run_async(invoke())

    require(
        result is not None,
        f"MCP tool returned no result: {tool_name}",
    )


def test_mcp_schema_memory_update_contains_all_fields():
    tools = run_async(mcp_server.mcp.list_tools())
    tool = next(
        item for item in tools
        if item.name == "memory_update"
    )

    properties = tool.input_schema.get("properties", {})

    expected = {
        "mem_id",
        "text",
        "normalized_text",
        "tokens",
        "token_count",
        "importance",
        "memory_type",
        "metadata",
        "entities",
        "relationships",
        "last_accessed",
    }

    require(
        set(properties) == expected,
        f"Unexpected memory_update schema fields: {set(properties)}",
    )


def test_mcp_schema_requires_memory_update_id():
    tools = run_async(mcp_server.mcp.list_tools())
    tool = next(
        item for item in tools
        if item.name == "memory_update"
    )

    required = tool.input_schema.get("required", [])

    require(
        "mem_id" in required,
        f"memory_update does not require mem_id: {required}",
    )


# --------------------------------------------------
# Error propagation
# --------------------------------------------------


def test_controller_error_propagates_from_direct_tool(monkeypatch):
    class FailingController:
        def recall(self, query):
            raise RuntimeError("controller failure")

    monkeypatch.setattr(mcp_server, "_controller", FailingController())

    try:
        mcp_server.memory_search("test")
    except RuntimeError as exc:
        require(
            str(exc) == "controller failure",
            f"Unexpected exception: {exc}",
        )
    else:
        raise RuntimeError("Controller exception was swallowed")


def test_mcp_wraps_tool_failure():
    class FailingController:
        def recall(self, query):
            raise RuntimeError("controller failure")

    mcp_server._controller = FailingController()

    async def invoke():
        return await mcp_server.mcp.call_tool(
            "memory_search",
            {"query": "test"},
        )

    try:
        result = run_async(invoke())
    except Exception as exc:
        text = str(exc)
        require(
            "memory_search" in text or "Error executing tool" in text,
            f"Unexpected MCP error: {exc}",
        )
        return

    if getattr(result, "is_error", False):
        return

    raise RuntimeError(
        f"MCP failure did not produce an error result: {result}"
    )


# --------------------------------------------------
# STDIO entry point
# --------------------------------------------------


def test_main_runs_mcp_server(monkeypatch):
    calls = []

    def fake_run():
        calls.append(True)

    monkeypatch.setattr(mcp_server.mcp, "run", fake_run)

    mcp_server.main()

    require(
        calls == [True],
        f"MCP main did not invoke mcp.run(): {calls}",
    )


def test_main_is_sync_function():
    require(
        not inspect.iscoroutinefunction(mcp_server.main),
        "MCP main unexpectedly became async",
    )
