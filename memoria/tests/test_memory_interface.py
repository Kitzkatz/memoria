from shared.memory_interface import MemoryInterface


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class FakeSystem:
    def __init__(self):
        self.calls = []

    def ingest_code(self, directory, max_files):
        self.calls.append(("ingest_code", directory, max_files))
        return {"ingested": directory, "max_files": max_files}

    def ingest_pdf(self, filepath, max_pages):
        self.calls.append(("ingest_pdf", filepath, max_pages))
        return {"ingested": filepath, "max_pages": max_pages}


class FakeController:
    def __init__(self):
        self.calls = []
        self.system = FakeSystem()
        self.plugin_manager = object()

    def remember(self, text):
        self.calls.append(("remember", text))
        return 42

    def remember_many(
        self,
        texts,
        metadatas=None,
        skip_embedding_build=False,
    ):
        self.calls.append(
            (
                "remember_many",
                texts,
                metadatas,
                skip_embedding_build,
            )
        )
        return [101 + index for index in range(len(texts))]

    def recall(self, query):
        self.calls.append(("recall", query))
        return {
            "results": [query],
            "diagnostics": {"source": "fake"},
        }

    def set_goal(self, goal, progress):
        self.calls.append(("set_goal", goal, progress))
        return 7

    def update_goal(self, goal_id, progress, status):
        self.calls.append(
            ("update_goal", goal_id, progress, status)
        )
        return "updated"

    def list_goals(self, status):
        self.calls.append(("list_goals", status))
        return [{"id": 1, "status": status}]

    def raw_chat(self, prompt):
        self.calls.append(("raw_chat", prompt))
        return "raw response"

    def chat(self, prompt):
        self.calls.append(("chat", prompt))
        return "chat response"

    def reflect(self):
        self.calls.append(("reflect",))
        return {"reflection": "ok"}

    def stats(self):
        self.calls.append(("stats",))
        return {"memories": 12}

    def raise_error(self, *args, **kwargs):
        raise RuntimeError("controller failure")


class InterfaceForTest(MemoryInterface):
    def __init__(self, controller):
        self.controller = controller


def make_interface():
    controller = FakeController()
    interface = InterfaceForTest(controller)
    return interface, controller


def test_initialization_creates_controller():
    interface = MemoryInterface()

    require(
        interface.controller is not None,
        "MemoryInterface did not create a controller",
    )


def test_plugin_manager_exposes_controller_plugin_manager():
    interface, controller = make_interface()

    require(
        interface.plugin_manager is controller.plugin_manager,
        "plugin_manager did not expose controller plugin manager",
    )


def test_plugin_manager_returns_none_when_controller_has_none():
    interface, controller = make_interface()
    controller.plugin_manager = None

    require(
        interface.plugin_manager is None,
        "plugin_manager should return None when unavailable",
    )


def test_remember_delegates_and_returns_id():
    interface, controller = make_interface()

    result = interface.remember("hello memory")

    require(result == 42, f"Unexpected remember result: {result}")
    require(
        controller.calls == [("remember", "hello memory")],
        f"Unexpected remember calls: {controller.calls}",
    )


def test_remember_many_delegates_without_metadata():
    interface, controller = make_interface()

    texts = ["one", "two", "three"]

    result = interface.remember_many(texts)

    require(
        result == [101, 102, 103],
        f"Unexpected remember_many result: {result}",
    )
    require(
        controller.calls == [
            ("remember_many", texts, None, False)
        ],
        f"Unexpected remember_many calls: {controller.calls}",
    )


def test_remember_many_delegates_with_metadata():
    interface, controller = make_interface()

    texts = ["one", "two"]
    metadatas = [
        {"type": "a"},
        {"type": "b"},
    ]

    result = interface.remember_many(
        texts,
        metadatas=metadatas,
    )

    require(
        result == [101, 102],
        f"Unexpected remember_many result: {result}",
    )
    require(
        controller.calls == [
            ("remember_many", texts, metadatas, False)
        ],
        f"Unexpected remember_many calls: {controller.calls}",
    )


def test_remember_many_preserves_skip_embedding_build():
    interface, controller = make_interface()

    texts = ["one"]

    interface.remember_many(
        texts,
        skip_embedding_build=True,
    )

    require(
        controller.calls == [
            ("remember_many", texts, None, True)
        ],
        f"skip_embedding_build was not preserved: {controller.calls}",
    )


def test_store_many_delegates_through_remember_many():
    interface, controller = make_interface()

    texts = ["one", "two"]
    metadatas = [{"x": 1}, {"x": 2}]

    result = interface.store_many(
        texts,
        metadatas=metadatas,
        skip_embedding_build=True,
    )

    require(
        result == [101, 102],
        f"Unexpected store_many result: {result}",
    )
    require(
        controller.calls == [
            ("remember_many", texts, metadatas, True)
        ],
        f"store_many did not delegate correctly: {controller.calls}",
    )


def test_recall_delegates_and_preserves_result():
    interface, controller = make_interface()

    result = interface.recall("what happened?")

    require(
        result == {
            "results": ["what happened?"],
            "diagnostics": {"source": "fake"},
        },
        f"Unexpected recall result: {result}",
    )
    require(
        controller.calls == [("recall", "what happened?")],
        f"Unexpected recall calls: {controller.calls}",
    )


def test_recall_many_queries_each_item_in_order():
    interface, controller = make_interface()

    queries = ["first", "second", "third"]

    result = interface.recall_many(queries)

    require(
        result == [
            {
                "results": ["first"],
                "diagnostics": {"source": "fake"},
            },
            {
                "results": ["second"],
                "diagnostics": {"source": "fake"},
            },
            {
                "results": ["third"],
                "diagnostics": {"source": "fake"},
            },
        ],
        f"Unexpected recall_many result: {result}",
    )

    require(
        controller.calls == [
            ("recall", "first"),
            ("recall", "second"),
            ("recall", "third"),
        ],
        f"Unexpected recall_many calls: {controller.calls}",
    )


def test_set_goal_delegates():
    interface, controller = make_interface()

    result = interface.set_goal(
        "finish release",
        progress="in progress",
    )

    require(result == 7, f"Unexpected goal ID: {result}")
    require(
        controller.calls == [
            ("set_goal", "finish release", "in progress")
        ],
        f"Unexpected set_goal calls: {controller.calls}",
    )


def test_set_goal_uses_default_progress():
    interface, controller = make_interface()

    interface.set_goal("finish release")

    require(
        controller.calls == [
            ("set_goal", "finish release", "started")
        ],
        f"Default progress was not preserved: {controller.calls}",
    )


def test_update_goal_delegates():
    interface, controller = make_interface()

    result = interface.update_goal(
        7,
        progress="complete",
        status="completed",
    )

    require(result == "updated", f"Unexpected update result: {result}")
    require(
        controller.calls == [
            ("update_goal", 7, "complete", "completed")
        ],
        f"Unexpected update_goal calls: {controller.calls}",
    )


def test_update_goal_preserves_none_values():
    interface, controller = make_interface()

    interface.update_goal(7)

    require(
        controller.calls == [
            ("update_goal", 7, None, None)
        ],
        f"None values were not preserved: {controller.calls}",
    )


def test_list_goals_delegates_with_status():
    interface, controller = make_interface()

    result = interface.list_goals("active")

    require(
        result == [{"id": 1, "status": "active"}],
        f"Unexpected list_goals result: {result}",
    )
    require(
        controller.calls == [("list_goals", "active")],
        f"Unexpected list_goals calls: {controller.calls}",
    )


def test_list_goals_preserves_default_status():
    interface, controller = make_interface()

    interface.list_goals()

    require(
        controller.calls == [("list_goals", None)],
        f"Default status was not preserved: {controller.calls}",
    )


def test_raw_chat_delegates():
    interface, controller = make_interface()

    result = interface.raw_chat("hello")

    require(result == "raw response", f"Unexpected raw_chat result: {result}")
    require(
        controller.calls == [("raw_chat", "hello")],
        f"Unexpected raw_chat calls: {controller.calls}",
    )


def test_chat_delegates():
    interface, controller = make_interface()

    result = interface.chat("hello")

    require(result == "chat response", f"Unexpected chat result: {result}")
    require(
        controller.calls == [("chat", "hello")],
        f"Unexpected chat calls: {controller.calls}",
    )


def test_reflect_delegates():
    interface, controller = make_interface()

    result = interface.reflect()

    require(
        result == {"reflection": "ok"},
        f"Unexpected reflection result: {result}",
    )
    require(
        controller.calls == [("reflect",)],
        f"Unexpected reflect calls: {controller.calls}",
    )


def test_stats_delegates():
    interface, controller = make_interface()

    result = interface.stats()

    require(
        result == {"memories": 12},
        f"Unexpected stats result: {result}",
    )
    require(
        controller.calls == [("stats",)],
        f"Unexpected stats calls: {controller.calls}",
    )


def test_ingest_code_delegates_to_controller_system():
    interface, controller = make_interface()

    result = interface.ingest_code(
        "/repo",
        max_files=25,
    )

    require(
        result == {
            "ingested": "/repo",
            "max_files": 25,
        },
        f"Unexpected ingest_code result: {result}",
    )
    require(
        controller.system.calls == [
            ("ingest_code", "/repo", 25)
        ],
        f"Unexpected ingest_code calls: {controller.system.calls}",
    )


def test_ingest_code_uses_default_max_files():
    interface, controller = make_interface()

    interface.ingest_code("/repo")

    require(
        controller.system.calls == [
            ("ingest_code", "/repo", 1000)
        ],
        f"Default max_files was not preserved: {controller.system.calls}",
    )


def test_ingest_pdf_delegates_to_controller_system():
    interface, controller = make_interface()

    result = interface.ingest_pdf(
        "/docs/test.pdf",
        max_pages=20,
    )

    require(
        result == {
            "ingested": "/docs/test.pdf",
            "max_pages": 20,
        },
        f"Unexpected ingest_pdf result: {result}",
    )
    require(
        controller.system.calls == [
            ("ingest_pdf", "/docs/test.pdf", 20)
        ],
        f"Unexpected ingest_pdf calls: {controller.system.calls}",
    )


def test_ingest_pdf_uses_default_max_pages():
    interface, controller = make_interface()

    interface.ingest_pdf("/docs/test.pdf")

    require(
        controller.system.calls == [
            ("ingest_pdf", "/docs/test.pdf", 100)
        ],
        f"Default max_pages was not preserved: {controller.system.calls}",
    )


def test_controller_exception_propagates():
    interface, controller = make_interface()

    controller.remember = controller.raise_error

    try:
        interface.remember("boom")
    except RuntimeError as exc:
        require(
            str(exc) == "controller failure",
            f"Unexpected exception: {exc}",
        )
        return

    raise RuntimeError(
        "MemoryInterface swallowed a controller exception"
    )


def test_repr_is_available():
    interface, _ = make_interface()

    value = repr(interface)

    require(
        "MemoryInterface" in value,
        f"Unexpected repr: {value}",
    )


def test_str_is_available():
    interface, _ = make_interface()

    value = str(interface)

    require(
        "MemoryInterface" in value,
        f"Unexpected str: {value}",
    )
