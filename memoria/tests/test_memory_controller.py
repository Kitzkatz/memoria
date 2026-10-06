from pathlib import Path

import memory.memory_controller as controller_module
from memory.memory_controller import MemoryController


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class FakeDB:
    def __init__(self):
        self.calls = []
        self.memory_count = 10

    def fetch(self, mem_id):
        self.calls.append(("fetch", mem_id))
        return {"id": mem_id, "text": "memory"}

    def update(self, mem_id, **kwargs):
        self.calls.append(("update", mem_id, kwargs))
        return "updated"

    def delete(self, mem_id):
        self.calls.append(("delete", mem_id))
        return "deleted"

    def count(self):
        self.calls.append(("count",))
        return self.memory_count


class FakeVectorStore:
    def __init__(self, count_value=20, error=None):
        self.count_value = count_value
        self.error = error
        self.calls = []

    def count(self):
        self.calls.append(("count",))
        if self.error:
            raise self.error
        return self.count_value


class FakeEmbeddingCache:
    def __init__(self, count_value=30, error=None):
        self.count_value = count_value
        self.error = error
        self.calls = []

    def count(self):
        self.calls.append(("count",))
        if self.error:
            raise self.error
        return self.count_value


class FakeRelevanceManager:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else {"signals": 3}
        self.error = error
        self.calls = []

    def stats(self):
        self.calls.append(("stats",))
        if self.error:
            raise self.error
        return self.result


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, prompt):
        self.calls.append(("chat", prompt))
        return "LLM response"


class FakeSystem:
    def __init__(
        self,
        db,
        vector_store=None,
        embedding_cache=None,
        relevance_manager=None,
        query_result=None,
        reflect_result=None,
        store_result=None,
        store_many_result=None,
        plugin_manager=None,
    ):
        self.db = db
        self.vector_store = vector_store
        self.embedding_cache = embedding_cache
        self.relevance_manager = relevance_manager
        self.query_result = (
            query_result
            if query_result is not None
            else {"results": [{"text": "memory one"}]}
        )
        self.reflect_result = (
            reflect_result
            if reflect_result is not None
            else {"reflection": "ok"}
        )
        self.store_result = (
            store_result if store_result is not None else 123
        )
        self.store_many_result = (
            store_many_result
            if store_many_result is not None
            else [123, 124]
        )
        self.plugin_manager = plugin_manager
        self.calls = []

    def store(self, text, metadata=None):
        self.calls.append(("store", text, metadata))
        return self.store_result

    def store_many(
        self,
        texts,
        metadatas=None,
        skip_embedding_build=False,
    ):
        self.calls.append(
            (
                "store_many",
                texts,
                metadatas,
                skip_embedding_build,
            )
        )
        return self.store_many_result

    def query(self, query):
        self.calls.append(("query", query))
        return self.query_result

    def reflect(self):
        self.calls.append(("reflect",))
        return self.reflect_result


class FakeGoals:
    def __init__(self):
        self.calls = []

    def set_goal(self, goal, progress):
        self.calls.append(("set_goal", goal, progress))
        return 77

    def update_goal(self, goal_id, progress=None, status=None):
        self.calls.append(
            ("update_goal", goal_id, progress, status)
        )
        return "goal updated"

    def list_goals(self, status=None):
        self.calls.append(("list_goals", status))
        return [{"id": 1, "status": status}]


class FakeEntityStore:
    def __init__(self, db):
        self.db = db


def make_controller(
    monkeypatch,
    *,
    query_result=None,
    vector_store=None,
    embedding_cache=None,
    relevance_manager=None,
    plugin_manager=None,
):
    db = FakeDB()
    llm = FakeLLM()
    goals = FakeGoals()
    system = FakeSystem(
        db,
        vector_store=vector_store,
        embedding_cache=embedding_cache,
        relevance_manager=relevance_manager,
        query_result=query_result,
        plugin_manager=plugin_manager,
    )

    created = {}

    def fake_bootstrap():
        created["bootstrap"] = True
        return db, "vector-store", "embedder", llm

    def fake_entity_store(received_db):
        require(
            received_db is db,
            "EntityStore received the wrong DB",
        )
        created["entity_store"] = True
        return "entity-store"

    def fake_memory_system(
        received_db,
        received_vs,
        received_embedder,
        received_entity_store,
        llm=None,
    ):
        require(
            received_db is db,
            "MemorySystem received the wrong DB",
        )
        require(
            received_vs == "vector-store",
            "MemorySystem received the wrong vector store",
        )
        require(
            received_embedder == "embedder",
            "MemorySystem received the wrong embedder",
        )
        require(
            received_entity_store == "entity-store",
            "MemorySystem received the wrong entity store",
        )
        require(
            llm is llm_object,
            "MemorySystem received the wrong LLM",
        )
        created["memory_system"] = True
        return system

    def fake_goal_tracker(received_db):
        require(
            received_db is db,
            "GoalTracker received the wrong DB",
        )
        created["goal_tracker"] = True
        return goals

    llm_object = llm

    monkeypatch.setattr(
        controller_module,
        "bootstrap",
        fake_bootstrap,
    )
    monkeypatch.setattr(
        controller_module,
        "EntityStore",
        fake_entity_store,
    )
    monkeypatch.setattr(
        controller_module,
        "MemorySystem",
        fake_memory_system,
    )
    monkeypatch.setattr(
        controller_module,
        "GoalTracker",
        fake_goal_tracker,
    )

    controller = MemoryController()

    return controller, system, goals, db, llm, created


def test_constructor_wires_bootstrap_and_components(monkeypatch):
    controller, system, goals, db, llm, created = make_controller(
        monkeypatch
    )

    require(created.get("bootstrap"), "bootstrap was not called")
    require(
        created.get("entity_store"),
        "EntityStore was not constructed",
    )
    require(
        created.get("memory_system"),
        "MemorySystem was not constructed",
    )
    require(
        created.get("goal_tracker"),
        "GoalTracker was not constructed",
    )

    require(
        controller.system is system,
        "Controller stored the wrong MemorySystem",
    )
    require(
        controller.goals is goals,
        "Controller stored the wrong GoalTracker",
    )
    require(
        controller.llm is llm,
        "Controller stored the wrong LLM",
    )


def test_plugin_manager_exposes_system_plugin_manager(monkeypatch):
    plugin_manager = object()

    controller, _, _, _, _, _ = make_controller(
        monkeypatch,
        plugin_manager=plugin_manager,
    )

    require(
        controller.plugin_manager is plugin_manager,
        "Controller did not expose system plugin manager",
    )


def test_plugin_manager_returns_none_when_unavailable(monkeypatch):
    controller, system, _, _, _, _ = make_controller(monkeypatch)

    system.plugin_manager = None

    require(
        controller.plugin_manager is None,
        "plugin_manager should return None when unavailable",
    )


def test_remember_without_metadata_delegates_to_system(monkeypatch):
    controller, system, _, _, _, _ = make_controller(monkeypatch)

    result = controller.remember("hello")

    require(result == 123, f"Unexpected remember result: {result}")
    require(
        system.calls == [("store", "hello", None)],
        f"Unexpected store calls: {system.calls}",
    )


def test_remember_with_metadata_delegates_to_system(monkeypatch):
    controller, system, _, _, _, _ = make_controller(monkeypatch)

    metadata = {"source": "test"}

    result = controller.remember(
        "hello",
        metadata=metadata,
    )

    require(result == 123, f"Unexpected remember result: {result}")
    require(
        system.calls == [
            ("store", "hello", metadata)
        ],
        f"Unexpected store calls: {system.calls}",
    )


def test_remember_empty_metadata_uses_normal_store_path(monkeypatch):
    controller, system, _, _, _, _ = make_controller(monkeypatch)

    controller.remember(
        "hello",
        metadata={},
    )

    require(
        system.calls == [("store", "hello", None)],
        f"Unexpected empty-metadata behavior: {system.calls}",
    )


def test_remember_many_without_metadata(monkeypatch):
    controller, system, _, _, _, _ = make_controller(monkeypatch)

    texts = ["one", "two"]

    result = controller.remember_many(texts)

    require(
        result == [123, 124],
        f"Unexpected remember_many result: {result}",
    )
    require(
        system.calls == [
            ("store_many", texts, None, False)
        ],
        f"Unexpected store_many calls: {system.calls}",
    )


def test_remember_many_with_metadata(monkeypatch):
    controller, system, _, _, _, _ = make_controller(monkeypatch)

    texts = ["one", "two"]
    metadatas = [{"a": 1}, {"b": 2}]

    result = controller.remember_many(
        texts,
        metadatas=metadatas,
    )

    require(
        result == [123, 124],
        f"Unexpected remember_many result: {result}",
    )
    require(
        system.calls == [
            ("store_many", texts, metadatas, False)
        ],
        f"Unexpected store_many calls: {system.calls}",
    )


def test_remember_many_preserves_skip_embedding_build(monkeypatch):
    controller, system, _, _, _, _ = make_controller(monkeypatch)

    texts = ["one"]

    controller.remember_many(
        texts,
        skip_embedding_build=True,
    )

    require(
        system.calls == [
            ("store_many", texts, None, True)
        ],
        f"skip_embedding_build was lost: {system.calls}",
    )


def test_remember_many_empty_metadata_uses_none(monkeypatch):
    controller, system, _, _, _, _ = make_controller(monkeypatch)

    texts = ["one"]

    controller.remember_many(
        texts,
        metadatas=[],
    )

    require(
        system.calls == [
            ("store_many", texts, None, False)
        ],
        f"Unexpected empty metadata behavior: {system.calls}",
    )


def test_set_goal_delegates(monkeypatch):
    controller, _, goals, _, _, _ = make_controller(monkeypatch)

    result = controller.set_goal(
        "finish release",
        progress="in progress",
    )

    require(result == 77, f"Unexpected goal result: {result}")
    require(
        goals.calls == [
            ("set_goal", "finish release", "in progress")
        ],
        f"Unexpected goal calls: {goals.calls}",
    )


def test_set_goal_preserves_default_progress(monkeypatch):
    controller, _, goals, _, _, _ = make_controller(monkeypatch)

    controller.set_goal("finish release")

    require(
        goals.calls == [
            ("set_goal", "finish release", "started")
        ],
        f"Default progress was lost: {goals.calls}",
    )


def test_update_goal_delegates(monkeypatch):
    controller, _, goals, _, _, _ = make_controller(monkeypatch)

    result = controller.update_goal(
        7,
        progress="complete",
        status="completed",
    )

    require(
        result == "goal updated",
        f"Unexpected update_goal result: {result}",
    )
    require(
        goals.calls == [
            (
                "update_goal",
                7,
                "complete",
                "completed",
            )
        ],
        f"Unexpected update_goal calls: {goals.calls}",
    )


def test_update_goal_preserves_none_values(monkeypatch):
    controller, _, goals, _, _, _ = make_controller(monkeypatch)

    controller.update_goal(7)

    require(
        goals.calls == [
            ("update_goal", 7, None, None)
        ],
        f"None values were lost: {goals.calls}",
    )


def test_list_goals_delegates(monkeypatch):
    controller, _, goals, _, _, _ = make_controller(monkeypatch)

    result = controller.list_goals("active")

    require(
        result == [{"id": 1, "status": "active"}],
        f"Unexpected list_goals result: {result}",
    )
    require(
        goals.calls == [
            ("list_goals", "active")
        ],
        f"Unexpected list_goals calls: {goals.calls}",
    )


def test_list_goals_preserves_default_status(monkeypatch):
    controller, _, goals, _, _, _ = make_controller(monkeypatch)

    controller.list_goals()

    require(
        goals.calls == [
            ("list_goals", None)
        ],
        f"Default status was lost: {goals.calls}",
    )


def test_fetch_delegates_to_database(monkeypatch):
    controller, _, _, db, _, _ = make_controller(monkeypatch)

    result = controller.fetch(42)

    require(
        result == {"id": 42, "text": "memory"},
        f"Unexpected fetch result: {result}",
    )
    require(
        db.calls == [("fetch", 42)],
        f"Unexpected fetch calls: {db.calls}",
    )


def test_update_delegates_to_database(monkeypatch):
    controller, _, _, db, _, _ = make_controller(monkeypatch)

    result = controller.update(
        42,
        text="updated",
        importance=0.9,
    )

    require(
        result == "updated",
        f"Unexpected update result: {result}",
    )
    require(
        db.calls == [
            (
                "update",
                42,
                {
                    "text": "updated",
                    "importance": 0.9,
                },
            )
        ],
        f"Unexpected update calls: {db.calls}",
    )


def test_delete_delegates_to_database(monkeypatch):
    controller, _, _, db, _, _ = make_controller(monkeypatch)

    result = controller.delete(42)

    require(
        result == "deleted",
        f"Unexpected delete result: {result}",
    )
    require(
        db.calls == [("delete", 42)],
        f"Unexpected delete calls: {db.calls}",
    )


def test_recall_delegates_to_system_query(monkeypatch):
    controller, system, _, _, _, _ = make_controller(monkeypatch)

    result = controller.recall("what happened?")

    require(
        result == {"results": [{"text": "memory one"}]},
        f"Unexpected recall result: {result}",
    )
    require(
        system.calls == [
            ("query", "what happened?")
        ],
        f"Unexpected query calls: {system.calls}",
    )


def test_reflect_delegates_to_system(monkeypatch):
    controller, system, _, _, _, _ = make_controller(monkeypatch)

    result = controller.reflect()

    require(
        result == {"reflection": "ok"},
        f"Unexpected reflect result: {result}",
    )
    require(
        system.calls == [("reflect",)],
        f"Unexpected reflect calls: {system.calls}",
    )


def test_stats_returns_core_counts(monkeypatch):
    controller, _, goals, db, _, _ = make_controller(monkeypatch)

    db.memory_count = 15
    goals.list_goals = lambda status=None: [
        {"id": 1},
        {"id": 2},
        {"id": 3},
    ]

    result = controller.stats()

    require(
        result["memory_count"] == 15,
        f"Unexpected memory_count: {result}",
    )
    require(
        result["goals"] == 3,
        f"Unexpected goal count: {result}",
    )


def test_stats_includes_vector_count(monkeypatch):
    vector_store = FakeVectorStore(count_value=55)

    controller, _, _, _, _, _ = make_controller(
        monkeypatch,
        vector_store=vector_store,
    )

    result = controller.stats()

    require(
        result["vector_count"] == 55,
        f"Unexpected vector_count: {result}",
    )
    require(
        vector_store.calls == [("count",)],
        f"Unexpected vector store calls: {vector_store.calls}",
    )


def test_stats_includes_embedding_cache_count(monkeypatch):
    embedding_cache = FakeEmbeddingCache(count_value=66)

    controller, _, _, _, _, _ = make_controller(
        monkeypatch,
        embedding_cache=embedding_cache,
    )

    result = controller.stats()

    require(
        result["embedding_cache"] == 66,
        f"Unexpected embedding cache count: {result}",
    )


def test_stats_includes_relevance_stats(monkeypatch):
    relevance = FakeRelevanceManager(
        result={"active": 4}
    )

    controller, _, _, _, _, _ = make_controller(
        monkeypatch,
        relevance_manager=relevance,
    )

    result = controller.stats()

    require(
        result["relevance"] == {"active": 4},
        f"Unexpected relevance stats: {result}",
    )


def test_stats_ignores_vector_store_errors(monkeypatch):
    vector_store = FakeVectorStore(
        error=RuntimeError("vector failure")
    )

    controller, _, _, _, _, _ = make_controller(
        monkeypatch,
        vector_store=vector_store,
    )

    result = controller.stats()

    require(
        "vector_count" not in result,
        f"Vector failure should be swallowed: {result}",
    )


def test_stats_ignores_embedding_cache_errors(monkeypatch):
    embedding_cache = FakeEmbeddingCache(
        error=RuntimeError("cache failure")
    )

    controller, _, _, _, _, _ = make_controller(
        monkeypatch,
        embedding_cache=embedding_cache,
    )

    result = controller.stats()

    require(
        "embedding_cache" not in result,
        f"Cache failure should be swallowed: {result}",
    )


def test_stats_ignores_relevance_manager_errors(monkeypatch):
    relevance = FakeRelevanceManager(
        error=RuntimeError("relevance failure")
    )

    controller, _, _, _, _, _ = make_controller(
        monkeypatch,
        relevance_manager=relevance,
    )

    result = controller.stats()

    require(
        "relevance" not in result,
        f"Relevance failure should be swallowed: {result}",
    )


def test_raw_chat_delegates_to_llm(monkeypatch):
    controller, _, _, _, llm, _ = make_controller(monkeypatch)

    result = controller.raw_chat("hello")

    require(
        result == "LLM response",
        f"Unexpected raw_chat result: {result}",
    )
    require(
        llm.calls == [("chat", "hello")],
        f"Unexpected LLM calls: {llm.calls}",
    )


def test_chat_retrieves_memories_and_sends_formatted_prompt(
    monkeypatch,
):
    query_result = {
        "results": [
            {"text": "memory one"},
            {"text": "memory two"},
        ]
    }

    controller, system, _, _, llm, _ = make_controller(
        monkeypatch,
        query_result=query_result,
    )

    controller._load_default_template = lambda: (
        "SYSTEM={system}\n"
        "CONTEXT={context}\n"
        "USER={user}\n"
        "ASSISTANT={assistant}"
    )

    result = controller.chat(
        "what happened?",
        top_n=2,
    )

    require(
        result == "LLM response",
        f"Unexpected chat result: {result}",
    )
    require(
        system.calls == [
            ("query", "what happened?")
        ],
        f"Unexpected chat retrieval calls: {system.calls}",
    )

    require(
        len(llm.calls) == 1,
        f"Expected one LLM call: {llm.calls}",
    )

    prompt = llm.calls[0][1]

    require(
        "memory one" in prompt,
        f"First memory missing from prompt: {prompt}",
    )
    require(
        "memory two" in prompt,
        f"Second memory missing from prompt: {prompt}",
    )
    require(
        "what happened?" in prompt,
        f"User prompt missing: {prompt}",
    )


def test_chat_respects_top_n(monkeypatch):
    query_result = {
        "results": [
            {"text": "one"},
            {"text": "two"},
            {"text": "three"},
        ]
    }

    controller, _, _, _, llm, _ = make_controller(
        monkeypatch,
        query_result=query_result,
    )

    controller._load_default_template = lambda: "{context}"

    controller.chat(
        "question",
        top_n=1,
    )

    prompt = llm.calls[0][1]

    require(
        "one" in prompt,
        f"Top result missing: {prompt}",
    )
    require(
        "two" not in prompt,
        f"Second result should not be included: {prompt}",
    )
    require(
        "three" not in prompt,
        f"Third result should not be included: {prompt}",
    )


def test_chat_uses_no_memory_fallback(monkeypatch):
    query_result = {"results": []}

    controller, _, _, _, llm, _ = make_controller(
        monkeypatch,
        query_result=query_result,
    )

    controller._load_default_template = lambda: "{context}"

    controller.chat("question")

    require(
        llm.calls[0][1] == "No relevant memories found.",
        f"Unexpected empty-context prompt: {llm.calls[0][1]}",
    )


def test_chat_accepts_template_string(monkeypatch):
    controller, _, _, _, llm, _ = make_controller(monkeypatch)

    controller.chat(
        "hello",
        template="USER={user}",
    )

    require(
        llm.calls[0][1] == "USER=hello",
        f"Template string was not used: {llm.calls[0][1]}",
    )


def test_chat_loads_template_from_existing_file(
    monkeypatch,
    tmp_path,
):
    template_path = tmp_path / "template.txt"
    template_path.write_text(
        "FILE USER={user}",
        encoding="utf-8",
    )

    controller, _, _, _, llm, _ = make_controller(monkeypatch)

    controller.chat(
        "hello",
        template=str(template_path),
    )

    require(
        llm.calls[0][1] == "FILE USER=hello",
        f"Template file was not loaded: {llm.calls[0][1]}",
    )


def test_chat_applies_custom_template_variables(monkeypatch):
    controller, _, _, _, llm, _ = make_controller(monkeypatch)

    controller.chat(
        "hello",
        template="CUSTOM={custom}",
        template_vars={"custom": "value"},
    )

    require(
        llm.calls[0][1] == "CUSTOM=value",
        f"Custom template variables were not applied: {llm.calls[0][1]}",
    )


def test_chat_missing_template_variable_uses_fallback(
    monkeypatch,
):
    controller, _, _, _, llm, _ = make_controller(monkeypatch)

    controller.chat(
        "hello",
        template="{missing_variable}",
    )

    prompt = llm.calls[0][1]

    require(
        "Context:" in prompt,
        f"Fallback context missing: {prompt}",
    )
    require(
        "User:" in prompt,
        f"Fallback user section missing: {prompt}",
    )
    require(
        "hello" in prompt,
        f"Fallback user prompt missing: {prompt}",
    )


def test_load_default_template_uses_configured_file(
    monkeypatch,
    tmp_path,
):
    template_path = tmp_path / "llama3.txt"
    template_path.write_text(
        "configured template",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        controller_module.settings,
        "CHAT_TEMPLATE_DIR",
        str(tmp_path),
    )
    monkeypatch.setattr(
        controller_module.settings,
        "CHAT_TEMPLATE_FILE",
        "llama3.txt",
    )

    controller, _, _, _, _, _ = make_controller(monkeypatch)

    result = controller._load_default_template()

    require(
        result == "configured template",
        f"Configured template was not loaded: {result}",
    )


def test_load_default_template_falls_back_when_file_missing(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        controller_module.settings,
        "CHAT_TEMPLATE_DIR",
        str(tmp_path),
    )
    monkeypatch.setattr(
        controller_module.settings,
        "CHAT_TEMPLATE_FILE",
        "missing.txt",
    )

    controller, _, _, _, _, _ = make_controller(monkeypatch)

    result = controller._load_default_template()

    require(
        "{system}" in result,
        "Built-in template missing system variable",
    )
    require(
        "{context}" in result,
        "Built-in template missing context variable",
    )
    require(
        "{user}" in result,
        "Built-in template missing user variable",
    )


def test_load_template_from_file_reads_utf8(
    monkeypatch,
    tmp_path,
):
    template_path = tmp_path / "template.txt"
    template_path.write_text(
        "héllo {user}",
        encoding="utf-8",
    )

    controller, _, _, _, _, _ = make_controller(monkeypatch)

    result = controller._load_template_from_file(
        template_path
    )

    require(
        result == "héllo {user}",
        f"UTF-8 template was not read correctly: {result}",
    )


def test_load_template_from_file_falls_back_on_error(
    monkeypatch,
):
    controller, _, _, _, _, _ = make_controller(monkeypatch)

    result = controller._load_template_from_file(
        Path("/definitely/missing/template.txt")
    )

    require(
        "{system}" in result,
        "Template load failure did not use built-in fallback",
    )


def test_builtin_llama_template_contains_expected_sections(
    monkeypatch,
):
    controller, _, _, _, _, _ = make_controller(monkeypatch)

    result = controller._get_builtin_llama3_template()

    require(
        "<|start_header_id|>system<|end_header_id|>" in result,
        "Built-in template missing system header",
    )
    require(
        "<|start_header_id|>user<|end_header_id|>" in result,
        "Built-in template missing user header",
    )
    require(
        "<|start_header_id|>assistant<|end_header_id|>" in result,
        "Built-in template missing assistant header",
    )
    require(
        "{context}" in result,
        "Built-in template missing context variable",
    )
    require(
        "{user}" in result,
        "Built-in template missing user variable",
    )
