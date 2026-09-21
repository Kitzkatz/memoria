# Plugin System for Memoria

Memoria's plugin system allows you to extend its core functionality without modifying the source code. Plugins can add new retrieval sources, ranking signals, storage backends, custom analyzers, and more.

---

## Overview

The plugin system is built on top of **pluggy**, a lightweight, production-ready plugin framework used by projects like `pytest`. It provides:

- **Hook specifications** — defined per subsystem, these are the contracts plugins implement.
- **Plugin discovery** — via Python entry points (for pip-installed packages) and a local `plugins/` directory.
- **Error isolation** — plugin failures are caught and logged, preventing crashes.

---

## Installation & Setup

### 1. Install Memoria with plugin support

The plugin system is enabled by default. You only need to ensure `pluggy` is installed:

```bash
pip install pluggy>=1.5.0
```

If you installed Memoria from source, it's already included in `requirements.txt`.

### 2. Enable plugins (configuration)

In `cache/config.py`, you can control plugin behavior:

```python
# Plugin System
PLUGIN_ENABLED: bool = True
PLUGIN_DIR: str = "plugins"
PLUGIN_AUTO_LOAD: bool = True
```

- `PLUGIN_ENABLED` — globally enable/disable plugins.
- `PLUGIN_DIR` — directory for local plugins (relative to project root).
- `PLUGIN_AUTO_LOAD` — if `True`, plugins are loaded at startup.

---

## Plugin Discovery

Plugins are loaded from two sources:

1. **Python entry points** — for plugins distributed as separate packages.
2. **Local `plugins/` directory** — for quick prototyping and local development.

### Entry Points (for pip-installed plugins)

In your `pyproject.toml` or `setup.py`, declare an entry point under the `memoria.plugins` group:

```toml
[project.entry-points."memoria.plugins"]
my_retriever = "my_retriever_plugin:MyRetrieverPlugin"
```

The plugin class must implement the relevant hook specifications.

### Local Directory

Place your plugin files (`.py`) inside the `plugins/` directory. Each file can contain a class that implements hooks. The plugin manager will automatically discover and load them.

---

## Writing a Plugin

### 1. Choose a subsystem and its hook

Memoria defines hook specifications for these subsystems:

| Subsystem | Hook Module | Main Hooks |
|-----------|-------------|------------|
| **Lifecycle** | `core.hooks.lifecycle` | `memoria_on_startup`, `memoria_on_shutdown`, `memoria_pre_query`, `memoria_post_query`, `memoria_pre_store`, `memoria_post_store` |
| **Retrieval** | `core.hooks.retrieval` | `memoria_register_retriever`, `memoria_retrieval_pre`, `memoria_retrieval_post` |
| **Ranking** | `core.hooks.ranking` | `memoria_register_ranking_signal`, `memoria_register_reranker`, `memoria_ranking_pre`, `memoria_ranking_post` |
| **Storage** | `core.hooks.storage` | `memoria_register_database_backend`, `memoria_register_vector_store`, `memoria_storage_pre`, `memoria_storage_post` |
| **Ingestion** | `core.hooks.ingestion` | `memoria_register_extractor`, `memoria_register_entity_recognizer`, `memoria_ingestion_pre`, `memoria_ingestion_post` |
| **Query** | `core.hooks.query` | `memoria_register_query_processor`, `memoria_query_process_pre`, `memoria_query_process_post` |
| **Scheduler** | `core.hooks.scheduler` | `memoria_register_worker`, `memoria_register_scheduler_policy`, `memoria_scheduler_pre`, `memoria_scheduler_post` |
| **Routing** | `core.hooks.routing` | `memoria_register_type_router`, `memoria_register_signal_router`, `memoria_routing_pre`, `memoria_routing_post` |
| **Feedback** | `core.hooks.feedback` | `memoria_register_feedback_recorder`, `memoria_feedback_pre`, `memoria_feedback_post` |
| **Evaluation** | `core.hooks.evaluation` | `memoria_register_benchmark_adapter`, `memoria_register_analyzer`, `memoria_analysis_pre`, `memoria_analysis_post` |

### 2. Create a plugin class

Your plugin class must use the `@hookimpl` decorator from `core.plugin_manager` to mark the methods that implement hooks.

**Example: A custom ranking signal**

```python
from core.plugin_manager import hookimpl

class MyRankingPlugin:
    @hookimpl
    def memoria_register_ranking_signal(self):
        return {
            "name": "my_signal",
            "score_func": self.compute_my_signal,
            "weight": 0.1,   # optional
        }

    def compute_my_signal(self, memory, query):
        # Compute a score based on custom logic
        return 0.5
```

**Example: A custom retriever**

```python
from core.plugin_manager import hookimpl

class MyRetrieverPlugin:
    @hookimpl
    def memoria_register_retriever(self):
        return {
            "name": "my_retriever",
            "retriever": self.retrieve,
        }

    def retrieve(self, query, limit):
        # Your retrieval logic
        return []  # list of (mem_id, score) or (mem_id, distance)
```

**Example: A pre-query hook**

```python
from core.plugin_manager import hookimpl

class MyQueryPlugin:
    @hookimpl
    def memoria_pre_query(self, text):
        # Modify or log the query before it's processed
        print(f"Query: {text}")
```

### 3. Register the plugin

**For entry points:** define the plugin class in your package and declare it in `pyproject.toml`.

**For local plugins:** place the file in `plugins/` — the plugin manager will auto-detect and load it.

---

## Testing Your Plugin

To test a plugin without installing it, you can place it in the `plugins/` directory and run Memoria.

You can also write unit tests using `pytest`:

```python
from core.plugin_manager import MemoriaPluginManager

def test_my_plugin():
    pm = MemoriaPluginManager()
    pm.discover_plugins()   # loads plugins
    # Check if your plugin is registered
    assert "my_plugin" in pm.list_plugins()
```

---

## Best Practices

1. **Keep hooks lightweight** — plugins should not block the main thread.
2. **Handle errors gracefully** — use try/except in your hook implementations.
3. **Avoid global state** — prefer stateless plugins.
4. **Document your plugin** — explain what it does and how to configure it.
5. **Use `plugin_manager` if needed** — if your plugin needs to interact with other plugins, you can access `plugin_manager` from the hook call context (though not provided directly, you can store it globally).
6. **Versioning** — if your plugin depends on a specific Memoria version, check the version in `memoria.__version__`.

---

## Example: A Complete Plugin

A full example plugin that adds a custom retriever, ranking signal, and pre-query hook can be found in the `examples/plugins/` directory of the Memoria repository.

---

## Reference: Hook Signatures

For each hook, the plugin manager expects a specific return type. Refer to the respective hook specification file in `core/hooks/` for details.

### Common Return Types

- **Retriever** — `{"name": str, "retriever": Callable}`
- **Ranking signal** — `{"name": str, "score_func": Callable, "weight": float}`
- **Reranker** — `{"name": str, "reranker": Callable}`
- **Worker** — `{"name": str, "worker": Callable}`
- **Policy** — `Callable` that returns a `CompletionPolicy` instance.

### Pre/Post Hooks

These hooks receive parameters (e.g. `query`, `candidates`, `records`) and may modify them in-place or return new values. The plugin manager will use the returned value if applicable, but for most hooks, modification is done in-place.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Plugin not loaded | Check `PLUGIN_ENABLED` and `PLUGIN_AUTO_LOAD` in `config.py`. Also ensure the plugin file is in the correct location. |
| Missing `pluggy` | Install with `pip install pluggy` |
| Hook not called | Ensure your plugin uses `@hookimpl` and the method name matches the hook specification. |
| Error in plugin | Check logs — plugin errors are logged with the `[Plugin]` prefix. |

---

## Next Steps

- Explore existing plugins in the Memoria repository.
- Use the cookiecutter template (if provided) to scaffold a new plugin project.
- Share your plugin with the community!

---

**Happy plugin building!**
