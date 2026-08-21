"""
Plugin manager for Memoria.

Uses pluggy for hook-based plugin system. Handles discovery and loading
of plugins from entry points and local plugins directory.
"""

import importlib.metadata
import importlib.util
from pathlib import Path
from typing import List, Optional, Type, Dict, Any
from cache.config import settings

import pluggy

# Hook specification marker (used to define hooks)
hookspec = pluggy.HookspecMarker("memoria")
# Hook implementation marker (used by plugins to implement hooks)
hookimpl = pluggy.HookimplMarker("memoria")


class MemoriaPluginManager:
    """
    Central plugin manager. Loads and manages all plugins.
    """

    def __init__(self):
        self.pm = pluggy.PluginManager("memoria")
        self.plugins = {}
        self._loaded = False

    def add_hookspecs(self, module):
        """Register hook specifications from a module."""
        self.pm.add_hookspecs(module)

    def register(self, plugin):
        """Register a plugin instance."""
        self.pm.register(plugin)
        name = getattr(plugin, "__name__", str(plugin))
        self.plugins[name] = plugin

    def discover_plugins(self):
        """
        Discover and load plugins from:
        1. Entry points (pip-installed plugins)
        2. Local plugins directory (if configured)
        """
        if self._loaded:
            return

        # 1. Import all hook spec modules to register them
        self._import_hook_specs()

        # 2. Load from entry points
        try:
            eps = importlib.metadata.entry_points()
            # For Python >=3.10, use select()
            if hasattr(eps, "select"):
                plugin_eps = eps.select(group="memoria.plugins")
            else:
                # Fallback for older Python
                plugin_eps = eps.get("memoria.plugins", [])
            for ep in plugin_eps:
                try:
                    plugin_class = ep.load()
                    plugin_instance = plugin_class()
                    self.register(plugin_instance)
                except Exception as e:
                    print(f"[PluginManager] Failed to load plugin {ep.name}: {e}")
        except Exception as e:
            print(f"[PluginManager] Entry point discovery error: {e}")

        # 3. Load from local plugins directory
        plugin_dir = getattr(settings, "PLUGIN_DIR", "plugins")
        if Path(plugin_dir).exists():
            for path in Path(plugin_dir).glob("*.py"):
                if path.stem.startswith("_"):
                    continue
                try:
                    spec = importlib.util.spec_from_file_location(path.stem, path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    # If module has a `plugin` instance, register it
                    if hasattr(module, "plugin"):
                        self.register(module.plugin)
                    # Alternatively, look for a `register` function
                    elif hasattr(module, "register"):
                        module.register(self)
                    else:
                        # Assume the module has a class with the same name as the file
                        class_name = "".join(word.capitalize() for word in path.stem.split("_"))
                        if hasattr(module, class_name):
                            plugin_class = getattr(module, class_name)
                            plugin_instance = plugin_class()
                            self.register(plugin_instance)
                except Exception as e:
                    print(f"[PluginManager] Failed to load local plugin {path.stem}: {e}")

        self._loaded = True

    def _import_hook_specs(self):
        """Import all hook spec modules to register them."""
        # These imports will register the specs via the hookspec decorator
        from .hooks import (
            retrieval,
            ranking,
            storage,
            ingestion,
            query,
            scheduler,
            routing,
            feedback,
            evaluation,
            lifecycle,
        )
        self.add_hookspecs(retrieval)
        self.add_hookspecs(ranking)
        self.add_hookspecs(storage)
        self.add_hookspecs(ingestion)
        self.add_hookspecs(query)
        self.add_hookspecs(scheduler)
        self.add_hookspecs(routing)
        self.add_hookspecs(feedback)
        self.add_hookspecs(evaluation)
        self.add_hookspecs(lifecycle)

    def get_hook(self, name):
        """Get a hook caller by name."""
        return getattr(self.pm.hook, name)

    def __getattr__(self, name):
        """Proxy to plugin manager's hook attribute."""
        return getattr(self.pm.hook, name)

    def list_plugins(self) -> List[str]:
        """Return names of loaded plugins."""
        return list(self.plugins.keys())

    def enable(self, name: str):
        """Enable a plugin (placeholder - not implemented yet)."""
        pass

    def disable(self, name: str):
        """Disable a plugin (placeholder - not implemented yet)."""
        pass
