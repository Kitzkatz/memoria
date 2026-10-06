"""
Plugin manager for Memoria.

Uses pluggy for hook-based plugins. Supports discovery from installed
entry points and the local plugins directory, with configuration-driven
plugin activation and runtime enable/disable controls.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List

import pluggy

from cache.config import settings


hookspec = pluggy.HookspecMarker("memoria")
hookimpl = pluggy.HookimplMarker("memoria")


class MemoriaPluginManager:
    """Central plugin manager for discovery, loading, and activation."""

    def __init__(self):
        self.pm = pluggy.PluginManager("memoria")

        self.plugins: Dict[str, object] = {}
        self.enabled: Dict[str, object] = {}

        self._loaded = False

    def add_hookspecs(self, module):
        """Register hook specifications from a module."""
        self.pm.add_hookspecs(module)

    def register(self, plugin, name: str | None = None) -> str:
        """
        Register a plugin object.

        Registration means the plugin is loaded and available. Activation
        determines whether its hooks are currently registered with Pluggy.
        """
        plugin_name = name or self._plugin_name(plugin)

        if plugin_name in self.plugins:
            existing = self.plugins[plugin_name]

            if existing is plugin:
                return plugin_name

            raise ValueError(
                f"Plugin name already registered: {plugin_name}"
            )

        self.plugins[plugin_name] = plugin

        if self._should_enable(plugin_name):
            self._activate(plugin_name)

        return plugin_name

    def discover_plugins(self):
        """
        Discover and load plugins from entry points and the local plugin
        directory according to the configured plugin policy.
        """
        if self._loaded:
            return

        if not settings.PLUGIN_ENABLED:
            self._import_hook_specs()
            self._loaded = True
            return

        self._import_hook_specs()

        if settings.PLUGIN_AUTO_LOAD:
            self._discover_entry_points()
            self._discover_local_plugins()

        self._loaded = True

    def _discover_entry_points(self):
        """Discover plugins registered through package entry points."""
        try:
            eps = importlib.metadata.entry_points()

            if hasattr(eps, "select"):
                plugin_eps = eps.select(group="memoria.plugins")
            else:
                plugin_eps = eps.get("memoria.plugins", [])

            for ep in plugin_eps:
                try:
                    plugin_class = ep.load()
                    plugin_instance = plugin_class()
                    self.register(plugin_instance, name=ep.name)
                except Exception as exc:
                    print(
                        f"[PluginManager] Failed to load plugin "
                        f"{ep.name}: {exc}"
                    )

        except Exception as exc:
            print(
                f"[PluginManager] Entry point discovery error: {exc}"
            )

    def _discover_local_plugins(self):
        """Discover plugins from the configured local plugin directory."""
        plugin_dir = Path(settings.PLUGIN_DIR)

        if not plugin_dir.exists():
            return

        for path in plugin_dir.glob("*.py"):
            if path.stem.startswith("_"):
                continue

            try:
                self._load_local_plugin(path)
            except Exception as exc:
                print(
                    f"[PluginManager] Failed to load local plugin "
                    f"{path.stem}: {exc}"
                )

    def _load_local_plugin(self, path: Path):
        """Load a single local plugin module."""
        spec = importlib.util.spec_from_file_location(path.stem, path)

        if spec is None or spec.loader is None:
            raise ImportError(
                f"Unable to create import specification for {path}"
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "plugin"):
            self.register(module.plugin, name=path.stem)
            return

        if hasattr(module, "register"):
            module.register(self)
            return

        class_name = "".join(
            word.capitalize()
            for word in path.stem.split("_")
        )

        if hasattr(module, class_name):
            plugin_class = getattr(module, class_name)
            plugin_instance = plugin_class()
            self.register(plugin_instance, name=path.stem)

    def _import_hook_specs(self):
        """Import all hook specification modules."""
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
            analysis,
        )

        for module in (
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
            analysis,
        ):
            self.add_hookspecs(module)

    def _activate(self, name: str):
        """Register a loaded plugin with Pluggy."""
        plugin = self.plugins[name]

        if name in self.enabled:
            return

        if self.pm.is_registered(plugin):
            self.enabled[name] = plugin
            return

        self.pm.register(plugin, name=name)
        self.enabled[name] = plugin

    def _deactivate(self, name: str):
        """Unregister an active plugin from Pluggy."""
        plugin = self.plugins[name]

        if self.pm.is_registered(plugin):
            self.pm.unregister(plugin=plugin)

        self.enabled.pop(name, None)

    def _should_enable(self, name: str) -> bool:
        """Return whether configuration allows a plugin to be enabled."""
        if not settings.PLUGIN_ENABLED:
            return False

        allowlist = set(settings.PLUGIN_ENABLED_PLUGINS)
        denylist = set(settings.PLUGIN_DISABLED_PLUGINS)

        if name in denylist:
            return False

        if allowlist and name not in allowlist:
            return False

        return True

    @staticmethod
    def _plugin_name(plugin) -> str:
        """Return a stable name for a plugin object."""
        name = getattr(plugin, "__plugin_name__", None)

        if name:
            return str(name)

        name = getattr(plugin, "__name__", None)

        if name:
            return str(name)

        return plugin.__class__.__name__

    def get_hook(self, name):
        """Get a hook caller by name."""
        return getattr(self.pm.hook, name)

    def __getattr__(self, name):
        """Proxy unknown attributes to Pluggy's hook relay."""
        return getattr(self.pm.hook, name)

    def list_plugins(self) -> List[str]:
        """Return names of all loaded plugins."""
        return list(self.plugins.keys())

    def list_enabled_plugins(self) -> List[str]:
        """Return names of currently active plugins."""
        return list(self.enabled.keys())

    def is_enabled(self, name: str) -> bool:
        """Return whether a loaded plugin is currently active."""
        return name in self.enabled

    def enable(self, name: str):
        """Enable a loaded plugin."""
        if name not in self.plugins:
            raise KeyError(f"Unknown plugin: {name}")

        self._activate(name)

    def disable(self, name: str):
        """Disable a loaded plugin."""
        if name not in self.plugins:
            raise KeyError(f"Unknown plugin: {name}")

        self._deactivate(name)

    def unload(self, name: str):
        """Unload a plugin completely."""
        if name not in self.plugins:
            raise KeyError(f"Unknown plugin: {name}")

        self._deactivate(name)
        self.plugins.pop(name, None)

    def load_plugin(self, name: str):
        """
        Activate an already discovered plugin.

        This is an explicit runtime operation and does not perform discovery.
        """
        if name not in self.plugins:
            raise KeyError(f"Unknown plugin: {name}")

        self._activate(name)
