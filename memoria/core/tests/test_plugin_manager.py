from types import SimpleNamespace

import pytest

from core.plugin_manager import MemoriaPluginManager, hookimpl


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class PluginFixture:
    __plugin_name__ = "test_plugin"

    def __init__(self):
        self.calls = 0

    @hookimpl
    def analyze(self, value):
        self.calls += 1
        return value + 1


class OtherPlugin:
    __plugin_name__ = "other_plugin"

    @hookimpl
    def analyze(self, value):
        return value + 10


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(
        "core.plugin_manager.settings",
        SimpleNamespace(
            PLUGIN_ENABLED=True,
            PLUGIN_DIR="plugins",
            PLUGIN_AUTO_LOAD=True,
            PLUGIN_ENABLED_PLUGINS=[],
            PLUGIN_DISABLED_PLUGINS=[],
        ),
    )

    return MemoriaPluginManager()


def test_manager_initializes(manager):
    require(manager.plugins == {}, "Plugin registry should start empty")
    require(manager.enabled == {}, "Enabled registry should start empty")
    require(manager._loaded is False, "Manager should start unloaded")


def test_register_loads_and_enables_plugin(manager):
    plugin = PluginFixture()

    name = manager.register(plugin)

    require(name == "test_plugin", "Plugin name was not resolved correctly")
    require(manager.plugins["test_plugin"] is plugin, "Plugin was not stored")
    require(
        manager.enabled["test_plugin"] is plugin,
        "Plugin should be enabled by default",
    )
    require(
        manager.pm.is_registered(plugin),
        "Plugin should be registered with Pluggy",
    )


def test_register_is_idempotent_for_same_plugin(manager):
    plugin = PluginFixture()

    first = manager.register(plugin)
    second = manager.register(plugin)

    require(first == "test_plugin", "First registration name is wrong")
    require(second == "test_plugin", "Second registration name is wrong")
    require(
        len(manager.plugins) == 1,
        "Duplicate registration created another plugin",
    )


def test_register_rejects_duplicate_name(manager):
    manager.register(PluginFixture())

    with pytest.raises(ValueError):
        manager.register(PluginFixture())


def test_disable_unregisters_plugin(manager):
    plugin = PluginFixture()
    manager.register(plugin)

    manager.disable("test_plugin")

    require(
        "test_plugin" not in manager.enabled,
        "Disabled plugin remains enabled",
    )
    require(
        not manager.pm.is_registered(plugin),
        "Disabled plugin remains registered with Pluggy",
    )


def test_enable_registers_disabled_plugin(manager):
    plugin = PluginFixture()
    manager.register(plugin)
    manager.disable("test_plugin")

    manager.enable("test_plugin")

    require(
        manager.is_enabled("test_plugin"),
        "Plugin was not enabled",
    )
    require(
        manager.pm.is_registered(plugin),
        "Enabled plugin was not registered with Pluggy",
    )


def test_unload_removes_plugin(manager):
    plugin = PluginFixture()
    manager.register(plugin)

    manager.unload("test_plugin")

    require(
        "test_plugin" not in manager.plugins,
        "Plugin remained in loaded registry",
    )
    require(
        "test_plugin" not in manager.enabled,
        "Plugin remained enabled after unload",
    )
    require(
        not manager.pm.is_registered(plugin),
        "Unloaded plugin remains registered with Pluggy",
    )


def test_unknown_plugin_operations_raise(manager):
    with pytest.raises(KeyError):
        manager.enable("missing")

    with pytest.raises(KeyError):
        manager.disable("missing")

    with pytest.raises(KeyError):
        manager.unload("missing")

    with pytest.raises(KeyError):
        manager.load_plugin("missing")


def test_allowlist_only_enables_selected_plugins(manager, monkeypatch):
    monkeypatch.setattr(
        manager,
        "_should_enable",
        lambda name: name == "test_plugin",
    )

    first = PluginFixture()
    second = OtherPlugin()

    manager.register(first)
    manager.register(second)

    require(
        manager.is_enabled("test_plugin"),
        "Allowlisted plugin was not enabled",
    )
    require(
        not manager.is_enabled("other_plugin"),
        "Non-allowlisted plugin was enabled",
    )


def test_denylist_overrides_allowlist(manager, monkeypatch):
    monkeypatch.setattr(
        manager,
        "_should_enable",
        lambda name: name == "test_plugin" and name != "other_plugin",
    )

    first = PluginFixture()
    second = OtherPlugin()

    manager.register(first)
    manager.register(second)

    require(
        manager.is_enabled("test_plugin"),
        "Allowed plugin was not enabled",
    )
    require(
        not manager.is_enabled("other_plugin"),
        "Denied plugin was enabled",
    )


def test_global_plugin_disable_prevents_activation(manager, monkeypatch):
    monkeypatch.setattr(
        manager,
        "_should_enable",
        lambda name: False,
    )

    plugin = PluginFixture()
    manager.register(plugin)

    require(
        not manager.is_enabled("test_plugin"),
        "Plugin system disabled but plugin was activated",
    )
    require(
        not manager.pm.is_registered(plugin),
        "Plugin system disabled but plugin was registered",
    )


def test_plugin_lists_are_reported(manager):
    first = PluginFixture()
    second = OtherPlugin()

    manager.register(first)
    manager.register(second)
    manager.disable("other_plugin")

    require(
        manager.list_plugins() == ["test_plugin", "other_plugin"],
        "Loaded plugin list is incorrect",
    )
    require(
        manager.list_enabled_plugins() == ["test_plugin"],
        "Enabled plugin list is incorrect",
    )
    require(
        manager.is_enabled("test_plugin"),
        "Enabled plugin was not reported as enabled",
    )
    require(
        not manager.is_enabled("other_plugin"),
        "Disabled plugin was reported as enabled",
    )


def test_plugin_hook_is_removed_when_disabled(manager):
    plugin = PluginFixture()
    manager.register(plugin)

    manager.disable("test_plugin")

    require(
        not manager.pm.is_registered(plugin),
        "Disabled plugin still participates in hooks",
    )


def test_plugin_hook_is_restored_when_reenabled(manager):
    plugin = PluginFixture()
    manager.register(plugin)

    manager.disable("test_plugin")
    manager.enable("test_plugin")

    require(
        manager.pm.is_registered(plugin),
        "Re-enabled plugin is not registered",
    )


def test_explicit_plugin_name_is_supported(manager):
    plugin = PluginFixture()

    name = manager.register(plugin, name="custom")

    require(name == "custom", "Explicit plugin name was ignored")
    require("custom" in manager.plugins, "Custom name was not stored")


def test_load_plugin_activates_loaded_plugin(manager):
    plugin = PluginFixture()

    manager.register(plugin)
    manager.disable("test_plugin")
    manager.load_plugin("test_plugin")

    require(
        manager.is_enabled("test_plugin"),
        "load_plugin did not activate the plugin",
    )


def test_global_disabled_configuration(manager, monkeypatch):
    monkeypatch.setattr(
        manager,
        "_should_enable",
        lambda name: False,
    )

    plugin = PluginFixture()
    manager.register(plugin)

    require(
        manager.list_plugins() == ["test_plugin"],
        "Plugin should still be discoverable while globally disabled",
    )
    require(
        manager.list_enabled_plugins() == [],
        "Globally disabled manager has active plugins",
    )
def test_entry_point_discovery(manager, monkeypatch):
    plugin = PluginFixture()

    entry_point = SimpleNamespace(
        name="entry_plugin",
        load=lambda: PluginFixture,
    )

    entry_points = SimpleNamespace(
        select=lambda group: [entry_point],
    )

    monkeypatch.setattr(
        "core.plugin_manager.importlib.metadata.entry_points",
        lambda: entry_points,
    )

    monkeypatch.setattr(
        manager,
        "_discover_local_plugins",
        lambda: None,
    )

    manager.discover_plugins()

    require(
        "entry_plugin" in manager.plugins,
        "Entry-point plugin was not discovered",
    )
    require(
        manager.is_enabled("entry_plugin"),
        "Entry-point plugin was not enabled",
    )


def test_local_plugin_instance_discovery(manager, tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    (plugin_dir / "local_plugin.py").write_text(
        """
from core.plugin_manager import hookimpl

class LocalPlugin:
    __plugin_name__ = "local_plugin"

    @hookimpl
    def analyze(self, value):
        return value + 1

plugin = LocalPlugin()
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "core.plugin_manager.settings.PLUGIN_DIR",
        str(plugin_dir),
    )

    monkeypatch.setattr(
        "core.plugin_manager.importlib.metadata.entry_points",
        lambda: SimpleNamespace(select=lambda group: []),
    )

    manager.discover_plugins()

    require(
        "local_plugin" in manager.plugins,
        "Local plugin instance was not discovered",
    )
    require(
        manager.is_enabled("local_plugin"),
        "Local plugin instance was not enabled",
    )


def test_local_plugin_register_function(manager, tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    (plugin_dir / "register_plugin.py").write_text(
        """
from core.plugin_manager import hookimpl

class RegisteredPlugin:
    __plugin_name__ = "registered_plugin"

    @hookimpl
    def analyze(self, value):
        return value + 2

def register(manager):
    manager.register(RegisteredPlugin())
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "core.plugin_manager.settings.PLUGIN_DIR",
        str(plugin_dir),
    )

    monkeypatch.setattr(
        "core.plugin_manager.importlib.metadata.entry_points",
        lambda: SimpleNamespace(select=lambda group: []),
    )

    manager.discover_plugins()

    require(
        "registered_plugin" in manager.plugins,
        "Plugin register() convention was not discovered",
    )


def test_local_plugin_class_discovery(manager, tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    (plugin_dir / "class_plugin.py").write_text(
        """
from core.plugin_manager import hookimpl

class ClassPlugin:
    __plugin_name__ = "class_plugin"

    @hookimpl
    def analyze(self, value):
        return value + 3
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "core.plugin_manager.settings.PLUGIN_DIR",
        str(plugin_dir),
    )

    monkeypatch.setattr(
        "core.plugin_manager.importlib.metadata.entry_points",
        lambda: SimpleNamespace(select=lambda group: []),
    )

    manager.discover_plugins()

    require(
        "class_plugin" in manager.plugins,
        "Plugin class convention was not discovered",
    )


def test_auto_load_disabled_skips_discovery(manager, monkeypatch):
    monkeypatch.setattr(
        "core.plugin_manager.settings.PLUGIN_AUTO_LOAD",
        False,
    )

    discover_entry_points = lambda: (_ for _ in ()).throw(
        RuntimeError("entry-point discovery should not run")
    )

    discover_local = lambda: (_ for _ in ()).throw(
        RuntimeError("local discovery should not run")
    )

    monkeypatch.setattr(
        manager,
        "_discover_entry_points",
        discover_entry_points,
    )
    monkeypatch.setattr(
        manager,
        "_discover_local_plugins",
        discover_local,
    )

    manager.discover_plugins()

    require(
        manager.list_plugins() == [],
        "Plugins were discovered with auto-load disabled",
    )
    require(
        manager._loaded,
        "Manager did not mark discovery complete",
    )


def test_global_plugin_disable_skips_discovery(manager, monkeypatch):
    monkeypatch.setattr(
        "core.plugin_manager.settings.PLUGIN_ENABLED",
        False,
    )

    discover_entry_points = lambda: (_ for _ in ()).throw(
        RuntimeError("entry-point discovery should not run")
    )

    discover_local = lambda: (_ for _ in ()).throw(
        RuntimeError("local discovery should not run")
    )

    monkeypatch.setattr(
        manager,
        "_discover_entry_points",
        discover_entry_points,
    )
    monkeypatch.setattr(
        manager,
        "_discover_local_plugins",
        discover_local,
    )

    manager.discover_plugins()

    require(
        manager.list_plugins() == [],
        "Plugins were discovered while plugin system was disabled",
    )
    require(
        manager._loaded,
        "Manager did not mark discovery complete",
    )
