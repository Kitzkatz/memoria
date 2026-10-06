from pathlib import Path

import pytest

from core.plugin_generator import (
    HOOK_GROUPS,
    generate_plugin,
    normalize_plugin_name,
    plugin_class_name,
    render_plugin_module,
    render_readme,
    render_tests,
    validate_hook_groups,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def test_normalize_plugin_name():
    require(
        normalize_plugin_name("GitHub Tools") == "github_tools",
        "GitHub Tools should normalize to github_tools",
    )

    require(
        normalize_plugin_name("github-tools") == "github_tools",
        "github-tools should normalize to github_tools",
    )

    require(
        normalize_plugin_name("  My Plugin  ") == "my_plugin",
        "Whitespace should be removed during normalization",
    )

    require(
        normalize_plugin_name("123plugin") == "plugin_123plugin",
        "Names beginning with a digit should receive a plugin prefix",
    )


def test_normalize_plugin_name_rejects_empty_name():
    with pytest.raises(ValueError):
        normalize_plugin_name("")


def test_plugin_class_name():
    require(
        plugin_class_name("github_tools") == "GithubToolsPlugin",
        "Class name conversion is incorrect",
    )

    require(
        plugin_class_name("obsidian") == "ObsidianPlugin",
        "Single-word class name conversion is incorrect",
    )


def test_validate_hook_groups():
    groups = validate_hook_groups(
        [
            "retrieval",
            "ranking",
            "retrieval",
        ]
    )

    require(
        groups == ["retrieval", "ranking"],
        "Hook groups should be normalized and deduplicated",
    )


def test_validate_hook_groups_rejects_unknown_group():
    with pytest.raises(ValueError):
        validate_hook_groups(["not_a_real_hook_group"])


def test_render_plugin_module():
    source = render_plugin_module(
        "example_plugin",
        "Example plugin for testing.",
        ["retrieval", "ranking"],
    )

    required_fragments = [
        "class ExamplePlugin:",
        '__plugin_name__ = "example_plugin"',
        "def memoria_register_retriever(",
        "def memoria_retrieval_pre(",
        "def memoria_retrieval_post(",
        "def memoria_register_ranking_signal(",
        "def memoria_register_reranker(",
        "def memoria_ranking_pre(",
        "def memoria_ranking_post(",
        "plugin = ExamplePlugin()",
    ]

    for fragment in required_fragments:
        require(
            fragment in source,
            f"Generated plugin source is missing: {fragment}",
        )

    compile(source, "example_plugin.py", "exec")


def test_render_readme():
    readme = render_readme(
        "example_plugin",
        "Example plugin for testing.",
        ["retrieval", "ranking"],
    )

    required_fragments = [
        "# Example Plugin",
        "Example plugin for testing.",
        "- `retrieval`",
        "- `ranking`",
        "`example_plugin.py`",
        "plugins/",
    ]

    for fragment in required_fragments:
        require(
            fragment in readme,
            f"Generated README is missing: {fragment}",
        )


def test_render_tests():
    test_source = render_tests(
        "example_plugin",
        ["retrieval"],
    )

    required_fragments = [
        "def test_plugin_module_loads():",
        "def test_plugin_is_pluggy_compatible():",
        "def test_generated_hooks_exist():",
        "memoria_register_retriever",
        "memoria_retrieval_pre",
        "memoria_retrieval_post",
    ]

    for fragment in required_fragments:
        require(
            fragment in test_source,
            f"Generated test source is missing: {fragment}",
        )

    compile(test_source, "test_example_plugin.py", "exec")


def test_generate_plugin(tmp_path: Path):
    generated = generate_plugin(
        plugin_name="Example Plugin",
        description="Example generated plugin.",
        hook_groups=["retrieval", "ranking"],
        output_dir=tmp_path,
        create_tests=True,
        create_readme=True,
    )

    expected_keys = {"plugin", "readme", "tests"}

    require(
        set(generated) == expected_keys,
        "Generator did not return the expected generated files",
    )

    plugin_file = tmp_path / "example_plugin.py"
    readme_file = tmp_path / "example_plugin.md"
    test_file = tmp_path / "tests" / "test_example_plugin.py"

    for path in (plugin_file, readme_file, test_file):
        require(
            path.exists(),
            f"Expected generated file does not exist: {path}",
        )

    plugin_source = plugin_file.read_text(encoding="utf-8")
    readme_source = readme_file.read_text(encoding="utf-8")
    test_source = test_file.read_text(encoding="utf-8")

    compile(plugin_source, str(plugin_file), "exec")
    compile(test_source, str(test_file), "exec")

    require(
        "ExamplePlugin" in plugin_source,
        "Generated plugin class is missing",
    )

    require(
        "Example generated plugin." in readme_source,
        "Generated README description is missing",
    )

    require(
        "test_plugin_module_loads" in test_source,
        "Generated test file is missing its load test",
    )


def test_generate_plugin_rejects_existing_plugin(tmp_path: Path):
    plugin_file = tmp_path / "example_plugin.py"
    plugin_file.write_text(
        "existing plugin",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        generate_plugin(
            plugin_name="example_plugin",
            output_dir=tmp_path,
        )


def test_generate_plugin_without_optional_files(tmp_path: Path):
    generated = generate_plugin(
        plugin_name="minimal_plugin",
        description="Minimal plugin.",
        hook_groups=[],
        output_dir=tmp_path,
        create_tests=False,
        create_readme=False,
    )

    require(
        set(generated) == {"plugin"},
        "Minimal generation should only create the plugin module",
    )

    plugin_file = tmp_path / "minimal_plugin.py"

    require(
        plugin_file.exists(),
        "Minimal plugin module was not generated",
    )

    source = plugin_file.read_text(encoding="utf-8")

    compile(source, str(plugin_file), "exec")

    require(
        "class MinimalPlugin:" in source,
        "Minimal plugin class is missing",
    )


def test_hook_groups_are_complete():
    expected_groups = {
        "analysis",
        "evaluation",
        "feedback",
        "ingestion",
        "lifecycle",
        "query",
        "ranking",
        "retrieval",
        "routing",
        "scheduler",
        "storage",
    }

    require(
        set(HOOK_GROUPS) == expected_groups,
        "HOOK_GROUPS does not contain the expected hook groups",
    )
