from pathlib import Path

import pytest

from benchmark.github.adapter import GitHubAdapter
from benchmark.github.parser import GitHubParser


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    (root / "README.md").write_text(
        "# Test Repository\n\nA repository used for GitHub adapter tests.\n",
        encoding="utf-8",
    )

    package = root / "memory"
    package.mkdir()

    (package / "__init__.py").write_text(
        '"""Memory package."""\n',
        encoding="utf-8",
    )

    (package / "controller.py").write_text(
        '''
class BaseController:
    """Base controller."""

    def base_method(self):
        return "base"


class MemoryController(BaseController):
    """Controls memory operations."""

    def __init__(self, name):
        self.name = name

    def recall(self, query):
        return query


def helper(value):
    return value * 2


async def async_helper(value):
    return value
'''.lstrip(),
        encoding="utf-8",
    )

    (package / "nested.py").write_text(
        '''
class Outer:
    class Inner:
        def run(self):
            return True

    def outer_method(self):
        def nested():
            return "nested"

        return nested()


def outer_function():
    def nested_function():
        return "nested"

    return nested_function()
'''.lstrip(),
        encoding="utf-8",
    )

    nested = package / "subsystem"
    nested.mkdir()

    (nested / "worker.py").write_text(
        '''
class Worker:
    def run(self):
        return True


def standalone():
    return 42
'''.lstrip(),
        encoding="utf-8",
    )

    return root

def test_parser_initializes(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    require(parser.repository_path == repo.resolve(), "Repository path mismatch")
    require(
        parser.repository_name == "example/test",
        "Repository name mismatch",
    )


def test_parse_repository(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    record = parser.parse_repository()

    require(record.name == "test", "Repository name mismatch")
    require(record.path == ".", "Repository path mismatch")
    require("Test Repository" in record.text, "README content missing")
    require(record.metadata["source"] == "github", "Source metadata missing")
    require(
        record.metadata["record_type"] == "repository",
        "Record type mismatch",
    )
    require(
        record.metadata["repository"] == "example/test",
        "Repository metadata missing",
    )


def test_discover_directories(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    records = parser.parse_directories()

    paths = {record.path for record in records}

    require("memory" in paths, "memory directory missing")
    require("memory/subsystem" in paths, "nested directory missing")


def test_parse_files(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    records = parser.parse_files()

    paths = {record.path for record in records}

    require("README.md" in paths, "README missing")
    require("memory/controller.py" in paths, "controller.py missing")
    require(
        "memory/subsystem/worker.py" in paths,
        "worker.py missing",
    )


def test_python_file_contains_source(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    records = parser.parse_files()

    controller = next(
        record
        for record in records
        if record.path == "memory/controller.py"
    )

    require("class MemoryController" in controller.text, "Class source missing")
    require("def recall" in controller.text, "Method source missing")


def test_parse_symbols_finds_classes(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    records = parser.parse_symbols()

    classes = {
        record.metadata["qualified_name"]
        for record in records
        if record.metadata["symbol_type"] == "class"
    }

    require("BaseController" in classes, "BaseController missing")
    require("MemoryController" in classes, "MemoryController missing")
    require("Worker" in classes, "Worker missing")


def test_parse_symbols_finds_functions(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    records = parser.parse_symbols()

    functions = {
        record.metadata["qualified_name"]
        for record in records
        if record.metadata["symbol_type"] in {
            "function",
            "async_function",
        }
    }

    require("helper" in functions, "helper missing")
    require("async_helper" in functions, "async_helper missing")
    require("standalone" in functions, "standalone missing")


def test_parse_symbols_finds_methods(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    records = parser.parse_symbols()

    methods = {
        record.metadata["qualified_name"]
        for record in records
        if record.metadata["symbol_type"] == "method"
    }

    require(
        "BaseController.base_method" in methods,
        "BaseController.base_method missing",
    )
    require(
        "MemoryController.recall" in methods,
        "MemoryController.recall missing",
    )
    require(
        "Worker.run" in methods,
        "Worker.run missing",
    )


def test_parse_symbols_finds_async_functions(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    records = parser.parse_symbols()

    async_symbols = {
        record.metadata["qualified_name"]
        for record in records
        if record.metadata["symbol_type"] == "async_function"
    }

    require("async_helper" in async_symbols, "async function missing")


def test_symbol_metadata_contains_location(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    records = parser.parse_symbols()

    recall = next(
        record
        for record in records
        if record.metadata["qualified_name"] == "MemoryController.recall"
    )

    require(
        recall.metadata["path"] == "memory/controller.py",
        "Symbol path missing",
    )
    require(
        recall.metadata["line_start"] > 0,
        "Symbol start line missing",
    )
    require(
        recall.metadata["line_end"] >= recall.metadata["line_start"],
        "Symbol end line invalid",
    )
    require(
        recall.metadata["parent"] == "MemoryController",
        "Symbol parent missing",
    )


def test_class_metadata_contains_bases(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    records = parser.parse_symbols()

    controller = next(
        record
        for record in records
        if record.metadata["qualified_name"] == "MemoryController"
    )

    require(
        controller.metadata["bases"] == ["BaseController"],
        "Class bases missing",
    )


def test_python_symbol_text_is_exact_source(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    records = parser.parse_symbols()

    helper = next(
        record
        for record in records
        if record.metadata["qualified_name"] == "helper"
    )

    require(
        "def helper(value):" in helper.text,
        "Function source missing",
    )
    require(
        'return value * 2' in helper.text,
        "Function body missing",
    )


def test_non_python_files_have_no_symbols(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    records = parser.parse_symbols()

    paths = {record.path for record in records}

    require("README.md" not in paths, "README produced symbols")


def test_parser_ignores_git_directory(repo):
    git_dir = repo / ".git"
    git_dir.mkdir()
    (git_dir / "ignored.py").write_text(
        "def ignored():\n    return False\n",
        encoding="utf-8",
    )

    parser = GitHubParser(repo, repository_name="example/test")

    files = parser.parse_files()
    symbols = parser.parse_symbols()

    require(
        all(".git" not in record.path for record in files),
        ".git file was included",
    )
    require(
        all(".git" not in record.path for record in symbols),
        ".git symbol was included",
    )
def test_nested_classes_have_qualified_names(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    records = parser.parse_symbols()

    qualified_names = {
        record.metadata["qualified_name"]
        for record in records
    }

    require("Outer" in qualified_names, "Outer class missing")
    require("Outer.Inner" in qualified_names, "Nested class missing")
    require(
        "Outer.Inner.run" in qualified_names,
        "Nested class method missing",
    )


def test_nested_class_metadata_tracks_parent(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    records = parser.parse_symbols()

    inner = next(
        record
        for record in records
        if record.metadata["qualified_name"] == "Outer.Inner"
    )

    require(
        inner.metadata["parent"] == "Outer",
        "Nested class parent missing",
    )
    require(
        inner.metadata["symbol_type"] == "class",
        "Nested class type incorrect",
    )


def test_nested_method_has_full_qualified_name(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    records = parser.parse_symbols()

    method = next(
        record
        for record in records
        if record.metadata["name"] == "run"
    )

    require(
        method.metadata["qualified_name"] == "Outer.Inner.run",
        "Nested method qualified name incorrect",
    )


def test_nested_function_has_full_qualified_name(repo):
    parser = GitHubParser(repo, repository_name="example/test")

    records = parser.parse_symbols()

    qualified_names = {
        record.metadata["qualified_name"]
        for record in records
    }

    require(
        "Outer.outer_method.nested" in qualified_names,
        "Nested method function missing",
    )
    require(
        "outer_function.nested_function" in qualified_names,
        "Nested function missing",
    )
def test_adapter_initializes(repo):
    adapter = GitHubAdapter(
        repo,
        repository_name="example/test",
    )

    require(
        adapter.repository_name == "example/test",
        "Adapter repository name mismatch",
    )
    require(
        adapter.parser.repository_name == "example/test",
        "Adapter parser repository name mismatch",
    )


def test_adapter_repository_returns_normalized_record(repo):
    adapter = GitHubAdapter(
        repo,
        repository_name="example/test",
    )

    record = adapter.repository()

    require(isinstance(record, dict), "Repository record is not a dict")
    require("text" in record, "Repository text missing")
    require("metadata" in record, "Repository metadata missing")
    require(
        record["text"].startswith("# Test Repository"),
        "Repository text incorrect",
    )
    require(
        record["metadata"]["record_type"] == "repository",
        "Repository record type incorrect",
    )
    require(
        record["metadata"]["repository"] == "example/test",
        "Repository metadata incorrect",
    )


def test_adapter_directories_returns_normalized_records(repo):
    adapter = GitHubAdapter(
        repo,
        repository_name="example/test",
    )

    records = adapter.directories()

    require(isinstance(records, list), "directories() did not return a list")

    paths = {
        record["metadata"]["path"]
        for record in records
    }

    require("memory" in paths, "memory directory missing")
    require(
        "memory/subsystem" in paths,
        "memory/subsystem directory missing",
    )

    for record in records:
        require(isinstance(record, dict), "Directory record is not a dict")
        require("text" in record, "Directory text missing")
        require("metadata" in record, "Directory metadata missing")
        require(
            record["metadata"]["record_type"] == "directory",
            "Directory record type incorrect",
        )


def test_adapter_files_returns_normalized_records(repo):
    adapter = GitHubAdapter(
        repo,
        repository_name="example/test",
    )

    records = adapter.files()

    require(isinstance(records, list), "files() did not return a list")

    paths = {
        record["metadata"]["path"]
        for record in records
    }

    require("README.md" in paths, "README missing")
    require(
        "memory/controller.py" in paths,
        "controller.py missing",
    )
    require(
        "memory/nested.py" in paths,
        "nested.py missing",
    )
    require(
        "memory/subsystem/worker.py" in paths,
        "worker.py missing",
    )

    for record in records:
        require(isinstance(record, dict), "File record is not a dict")
        require("text" in record, "File text missing")
        require("metadata" in record, "File metadata missing")
        require(
            record["metadata"]["record_type"] == "file",
            "File record type incorrect",
        )


def test_adapter_symbols_returns_normalized_records(repo):
    adapter = GitHubAdapter(
        repo,
        repository_name="example/test",
    )

    records = adapter.symbols()

    require(isinstance(records, list), "symbols() did not return a list")

    qualified_names = {
        record["metadata"]["qualified_name"]
        for record in records
    }

    require(
        "MemoryController" in qualified_names,
        "MemoryController missing",
    )
    require(
        "MemoryController.recall" in qualified_names,
        "MemoryController.recall missing",
    )
    require(
        "Outer.Inner" in qualified_names,
        "Nested class missing",
    )
    require(
        "Outer.Inner.run" in qualified_names,
        "Nested method missing",
    )
    require(
        "Outer.outer_method.nested" in qualified_names,
        "Nested function missing",
    )


def test_adapter_iter_records_returns_all_record_types(repo):
    adapter = GitHubAdapter(
        repo,
        repository_name="example/test",
    )

    records = list(adapter.iter_records())

    require(records, "iter_records() returned no records")

    record_types = {
        record["metadata"]["record_type"]
        for record in records
    }

    require(
        record_types == {
            "repository",
            "directory",
            "file",
            "symbol",
        },
        "iter_records() did not return all record types",
    )


def test_adapter_records_returns_materialized_records(repo):
    adapter = GitHubAdapter(
        repo,
        repository_name="example/test",
    )

    records = adapter.records()

    require(isinstance(records, list), "records() did not return a list")
    require(records, "records() returned no records")

    for record in records:
        require(isinstance(record, dict), "Record is not a dict")
        require("text" in record, "Record text missing")
        require("metadata" in record, "Record metadata missing")


def test_adapter_load_inserts_every_record(repo):
    adapter = GitHubAdapter(
        repo,
        repository_name="example/test",
    )

    inserted = []

    def insert(text, metadata):
        inserted.append((text, metadata))

    count = adapter.load(insert)

    require(count > 0, "load() inserted no records")
    require(
        count == len(inserted),
        "load() count does not match inserted records",
    )

    for text, metadata in inserted:
        require(isinstance(text, str), "Inserted text is not a string")
        require(
            isinstance(metadata, dict),
            "Inserted metadata is not a dict",
        )


def test_adapter_load_preserves_symbol_metadata(repo):
    adapter = GitHubAdapter(
        repo,
        repository_name="example/test",
    )

    inserted = []

    def insert(text, metadata):
        inserted.append((text, metadata))

    adapter.load(insert)

    inner = next(
        metadata
        for _, metadata in inserted
        if metadata.get("qualified_name") == "Outer.Inner"
    )

    require(
        inner["source"] == "github",
        "Source metadata lost",
    )
    require(
        inner["repository"] == "example/test",
        "Repository metadata lost",
    )
    require(
        inner["record_type"] == "symbol",
        "Record type lost",
    )
    require(
        inner["symbol_type"] == "class",
        "Symbol type lost",
    )
    require(
        inner["parent"] == "Outer",
        "Parent metadata lost",
    )


def test_adapter_load_rejects_non_callable_insert(repo):
    adapter = GitHubAdapter(
        repo,
        repository_name="example/test",
    )

    with pytest.raises(TypeError, match="insert must be callable"):
        adapter.load(None)


def test_adapter_stats_reports_record_counts(repo):
    adapter = GitHubAdapter(
        repo,
        repository_name="example/test",
    )

    stats = adapter.stats()

    require(
        stats["repository"] == "example/test",
        "Stats repository incorrect",
    )
    require(
        stats["records"] > 0,
        "Stats reported no records",
    )
    require(
        stats["repositories"] == 1,
        "Repository count incorrect",
    )
    require(
        stats["directories"] == 2,
        "Directory count incorrect",
    )
    require(
        stats["files"] == 5,
        "File count incorrect",
    )
    require(
        stats["symbols"] > 0,
        "Symbol count incorrect",
    )


def test_adapter_normalization_does_not_expose_dataclasses(repo):
    adapter = GitHubAdapter(
        repo,
        repository_name="example/test",
    )

    records = adapter.records()

    for record in records:
        require(
            isinstance(record, dict),
            "Normalized record is not a dict",
        )
        require(
            isinstance(record["text"], str),
            "Normalized text is not a string",
        )
        require(
            isinstance(record["metadata"], dict),
            "Normalized metadata is not a dict",
        )
def test_adapter_is_exported_from_package():
    from benchmark.github import GitHubAdapter as ExportedGitHubAdapter

    require(
        ExportedGitHubAdapter is GitHubAdapter,
        "GitHubAdapter package export does not match the implementation",
    )
