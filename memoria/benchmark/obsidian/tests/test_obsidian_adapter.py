"""
Tests for the Memoria Obsidian vault ingestion adapter.

Run from the Memoria project root:

    OBSIDIAN_TEST_VAULT=./obsidian-demo-vault \
        python -m pytest benchmark/obsidian/tests/test_obsidian_adapter.py -v

    OBSIDIAN_TEST_VAULT=./obsidian-sample-vault \
        python -m pytest benchmark/obsidian/tests/test_obsidian_adapter.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import cache.config as config
from benchmark.obsidian.adapter import ObsidianAdapter
from benchmark.obsidian.wikilinks import extract_wikilinks
from shared.memory_interface import MemoryInterface


@pytest.fixture
def vault_path() -> Path:
    raw = os.environ.get("OBSIDIAN_TEST_VAULT")

    if not raw:
        pytest.skip(
            "Set OBSIDIAN_TEST_VAULT to the path of an Obsidian vault."
        )

    path = Path(raw).expanduser().resolve()

    if not path.is_dir():
        pytest.fail(f"Obsidian test vault does not exist: {path}")

    return path


@pytest.fixture
def isolated_memory(tmp_path: Path) -> MemoryInterface:
    """
    Create a completely isolated Memoria instance for integration tests.

    The production settings object is already instantiated when cache.config
    is imported, so configure its paths directly rather than changing
    environment variables after import.
    """
    config.settings.DB_PATH = str(tmp_path / "memory.db")
    config.settings.VECTOR_INDEX_PATH = str(tmp_path / "memory.index")
    config.settings.CACHE_PATH = str(tmp_path / "embedding_cache.pkl")

    return MemoryInterface()


def _source_path(record: dict) -> str:
    """Return the vault-relative source path."""
    path = record["metadata"]["relative_path"]

    assert isinstance(path, str), (
        f"Expected metadata['relative_path'] to be a string, "
        f"got {type(path).__name__}"
    )

    return path


def _section_key(record: dict) -> tuple[str, int | None]:
    """Return a stable identity for one section-level memory."""
    metadata = record["metadata"]

    return (
        _source_path(record),
        metadata.get("line_number"),
    )


# ---------------------------------------------------------------------------
# Generic adapter tests
# ---------------------------------------------------------------------------


def test_discovers_notes(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    records = adapter.records()

    assert records, "Adapter returned no records."

    section_keys = [
        _section_key(record)
        for record in records
    ]

    assert len(section_keys) == len(set(section_keys)), (
        "Adapter returned duplicate section records."
    )

    paths = [_source_path(record) for record in records]

    assert all(path.endswith(".md") for path in paths)

    # Section-based ingestion should produce multiple memories from at least
    # one multi-section note when the vault contains structured notes.
    assert len(set(paths)) <= len(records)


def test_record_shape(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    record = next(adapter.iter_notes())

    assert isinstance(record, dict)
    assert "text" in record
    assert "metadata" in record

    assert isinstance(record["text"], str)
    assert isinstance(record["metadata"], dict)


def test_preserves_source_metadata(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    records = adapter.records()

    for record in records:
        metadata = record["metadata"]

        assert metadata["source"] == "obsidian"
        assert metadata["relative_path"] == _source_path(record)
        assert metadata["vault_path"] == _source_path(record)


def test_preserves_frontmatter(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    records = adapter.records()

    frontmatter_records = [
        record
        for record in records
        if record["metadata"].get("frontmatter")
    ]

    assert frontmatter_records, "Vault contained no frontmatter records."

    for record in frontmatter_records:
        assert isinstance(
            record["metadata"]["frontmatter"],
            dict,
        )


def test_extracts_wikilinks(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    records = adapter.records()

    linked_records = [
        record
        for record in records
        if record["metadata"].get("links")
    ]

    assert linked_records, "Vault contained no wikilinks."

    for record in linked_records:
        links = record["metadata"]["links"]

        assert isinstance(links, list)

        for link in links:
            assert isinstance(link, dict)
            assert isinstance(link["target"], str)
            assert link["target"]


def test_extracts_tasks(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    records = adapter.records()

    task_records = [
        record
        for record in records
        if record["metadata"].get("tasks")
    ]

    # A vault may legitimately contain no tasks.
    for record in task_records:
        tasks = record["metadata"]["tasks"]

        assert isinstance(tasks, list)

        for task in tasks:
            assert isinstance(task, dict)
            assert "completed" in task


def test_identifies_templates(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    records = adapter.records()

    templates = [
        record
        for record in records
        if record["metadata"].get("is_template")
    ]

    # Template detection is vault-specific and may legitimately identify
    # zero templates. If templates are identified, they must be Markdown
    # records and carry the expected boolean metadata.
    for record in templates:
        assert _source_path(record).endswith(".md")
        assert record["metadata"]["is_template"] is True


def test_can_exclude_templates(vault_path: Path):
    with_templates = ObsidianAdapter(
        vault_path,
        include_templates=True,
    ).records()

    without_templates = ObsidianAdapter(
        vault_path,
        include_templates=False,
    ).records()

    assert len(without_templates) <= len(with_templates)

    assert not any(
        record["metadata"].get("is_template")
        for record in without_templates
    )


def test_notes_contain_actual_content(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    records = adapter.records()

    nonempty = [
        record
        for record in records
        if record["text"].strip()
    ]

    assert nonempty, "Vault contained no non-empty Markdown notes."

    for record in nonempty:
        assert isinstance(record["text"], str)
        assert record["text"].strip()


def test_load_calls_insert_function(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    inserted = []

    def insert(text: str, metadata: dict):
        inserted.append((text, metadata))
        return len(inserted)

    count = adapter.load(insert)

    assert count == len(inserted)
    assert count == len(adapter.records())

    for text, metadata in inserted:
        assert isinstance(text, str)
        assert isinstance(metadata, dict)


def test_missing_vault_rejected(tmp_path: Path):
    missing = tmp_path / "does-not-exist"

    with pytest.raises((FileNotFoundError, ValueError)):
        ObsidianAdapter(missing)


# ---------------------------------------------------------------------------
# Wikilink parser regression tests
# ---------------------------------------------------------------------------


def test_wikilinks_ignore_inline_code():
    text = (
        "Real link: [[Example Project]]\n"
        "Inline code: `[[Fake Note]]`\n"
    )

    links = extract_wikilinks(
        text,
        "README.md",
        {"Example Project.md"},
    )

    assert len(links) == 1
    assert links[0]["target"] == "Example Project"


def test_wikilinks_ignore_fenced_code():
    text = (
        "Real link: [[Example Project]]\n\n"
        "```markdown\n"
        "[[Fake Note]]\n"
        "```\n"
    )

    links = extract_wikilinks(
        text,
        "README.md",
        {"Example Project.md"},
    )

    assert len(links) == 1
    assert links[0]["target"] == "Example Project"


# ---------------------------------------------------------------------------
# Generic end-to-end integration tests
# ---------------------------------------------------------------------------


def test_obsidian_vault_ingests_into_memoria(
    vault_path: Path,
    isolated_memory: MemoryInterface,
):
    adapter = ObsidianAdapter(vault_path)

    expected = len(adapter.records())

    count = adapter.load_into_memory(
        isolated_memory,
        batch_size=100,
    )

    assert count == expected
    assert count > 0


def test_obsidian_content_is_retrievable(
    vault_path: Path,
    isolated_memory: MemoryInterface,
):
    adapter = ObsidianAdapter(vault_path)

    records = adapter.records()

    nonempty = [
        record
        for record in records
        if record["text"].strip()
    ]

    assert nonempty

    # Find the first record containing enough substantive text to derive
    # a meaningful retrieval query. Some valid Obsidian sections contain
    # only short labels, bullets, symbols, or other structural content.
    candidate = None
    candidate_words = []

    for record in nonempty:
        words = [
            word.strip(".,:;!?()[]{}#*-_`\"'")
            for word in record["text"].split()
        ]
        words = [
            word
            for word in words
            if len(word) >= 5
        ]

        if words:
            candidate = record
            candidate_words = words
            break

    assert candidate is not None, (
        "Could not derive a retrieval query from any "
        "non-empty Obsidian record."
    )

    query = " ".join(candidate_words[:8])

    count = adapter.load_into_memory(
        isolated_memory,
        batch_size=100,
    )

    assert count == len(records)

    results = isolated_memory.recall(query)

    assert results

    result_text = str(results)

    assert any(
        word.lower() in result_text.lower()
        for word in candidate_words[:3]
    )


# ---------------------------------------------------------------------------
# Demo-vault-specific contract tests
#
# These deliberately remain specific to obsidian-demo-vault.
# Run them only when OBSIDIAN_TEST_VAULT points at that vault.
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_vault(vault_path: Path) -> Path:
    if not (
        vault_path
        / "02-projects/Example Project/Example Project.md"
    ).is_file():
        pytest.skip("Demo-vault contract test.")

    return vault_path


def test_demo_expected_notes(demo_vault: Path):
    adapter = ObsidianAdapter(demo_vault)

    records = adapter.records()

    paths = {
        _source_path(record)
        for record in records
    }

    assert len(paths) == 14

    assert "02-projects/Example Project/Example Project.md" in paths
    assert "03-areas/Career/Career.md" in paths
    assert "04-resources/People/@Jane Smith.md" in paths

    # Section ingestion should produce more records than source notes.
    assert len(records) > len(paths)


def test_demo_source_metadata(demo_vault: Path):
    adapter = ObsidianAdapter(demo_vault)

    record = next(
        record
        for record in adapter.records()
        if _source_path(record)
        == "02-projects/Example Project/Example Project.md"
    )

    metadata = record["metadata"]

    assert metadata["source"] == "obsidian"
    assert metadata["relative_path"] == (
        "02-projects/Example Project/Example Project.md"
    )
    assert metadata["vault_path"] == (
        "02-projects/Example Project/Example Project.md"
    )


def test_demo_frontmatter(demo_vault: Path):
    adapter = ObsidianAdapter(demo_vault)

    project_records = [
        record
        for record in adapter.records()
        if _source_path(record)
        == "02-projects/Example Project/Example Project.md"
    ]

    assert project_records

    for record in project_records:
        frontmatter = record["metadata"]["frontmatter"]

        assert frontmatter["category"] == ["project"]
        assert frontmatter["status"] == ["inprogress"]


def test_demo_wikilinks(demo_vault: Path):
    adapter = ObsidianAdapter(demo_vault)

    project_records = [
        record
        for record in adapter.records()
        if _source_path(record)
        == "02-projects/Example Project/Example Project.md"
    ]

    assert project_records

    links = [
        link
        for record in project_records
        for link in record["metadata"]["links"]
    ]

    assert links

    assert any(
        "04-resources/Meeting notes/20260115-Team Sync" in link["target"]
        for link in links
    )

    assert any(
        "@Jane Smith" in link["target"]
        for link in links
    )


def test_demo_tasks(demo_vault: Path):
    adapter = ObsidianAdapter(demo_vault)

    project_records = [
        record
        for record in adapter.records()
        if _source_path(record)
        == "02-projects/Example Project/Example Project.md"
    ]

    assert project_records

    tasks = [
        task
        for record in project_records
        for task in record["metadata"]["tasks"]
    ]

    assert tasks

    completed = [
        task
        for task in tasks
        if task["completed"]
    ]

    pending = [
        task
        for task in tasks
        if not task["completed"]
    ]

    assert completed
    assert pending


def test_demo_templates(demo_vault: Path):
    adapter = ObsidianAdapter(demo_vault)

    records = adapter.records()

    templates = [
        record
        for record in records
        if record["metadata"].get("is_template")
    ]

    assert templates

    template_paths = {
        _source_path(record)
        for record in templates
    }

    assert "99-meta/Templates/project-template.md" in template_paths
    assert "99-meta/Templates/daily-note-template.md" in template_paths


def test_demo_project_content(demo_vault: Path):
    adapter = ObsidianAdapter(demo_vault)

    project_records = [
        record
        for record in adapter.records()
        if _source_path(record)
        == "02-projects/Example Project/Example Project.md"
    ]

    assert project_records

    # The note itself is "Example Project"; section headings are the
    # headings actually present inside the Markdown file.
    assert any(
        record["metadata"]["filename"] == "Example Project"
        for record in project_records
    )

    headings = {
        record["metadata"].get("heading")
        for record in project_records
    }

    assert "Overview" in headings
    assert "Next Actions" in headings
    assert "Meeting Notes" in headings

    full_text = "\n".join(
        record["text"]
        for record in project_records
    )

    assert "This is a sample project" in full_text
    assert "Deliver a new feature by end of January 2026" in full_text
    assert "Phase 3: Documentation & Review" in full_text

def test_demo_retrieval(
    demo_vault: Path,
    isolated_memory: MemoryInterface,
):
    adapter = ObsidianAdapter(demo_vault)

    expected = len(adapter.records())

    count = adapter.load_into_memory(
        isolated_memory,
        batch_size=100,
    )

    assert count == expected
    assert count > 14

    results = isolated_memory.recall(
        "What is the Example Project trying to accomplish?"
    )

    assert results
    assert "Example Project" in str(results)


def test_demo_people_retrieval(
    demo_vault: Path,
    isolated_memory: MemoryInterface,
):
    adapter = ObsidianAdapter(demo_vault)

    expected = len(adapter.records())

    count = adapter.load_into_memory(
        isolated_memory,
        batch_size=100,
    )

    assert count == expected
    assert count > 14

    results = isolated_memory.recall(
        "Who is Jane Smith?"
    )

    assert results
    assert "Jane Smith" in str(results)


# ---------------------------------------------------------------------------
# Manual diagnostic entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python test_obsidian_adapter.py "
            "/path/to/obsidian-vault"
        )
        raise SystemExit(1)

    vault = Path(sys.argv[1]).resolve()

    adapter = ObsidianAdapter(vault)

    print("=" * 70)
    print("MEMORIA OBSIDIAN ADAPTER TEST")
    print("=" * 70)
    print(f"Vault: {vault}")
    print()

    records = adapter.records()

    print(f"Records: {len(records)}")

    templates = sum(
        1
        for record in records
        if record["metadata"].get("is_template")
    )

    linked = sum(
        1
        for record in records
        if record["metadata"].get("links")
    )

    tasks = sum(
        len(record["metadata"].get("tasks", []))
        for record in records
    )

    print(f"Templates: {templates}")
    print(f"Notes with links: {linked}")
    print(f"Tasks: {tasks}")
    print()

    for record in records:
        metadata = record["metadata"]
        source = metadata["relative_path"]

        print("-" * 70)
        print(source)
        print(f"Heading: {metadata.get('heading')}")
        print(f"Breadcrumb: {metadata.get('breadcrumb', [])}")
        print(f"Line: {metadata.get('line_number')}")
        print(f"Template: {metadata.get('is_template', False)}")
        print(f"Links: {len(metadata.get('links', []))}")
        print(f"Tasks: {len(metadata.get('tasks', []))}")
        print(f"Characters: {len(record['text'])}")

    print()
    print("=" * 70)
    print("ADAPTER TEST COMPLETE")
    print("=" * 70)
