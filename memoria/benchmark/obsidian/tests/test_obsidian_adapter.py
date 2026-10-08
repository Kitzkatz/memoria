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


def require(condition, message):
    """Fail with a useful diagnostic without using bare assertions."""
    if not condition:
        raise RuntimeError(message)


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

    require(
        isinstance(path, str),
        (
            "Expected metadata['relative_path'] to be a string, "
            f"got {type(path).__name__}"
        ),
    )

    return path


def _section_key(record: dict) -> tuple[str, int | None, int]:
    """Return a stable identity for one section-level memory."""
    metadata = record["metadata"]

    return (
        _source_path(record),
        metadata.get("line_number"),
        metadata.get("chunk_index", 0),
    )


# ---------------------------------------------------------------------------
# Generic adapter tests
# ---------------------------------------------------------------------------


def test_discovers_notes(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    records = adapter.records()

    require(records, "Adapter returned no records.")

    section_keys = [
        _section_key(record)
        for record in records
    ]

    require(
        len(section_keys) == len(set(section_keys)),
        "Adapter returned duplicate section records.",
    )

    paths = [_source_path(record) for record in records]

    require(
        all(path.endswith(".md") for path in paths),
        "Adapter returned a non-Markdown source path.",
    )

    # Section-based ingestion should produce multiple memories from at least
    # one multi-section note when the vault contains structured notes.
    require(
        len(set(paths)) <= len(records),
        "Record count is smaller than the number of source notes.",
    )


def test_record_shape(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    record = next(adapter.iter_notes())

    require(isinstance(record, dict), "Record is not a dict.")
    require("text" in record, "Record is missing 'text'.")
    require("metadata" in record, "Record is missing 'metadata'.")
    require(
        isinstance(record["text"], str),
        "Record 'text' is not a string.",
    )
    require(
        isinstance(record["metadata"], dict),
        "Record 'metadata' is not a dict.",
    )


def test_preserves_source_metadata(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    records = adapter.records()

    for record in records:
        metadata = record["metadata"]

        require(
            metadata["source"] == "obsidian",
            "Record source metadata is not 'obsidian'.",
        )
        require(
            metadata["relative_path"] == _source_path(record),
            "relative_path does not match the record source path.",
        )
        require(
            metadata["vault_path"] == _source_path(record),
            "vault_path does not match the record source path.",
        )


def test_preserves_frontmatter(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    records = adapter.records()

    frontmatter_records = [
        record
        for record in records
        if record["metadata"].get("frontmatter")
    ]

    require(
        frontmatter_records,
        "Vault contained no frontmatter records.",
    )

    for record in frontmatter_records:
        require(
            isinstance(record["metadata"]["frontmatter"], dict),
            "Frontmatter metadata is not a dict.",
        )


def test_extracts_wikilinks(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    records = adapter.records()

    linked_records = [
        record
        for record in records
        if record["metadata"].get("links")
    ]

    require(
        linked_records,
        "Vault contained no wikilinks.",
    )

    for record in linked_records:
        links = record["metadata"]["links"]

        require(
            isinstance(links, list),
            "Record links metadata is not a list.",
        )

        for link in links:
            require(
                isinstance(link, dict),
                "Wikilink metadata entry is not a dict.",
            )
            require(
                isinstance(link["target"], str),
                "Wikilink target is not a string.",
            )
            require(
                bool(link["target"]),
                "Wikilink target is empty.",
            )


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

        require(
            isinstance(tasks, list),
            "Tasks metadata is not a list.",
        )

        for task in tasks:
            require(
                isinstance(task, dict),
                "Task metadata entry is not a dict.",
            )
            require(
                "completed" in task,
                "Task metadata is missing 'completed'.",
            )


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
        require(
            _source_path(record).endswith(".md"),
            "Template source path is not Markdown.",
        )
        require(
            record["metadata"]["is_template"] is True,
            "Template record does not have is_template=True.",
        )


def test_can_exclude_templates(vault_path: Path):
    with_templates = ObsidianAdapter(
        vault_path,
        include_templates=True,
    ).records()

    without_templates = ObsidianAdapter(
        vault_path,
        include_templates=False,
    ).records()

    require(
        len(without_templates) <= len(with_templates),
        "Excluding templates increased the record count.",
    )

    require(
        not any(
            record["metadata"].get("is_template")
            for record in without_templates
        ),
        "A template record remained after template exclusion.",
    )


def test_notes_contain_actual_content(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    records = adapter.records()

    nonempty = [
        record
        for record in records
        if record["text"].strip()
    ]

    require(
        nonempty,
        "Vault contained no non-empty Markdown notes.",
    )

    for record in nonempty:
        require(
            isinstance(record["text"], str),
            "Non-empty record text is not a string.",
        )
        require(
            bool(record["text"].strip()),
            "Record text unexpectedly became empty.",
        )


def test_load_calls_insert_function(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    inserted = []

    def insert(text: str, metadata: dict):
        inserted.append((text, metadata))
        return len(inserted)

    count = adapter.load(insert)

    require(
        count == len(inserted),
        "load() count does not match inserted callback count.",
    )
    require(
        count == len(adapter.records()),
        "load() count does not match adapter record count.",
    )

    for text, metadata in inserted:
        require(
            isinstance(text, str),
            "load() supplied non-string text.",
        )
        require(
            isinstance(metadata, dict),
            "load() supplied non-dict metadata.",
        )


def test_missing_vault_rejected(tmp_path: Path):
    missing = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError):
        ObsidianAdapter(missing)


def test_file_path_rejected(tmp_path: Path):
    not_a_vault = tmp_path / "vault.md"
    not_a_vault.write_text("# Not a vault", encoding="utf-8")

    with pytest.raises(ValueError):
        ObsidianAdapter(not_a_vault)


def test_load_rejects_non_callable(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    with pytest.raises(TypeError):
        adapter.load(None)


def test_records_empty_vault(tmp_path: Path):
    vault = tmp_path / "empty-vault"
    vault.mkdir()

    adapter = ObsidianAdapter(vault)

    records = adapter.records()

    require(
        records == [],
        f"Empty vault returned {len(records)} records.",
    )


def test_load_empty_vault(tmp_path: Path):
    vault = tmp_path / "empty-vault"
    vault.mkdir()

    adapter = ObsidianAdapter(vault)

    inserted = []

    def insert(text: str, metadata: dict):
        inserted.append((text, metadata))

    count = adapter.load(insert)

    require(
        count == 0,
        f"Empty vault load returned count={count}.",
    )
    require(
        inserted == [],
        "Empty vault invoked the insertion callback.",
    )


def test_text_before_first_heading_is_kept(tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()

    (vault / "plain.md").write_text(
        "No heading here.\n- [ ] loose task\n",
        encoding="utf-8",
    )
    (vault / "intro.md").write_text(
        "---\ntags: [a]\n---\nIntro text.\n\n# Title\n\nBody.\n- [ ] late task\n",
        encoding="utf-8",
    )

    records = {
        (record["metadata"]["filename"], record["metadata"]["heading"]): record
        for record in ObsidianAdapter(vault).records()
    }

    require(
        set(records) == {("plain", ""), ("intro", ""), ("intro", "Title")},
        f"Unexpected records: {sorted(records)}",
    )
    require(
        records[("plain", "")]["metadata"]["tasks"][0]["text"] == "loose task",
        "A headingless note lost its task.",
    )
    require(
        records[("intro", "")]["text"] == "Intro text."
        and records[("intro", "")]["metadata"]["tasks"] == [],
        "The text before the first heading was not kept on its own.",
    )


def test_stats_reports_vault_counts(vault_path: Path):
    adapter = ObsidianAdapter(vault_path)

    records = adapter.records()
    stats = adapter.stats()

    expected_templates = sum(
        1
        for record in records
        if record["metadata"].get("is_template", False)
    )

    expected_linked = sum(
        1
        for record in records
        if record["metadata"].get("links")
    )

    expected_tasks = sum(
        len(record["metadata"].get("tasks", []))
        for record in records
    )

    require(
        stats["vault_path"] == str(adapter.vault_path),
        "stats() returned the wrong vault_path.",
    )
    require(
        stats["records"] == len(records),
        "stats() returned the wrong record count.",
    )
    require(
        stats["templates"] == expected_templates,
        "stats() returned the wrong template count.",
    )
    require(
        stats["linked_notes"] == expected_linked,
        "stats() returned the wrong linked-note count.",
    )
    require(
        stats["tasks"] == expected_tasks,
        "stats() returned the wrong task count.",
    )


def test_empty_vault_stats(tmp_path: Path):
    vault = tmp_path / "empty-vault"
    vault.mkdir()

    adapter = ObsidianAdapter(vault)

    stats = adapter.stats()

    require(
        stats["vault_path"] == str(adapter.vault_path),
        "Empty-vault stats returned the wrong vault_path.",
    )
    require(
        stats["records"] == 0,
        "Empty-vault stats returned a non-zero record count.",
    )
    require(
        stats["templates"] == 0,
        "Empty-vault stats returned a non-zero template count.",
    )
    require(
        stats["linked_notes"] == 0,
        "Empty-vault stats returned a non-zero linked-note count.",
    )
    require(
        stats["tasks"] == 0,
        "Empty-vault stats returned a non-zero task count.",
    )


# ---------------------------------------------------------------------------
# Adapter validation and parser contract tests
# ---------------------------------------------------------------------------


def test_iter_notes_validates_parser_records(
    vault_path: Path,
    monkeypatch,
):
    adapter = ObsidianAdapter(vault_path)

    def bad_parse(path):
        return [
            {
                "text": 123,
                "metadata": {},
            }
        ]

    monkeypatch.setattr(adapter.parser, "parse", bad_parse)

    with pytest.raises(TypeError, match="text.*string"):
        list(adapter.iter_notes())


def test_iter_notes_rejects_missing_text(
    vault_path: Path,
    monkeypatch,
):
    adapter = ObsidianAdapter(vault_path)

    def bad_parse(path):
        return [
            {
                "metadata": {},
            }
        ]

    monkeypatch.setattr(adapter.parser, "parse", bad_parse)

    with pytest.raises(ValueError, match="missing required 'text'"):
        list(adapter.iter_notes())


def test_iter_notes_rejects_missing_metadata(
    vault_path: Path,
    monkeypatch,
):
    adapter = ObsidianAdapter(vault_path)

    def bad_parse(path):
        return [
            {
                "text": "content",
            }
        ]

    monkeypatch.setattr(adapter.parser, "parse", bad_parse)

    with pytest.raises(ValueError, match="missing required 'metadata'"):
        list(adapter.iter_notes())


def test_iter_notes_propagates_parser_failure(
    vault_path: Path,
    monkeypatch,
):
    adapter = ObsidianAdapter(vault_path)

    def broken_parse(path):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(adapter.parser, "parse", broken_parse)

    with pytest.raises(RuntimeError, match="parser exploded"):
        list(adapter.iter_notes())


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

    require(
        len(links) == 1,
        f"Expected one wikilink, got {len(links)}.",
    )
    require(
        links[0]["target"] == "Example Project",
        f"Unexpected wikilink target: {links[0]['target']}",
    )


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

    require(
        len(links) == 1,
        f"Expected one wikilink, got {len(links)}.",
    )
    require(
        links[0]["target"] == "Example Project",
        f"Unexpected wikilink target: {links[0]['target']}",
    )


def _write_long_section(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()

    lines = [f"Line {i} about soil moisture and watering." for i in range(30)]
    lines.append("x" * 700)
    lines.append("The backup pump is in the blue shed.")

    (vault / "garden.md").write_text(
        "# Garden\n\n## Irrigation\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return vault


def test_long_sections_are_split_to_max_chars(tmp_path: Path):
    vault = _write_long_section(tmp_path)

    records = ObsidianAdapter(vault).records()

    require(
        len(records) > 1,
        f"Expected the long section to be split, got {len(records)} record.",
    )
    require(
        all(len(record["text"]) <= 512 for record in records),
        "A chunk is longer than max_chars.",
    )
    require(
        "blue shed" in records[-1]["text"],
        "The end of the section is missing from the last chunk.",
    )
    require(
        [record["metadata"]["chunk_index"] for record in records]
        == list(range(len(records))),
        "Chunks are not numbered in order.",
    )
    require(
        all(
            record["metadata"]["heading"] == "Irrigation"
            and record["metadata"]["chunk_count"] == len(records)
            for record in records
        ),
        "Chunks lost their section metadata.",
    )


def test_max_chars_none_keeps_one_record_per_section(tmp_path: Path):
    vault = _write_long_section(tmp_path)

    records = ObsidianAdapter(vault, max_chars=None).records()

    require(
        len(records) == 1,
        f"Expected one unsplit record, got {len(records)}.",
    )
    require(
        "chunk_index" not in records[0]["metadata"],
        "An unsplit record has chunk metadata.",
    )


# ---------------------------------------------------------------------------
# load_into_memory contract tests
# ---------------------------------------------------------------------------


def test_load_into_memory_passes_batch_configuration(
    vault_path: Path,
    monkeypatch,
):
    adapter = ObsidianAdapter(vault_path)

    class FakeBatchLoader:
        captured = None

        def __init__(self, memory):
            self.memory = memory

        def insert_batch(
            self,
            texts,
            batch_size=100,
            skip_embedding=False,
            parallel_extract=True,
            max_workers=4,
            metadatas=None,
            skip_embedding_build=False,
        ):
            FakeBatchLoader.captured = {
                "memory": self.memory,
                "texts": texts,
                "batch_size": batch_size,
                "metadatas": metadatas,
                "skip_embedding_build": skip_embedding_build,
            }
            return len(texts)

    monkeypatch.setattr(
        "benchmark.batch_loader.BatchLoader",
        FakeBatchLoader,
    )

    fake_memory = object()

    count = adapter.load_into_memory(
        fake_memory,
        batch_size=7,
        skip_embedding_build=True,
    )

    captured = FakeBatchLoader.captured

    require(captured is not None, "BatchLoader.insert_batch was not called.")
    require(
        captured["memory"] is fake_memory,
        "load_into_memory() passed the wrong memory interface.",
    )
    require(
        captured["batch_size"] == 7,
        "load_into_memory() did not forward batch_size.",
    )
    require(
        captured["skip_embedding_build"] is True,
        "load_into_memory() did not forward skip_embedding_build.",
    )
    require(
        captured["texts"] == [
            record["text"]
            for record in adapter.records()
        ],
        "load_into_memory() passed unexpected texts.",
    )
    require(
        captured["metadatas"] == [
            record["metadata"]
            for record in adapter.records()
        ],
        "load_into_memory() passed unexpected metadata.",
    )
    require(
        count == len(captured["texts"]),
        "load_into_memory() returned an unexpected count.",
    )


def test_load_into_memory_empty_vault_returns_zero(
    tmp_path: Path,
    monkeypatch,
):
    vault = tmp_path / "empty-vault"
    vault.mkdir()

    adapter = ObsidianAdapter(vault)

    class ExplodingBatchLoader:
        def __init__(self, memory):
            raise RuntimeError(
                "BatchLoader should not be constructed for an empty vault"
            )

    monkeypatch.setattr(
        "benchmark.batch_loader.BatchLoader",
        ExplodingBatchLoader,
    )

    count = adapter.load_into_memory(object())

    require(
        count == 0,
        f"Empty-vault load_into_memory returned {count}.",
    )


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

    require(
        count == expected,
        f"Expected {expected} inserted records, got {count}.",
    )
    require(
        count > 0,
        "Obsidian integration inserted zero records.",
    )


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

    require(
        nonempty,
        "Vault contained no non-empty records.",
    )

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

    require(
        candidate is not None,
        (
            "Could not derive a retrieval query from any "
            "non-empty Obsidian record."
        ),
    )

    query = " ".join(candidate_words[:8])

    count = adapter.load_into_memory(
        isolated_memory,
        batch_size=100,
    )

    require(
        count == len(records),
        f"Expected {len(records)} records, got {count}.",
    )

    results = isolated_memory.recall(query)

    require(
        results,
        "Memoria recall returned no results after Obsidian ingestion.",
    )

    result_text = str(results)

    require(
        any(
            word.lower() in result_text.lower()
            for word in candidate_words[:3]
        ),
        "Retrieved results did not contain expected source content.",
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

    require(
        len(paths) == 14,
        f"Expected 14 demo source notes, got {len(paths)}.",
    )

    require(
        "02-projects/Example Project/Example Project.md" in paths,
        "Demo project note is missing.",
    )
    require(
        "03-areas/Career/Career.md" in paths,
        "Demo career note is missing.",
    )
    require(
        "04-resources/People/@Jane Smith.md" in paths,
        "Demo Jane Smith note is missing.",
    )

    # Section ingestion should produce more records than source notes.
    require(
        len(records) > len(paths),
        "Demo vault did not produce section-level records.",
    )


def test_demo_source_metadata(demo_vault: Path):
    adapter = ObsidianAdapter(demo_vault)

    record = next(
        record
        for record in adapter.records()
        if _source_path(record)
        == "02-projects/Example Project/Example Project.md"
    )

    metadata = record["metadata"]

    require(
        metadata["source"] == "obsidian",
        "Demo source metadata is incorrect.",
    )
    require(
        metadata["relative_path"]
        == "02-projects/Example Project/Example Project.md",
        "Demo relative_path is incorrect.",
    )
    require(
        metadata["vault_path"]
        == "02-projects/Example Project/Example Project.md",
        "Demo vault_path is incorrect.",
    )


def test_demo_frontmatter(demo_vault: Path):
    adapter = ObsidianAdapter(demo_vault)

    project_records = [
        record
        for record in adapter.records()
        if _source_path(record)
        == "02-projects/Example Project/Example Project.md"
    ]

    require(
        project_records,
        "Demo project produced no records.",
    )

    for record in project_records:
        frontmatter = record["metadata"]["frontmatter"]

        require(
            frontmatter["category"] == ["project"],
            f"Unexpected project category: {frontmatter.get('category')}",
        )
        require(
            frontmatter["status"] == ["inprogress"],
            f"Unexpected project status: {frontmatter.get('status')}",
        )


def test_demo_wikilinks(demo_vault: Path):
    adapter = ObsidianAdapter(demo_vault)

    project_records = [
        record
        for record in adapter.records()
        if _source_path(record)
        == "02-projects/Example Project/Example Project.md"
    ]

    require(
        project_records,
        "Demo project produced no records.",
    )

    links = [
        link
        for record in project_records
        for link in record["metadata"]["links"]
    ]

    require(
        links,
        "Demo project contains no wikilinks.",
    )

    require(
        any(
            "04-resources/Meeting notes/20260115-Team Sync"
            in link["target"]
            for link in links
        ),
        "Demo project is missing the Team Sync wikilink.",
    )

    require(
        any(
            "@Jane Smith" in link["target"]
            for link in links
        ),
        "Demo project is missing the Jane Smith wikilink.",
    )


def test_demo_tasks(demo_vault: Path):
    adapter = ObsidianAdapter(demo_vault)

    project_records = [
        record
        for record in adapter.records()
        if _source_path(record)
        == "02-projects/Example Project/Example Project.md"
    ]

    require(
        project_records,
        "Demo project produced no records.",
    )

    tasks = [
        task
        for record in project_records
        for task in record["metadata"]["tasks"]
    ]

    require(
        tasks,
        "Demo project contains no tasks.",
    )

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

    require(
        completed,
        "Demo project contains no completed tasks.",
    )
    require(
        pending,
        "Demo project contains no pending tasks.",
    )


def test_demo_templates(demo_vault: Path):
    adapter = ObsidianAdapter(demo_vault)

    records = adapter.records()

    templates = [
        record
        for record in records
        if record["metadata"].get("is_template")
    ]

    require(
        templates,
        "Demo vault contains no detected templates.",
    )

    template_paths = {
        _source_path(record)
        for record in templates
    }

    require(
        "99-meta/Templates/project-template.md" in template_paths,
        "Demo project template was not detected.",
    )
    require(
        "99-meta/Templates/daily-note-template.md" in template_paths,
        "Demo daily-note template was not detected.",
    )


def test_demo_project_content(demo_vault: Path):
    adapter = ObsidianAdapter(demo_vault)

    project_records = [
        record
        for record in adapter.records()
        if _source_path(record)
        == "02-projects/Example Project/Example Project.md"
    ]

    require(
        project_records,
        "Demo project produced no records.",
    )

    # The note itself is "Example Project"; section headings are the
    # headings actually present inside the Markdown file.
    require(
        any(
            record["metadata"]["filename"] == "Example Project"
            for record in project_records
        ),
        "Demo project filename metadata is missing.",
    )

    headings = {
        record["metadata"].get("heading")
        for record in project_records
    }

    require(
        "Overview" in headings,
        "Demo project is missing the Overview section.",
    )
    require(
        "Next Actions" in headings,
        "Demo project is missing the Next Actions section.",
    )
    require(
        "Meeting Notes" in headings,
        "Demo project is missing the Meeting Notes section.",
    )

    full_text = "\n".join(
        record["text"]
        for record in project_records
    )

    require(
        "This is a sample project" in full_text,
        "Demo project content is missing the sample-project text.",
    )
    require(
        "Deliver a new feature by end of January 2026" in full_text,
        "Demo project content is missing the expected delivery text.",
    )
    require(
        "Phase 3: Documentation & Review" in full_text,
        "Demo project content is missing the expected phase text.",
    )


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

    require(
        count == expected,
        f"Expected {expected} inserted records, got {count}.",
    )
    require(
        count > 14,
        f"Expected section-level ingestion to exceed 14 records, got {count}.",
    )

    results = isolated_memory.recall(
        "What is the Example Project trying to accomplish?"
    )

    require(
        results,
        "Example Project retrieval returned no results.",
    )
    require(
        "Example Project" in str(results),
        "Example Project retrieval did not contain the project name.",
    )


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

    require(
        count == expected,
        f"Expected {expected} inserted records, got {count}.",
    )
    require(
        count > 14,
        f"Expected section-level ingestion to exceed 14 records, got {count}.",
    )

    results = isolated_memory.recall(
        "Who is Jane Smith?"
    )

    require(
        results,
        "Jane Smith retrieval returned no results.",
    )
    require(
        "Jane Smith" in str(results),
        "Jane Smith retrieval did not contain Jane Smith.",
    )


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
