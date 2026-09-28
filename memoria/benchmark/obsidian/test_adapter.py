"""Small structural tests for the Obsidian adapter."""

from pathlib import Path

from .adapter import ObsidianAdapter


def test_demo_vault(vault_path: str):
    adapter = ObsidianAdapter(vault_path)
    records = adapter.records()

    assert records

    project = next(
        record for record in records
        if record["metadata"]["filename"] == "Example Project"
    )

    assert "project" in project["metadata"]["category"]
    assert project["metadata"]["links"]

    jane_links = [
        link for link in project["metadata"]["links"]
        if link["target"] == "@Jane Smith"
    ]
    assert jane_links
