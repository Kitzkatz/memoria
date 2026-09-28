
"""Normalized structures emitted by the Obsidian adapter."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


def _json_safe(value: Any) -> Any:
    """Convert YAML-native values into JSON-serializable values."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return value


@dataclass
class ObsidianLink:
    target: str
    alias: str | None = None
    source: str = ""
    unresolved: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "alias": self.alias,
            "source": self.source,
            "unresolved": self.unresolved,
        }


@dataclass
class ObsidianNote:
    text: str
    path: str
    filename: str
    folder: str
    frontmatter: dict[str, Any] = field(default_factory=dict)
    headings: list[str] = field(default_factory=list)
    links: list[ObsidianLink] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    is_template: bool = False

    def as_memory(self) -> dict[str, Any]:
        category = _json_safe(
            self.frontmatter.get("category", [])
        )
        if isinstance(category, str):
            category = [category]

        status = _json_safe(
            self.frontmatter.get("status", [])
        )
        if isinstance(status, str):
            status = [status]

        return {
            "text": self.text,
            "metadata": {
                "source": "obsidian",
                "vault_path": self.path,
                "relative_path": self.path,
                "filename": self.filename,
                "folder": self.folder,
                "category": category,
                "status": status,
                "frontmatter": _json_safe(self.frontmatter),
                "headings": self.headings,
                "links": [link.as_dict() for link in self.links],
                "tasks": _json_safe(self.tasks),
                "tags": self.tags,
                "is_template": self.is_template,
            },
        }

