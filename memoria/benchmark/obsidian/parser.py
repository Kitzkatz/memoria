"""
Obsidian structure parser for Memoria.

Uses the maintained ``obsidianmd-parser`` package for Obsidian-specific
Markdown parsing and translates its Note/Section objects into the normalized
records expected by ObsidianAdapter.

The parser does not perform storage, embedding, ranking, or retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from obsidian_parser import Vault
from obsidian_parser.note import Section


class ObsidianParser:
    """Translate an Obsidian vault into Memoria-shaped section records."""

    def __init__(self, vault_root: str | Path):
        self.vault_root = Path(vault_root).expanduser().resolve()

        if not self.vault_root.exists():
            raise FileNotFoundError(
                f"Obsidian vault does not exist: {self.vault_root}"
            )

        if not self.vault_root.is_dir():
            raise ValueError(
                f"Obsidian vault path is not a directory: {self.vault_root}"
            )

        self.vault = Vault(self.vault_root)

    # --------------------------------------------------
    # DISCOVERY
    # --------------------------------------------------

    def discover_notes(self) -> list[Path]:
        """Return Markdown files discovered by the external parser."""

        return sorted(
            Path(path).resolve()
            for path in self.vault.file_paths
            if Path(path).suffix.lower() == ".md"
            and ".git" not in Path(path).parts
            and ".obsidian" not in Path(path).parts
        )

    # --------------------------------------------------
    # NOTE PARSING
    # --------------------------------------------------

    def parse(self, path: str | Path, known_notes=None) -> list[dict]:
        """
        Parse one Obsidian note into section-level Memoria records.

        ``known_notes`` is retained for compatibility with the previous
        parser API. The external parser resolves Obsidian structure itself.
        """

        path = Path(path).expanduser().resolve()

        try:
            relative_path = path.relative_to(self.vault_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Note path is outside vault: {path}"
            ) from exc

        note = self.vault.get_note(relative_path, strategy="exact")

        if note is None:
            raise FileNotFoundError(
                f"Obsidian note could not be parsed: {relative_path}"
            )

        is_template = self._is_template(relative_path)

        records: list[dict] = []

        preamble, first_heading = self._preamble(note)

        if preamble is not None:
            record = self._section_record(
                note=note,
                section=preamble,
                relative_path=relative_path,
                is_template=is_template,
            )
            record["metadata"]["tasks"] = self._tasks(
                [
                    task
                    for task in note.tasks
                    if task.line_number < first_heading
                ]
            )
            records.append(record)

        for section in self._root_sections(note):
            self._collect_section_records(
                note=note,
                section=section,
                relative_path=relative_path,
                is_template=is_template,
                records=records,
            )

        return records

    # --------------------------------------------------
    # SECTION WALKING
    # --------------------------------------------------

    @staticmethod
    def _preamble(note) -> tuple[Section | None, int]:
        """
        Return the text before the first heading as a headingless section.

        obsidianmd-parser only creates sections under headings, so a note
        without headings, or the text above its first heading, would
        otherwise be dropped. Line numbers count from the end of the
        frontmatter, as they do for the parser's own sections and tasks.
        """

        content = note.content

        if content.startswith("---\n"):
            parts = content.split("---\n", 2)

            if len(parts) == 3:
                content = parts[2]

        lines = content.splitlines()
        first_heading = min(
            (section.line_number for section in note.sections),
            default=len(lines),
        )
        text = "\n".join(lines[:first_heading]).strip()

        if not text:
            return None, first_heading

        return (
            Section(heading="", level=0, content=text, line_number=0),
            first_heading,
        )

    @staticmethod
    def _root_sections(note) -> Iterator:
        """
        Yield only root sections.

        ``note.sections`` is flattened by obsidianmd-parser, so recursively
        walking it directly would duplicate child sections.
        """

        for section in note.sections:
            if section.parent is None:
                yield section

    def _collect_section_records(
        self,
        *,
        note,
        section,
        relative_path: str,
        is_template: bool,
        records: list[dict],
    ) -> None:
        """Recursively convert non-empty sections into Memoria records."""

        text = section.content.strip()

        if text:
            records.append(
                self._section_record(
                    note=note,
                    section=section,
                    relative_path=relative_path,
                    is_template=is_template,
                )
            )

        for child in section.subsections:
            self._collect_section_records(
                note=note,
                section=child,
                relative_path=relative_path,
                is_template=is_template,
                records=records,
            )

    # --------------------------------------------------
    # RECORD CONVERSION
    # --------------------------------------------------

    def _section_record(
        self,
        *,
        note,
        section,
        relative_path: str,
        is_template: bool,
    ) -> dict:
        """Convert one external-parser section into a Memoria record."""

        path = Path(relative_path)

        metadata = {
            "source": "obsidian",
            "vault_path": relative_path,
            "relative_path": relative_path,
            "filename": path.stem,
            "folder": (
                path.parent.as_posix()
                if path.parent != Path(".")
                else ""
            ),
            "heading": section.heading,
            "breadcrumb": list(section.parent_headings),
            "section_path": section.full_path,
            "line_number": section.line_number,
            "frontmatter": self._json_safe(note.frontmatter),
            "headings": list(section.parent_headings),
            "links": self._links(section.wikilinks),
            "tasks": self._section_tasks(note.tasks, section),
            "tags": self._tags(note.tags),
            "is_template": is_template,
        }

        return {
            "text": section.content.strip(),
            "metadata": metadata,
        }

    # --------------------------------------------------
    # OBISIDIAN OBJECT CONVERSION
    # --------------------------------------------------

    @staticmethod
    def _links(links) -> list[dict]:
        """Convert obsidianmd-parser WikiLink objects to plain dictionaries."""

        result = []

        for link in links or []:
            result.append(
                {
                    "target": str(link.target),
                    "alias": link.alias,
                    "heading": link.heading,
                    "block_id": link.block_id,
                    "display_text": link.display_text,
                }
            )

        return result

    @staticmethod
    def _tasks(tasks) -> list[dict]:
        """Convert obsidianmd-parser Task objects to plain dictionaries."""

        result = []

        for task in tasks or []:
            result.append(
                {
                    "text": str(task.text),
                    "completed": bool(task.completed),
                    "status": task.status,
                    "line_number": task.line_number,
                    "indent_level": task.indent_level,
                    "status_meaning": task.status_meaning,
                }
            )

        return result

    @classmethod
    def _section_tasks(cls, tasks, section) -> list[dict]:
        """Convert only tasks belonging to this section."""

        content = section.content.strip()

        if not content:
            return []

        content_lines = content.splitlines()
        start_line = section.line_number
        end_line = start_line + len(content_lines) - 1

        section_tasks = [
            task
            for task in (tasks or [])
            if start_line + 2 <= task.line_number <= end_line + 2
        ]

        return cls._tasks(section_tasks)

    @staticmethod
    def _tags(tags) -> list[str]:
        """Convert parser Tag objects into stable string representations."""

        result = []

        for tag in tags or []:
            if isinstance(tag, str):
                result.append(tag)
                continue

            value = getattr(tag, "name", None)

            if value is None:
                value = getattr(tag, "tag", None)

            if value is None:
                value = str(tag)

            result.append(str(value).lstrip("#"))

        return result

    # --------------------------------------------------
    # TEMPLATE DETECTION
    # --------------------------------------------------

    @staticmethod
    def _is_template(relative_path: str) -> bool:
        """Preserve Memoria's existing template convention."""

        normalized = relative_path.replace("\\", "/")

        return (
            "99-meta/Templates/" in normalized
            or "00 Meta/templates/" in normalized
        )

    # --------------------------------------------------
    # JSON-SAFE METADATA
    # --------------------------------------------------

    @classmethod
    def _json_safe(cls, value):
        """Convert parser metadata into values safe for Memoria metadata."""

        from datetime import date, datetime

        if isinstance(value, (datetime, date)):
            return value.isoformat()

        if isinstance(value, dict):
            return {
                str(key): cls._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                cls._json_safe(item)
                for item in value
            ]

        return value
