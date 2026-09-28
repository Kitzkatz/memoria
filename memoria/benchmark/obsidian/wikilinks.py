"""Obsidian wikilink extraction and vault resolution."""

from __future__ import annotations

import re
from pathlib import Path


WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
FENCED_CODE_RE = re.compile(
    r"(?ms)^(`{3,}|~{3,}).*?^\1[ \t]*$"
)
INLINE_CODE_RE = re.compile(r"`+[^`\n]*`+")


def _mask_markdown_code(text: str) -> str:
    """Replace Markdown code spans/blocks with whitespace.

    This preserves character positions so regex match offsets remain
    aligned with the original text.
    """
    text = FENCED_CODE_RE.sub(
        lambda match: " " * len(match.group(0)),
        text,
    )

    text = INLINE_CODE_RE.sub(
        lambda match: " " * len(match.group(0)),
        text,
    )

    return text


def extract_wikilinks(
    text: str,
    source_path: str,
    known_notes: set[str],
) -> list[dict]:
    links = []

    searchable = _mask_markdown_code(text)

    for match in WIKILINK_RE.finditer(searchable):
        raw = text[match.start(1):match.end(1)].strip()

        if "|" in raw:
            target, alias = raw.split("|", 1)
            target = target.strip()
            alias = alias.strip()
        else:
            target = raw
            alias = None

        if "#" in target:
            target, heading = target.split("#", 1)
            target = target.strip()
            heading = heading.strip()
        else:
            heading = None

        resolution_target = target

        # Heading-only links such as [[#Numbers]] have no note target.
        # Preserve the heading as the target so the link itself is not lost.
        if not resolution_target and heading:
            link_target = f"#{heading}"
        else:
            link_target = resolution_target

        resolved = resolve_target(
            resolution_target,
            source_path,
            known_notes,
        )

        links.append({
            "target": link_target,
            "alias": alias,
            "source": source_path,
            "unresolved": resolved is None,
            "resolved_path": resolved,
            "heading": heading,
        })

    return links

def resolve_target(
    target: str,
    source_path: str,
    known_notes: set[str],
) -> str | None:
    if not target:
        return None

    normalized = target.replace("\\", "/").strip("/")

    candidates = [
        normalized,
        normalized + ".md",
    ]

    for candidate in candidates:
        if candidate in known_notes:
            return candidate

    basename = Path(normalized).name

    matches = [
        note
        for note in known_notes
        if Path(note).stem == basename
    ]

    if len(matches) == 1:
        return matches[0]

    return None
