"""Markdown structure extraction used by the Obsidian adapter."""

from __future__ import annotations

import re

HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
TASK_RE = re.compile(r"^(\s*)[-*+]\s+\[([ xX])\]\s+(.*)$")
TAG_RE = re.compile(r"(?<![\w#])#([A-Za-z0-9_/-]+)")
WIKILINK_RE = re.compile(r"!?\[\[.*?\]\]")


def extract_headings(text: str) -> list[str]:
    return [
        match.group(1).strip()
        for line in text.splitlines()
        if (match := HEADING_RE.match(line))
    ]


def extract_tasks(text: str) -> list[dict]:
    tasks = []

    for line in text.splitlines():
        match = TASK_RE.match(line)
        if not match:
            continue

        tasks.append({
            "text": match.group(3).strip(),
            "completed": match.group(2).lower() == "x",
            "indent": len(match.group(1)),
        })

    return tasks


def extract_tags(text: str) -> list[str]:
    """Extract Obsidian-style inline tags outside Markdown code/headings."""
    seen = set()
    tags = []

    in_fence = False
    fence_marker = None

    for line in text.splitlines():
        stripped = line.lstrip()

        # Toggle fenced code blocks.
        fence_match = re.match(r"(`{3,}|~{3,})", stripped)
        if fence_match:
            marker = fence_match.group(1)[0]

            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = None

            continue

        if in_fence:
            continue

        # A line beginning with Markdown heading syntax is not a tag.
        if re.match(r"^#{1,6}(?:\s|$)", stripped):
            continue

        # Remove inline code and Obsidian wikilinks/embeds
        # before searching for tags.
        searchable = re.sub(r"`+[^`\n]*`+", "", line)
        searchable = WIKILINK_RE.sub("", searchable)

        for match in TAG_RE.finditer(searchable):
            tag = match.group(1)

            if tag not in seen:
                seen.add(tag)
                tags.append(tag)

    return tags
