

"""Small YAML frontmatter parser.

PyYAML is used when available. A conservative fallback handles the simple
mapping/list frontmatter commonly found in Obsidian vaults.
"""

from __future__ import annotations

from typing import Any


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break

    if end is None:
        return {}, text

    raw = "\n".join(lines[1:end])

    try:
        import yaml

        data = yaml.safe_load(raw) or {}
        if not isinstance(data, dict):
            data = {"_frontmatter": data}
        return data, "\n".join(lines[end + 1:])
    except ImportError:
        return _fallback_yaml(raw), "\n".join(lines[end + 1:])
    except yaml.YAMLError:
        return _fallback_yaml(raw), "\n".join(lines[end + 1:])


def _fallback_yaml(raw: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_list_key: str | None = None

    for line in raw.splitlines():
        if not line.strip():
            continue

        if line.startswith("  - ") and current_list_key:
            result.setdefault(current_list_key, []).append(
                _scalar(line[4:].strip())
            )
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value:
            result[key] = _scalar(value)
            current_list_key = None
        else:
            result[key] = []
            current_list_key = key

    return result


def _scalar(value: str) -> Any:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]

    if value.lower() in {"true", "false"}:
        return value.lower() == "true"

    if value.lower() in {"null", "none", "~"}:
        return None

    return value
