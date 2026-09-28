# Obsidian Adapter

Initial Obsidian → Memoria ingestion adapter.

## Supported

- `.md` note discovery
- YAML frontmatter
- headings
- Markdown tasks/checklists
- `[[wikilinks]]`
- aliased links: `[[target|display text]]`
- vault-relative link resolution
- bare-note-name resolution when unambiguous
- unresolved-link preservation
- note path/folder/filename metadata
- tags
- template detection
- `.git` / `.obsidian` exclusion from Markdown ingestion

## Design

The adapter does **not** implement retrieval, ranking, FAISS, BM25, or graph
search. It converts Obsidian's source structure into Memoria records.

That keeps the source adapter separate from the retrieval substrate.

## Basic usage

```python
from adapters.obsidian import ObsidianAdapter

adapter = ObsidianAdapter("./obsidian-demo-vault")

for record in adapter.records():
    print(record["metadata"]["vault_path"])
    print(record["text"])
```

## Memoria insertion

The adapter deliberately accepts a callback so it can be wired to the
existing MemoryInterface/BatchLoader without duplicating ingestion logic:

```python
adapter.load(
    lambda text, metadata: memory.store(
        text,
        metadata=metadata,
    )
)
```

Adjust the callback to the actual Memoria insertion API.

## Current scope

This is the first structural pass. Canvas, Bases, attachments, embeds,
Dataview-specific semantics, Tasks plugin metadata, and other plugin-specific
formats should be added only after the core Markdown/wiki-link path is stable.
