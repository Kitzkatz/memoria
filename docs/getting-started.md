---
title: Getting Started
description: Install Memoria, run your first query, and explore the CLI, TUI, GUI, and API.
---

# Getting Started

This page covers installation, first queries, and the four interfaces Memoria exposes.

**Requirements**

- Python 3.12+
- 4GB RAM minimum target
- CPU-compatible (no GPU required)
- Local embedding model (bundled with the repo)
- No cloud service or API keys required

---

## Installation

```bash
git clone https://github.com/Kitzkatz/memoria.git
cd memoria
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Your First Query

```bash
python cli.py store "Kevin Johnson likes ramen."
python cli.py recall "What does Kevin Johnson like?"
```

That's it. Memoria stores the memory, indexes it, and retrieves it.

Expected output:

<!-- adjust these two blocks to match your actual CLI output before publishing -->

```text
$ python cli.py store "Kevin Johnson likes ramen."
Stored memory 4f8a2c1e
  embeddings: 1
  entities:   3
  relations:  2

$ python cli.py recall "What does Kevin Johnson like?"
1. Kevin Johnson likes ramen.
   score:  0.94
   source: fusion
```

---

## Interfaces

| Interface | Command | URL |
|-----------|---------|-----|
| CLI | `python cli.py` | Terminal |
| TUI | `python tui.py` | Terminal |
| GUI | `python gui.py` | http://localhost:5000 |
| API | `python main.py` | http://localhost:8000/docs |

---

## CLI Commands

```bash
python cli.py store "Your memory here"
python cli.py recall "What did I say?" --limit 5
python cli.py chat "What does Kevin Johnson like?"
python cli.py set-goal "Finish V4 release" --progress started
python cli.py list-goals --status active
python cli.py info
python cli.py doctor
python cli.py benchmark --limit 100
python cli.py serve --port 8000
python cli.py export memories.json
python cli.py import memories.json
python cli.py config
```

- `store` — insert a memory
- `recall` — retrieve memories for a query
- `chat` — full LLM-backed response using retrieved context
- `set-goal` / `list-goals` — track long-running goals
- `info` — system overview
- `doctor` — health and integrity check
- `benchmark` — run the synthetic benchmark
- `serve` — start the API
- `export` / `import` — serialize and restore memories
- `config` — dump effective configuration

---

## TUI

```bash
python tui.py
```

Commands available inside the TUI:

`store`, `recall`, `chat`, `set-goal`, `list-goals`, `graph`, `stats`, `doctor`, `export`, `import`, `quit`

---

## GUI

```bash
python gui.py
```

Then open:

```text
http://localhost:5000
```

---

## API

```bash
python main.py
```

Then open:

```text
http://localhost:8000/docs
```

Endpoints include:

- `/memory/store`
- `/memory/query`
- `/memory/batch_store`
- `/chat`
- `/chat/raw`
- `/debug/stats`
- `/debug/health`
- `/maintenance/rebuild_index`
- `/benchmark/run`

---

## Configuration Overview

Memoria is configured through Pydantic settings in `cache/config.py`. Environment variables can override settings with a `MEMORY_` prefix:

```bash
export MEMORY_TOP_K=1000
export MEMORY_CONTEXT_TOKEN_BUDGET=20000
```

See [Configuration](configuration.md) for the full list.

---

## Where To Next

- Tune the system: [Configuration](configuration.md)
- Add a dataset: [Adapters](adapters.md)
- Understand retrieval: [Retrieval](retrieval.md)
- Reproduce the benchmarks: [Benchmarks](benchmarks.md)
