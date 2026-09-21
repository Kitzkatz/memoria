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

The repository is named `memoria` and contains the system in a subdirectory that is also named `memoria`. The clone lands at the repo root, `requirements.txt` lives at the repo root, and the runnable system lives one level down.

```bash
git clone https://github.com/Kitzkatz/memoria.git
cd memoria                          # repo root
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd memoria                          # into the system
```

All commands in the rest of this page run from inside the inner `memoria/` directory.
---

## Your First Query

```bash
python cli.py store "Kevin Johnson likes ramen."
python cli.py recall "What does Kevin Johnson like?"
```

That's it. Memoria stores the memory, indexes it, and retrieves it.

Expected output:

```text
$ python cli.py store "Kevin Johnson likes ramen."
Stored memory with ID: <memory_id>

$ python cli.py recall "What does Kevin Johnson like?"
Found 1 results, showing first 3:
Rank   Score      Text
--------------------------------------------------------------------------------
1      0.9542     Kevin Johnson likes ramen.
```

The score column shows the `final_score` attached to each result. `Rank` is the sorted position. Text is truncated to fit the configured table width (`CLI_TABLE_WIDTH`, default 80).

---

## Interfaces

| Interface | Command | URL |
|-----------|---------|-----|
| CLI | `python cli.py <command>` | Terminal |
| TUI | `python tui.py` | Terminal |
| GUI | `python gui.py` | http://localhost:5000 |
| API | `python cli.py serve` | http://localhost:8000/docs |

All four interfaces talk to the same `MemoryInterface`. Memory state is shared through the same SQLite database and FAISS index.

---

## CLI Commands

```bash
python cli.py store "Your memory here"
python cli.py recall "What did I say?" --limit 5
python cli.py store-many memories.json
python cli.py chat "What does Kevin Johnson like?"
python cli.py set-goal "Finish V4 release" --progress started
python cli.py update-goal 1 --status completed
python cli.py list-goals --status active
python cli.py graph "Kevin Johnson" --depth 2
python cli.py info
python cli.py doctor
python cli.py benchmark --limit 100
python cli.py serve --port 8000
python cli.py export memories.json
python cli.py import memories.json
python cli.py config
```

| Command | Description |
|---------|-------------|
| `store <text>` | Insert a memory |
| `recall <query>` | Retrieve memories for a query |
| `store-many <file>` | Batch-insert memories from a JSON list of strings |
| `chat [prompt]` | One-shot or interactive LLM response using retrieved context |
| `set-goal <goal>` | Create a goal with a progress label |
| `update-goal <id>` | Update a goal's progress or status |
| `list-goals` | List goals, optionally filtered by `--status` |
| `graph <entity>` | Show graph neighbors for an entity |
| `info` | System overview and memory count |
| `doctor` | Integrity and sanity checks on the database |
| `benchmark` | Run the synthetic benchmark |
| `serve` | Start the API server |
| `export <file>` | Serialize all memories to JSON |
| `import <file>` | Restore memories from JSON |
| `config` | Dump effective configuration |

### `recall` options

| Flag | Default | Purpose |
|------|---------|---------|
| `--limit` | `CLI_DEFAULT_LIMIT` (3) | Number of results to display |
| `--format` | `CLI_OUTPUT_FORMAT` (table) | `table`, `json`, or `raw` |

### `serve` options

| Flag | Default | Purpose |
|------|---------|---------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Port |
| `--reload` | off | Auto-reload on code changes (development) |

---

## TUI

```bash
python tui.py
```

The TUI prompt is `Memory> `. Commands use **underscores** for multi-word names, unlike the CLI's hyphens.

| Command | Description |
|---------|-------------|
| `store <text>` | Store a memory |
| `recall <query> [limit]` | Recall with optional inline limit |
| `recall_json <query>` | Recall and print full JSON response |
| `store_many <file>` | Batch-insert from JSON |
| `chat [prompt]` | One-shot, or enter persistent chat mode |
| `set_goal <goal> [progress]` | Create a goal |
| `update_goal <id>` | Update progress/status |
| `list_goals [--status <status>]` | List goals |
| `graph <entity> [depth]` | Graph neighbors |
| `stats` | Memory count only |
| `info` | Full system overview |
| `doctor` | Integrity checks |
| `export <file>` / `import <file>` | JSON round-trip |
| `signals [type]` | Show active ranking signals |
| `signal_toggle <name>` | Toggle a signal |
| `signal_enable <name>` / `signal_disable <name>` | Explicitly set signal state |
| `signal_reset` | Reset registry to defaults |
| `history [subcommand]` | Query history management |
| `autostore [on\|off\|threshold\|max\|types\|status]` | Auto-store settings |
| `back` | Exit chat mode (when in chat mode) |
| `quit` / `q` | Exit the TUI |
| `help` / `h` | Command list |

### Chat mode

Typing `chat` with no argument enters persistent chat mode with the prompt `Chat> `. Type `.back` or `.exit` to return to the main shell.

Chat-mode dot commands:

| Command | Effect |
|---------|--------|
| `.back` / `.exit` | Return to main shell |
| `.history` | Show this session's chat history |
| `.clear` | Clear chat history |
| `.info` | Message count for this session |
| `.auto-on` / `.auto-off` | Override auto-store for this session |
| `.auto-status` | Show current auto-store state |
| `.help` | Chat-mode help |
| any other text | Sent to the assistant |

---

## GUI

```bash
python gui.py
```

Then open:

```text
http://localhost:5000
```

The GUI is a small FastAPI app. Interactive API docs are auto-generated at:

```text
http://localhost:5000/docs
```

Endpoints exposed by the GUI:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/query` | Recall memories |
| `POST` | `/store` | Store a single memory |
| `POST` | `/store_many` | Batch store |
| `POST` | `/chat` | Chat with the LLM |
| `POST` | `/ingest_code` | Ingest a code directory |
| `POST` | `/ingest_pdf` | Ingest a PDF |
| `POST` | `/set_goal` | Create a goal |
| `GET`  | `/list_goals` | List goals |
| `GET`  | `/signals` | Active signals for a type |
| `POST` | `/signals/toggle` | Toggle a signal |
| `GET`  | `/history` | Search query history |
| `GET`  | `/history/stats` | History statistics |
| `GET`  | `/settings/auto-store` | Read auto-store settings |
| `POST` | `/settings/auto-store` | Update auto-store settings |
| `GET`  | `/health` | Health check |
| `GET`  | `/stats` | System statistics |

---

## API Server

The API server is separate from the GUI. Start it with:

```bash
python cli.py serve --port 8000
```

Then open:

```text
http://localhost:8000/docs
```

The `/docs` page lists every endpoint the running app exposes. That page is generated by FastAPI from the routes actually registered at startup, so it is always the current source of truth for the API surface.

---

## Configuration Overview

Memoria is configured through Pydantic settings in `cache/config.py`. Every setting can be overridden with an environment variable prefixed by `MEMORY_`:

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
