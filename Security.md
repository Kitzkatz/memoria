# Security Policy

## Supported Versions

Memoria is under active development. Security fixes are applied to the latest release on `main`. Older tagged releases are not backported unless a fix is trivial to cherry-pick.

| Version        | Supported          |
| -------------- | ------------------- |
| latest (main)  | :white_check_mark:  |
| older releases | :x:                 |

## Scope

Memoria is a local-first memory/retrieval engine. It stores data in a local SQLite database and FAISS index, exposes an optional HTTP API (FastAPI), an optional MCP server, a CLI/TUI, and supports a local plugin system. Security reports are in scope if they concern any of the following:

- **SQL injection or unsafe query construction**, including anywhere a table or column name is built via string interpolation (several code paths build queries against per-type tables, e.g. `memories_<type>`, using f-strings — these are internally-controlled today, but any path where a type name, filter, or identifier reaches these queries from user/plugin input is a valid report).
- **Plugin loading and execution** (`core/plugin_manager.py`, `PLUGIN_DIR` auto-discovery). Plugins are Python code executed with full process privileges. Report anything that could let an untrusted plugin be loaded automatically, or that could let query/document content influence which plugin code runs.
- **MCP server and HTTP API surfaces** — auth bypass, path traversal in file-ingestion endpoints (PDF/code/Obsidian ingestion), unbounded resource consumption (e.g. a crafted request causing unbounded FAISS/BM25 rebuilds or unbounded memory growth), or any endpoint that reads/writes outside the configured data directory.
- **Deserialization issues** — including the legacy pickle fallback path in graph loading (`NumpyGraph._load_pickle`) and any BM25/embedding cache pickle files. Loading a pickle file from an untrusted source is unsafe by design; report any code path that could cause an untrusted pickle to be loaded automatically.
- **Data exposure** — any way a query, plugin, or API caller could read memories, embeddings, or metadata outside their intended scope (e.g. across shards, across sessions in benchmark isolation, or across a multi-tenant deployment if one exists).
- **Denial of service** via crafted input to ingestion (PDF/code/Obsidian parsing), regex-based extraction, or query processing.

Out of scope: vulnerabilities that require an attacker to already have local filesystem access equal to the user running Memoria (this is a local-first, single-user-trust-model tool by default), and issues in third-party dependencies (FAISS, SentenceTransformers, FastAPI, pluggy) that should be reported upstream — though we do want to know if Memoria uses them in an unsafe way.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security reports.**

Report vulnerabilities privately using one of the following:

- **GitHub private vulnerability reporting**: open a draft security advisory via the repository's Security tab ("Report a vulnerability"). This is the preferred method.
- **Email**: [security@REPLACE_ME.example] — if you don't have a dedicated address yet, set one up before publishing this policy, or route to a maintainer email you check regularly.

When reporting, please include:

- A description of the issue and its potential impact.
- Steps to reproduce, or a proof-of-concept if available.
- The affected version/commit.
- Whether the issue requires local access, plugin installation, network access to the API/MCP server, or a crafted document/query.

### What to expect

- **Acknowledgment** within 5 business days.
- **Initial assessment** (confirmed, not a vulnerability, or needs more information) within 10 business days.
- We'll keep you updated as a fix is developed, and credit you in the release notes / advisory unless you prefer to stay anonymous.
- We ask for a coordinated disclosure window and will work with you on timing — as a rule of thumb, please allow at least 90 days before public disclosure, or until a fix ships, whichever is sooner.

## Preferred Languages

Reports in English are preferred.

## Hardening Notes for Deployers

If you're running Memoria's HTTP API or MCP server somewhere other than your own machine:

- Do not point `PLUGIN_DIR` at a directory writable by untrusted users.
- Do not enable `ENABLE_CORS` with `allow_origins=["*"]` in any deployment reachable from the internet.
- Treat the SQLite database and FAISS index files as sensitive — they contain the full content of stored memories.
- If exposing the API beyond localhost, put it behind your own authentication layer; Memoria does not currently implement request-level auth.
