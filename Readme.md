# Memoria V4

**Local-first, LLM-agnostic memory system with parallel hybrid retrieval, multi-signal ranking, and a declarative type-routing architecture.**

~122ms average query latency · 99.5% retrieval · 4GB RAM · CPU-only · No cloud

---

## What It Is

Memoria is a fully local memory system for LLMs. It stores memories, retrieves them using parallel workers (FAISS, BM25, Graph, Phrase, Attribute) coordinated through a blackboard scheduler, ranks candidates with multiple signals, and routes queries by memory type.

**You own your data. No cloud. No API keys. No subscription.**

---

## Quick Start

```bash
git clone https://github.com/Kitzkatz/memoria.git
cd memoria
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python cli.py store "Kevin Johnson likes ramen."
python cli.py recall "What does Kevin Johnson like?"
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

---

## TUI

```bash
python tui.py
```

Commands: `store`, `recall`, `chat`, `set-goal`, `list-goals`, `graph`, `stats`, `doctor`, `export`, `import`, `quit`

---

## GUI

```bash
python gui.py
# http://localhost:5000
```

---

## API

```bash
python main.py
# http://localhost:8000/docs
```

Endpoints: `/memory/store`, `/memory/query`, `/memory/batch_store`, `/chat`, `/chat/raw`, `/debug/stats`, `/debug/health`, `/maintenance/rebuild_index`, `/benchmark/run`

---

## Synthetic World Demo

```bash
cd synthetic_world && python run_benchmark.py
cp benchmark_output/*.json ../memoria/benchmark_output/
cd ../memoria

python benchmark/test_batch_load.py
python benchmark/benchmark_runner.py --limit 50
python benchmark/benchmark_analyzer.py benchmark_output/results/benchmark_*.json
```

---

## Benchmark Results

Full 4,632-question synthetic benchmark:

```
Questions:  4632
Retrieved:  99.46%

Recall@1:   32.60%
Recall@3:   39.98%
Recall@5:   52.03%
Recall@10:  78.76%

Avg query latency: 122.3ms
Hardware:           4GB RAM, CPU-only
```

Ranking accuracy is under active tuning — retrieval coverage and latency are strong; top-1/top-3 precision is the current focus area.

---

## Configuration

All settings are managed via Pydantic in `cache/config.py`. Override any value with `MEMORY_` environment variables (e.g. `MEMORY_TOP_K=100`).

You must provide an embedding model inside `memory/models/` and set the path in `cache/config.py`.

### Paths & Models
- `DB_PATH` — SQLite file (`memory.db`)
- `VECTOR_INDEX_PATH` — FAISS index (`memory.index`)
- `CACHE_PATH` — Embedding cache (`cache/embedding_cache.pkl`)
- `EMBEDDING_MODEL` — Sentence-Transformer model (`memory/models/all-MiniLM-L6-v2`)
- `VECTOR_DIM` — Embedding dimension (384)
- `CHAT_TEMPLATE_DIR` — Chat template folder (`chat_templates`)
- `CHAT_TEMPLATE_FILE` — Default template (`llama3.txt`)

### Retrieval & Ranking
- `TOP_K` — Max candidates returned (500)
- `TOP_N` — Max memories passed to the LLM (25)
- `TOP_K_PER_SHARD` — Candidates per shard (150 by default — increase toward 500 for higher recall)
- `CONTEXT_TOKEN_BUDGET` — Token budget for the context builder (10000)
- `USE_BLACKBOARD` — Enable multi-source retrieval (True)
- `MMR_ENABLED` — Enable MMR reranking (True)

### Ranking Weights (sum ~1.0)

| Signal | Default |
|--------|---------|
| `RANKING_SEMANTIC` | 0.1195 |
| `RANKING_TOKEN` | 0.3107 |
| `RANKING_TFIDF` | 0.2929 |
| `RANKING_ENTITY` | 0.0272 |
| `RANKING_SUBJECT` | 0.0980 |
| `RANKING_BM25` | 0.0762 |

Use `--optimize` to auto-tune these.

### Performance Tuning

| Goal | Action |
|------|--------|
| **Speed** | Set `SKIP_EMBEDDING=True` or use `--skip-embedding` |
| **Recall** | Increase `TOP_K_PER_SHARD` and `TOP_K` |
| **Quality** | Enable `MMR_ENABLED` and tune ranking weights |
| **Memory** | Increase `CONTEXT_TOKEN_BUDGET` (more candidates in context) |

### Environment Overrides

```bash
export MEMORY_TOP_K=1000
export MEMORY_CONTEXT_TOKEN_BUDGET=20000
```

---

## Chat Templates & LLM Configuration

The system is pre-configured to work with a Llama 3 server. To switch to a different model or template:

1. **Create a custom template file** for your model and place it inside `memory/chat_templates/`. Use the placeholders `{system}`, `{context}`, `{user}`, and `{assistant}`.
2. **Update `cache/config.py`** by setting `CHAT_TEMPLATE_FILE` to the name of your new template (e.g. `"mistral.txt"`). The system will automatically use that file.

If the specified template file is not found, the system falls back to the built-in Llama 3 template. This works best with instruction-tuned models, but you can adapt it to any chat-style LLM by adjusting the endpoint (`LLM_URL`) and stop tokens (`LLM_STOP_TOKENS`).

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No module named 'faiss'` | `pip install faiss-cpu` |
| `No cache file found` | Cache builds on first run |
| `LLM connection failed` | Set `LLM_URL` in config |

---

## Requirements

- Python 3.12+
- 4GB RAM
- CPU only

---

## License

MIT

---

**Built solo, from scratch, in 7 weeks. Local-first. No cloud.**
