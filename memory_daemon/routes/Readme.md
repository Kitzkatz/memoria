# Memory Daemon V4

**Local-first, LLM-agnostic memory system with parallel hybrid retrieval, multi-signal ranking, and a declarative type-routing architecture.**

~124ms average query latency · 95.8% retrieval · 4GB RAM · CPU-only · No cloud

---

## What It Is

Memory Daemon is a fully local memory system for LLMs. It stores memories, retrieves them using parallel workers (FAISS, BM25, Graph, Phrase, Attribute) coordinated through a blackboard scheduler, ranks candidates with multiple signals, and routes queries by memory type.

**You own your data. No cloud. No API keys. No subscription.**

---

## Quick Start (5 Minutes)

```bash
git clone https://github.com/Kitzkatz/memory_daemon.git
cd memory_daemon
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
cp benchmark_output/*.json ../memory_daemon/benchmark_output/
cd ../memory_daemon

python benchmark/test_batch_load.py
python benchmark/benchmark_runner.py --limit 50
python benchmark/benchmark_analyzer.py benchmark_output/results/benchmark_*.json
```

---

## Benchmark Results

Full 4,612-question synthetic benchmark:

```
Questions:  4612
Retrieved:  95.77%

Recall@1:   19.84%
Recall@3:   43.13%
Recall@5:   52.54%
Recall@10:  62.90%

Avg query latency: 123.6ms
Hardware:           4GB RAM, CPU-only
```

Ranking accuracy is under active tuning — retrieval coverage and latency are the current strong points; top-1 precision is the next area of focus.

---

## Configuration

Edit `cache/config.py`:

```python
TOP_K: int = 500
TOP_N: int = 25
MMR_ENABLED: bool = True
EMBEDDING_MODEL: str = "memory/models/all-MiniLM-L6-v2"
EMBEDDING_CACHE_MAX_SIZE: int = 100000
```

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
