#!/usr/bin/env python3
"""
Memory Daemon CLI – configurable, command-line interface (like git).
"""
import argparse
import json
import sys
import os
import time
from datetime import datetime

from shared.memory_interface import MemoryInterface
from cache.config import settings
from core.logger import debug, info

# Optional dependencies
try:
    import uvicorn
    HAS_UVICORN = True
except ImportError:
    HAS_UVICORN = False

try:
    from benchmark.benchmark_runner import BenchmarkRunner
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False


def print_table(results, limit, show_scores=True, width=80):
    """Pretty‑print recall results as a table."""
    if not results:
        print("No results found.")
        return
    if show_scores:
        print(f"{'Rank':<6} {'Score':<10} {'Text'}")
        print("-" * width)
        for item in results[:limit]:
            score = item.get('final_score', 0)
            text = item.get('text', '')[:width - 20]
            print(f"{item['rank']:<6} {score:<10.4f} {text}")
    else:
        print(f"{'Rank':<6} {'Text'}")
        print("-" * width)
        for item in results[:limit]:
            text = item.get('text', '')[:width - 10]
            print(f"{item['rank']:<6} {text}")


def print_goals(goals):
    """Pretty‑print goals."""
    if not goals:
        print("No goals found.")
        return
    print(f"{'ID':<6} {'Goal':<30} {'Progress':<12} {'Status'}")
    print("-" * 70)
    for g in goals:
        print(f"{g['id']:<6} {g.get('goal', '')[:28]:<30} "
              f"{g.get('progress', '')[:12]:<12} {g.get('status', '')}")


def main():
    parser = argparse.ArgumentParser(
        prog="memory",
        description="Memory Daemon CLI – local memory engine for LLMs",
        usage="memory <command> [options]"
    )
    parser.add_argument("--version", action="version", version="Memory Daemon v4.0")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand")

    # ---- store ----
    p_store = subparsers.add_parser("store", help="Store a memory")
    p_store.add_argument("text", type=str, help="Memory text")

    # ---- recall ----
    p_recall = subparsers.add_parser("recall", help="Recall memories")
    p_recall.add_argument("query", type=str, help="Query text")
    p_recall.add_argument("--limit", type=int, default=settings.CLI_DEFAULT_LIMIT,
                          help=f"Number of results (default {settings.CLI_DEFAULT_LIMIT})")
    p_recall.add_argument("--format", choices=["table", "json", "raw"],
                          default=settings.CLI_OUTPUT_FORMAT,
                          help="Output format (default from config)")

    # ---- store-many ----
    p_many = subparsers.add_parser("store-many", help="Store memories from a JSON file")
    p_many.add_argument("file", type=str, help="JSON file containing a list of text strings")

    # ---- set-goal ----
    p_goal = subparsers.add_parser("set-goal", help="Set a goal")
    p_goal.add_argument("goal", type=str, help="Goal description")
    p_goal.add_argument("--progress", type=str, default="started", help="Progress status")

    # ---- update-goal ----
    p_upd = subparsers.add_parser("update-goal", help="Update an existing goal")
    p_upd.add_argument("goal_id", type=int, help="Goal ID")
    p_upd.add_argument("--progress", type=str, help="New progress")
    p_upd.add_argument("--status", type=str, help="New status (active, completed, etc.)")

    # ---- list-goals ----
    p_list = subparsers.add_parser("list-goals", help="List all goals")
    p_list.add_argument("--status", type=str, help="Filter by status")

    # ---- chat ----
    p_chat = subparsers.add_parser("chat", help="Chat with memory (one-turn or interactive)")
    p_chat.add_argument("prompt", nargs="?", default=None, help="User prompt (optional; omit for interactive mode)")

    # ---- info ----
    subparsers.add_parser("info", help="Show system information and memory statistics")

    # ---- graph ----
    p_graph = subparsers.add_parser("graph", help="Show graph neighbours for an entity")
    p_graph.add_argument("entity", type=str, help="Entity name")
    p_graph.add_argument("--depth", type=int, default=1, help="Depth of graph traversal")

    # ---- doctor ----
    subparsers.add_parser("doctor", help="Run integrity checks and diagnose issues")

    # ---- serve ----
    p_serve = subparsers.add_parser("serve", help="Start the HTTP API server")
    p_serve.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    p_serve.add_argument("--port", type=int, default=8000, help="Port to listen on")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")

    # ---- benchmark ----
    if HAS_BENCHMARK:
        p_bench = subparsers.add_parser("benchmark", help="Run the benchmark suite")
        p_bench.add_argument("--limit", type=int, help="Limit number of queries")

    # ---- export ----
    p_export = subparsers.add_parser("export", help="Export all memories to a JSON file")
    p_export.add_argument("file", type=str, help="Output file path")

    # ---- import ----
    p_import = subparsers.add_parser("import", help="Import memories from a JSON file")
    p_import.add_argument("file", type=str, help="Input file path")

    # ---- config ----
    subparsers.add_parser("config", help="Display current configuration")

    args = parser.parse_args()
    interface = MemoryInterface()

    try:
        if args.command == "store":
            mid = interface.remember(args.text)
            print(f"Stored memory with ID: {mid}")

        elif args.command == "recall":
            resp = interface.recall(args.query)
            results = resp.get("results", [])

            if args.format == "json":
                print(json.dumps(resp, indent=2))
            elif args.format == "raw":
                print(resp)
            else:  # table
                print(f"Found {len(results)} results, showing first {args.limit}:")
                print_table(results, args.limit,
                            show_scores=settings.CLI_SHOW_SCORES,
                            width=settings.CLI_TABLE_WIDTH)

        elif args.command == "store-many":
            with open(args.file, "r", encoding="utf8") as f:
                texts = json.load(f)
            if not isinstance(texts, list):
                print("Error: JSON file must contain a list of strings.")
                return
            ids = interface.remember_many(texts)
            print(f"Stored {len(ids)} memories")

        elif args.command == "set-goal":
            gid = interface.set_goal(args.goal, args.progress)
            print(f"Goal set with ID: {gid}")

        elif args.command == "update-goal":
            if not args.progress and not args.status:
                print("Error: At least one of --progress or --status is required.")
                return
            interface.update_goal(args.goal_id, progress=args.progress, status=args.status)
            print(f"Goal {args.goal_id} updated.")

        elif args.command == "list-goals":
            goals = interface.list_goals(status=args.status)
            print_goals(goals)

        # ---- Chat ----
        elif args.command == "chat":
            if args.prompt:
                # One-shot chat
                response = interface.chat(args.prompt)
                print(response)
            else:
                # Interactive chat mode
                print("Entering interactive chat mode. Type 'exit' to quit.")
                print("-" * 50)
                while True:
                    try:
                        user_input = input("You: ")
                        if user_input.lower() in ("exit", "quit", "q"):
                            print("Goodbye.")
                            break
                        if user_input.strip():
                            response = interface.chat(user_input)
                            print(f"Assistant: {response}")
                            print("-" * 50)
                    except KeyboardInterrupt:
                        print("\nGoodbye.")
                        break
                    except Exception as e:
                        print(f"Error: {e}")

        elif args.command == "info":
            db = interface.controller.system.db
            count = db.count()
            print(f"Memory Daemon v4.0")
            print(f"Database: {settings.DB_PATH}")
            print(f"Total memories: {count}")
            print(f"Embedding model: {settings.EMBEDDING_MODEL}")
            print(f"LLM URL: {settings.LLM_URL}{settings.LLM_ENDPOINT}")
            print(f"Top K: {settings.TOP_K}")
            print(f"Debug mode: {settings.DEBUG}")
            print(f"CLI output format: {settings.CLI_OUTPUT_FORMAT}")

        elif args.command == "graph":
            graph_search = interface.controller.system.graph_search
            entity = graph_search.find_entity(args.entity)
            if not entity:
                print(f"Entity '{args.entity}' not found.")
                return
            # neighbors expects entity name (string), not ID
            neighbors = graph_search.neighbors(args.entity, depth=args.depth)
            print(f"Neighbors of '{args.entity}' (depth {args.depth}):")
            for n in neighbors:
                print(f"  {n['relation']} → {n['target']} (source: {n['source']})")

        elif args.command == "doctor":
            db = interface.controller.system.db
            print("Running integrity checks...")
            try:
                result = db.integrity_check()
                print(f"Integrity check: {result}")
            except Exception as e:
                print(f"Integrity check error: {e}")
            try:
                sanity = db.sanity_check()
                print(f"DB count: {sanity['db_count']}")
                print(f"Columns: {sanity['columns']}")
            except Exception as e:
                print(f"Sanity check error: {e}")

        elif args.command == "serve":
            if not HAS_UVICORN:
                print("Error: uvicorn is not installed. Install with 'pip install uvicorn[standard]'")
                return
            os.environ.setdefault("MEMORY_DAEMON_CONFIG", os.path.abspath("."))
            info("[CLI] Starting server on {args.host}:{args.port}", category="cli")
            uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)

        elif args.command == "benchmark":
            if not HAS_BENCHMARK:
                print("Error: benchmark module not available.")
                return
            runner = BenchmarkRunner()
            outfile = runner.run(limit=args.limit)
            print(f"Benchmark results written to: {outfile}")

        elif args.command == "export":
            all_memories = interface.controller.system.db.fetch_all()
            with open(args.file, "w", encoding="utf8") as f:
                json.dump(all_memories, f, indent=2, default=str)
            print(f"Exported {len(all_memories)} memories to {args.file}")

        elif args.command == "import":
            with open(args.file, "r", encoding="utf8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                print("Error: import file must contain a list of memory objects.")
                return
            texts = [item.get('text', item.get('normalized_text', '')) for item in data if item.get('text')]
            ids = interface.remember_many(texts)
            print(f"Imported {len(ids)} memories from {args.file}")

        elif args.command == "config":
            print("Current configuration:")
            for key, value in settings.model_dump().items():
                print(f"  {key}: {value}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.command != "serve":  # hide traceback for serve errors
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
