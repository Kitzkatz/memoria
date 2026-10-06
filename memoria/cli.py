#!/usr/bin/env python3
"""
Memoria CLI – configurable, command-line interface (like git).
"""
import argparse
import json
import sys
import os
import time
from datetime import datetime


from cache.config import settings
from core.logger import debug, info
from core.plugin_generator import (
    HOOK_GROUPS,
    generate_plugin,
)
from shared.memory_interface import MemoryInterface


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

# Signal registry (optional, for power users)
try:
    from ranking.signal_registry import get_registry
    from ranking.signal_router import SignalRouter
    HAS_SIGNAL_REGISTRY = True
except ImportError:
    HAS_SIGNAL_REGISTRY = False

# Query history

try:
    from memory.query_history import QueryHistory
    HAS_QUERY_HISTORY = True
except ImportError:
    HAS_QUERY_HISTORY = False


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
    if not goals:
        print("No goals found.")
        return

    print(f"{'ID':<6} {'Goal':<30} {'Progress':<12} {'Status'}")
    print("-" * 70)

    for g in goals:
        goal_id = getattr(g, "id", "")
        goal = getattr(g, "goal", "")
        progress = getattr(g, "progress", "")
        status = getattr(g, "status", "")

        print(
            f"{goal_id:<6} "
            f"{str(goal)[:28]:<30} "
            f"{str(progress)[:12]:<12} "
            f"{status}"
        )
def print_signals(signals, memory_type="general"):
    """Pretty‑print signal registry."""
    if not signals:
        print("No signals found.")
        return

    registry = get_registry() if HAS_SIGNAL_REGISTRY else None

    print(f"\nSignals for type: {memory_type}")
    print(f"{'Signal':<18} {'Weight':<10} {'Cost':<8} {'Enabled':<8}")
    print("-" * 55)

    for name, weight in signals.items():
        if registry:
            cost = registry.get_cost(name)
            enabled = "✅" if registry.is_enabled(name) else "❌"
        else:
            cost = "unknown"
            enabled = "?"
        print(f"{name:<18} {weight:<10.4f} {cost:<8} {enabled:<8}")
    print()


def print_query_history(entries, limit=20):
    """Pretty-print recent query history entries."""
    if not entries:
        print("No history entries found.")
        return

    print(f"{'Timestamp':<20} {'Results':<8} {'Query'}")
    print("-" * 80)

    for entry in entries[:limit]:
        timestamp = datetime.fromtimestamp(
            entry.get("timestamp", 0)
        ).isoformat(timespec="minutes")
        result_count = len(entry.get("results", []))
        query = entry.get("query", "")[:50]

        print(f"{timestamp:<20} {result_count:<8} {query}")

        
def print_query_diff(diff_result):
    """Pretty‑print query diff results."""
    if "error" in diff_result:
        print(f"Error: {diff_result['error']}")
        return

    print(f"\nComparing {diff_result['entry1']['id']} vs {diff_result['entry2']['id']}")
    print(f"  Query 1: {diff_result['entry1']['query'][:60]}...")
    print(f"  Query 2: {diff_result['entry2']['query'][:60]}...")
    print(f"  Common results: {diff_result['common_results']}")
    print(f"  Only in first: {diff_result['only_in_first']}")
    print(f"  Only in second: {diff_result['only_in_second']}")

    if diff_result['score_changes']:
        print("\nTop score changes:")
        for change in diff_result['score_changes'][:5]:
            text = change.get('text', '')[:30]
            delta = change.get('delta', 0)
            old = change.get('score_old', 0)
            new = change.get('score_new', 0)
            print(f"  {text}... {old:.3f} → {new:.3f} (Δ{delta:+.3f})")


def print_auto_store_status():
    """Pretty‑print auto-store status."""
    status = "enabled" if settings.AUTO_STORE_MEMORIES else "disabled"
    print(f"Auto-store: {status}")
    print(f"Threshold: {settings.AUTO_STORE_THRESHOLD}")
    print(f"Max per session: {settings.AUTO_STORE_MAX_PER_SESSION}")
    print(f"Types: {settings.AUTO_STORE_TYPES}")

def _run_plugin_create():
    """Interactively create a new Memoria plugin."""

    print()
    print("Memoria Plugin Generator")
    print("========================")
    print()

    plugin_name = input("Plugin name: ").strip()

    if not plugin_name:
        print("Plugin name cannot be empty.")
        return

    description = input("Description: ").strip()

    print()
    print("Select hook groups.")
    print("Enter y to include a group, or n to skip it.")
    print()

    selected_groups = []

    for group in HOOK_GROUPS:
        answer = input(f"Include {group} hooks? [y/N]: ").strip().lower()

        if answer in {"y", "yes"}:
            selected_groups.append(group)

    print()

    create_tests = (
        input("Create tests? [Y/n]: ").strip().lower()
        not in {"n", "no"}
    )

    create_readme = (
        input("Create README? [Y/n]: ").strip().lower()
        not in {"n", "no"}
    )

    default_output = settings.PLUGIN_DIR

    output_input = input(
        f"Output directory [{default_output}]: "
    ).strip()

    output_dir = output_input or default_output

    print()
    print("Plugin configuration")
    print("--------------------")
    print(f"Name:        {plugin_name}")
    print(f"Description: {description or '(none)'}")
    print(
        "Hooks:       "
        + (", ".join(selected_groups) if selected_groups else "(none)")
    )
    print(f"Tests:       {'yes' if create_tests else 'no'}")
    print(f"README:      {'yes' if create_readme else 'no'}")
    print(f"Output:      {output_dir}")
    print()

    confirm = input("Create plugin? [Y/n]: ").strip().lower()

    if confirm in {"n", "no"}:
        print("Plugin creation cancelled.")
        return

    try:
        generated = generate_plugin(
            plugin_name=plugin_name,
            description=description,
            hook_groups=selected_groups,
            output_dir=output_dir,
            create_tests=create_tests,
            create_readme=create_readme,
        )
    except (ValueError, FileExistsError, OSError) as exc:
        print(f"Plugin creation failed: {exc}")
        return

    print()
    print("Plugin created successfully.")
    print()

    for kind, path in generated.items():
        print(f"{kind:>8}: {path}")

    print()

    
def main():
    parser = argparse.ArgumentParser(
        prog="memory",
        description="Memoria CLI – local memory engine for LLMs",
        usage="memory <command> [options]"
    )
    parser.add_argument("--version", action="version", version="Memoria v1.0")

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
    p_chat.add_argument("--auto-store", action="store_true", help="Enable auto-store for this chat session")
    p_chat.add_argument("--no-auto-store", action="store_true", help="Disable auto-store for this chat session")

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

    # ---- signals (NEW) ----
    if HAS_SIGNAL_REGISTRY:
        p_signals = subparsers.add_parser("signals", help="Manage ranking signals")
        p_signals.add_argument("--list", action="store_true", help="List all signals")
        p_signals.add_argument("--type", type=str, default="general", help="Memory type to list")
        p_signals.add_argument("--toggle", type=str, help="Signal name to toggle")
        p_signals.add_argument("--enable", action="store_true", help="Enable signal")
        p_signals.add_argument("--disable", action="store_true", help="Disable signal")
        p_signals.add_argument("--export", type=str, help="Export registry to JSON")
        p_signals.add_argument("--import", dest="import_file", type=str, help="Import registry from JSON")
        p_signals.add_argument("--reset", action="store_true", help="Reset registry to defaults")

    # ---- query-history (NEW) ----
    
    if HAS_QUERY_HISTORY:
        p_history = subparsers.add_parser(
            "query-history",
            help="Show query history"
        )
        p_history.add_argument(
            "--limit",
            type=int,
            default=20,
            help="Maximum number of entries to show"
        )
        p_history.add_argument(
            "--frequent",
            action="store_true",
            help="Show most frequent queries instead of recent queries"
        )
    # ---- auto-store (NEW) ----
    p_autostore = subparsers.add_parser("auto-store", help="Manage auto-store settings")
    p_autostore.add_argument("--enable", action="store_true", help="Enable auto-store")
    p_autostore.add_argument("--disable", action="store_true", help="Disable auto-store")
    p_autostore.add_argument("--threshold", type=float, help="Set confidence threshold (0.0-1.0)")
    p_autostore.add_argument("--max", type=int, help="Set max auto-stores per session")
    p_autostore.add_argument("--types", type=str, help="Comma-separated list of memory types to auto-store")
    p_autostore.add_argument("--status", action="store_true", help="Show current auto-store status")


    p_plugin = subparsers.add_parser(
        "plugin",
        help="Manage Memoria plugins",
    )

    plugin_subparsers = p_plugin.add_subparsers(
        dest="plugin_command",
        required=True,
    )

    p_plugin_create = plugin_subparsers.add_parser(
        "create",
        help="Create a new Memoria plugin",
    )

    p_plugin_create.set_defaults(plugin_action="create")
    
    args = parser.parse_args()

    if getattr(args, "plugin_action", None) == "create":
        _run_plugin_create()
        return

    try:
        
        interface = MemoryInterface()
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
            # Determine auto-store override
            auto_store_override = None
            if args.auto_store:
                auto_store_override = True
            elif args.no_auto_store:
                auto_store_override = False

            if args.prompt:
                # One-shot chat
                response = interface.chat(args.prompt, auto_store=auto_store_override)
                print(response)
            else:
                # Interactive chat mode
                print("Entering interactive chat mode. Type 'exit' to quit.")
                if auto_store_override is not None:
                    print(f"Auto-store: {'enabled' if auto_store_override else 'disabled'}")
                else:
                    print(f"Auto-store: {'enabled' if settings.AUTO_STORE_MEMORIES else 'disabled'} (from config)")
                print("-" * 50)
                while True:
                    try:
                        user_input = input("You: ")
                        if user_input.lower() in ("exit", "quit", "q"):
                            print("Goodbye.")
                            break
                        if user_input.strip():
                            response = interface.chat(user_input, auto_store=auto_store_override)
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
            print(f"Memoria v1.0")
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
            info(f"[CLI] Starting server on {args.host}:{args.port}", category="cli")
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
            texts = [
                item.get("text") or item.get("normalized_text", "")
                for item in data
                if item.get("text") or item.get("normalized_text")
            ]
            ids = interface.remember_many(texts)
            print(f"Imported {len(ids)} memories from {args.file}")

        elif args.command == "config":
            print("Current configuration:")
            for key, value in settings.model_dump().items():
                print(f"  {key}: {value}")

        # ---- Signals ----
        elif args.command == "signals":
            if not HAS_SIGNAL_REGISTRY:
                print("Error: Signal registry not available.")
                return

            registry = get_registry()
            router = SignalRouter(registry)
            memory_type = args.type

            if args.list:
                signals = router.get_active_signals(memory_type)
                print_signals(signals, memory_type)

            elif args.toggle:
                if args.enable and args.disable:
                    print("Error: Cannot use both --enable and --disable")
                    return
                if args.enable:
                    registry.enable(args.toggle)
                    print(f"Enabled signal: {args.toggle}")
                elif args.disable:
                    registry.disable(args.toggle)
                    print(f"Disabled signal: {args.toggle}")
                else:
                    current = registry.is_enabled(args.toggle)
                    if current:
                        registry.disable(args.toggle)
                        print(f"Disabled signal: {args.toggle}")
                    else:
                        registry.enable(args.toggle)
                        print(f"Enabled signal: {args.toggle}")
                router.clear_cache()

            elif args.export:
                registry.save(args.export)
                print(f"Exported registry to: {args.export}")

            elif getattr(args, 'import_file', None):  # <-- Fixed attribute access
                registry.load(args.import_file)
                print(f"Imported registry from: {args.import_file}")
                router.clear_cache()

            elif args.reset:
                registry.reload()
                router.clear_cache()
                print("Registry reset to defaults")

            else:
                signals = router.get_active_signals(memory_type)
                print_signals(signals, memory_type)

        # ---- query-history (NEW) ----
        
        elif args.command == "query-history":
            if not HAS_QUERY_HISTORY:
                print("Error: Query history module not available.")
                return

            history = QueryHistory()

            if args.frequent:
                queries = history.get_frequent(args.limit)

                if not queries:
                    print("No frequent queries found.")
                    return

                print(f"Most frequent queries (top {len(queries)}):")
                for index, query in enumerate(queries, 1):
                    print(f"{index:<4} {query}")
                return

            entries = history.get_recent(args.limit)
            print_query_history(entries, limit=args.limit)


        # ---- auto-store (NEW) ----
        elif args.command == "auto-store":
            if args.status or (not args.enable and not args.disable and args.threshold is None and args.max is None and args.types is None):
                print_auto_store_status()
                return

            if args.enable:
                settings.AUTO_STORE_MEMORIES = True
                print("Auto-store enabled")

            if args.disable:
                settings.AUTO_STORE_MEMORIES = False
                print("Auto-store disabled")

            if args.threshold is not None:
                if 0.0 <= args.threshold <= 1.0:
                    settings.AUTO_STORE_THRESHOLD = args.threshold
                    print(f"Auto-store threshold set to {args.threshold}")
                else:
                    print("Error: Threshold must be between 0.0 and 1.0")

            if args.max is not None:
                if args.max > 0:
                    settings.AUTO_STORE_MAX_PER_SESSION = args.max
                    print(f"Auto-store max per session set to {args.max}")
                else:
                    print("Error: Max must be > 0")

            if args.types is not None:
                types = [t.strip() for t in args.types.split(',')]
                settings.AUTO_STORE_TYPES = types
                print(f"Auto-store types set to: {', '.join(types)}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.command != "serve":
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
