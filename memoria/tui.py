#!/usr/bin/env python3
"""
Memoria TUI – interactive text interface with persistent chat mode.
"""
import cmd
import json
import sys
import time
from datetime import datetime
from shared.memory_interface import MemoryInterface
from cache.config import settings
from core.logger import debug, info

# Signal registry
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


def format_table(results, limit, show_scores=True, width=80):
    """Format recall results as a table string."""
    if not results:
        return "No results found."
    lines = []
    if show_scores:
        lines.append(f"{'Rank':<6} {'Score':<10} {'Text'}")
        lines.append("-" * width)
        for item in results[:limit]:
            score = item.get('final_score', 0)
            text = item.get('text', '')[:width - 20]
            lines.append(f"{item['rank']:<6} {score:<10.4f} {text}")
    else:
        lines.append(f"{'Rank':<6} {'Text'}")
        lines.append("-" * width)
        for item in results[:limit]:
            text = item.get('text', '')[:width - 10]
            lines.append(f"{item['rank']:<6} {text}")
    return "\n".join(lines)


def format_goals(goals):
    """Format goals as a table string."""
    if not goals:
        return "No goals found."
    lines = [f"{'ID':<6} {'Goal':<30} {'Progress':<12} {'Status'}"]
    lines.append("-" * 70)
    for g in goals:
        lines.append(
            f"{g['id']:<6} {g.get('goal', '')[:28]:<30} "
            f"{g.get('progress', '')[:12]:<12} {g.get('status', '')}"
        )
    return "\n".join(lines)


def format_signals(signals, memory_type="general"):
    """Format signals as a table string."""
    if not signals:
        return "No signals found."

    registry = get_registry() if HAS_SIGNAL_REGISTRY else None
    lines = []
    lines.append(f"Signals for type: {memory_type}")
    lines.append(f"{'Signal':<18} {'Weight':<10} {'Cost':<8} {'Enabled':<8}")
    lines.append("-" * 55)

    for name, weight in signals.items():
        if registry:
            cost = registry.get_cost(name)
            enabled = "✅" if registry.is_enabled(name) else "❌"
        else:
            cost = "unknown"
            enabled = "?"
        lines.append(f"{name:<18} {weight:<10.4f} {cost:<8} {enabled:<8}")

    return "\n".join(lines)


def format_query_history(entries, limit=20):
    """Format query history entries from QueryHistory."""
    if not entries:
        return "No history entries found."

    lines = []
    lines.append(f"{'Timestamp':<20} {'Results':<8} {'Query'}")
    lines.append("-" * 80)

    for entry in entries[:limit]:
        timestamp = entry.get("timestamp", "")

        if isinstance(timestamp, (int, float)):
            try:
                timestamp = datetime.fromtimestamp(timestamp).strftime(
                    "%Y-%m-%d %H:%M"
                )
            except (OverflowError, OSError, ValueError):
                timestamp = str(timestamp)
        else:
            timestamp = str(timestamp)[:16]

        results = entry.get("results", [])
        result_count = len(results) if isinstance(results, list) else 0
        query = str(entry.get("query", ""))

        lines.append(
            f"{timestamp:<20} "
            f"{result_count:<8} "
            f"{query}"
        )

    return "\n".join(lines)

def format_query_diff(diff_result):
    """Format query diff results as a string."""
    if "error" in diff_result:
        return f"Error: {diff_result['error']}"

    lines = []
    lines.append(f"\nComparing {diff_result['entry1']['id']} vs {diff_result['entry2']['id']}")
    lines.append(f"  Query 1: {diff_result['entry1']['query'][:60]}...")
    lines.append(f"  Query 2: {diff_result['entry2']['query'][:60]}...")
    lines.append(f"  Common results: {diff_result['common_results']}")
    lines.append(f"  Only in first: {diff_result['only_in_first']}")
    lines.append(f"  Only in second: {diff_result['only_in_second']}")

    if diff_result['score_changes']:
        lines.append("\nTop score changes:")
        for change in diff_result['score_changes'][:5]:
            text = change.get('text', '')[:30]
            delta = change.get('delta', 0)
            old = change.get('score_old', 0)
            new = change.get('score_new', 0)
            lines.append(f"  {text}... {old:.3f} → {new:.3f} (Δ{delta:+.3f})")

    return "\n".join(lines)


def format_auto_store_status():
    """Format auto-store status as a string."""
    lines = []
    status = "enabled" if settings.AUTO_STORE_MEMORIES else "disabled"
    lines.append(f"Auto-store: {status}")
    lines.append(f"Threshold: {settings.AUTO_STORE_THRESHOLD}")
    lines.append(f"Max per session: {settings.AUTO_STORE_MAX_PER_SESSION}")
    lines.append(f"Types: {', '.join(settings.AUTO_STORE_TYPES)}")
    return "\n".join(lines)


class MemoryShell(cmd.Cmd):
    intro = f"Memoria TUI v4.5. Type 'help' for commands.\n"
    prompt = "Memory> "
    chat_prompt = "Chat> "

    def __init__(self):
        super().__init__()
        self.mem = MemoryInterface()
        self.default_limit = settings.CLI_DEFAULT_LIMIT
        self.show_scores = settings.CLI_SHOW_SCORES
        self.table_width = settings.CLI_TABLE_WIDTH
        self.in_chat_mode = False
        self.chat_history = []  # list of {"user": str, "assistant": str}
        self.auto_store_override = None  # None = use config, True/False = override
        info("[TUI] Initialized", category="tui")

        # Signal registry
        if HAS_SIGNAL_REGISTRY:
            self.registry = get_registry()
            self.signal_router = SignalRouter(self.registry)
        else:
            self.registry = None
            self.signal_router = None

        # Query history
        if HAS_QUERY_HISTORY:
            self.history = QueryHistory()
        else:
            self.history = None

    # ---- Chat Mode Management ----
    def enter_chat_mode(self, initial_prompt=None):
        self.in_chat_mode = True
        self.prompt = self.chat_prompt
        print("Entering chat mode. Type '.back' or '.exit' to return to main shell.")
        auto_status = "enabled" if settings.AUTO_STORE_MEMORIES else "disabled"
        if self.auto_store_override is not None:
            auto_status = "enabled" if self.auto_store_override else "disabled (override)"
        print(f"Auto-store: {auto_status}")
        if initial_prompt:
            self._send_chat_message(initial_prompt)

    def exit_chat_mode(self):
        self.in_chat_mode = False
        self.prompt = "Memory> "
        print("Exited chat mode. Type 'chat' to re-enter.")

    def _send_chat_message(self, message):
        """Send a chat message, store history, and print response."""
        print("Thinking...", end="", flush=True)
        try:
            response = self.mem.chat(message, auto_store=self.auto_store_override)
            # Clear "Thinking..."
            print("\r" + " " * 12 + "\r", end="")
            if response:
                print(response)
                self.chat_history.append({"user": message, "assistant": response})
            else:
                print("[No response from assistant]")
        except Exception as e:
            print("\r" + " " * 12 + "\r", end="")
            print(f"[Error: {e}]")

    def _show_chat_history(self):
        if not self.chat_history:
            print("No chat history.")
            return
        for idx, turn in enumerate(self.chat_history, 1):
            print(f"[{idx}] User: {turn['user']}")
            print(f"    Assistant: {turn['assistant']}")
            print()

    def _chat_help(self):
        print("Chat mode commands:")
        print("  .back        Exit chat mode")
        print("  .exit        Exit chat mode (same as .back)")
        print("  .history     Show chat history")
        print("  .clear       Clear chat history")
        print("  .info        Show chat stats")
        print("  .auto-on     Enable auto-store for this session")
        print("  .auto-off    Disable auto-store for this session")
        print("  .auto-status Show current auto-store status")
        print("  .help        Show this help")
        print("  <any text>   Send as a message to the assistant")

    # ---- Commands ----
    def do_chat(self, arg):
        if not arg:
            self.enter_chat_mode()
        else:
            self.enter_chat_mode(arg)

    def do_back(self, arg):
        if self.in_chat_mode:
            self.exit_chat_mode()
        else:
            print("Not in chat mode.")

    def do_exit(self, arg):
        print("Goodbye.")
        return True

    def default(self, line):
        if self.in_chat_mode:
            if line in [".back", ".exit"]:
                self.exit_chat_mode()
                return
            if line == ".history":
                self._show_chat_history()
                return
            if line == ".clear":
                self.chat_history = []
                print("Chat history cleared.")
                return
            if line == ".help":
                self._chat_help()
                return
            if line == ".info":
                print(f"Chat history: {len(self.chat_history)} messages.")
                return
            if line == ".auto-on":
                self.auto_store_override = True
                print("Auto-store enabled for this chat session.")
                return
            if line == ".auto-off":
                self.auto_store_override = False
                print("Auto-store disabled for this chat session.")
                return
            if line == ".auto-status":
                status = "enabled" if settings.AUTO_STORE_MEMORIES else "disabled"
                if self.auto_store_override is not None:
                    status = "enabled" if self.auto_store_override else "disabled (override)"
                print(f"Auto-store: {status}")
                print(f"Threshold: {settings.AUTO_STORE_THRESHOLD}")
                print(f"Max per session: {settings.AUTO_STORE_MAX_PER_SESSION}")
                return
            self._send_chat_message(line)
        else:
            print(f"Unknown command: {line}. Type 'help' for available commands.")

    # ---- Existing Commands ----
    def do_store(self, arg):
        if not arg:
            print("Usage: store <text>")
            return
        try:
            mid = self.mem.remember(arg)
            print(f"Stored ID: {mid}")
        except Exception as e:
            print(f"Error: {e}")

    def do_recall(self, arg):
        if not arg:
            print("Usage: recall <query> [limit]")
            return
        parts = arg.split()
        if parts[-1].isdigit():
            limit = int(parts[-1])
            query = " ".join(parts[:-1])
        else:
            limit = self.default_limit
            query = " ".join(parts)
        if not query:
            print("Usage: recall <query> [limit]")
            return
        try:
            resp = self.mem.recall(query)
            results = resp.get("results", [])
            print(f"Found {len(results)} results, showing first {limit}:")
            print(format_table(results, limit, self.show_scores, self.table_width))
        except Exception as e:
            print(f"Error: {e}")

    def do_recall_json(self, arg):
        if not arg:
            print("Usage: recall_json <query>")
            return
        try:
            resp = self.mem.recall(arg)
            print(json.dumps(resp, indent=2))
        except Exception as e:
            print(f"Error: {e}")

    def do_store_many(self, arg):
        if not arg:
            print("Usage: store-many <file>")
            return
        try:
            with open(arg, "r", encoding="utf8") as f:
                texts = json.load(f)
            if not isinstance(texts, list):
                print("Error: JSON must contain a list of strings.")
                return
            ids = self.mem.remember_many(texts)
            print(f"Stored {len(ids)} memories")
        except FileNotFoundError:
            print(f"Error: File '{arg}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in '{arg}'.")
        except Exception as e:
            print(f"Error: {e}")

    def do_set_goal(self, arg):
        parts = arg.split()
        if not parts:
            print("Usage: set-goal <goal> [progress]")
            return
        goal = parts[0]
        progress = parts[1] if len(parts) > 1 else "started"
        try:
            gid = self.mem.set_goal(goal, progress)
            print(f"Goal ID: {gid}")
        except Exception as e:
            print(f"Error: {e}")

    def do_update_goal(self, arg):
        parts = arg.split()
        if len(parts) < 1:
            print("Usage: update-goal <id> [--progress <status>] [--status <status>]")
            return
        try:
            goal_id = int(parts[0])
            progress = None
            status = None
            i = 1
            while i < len(parts):
                if parts[i] == "--progress" and i + 1 < len(parts):
                    progress = parts[i + 1]
                    i += 2
                elif parts[i] == "--status" and i + 1 < len(parts):
                    status = parts[i + 1]
                    i += 2
                else:
                    i += 1
            if not progress and not status:
                print("Error: At least one of --progress or --status required.")
                return
            self.mem.update_goal(goal_id, progress=progress, status=status)
            print(f"Goal {goal_id} updated.")
        except ValueError:
            print("Error: Goal ID must be a number.")
        except Exception as e:
            print(f"Error: {e}")

    def do_list_goals(self, arg):
        status = None
        parts = arg.split()
        if len(parts) >= 2 and parts[0] == "--status":
            status = parts[1]
        try:
            goals = self.mem.list_goals(status=status)
            print(format_goals(goals))
        except Exception as e:
            print(f"Error: {e}")

    def do_stats(self, arg):
        try:
            db = self.mem.controller.system.db
            count = db.count()
            print(f"Total memories: {count}")
        except Exception as e:
            print(f"Error: {e}")

    def do_info(self, arg):
        try:
            db = self.mem.controller.system.db
            count = db.count()
            print(f"Memoria v1.0")
            print(f"Database: {settings.DB_PATH}")
            print(f"Total memories: {count}")
            print(f"Embedding model: {settings.EMBEDDING_MODEL}")
            print(f"LLM URL: {settings.LLM_URL}{settings.LLM_ENDPOINT}")
            print(f"Top K: {settings.TOP_K}")
            print(f"Debug mode: {settings.DEBUG}")
        except Exception as e:
            print(f"Error: {e}")

    def do_graph(self, arg):
        parts = arg.split()
        if not parts:
            print("Usage: graph <entity> [depth]")
            return
        entity_name = parts[0]

        try:
            depth = int(parts[1]) if len(parts) > 1 else 1
        except ValueError:
            print("Error: depth must be a number")
            return
        try:
            graph_search = self.mem.controller.system.graph_search
            entity = graph_search.find_entity(entity_name)
            if not entity:
                print(f"Entity '{entity_name}' not found.")
                return
            neighbors = graph_search.neighbors(entity_name, depth=depth)
            print(f"Neighbors of '{entity_name}' (depth {depth}):")
            for n in neighbors:
                print(f"  {n['relation']} → {n['target']} (source: {n['source']})")
        except Exception as e:
            print(f"Error: {e}")

    def do_doctor(self, arg):
        try:
            db = self.mem.controller.system.db
            print("Running integrity checks...")
            result = db.integrity_check()
            print(f"Integrity check: {result}")
            sanity = db.sanity_check()
            print(f"DB count: {sanity['db_count']}")
            print(f"Columns: {sanity['columns']}")
        except Exception as e:
            print(f"Error: {e}")

    def do_export(self, arg):
        if not arg:
            print("Usage: export <file>")
            return
        try:
            all_memories = self.mem.controller.system.db.fetch_all()
            with open(arg, "w", encoding="utf8") as f:
                json.dump(all_memories, f, indent=2, default=str)
            print(f"Exported {len(all_memories)} memories to {arg}")
        except Exception as e:
            print(f"Error: {e}")

    def do_import(self, arg):
        if not arg:
            print("Usage: import <file>")
            return
        try:
            with open(arg, "r", encoding="utf8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                print("Error: JSON must contain a list of memory objects.")
                return
            texts = [
                item.get("text") or item.get("normalized_text", "")
                for item in data
                if item.get("text") or item.get("normalized_text")
            ]
            ids = self.mem.remember_many(texts)
            print(f"Imported {len(ids)} memories from {arg}")
        except FileNotFoundError:
            print(f"Error: File '{arg}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in '{arg}'.")
        except Exception as e:
            print(f"Error: {e}")

    # ---- Signals Commands ----
    def do_signals(self, arg):
        """Show active ranking signals."""
        if not HAS_SIGNAL_REGISTRY:
            print("Signal registry not available.")
            return

        parts = arg.split()
        memory_type = parts[0] if parts else "general"

        signals = self.signal_router.get_active_signals(memory_type)
        print(format_signals(signals, memory_type))

    def do_signal_toggle(self, arg):
        """Toggle a signal on/off. Usage: signal_toggle <signal_name>"""
        if not HAS_SIGNAL_REGISTRY:
            print("Signal registry not available.")
            return

        parts = arg.split()
        if not parts:
            print("Usage: signal_toggle <signal_name>")
            return

        name = parts[0]
        current = self.registry.is_enabled(name)
        if current:
            self.registry.disable(name)
            print(f"Disabled signal: {name}")
        else:
            self.registry.enable(name)
            print(f"Enabled signal: {name}")
        self.signal_router.clear_cache()

    def do_signal_enable(self, arg):
        """Enable a signal. Usage: signal_enable <signal_name>"""
        if not HAS_SIGNAL_REGISTRY:
            print("Signal registry not available.")
            return

        parts = arg.split()
        if not parts:
            print("Usage: signal_enable <signal_name>")
            return

        name = parts[0]
        self.registry.enable(name)
        self.signal_router.clear_cache()
        print(f"Enabled signal: {name}")

    def do_signal_disable(self, arg):
        """Disable a signal. Usage: signal_disable <signal_name>"""
        if not HAS_SIGNAL_REGISTRY:
            print("Signal registry not available.")
            return

        parts = arg.split()
        if not parts:
            print("Usage: signal_disable <signal_name>")
            return

        name = parts[0]
        self.registry.disable(name)
        self.signal_router.clear_cache()
        print(f"Disabled signal: {name}")

    def do_signal_reset(self, arg):
        """Reset signals to defaults. Usage: signal_reset"""
        if not HAS_SIGNAL_REGISTRY:
            print("Signal registry not available.")
            return

        self.registry.reload()
        self.signal_router.clear_cache()
        print("Signals reset to defaults.")

    # ---- Query History Commands (NEW) ----
    def do_history(self, arg):
        """Query history. Usage: history [recent|frequent|context|previous]"""
        if not HAS_QUERY_HISTORY:
            print("Query history not available.")
            return

        parts = arg.split()

        if not parts:
            entries = self.history.get_recent(n=10)
            print(format_query_history(entries, limit=10))
            return

        subcmd = parts[0].lower()

        if subcmd == "recent":
            limit = 10

            if len(parts) > 1:
                try:
                    limit = int(parts[1])
                except ValueError:
                    print("Error: limit must be a number")
                    return

                if limit <= 0:
                    print("Error: limit must be greater than 0")
                    return

            entries = self.history.get_recent(n=limit)
            print(format_query_history(entries, limit=limit))
            return

        if subcmd == "frequent":
            limit = 10

            if len(parts) > 1:
                try:
                    limit = int(parts[1])
                except ValueError:
                    print("Error: limit must be a number")
                    return

                if limit <= 0:
                    print("Error: limit must be greater than 0")
                    return

            queries = self.history.get_frequent(n=limit)

            if not queries:
                print("No frequent queries found.")
                return

            print("Frequent queries:")
            for index, query in enumerate(queries, 1):
                print(f"  {index}. {query}")
            return

        if subcmd == "context":
            queries = self.history.get_context()

            if not queries:
                print("No query context available.")
                return

            print("Recent query context:")
            for index, query in enumerate(queries, 1):
                print(f"  {index}. {query}")
            return

        if subcmd == "previous":
            query = self.history.get_previous_query()

            if query is None:
                print("No previous query available.")
                return

            print(f"Previous query: {query}")
            return

        print(f"Unknown history subcommand: {subcmd}")
        print("Available: recent, frequent, context, previous")
    # ---- Auto-Store Commands (NEW) ----
    def do_autostore(self, arg):
        """Manage auto-store settings. Usage: autostore [on|off|threshold <value>|max <value>|types <list>|status]"""
        if not arg or arg == "status":
            print(format_auto_store_status())
            return

        parts = arg.split()
        subcmd = parts[0].lower()

        if subcmd == "on":
            settings.AUTO_STORE_MEMORIES = True
            print("Auto-store enabled")
        elif subcmd == "off":
            settings.AUTO_STORE_MEMORIES = False
            print("Auto-store disabled")
        elif subcmd == "threshold" and len(parts) > 1:
            try:
                value = float(parts[1])
                if 0.0 <= value <= 1.0:
                    settings.AUTO_STORE_THRESHOLD = value
                    print(f"Auto-store threshold set to {value}")
                else:
                    print("Error: Threshold must be between 0.0 and 1.0")
            except ValueError:
                print("Error: Threshold must be a number")
        elif subcmd == "max" and len(parts) > 1:
            try:
                value = int(parts[1])
                if value > 0:
                    settings.AUTO_STORE_MAX_PER_SESSION = value
                    print(f"Auto-store max per session set to {value}")
                else:
                    print("Error: Max must be > 0")
            except ValueError:
                print("Error: Max must be a number")
        elif subcmd == "types" and len(parts) > 1:
            types = [t.strip() for t in parts[1].split(',')]
            settings.AUTO_STORE_TYPES = types
            print(f"Auto-store types set to: {', '.join(types)}")
        else:
            print(f"Unknown autostore subcommand: {subcmd}")
            print("Available: on, off, threshold <value>, max <value>, types <list>, status")

    def do_quit(self, arg):
        if self.in_chat_mode:
            self.exit_chat_mode()
        print("Goodbye.")
        return True

    def do_q(self, arg):
        return self.do_quit(arg)

    def do_h(self, arg):
        self.do_help(arg)

    def help_help(self):
        print("List available commands. Type 'help <command>' for details.")

    def emptyline(self):
        pass


def main():
    try:
        MemoryShell().cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye.")
        sys.exit(0)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
