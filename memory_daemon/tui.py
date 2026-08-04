#!/usr/bin/env python3
"""
Memory Daemon TUI – interactive text interface with persistent chat mode.
"""
import cmd
import json
import sys
import time
from datetime import datetime
from shared.memory_interface import MemoryInterface
from cache.config import settings


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


class MemoryShell(cmd.Cmd):
    intro = f"Memory Daemon TUI v0.3.0. Type 'help' for commands.\n"
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

    # ---- Chat Mode Management ----
    def enter_chat_mode(self, initial_prompt=None):
        self.in_chat_mode = True
        self.prompt = self.chat_prompt
        print("Entering chat mode. Type '.back' or '.exit' to return to main shell.")
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
            response = self.mem.chat(message)
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
            self._send_chat_message(line)
        else:
            print(f"Unknown command: {line}. Type 'help' for available commands.")

    # ---- Existing Commands (unchanged) ----
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
            print(f"Memory Daemon v0.3.0")
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
        depth = int(parts[1]) if len(parts) > 1 else 1
        try:
            graph_search = self.mem.controller.system.graph_search
            entity = graph_search.find_entity(entity_name)
            if not entity:
                print(f"Entity '{entity_name}' not found.")
                return
            neighbors = graph_search.neighbors(entity.id, depth=depth)
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
                item.get('text', item.get('normalized_text', ''))
                for item in data
                if item.get('text')
            ]
            ids = self.mem.remember_many(texts)
            print(f"Imported {len(ids)} memories from {arg}")
        except FileNotFoundError:
            print(f"Error: File '{arg}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in '{arg}'.")
        except Exception as e:
            print(f"Error: {e}")

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


if __name__ == "__main__":
    main()
