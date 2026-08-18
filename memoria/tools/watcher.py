#!/usr/bin/env python3
"""
Development watcher — runs regression tests on file changes.

Usage:
    python -m tools.watcher

Monitors:
    - memory/ (all Python files)
    - system/ (all Python files)
    - retrieval/ (all Python files)
    - ranking/ (all Python files)
    - app.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from core.logger import debug, info, error


class DevWatcher(FileSystemEventHandler):
    """Watches for file changes and triggers regression tests."""

    def __init__(self, debounce_seconds: float = 2.0):
        self.debounce_seconds = debounce_seconds
        self.last_run = 0
        self.test_command = ["python", "tests/regression.py"]

        # Track which files are being watched
        info(f"[Watcher] Debounce: {debounce_seconds}s", category="watcher")

    def run_tests(self):
        """Run the regression test suite."""
        debug("[Watcher] Running regression tests...", category="watcher")

        try:
            result = subprocess.run(
                self.test_command,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                info("[Watcher] Regression tests passed ✓", category="watcher")
                if result.stdout:
                    debug(f"[Watcher] Output: {result.stdout[:200]}...", category="watcher")
            else:
                error(f"[Watcher] Regression tests failed (code {result.returncode})", category="watcher")
                if result.stderr:
                    error(f"[Watcher] Errors: {result.stderr[:500]}", category="watcher")

        except subprocess.TimeoutExpired:
            error("[Watcher] Regression tests timed out after 60s", category="watcher")
        except Exception as e:
            error(f"[Watcher] Error running tests: {e}", category="watcher")

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        # Skip non-Python files
        if not event.src_path.endswith('.py'):
            return

        # Debounce
        now = time.time()
        if now - self.last_run < self.debounce_seconds:
            return

        self.last_run = now

        debug(f"[Watcher] Change detected: {os.path.basename(event.src_path)}", category="watcher")
        self.run_tests()


def get_watch_paths():
    """Get list of paths to watch."""
    base_dir = Path(__file__).parent.parent

    paths = [
        base_dir / "memory",
        base_dir / "system",
        base_dir / "retrieval",
        base_dir / "ranking",
        base_dir / "ingestion",
        base_dir / "graph",
        base_dir / "blackboard",
        base_dir / "routing",
        base_dir / "core",
        base_dir / "cache",
        base_dir / "app.py",
    ]

    # Only return paths that exist
    return [str(p) for p in paths if p.exists()]


def main():
    """Start the watcher."""
    watch_paths = get_watch_paths()

    if not watch_paths:
        error("[Watcher] No valid paths to watch", category="watcher")
        return

    info("========================================", category="watcher")
    info("      Memory Daemon Dev Watcher", category="watcher")
    info("========================================", category="watcher")
    info(f"   Watching {len(watch_paths)} paths", category="watcher")
    for path in watch_paths:
        info(f"     - {path}", category="watcher")
    info("========================================", category="watcher")
    info("   Runs regression tests on file change", category="watcher")
    info("   Press Ctrl+C to stop", category="watcher")
    info("========================================", category="watcher")

    handler = DevWatcher()
    observer = Observer()

    # Schedule watches
    for path in watch_paths:
        if os.path.isdir(path):
            observer.schedule(handler, path, recursive=True)
        else:
            observer.schedule(handler, os.path.dirname(path), recursive=False)

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        info("\n[Watcher] Stopping...", category="watcher")
        observer.stop()

    observer.join()
    info("[Watcher] Stopped", category="watcher")


if __name__ == "__main__":
    main()
