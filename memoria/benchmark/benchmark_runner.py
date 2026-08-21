"""
benchmark_runner.py

Runs benchmark questions against the local memory system.

No HTTP.
No API.
Writes flight recorder output.
"""

import json
import time
import argparse
import sys
from pathlib import Path

from benchmark.benchmark_writer import BenchmarkWriter
from shared.memory_interface import MemoryInterface
from core.logger import debug, info, error

# Default paths
DEFAULT_QUESTION_FILE = "benchmark_output/benchmark_questions.json"
PROGRESS_INTERVAL = 5


class BenchmarkRunner:

    def __init__(self, question_file: str = None):
        """
        Initialize benchmark runner.

        Args:
            question_file: Path to questions JSON file
        """
        self.memory = MemoryInterface()
        self.writer = BenchmarkWriter()
        self.question_file = question_file or DEFAULT_QUESTION_FILE

        # ---- Plugin support ----
        self.plugin_manager = getattr(self.memory, 'plugin_manager', None)
        self.custom_adapters = []
        self.custom_analyzers = []
        if self.plugin_manager:
            self._register_custom_adapters()
            self._register_custom_analyzers()

        info(f"[Benchmark] Using questions from: {self.question_file}", category="benchmark")

    def _register_custom_adapters(self):
        """Register custom benchmark adapters from plugins."""
        try:
            adapters = self.plugin_manager.memoria_register_benchmark_adapter()
            for adapter_config in adapters:
                if isinstance(adapter_config, dict) and 'name' in adapter_config and 'adapter' in adapter_config:
                    self.custom_adapters.append(adapter_config)
                    info(f"[Benchmark] Registered custom adapter: {adapter_config['name']}", category="benchmark")
        except Exception as e:
            error(f"[Benchmark] Failed to register custom adapters: {e}", category="benchmark")

    def _register_custom_analyzers(self):
        """Register custom analyzers from plugins."""
        try:
            analyzers = self.plugin_manager.memoria_register_analyzer()
            for analyzer_config in analyzers:
                if isinstance(analyzer_config, dict) and 'name' in analyzer_config and 'analyzer' in analyzer_config:
                    self.custom_analyzers.append(analyzer_config)
                    info(f"[Benchmark] Registered custom analyzer: {analyzer_config['name']}", category="benchmark")
        except Exception as e:
            error(f"[Benchmark] Failed to register custom analyzers: {e}", category="benchmark")

    # -------------------------------------
    # LOAD QUESTIONS
    # -------------------------------------

    def load_questions(self):
        """Load benchmark questions from JSON file."""
        try:
            with open(self.question_file, "r", encoding="utf8") as f:
                return json.load(f)
        except FileNotFoundError:
            error(f"[Benchmark] Question file not found: {self.question_file}", category="benchmark")
            return []
        except json.JSONDecodeError as e:
            error(f"[Benchmark] Invalid JSON in question file: {e}", category="benchmark")
            return []

    # -------------------------------------
    # FIND EXPECTED
    # -------------------------------------

    def find_expected_rank(self, results, expected):
        """
        Find the rank of the expected memory in results.
        Checks text AND metadata for matches.
        """
        expected = str(expected).lower()

        for result in results:
            # Check text first (existing behavior)
            text = (
                result.get("normalized_text")
                or result.get("text", "")
            ).lower()

            if expected in text:
                return result.get("rank")

            # Check metadata for session IDs and other fields
            metadata = result.get("metadata", {})
            if metadata:
                # Check all string values in metadata
                for value in metadata.values():
                    if isinstance(value, str) and expected in value.lower():
                        return result.get("rank")
                    # Check if value is a list of strings
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and expected in item.lower():
                                return result.get("rank")

        return None

    def find_expected_by_ids(self, results, expected_ids):
        """
        Find the rank of a memory by checking metadata IDs.

        Args:
            results: List of result dicts
            expected_ids: List of metadata IDs to match

        Returns:
            int: Rank (1-indexed) or None if not found
        """
        if not expected_ids:
            return None

        expected_ids = [str(id) for id in expected_ids]

        for result in results:
            metadata = result.get("metadata", {})
            # Try different possible ID fields
            for key in ["id", "session_id", "question_id", "answer_id"]:
                if key in metadata:
                    if str(metadata[key]) in expected_ids:
                        return result.get("rank")

        return None

    # -------------------------------------
    # RUN
    # -------------------------------------

    def run(self, limit=None):
        """
        Run the benchmark.

        Args:
            limit: Optional limit on number of questions to run

        Returns:
            str: Path to output file
        """
        print()
        print("=" * 60)
        print("[BENCHMARK START]")
        print("=" * 60)

        # ---- Plugin hook: pre-run ----
        if self.plugin_manager:
            try:
                self.plugin_manager.memoria_pre_query(None)  # Placeholder for pre-benchmark
            except Exception as e:
                error(f"[Plugin] pre-run hook error: {e}", category="benchmark")

        questions = self.load_questions()
        if not questions:
            error("[Benchmark] No questions loaded, exiting", category="benchmark")
            return None

        if limit:
            questions = questions[:limit]

        # ---- Allow custom adapters to modify questions ----
        for adapter_config in self.custom_adapters:
            try:
                adapter = adapter_config['adapter']
                if hasattr(adapter, 'modify_questions'):
                    questions = adapter.modify_questions(questions)
                if hasattr(adapter, 'pre_run'):
                    adapter.pre_run(self)
            except Exception as e:
                error(f"[Benchmark] Custom adapter error: {e}", category="benchmark")

        total = len(questions)
        info(f"[Benchmark] {total} questions loaded", category="benchmark")

        start = time.perf_counter()
        next_progress = PROGRESS_INTERVAL

        for index, item in enumerate(questions, start=1):
            query = item.get("query")
            expected = item.get("expected")
            expected_ids = item.get("expected_ids", [])

            if not query:
                debug(f"[Benchmark] Skipping item {index}: missing query", category="benchmark")
                continue

            # For LongMemEval, expected may be empty but expected_ids has the answer
            if not expected and not expected_ids:
                debug(f"[Benchmark] Skipping item {index}: missing expected and expected_ids", category="benchmark")
                continue

            query_start = time.perf_counter()

            try:
                response = self.memory.recall(query)
            except Exception as e:
                error(f"[Benchmark] Query failed: {e}", category="benchmark")
                # Record failure
                self.writer.record(
                    query=query,
                    expected=expected or "",
                    expected_ids=expected_ids,
                    expected_rank=None,
                    retrieved=False,
                    candidates=[],
                    runtime_ms=0.0,
                    diagnostics={"error": str(e)}
                )
                continue

            query_time = (time.perf_counter() - query_start) * 1000

            # Extract results and diagnostics
            if isinstance(response, dict):
                results = response.get("results", [])
                diagnostics = response.get("diagnostics", {})
            else:
                results = response
                diagnostics = {}

            # Try text match first
            expected_rank = None
            if expected:
                expected_rank = self.find_expected_rank(results, expected)

            # If text match fails, try ID match
            if expected_rank is None and expected_ids:
                expected_rank = self.find_expected_by_ids(results, expected_ids)

            # Record result
            self.writer.record(
                query=query,
                expected=expected or "",
                expected_ids=expected_ids,
                expected_rank=expected_rank,
                retrieved=(expected_rank is not None),
                candidates=results,
                runtime_ms=query_time,
                diagnostics=diagnostics
            )

            # ---- Custom adapter hook per query ----
            for adapter_config in self.custom_adapters:
                try:
                    adapter = adapter_config['adapter']
                    if hasattr(adapter, 'on_query'):
                        adapter.on_query(index, item, response)
                except Exception as e:
                    error(f"[Benchmark] Custom adapter on_query error: {e}", category="benchmark")

            # Progress reporting
            percent = int(index / total * 100)
            if percent >= next_progress:
                elapsed = time.perf_counter() - start
                rate = index / max(elapsed, 0.001)
                eta = (total - index) / rate

                info(
                    f"[Benchmark] {percent}% {index}/{total} "
                    f"ETA {eta:.1f}s",
                    category="benchmark"
                )

                next_progress += PROGRESS_INTERVAL

        runtime = time.perf_counter() - start
        info(f"[Benchmark] Total runtime: {runtime:.2f}s", category="benchmark")

        outfile = self.writer.write()
        info(f"[Benchmark] Results written to: {outfile}", category="benchmark")

        # ---- Plugin hook: post-run ----
        if self.plugin_manager:
            try:
                self.plugin_manager.memoria_post_query(None, {"results_file": outfile})
            except Exception as e:
                error(f"[Plugin] post-run hook error: {e}", category="benchmark")

        # ---- Run custom analyzers (optional) ----
        for analyzer_config in self.custom_analyzers:
            try:
                analyzer = analyzer_config['analyzer']
                if hasattr(analyzer, 'analyze'):
                    analyzer.analyze(outfile)
                else:
                    analyzer(outfile)
            except Exception as e:
                error(f"[Benchmark] Custom analyzer error: {e}", category="benchmark")

        return outfile


# -----------------------------------------
# MAIN
# -----------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark against the memory system"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run first N benchmark questions"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Path to questions JSON file (default: benchmark_output/benchmark_questions.json)"
    )
    args = parser.parse_args()

    runner = BenchmarkRunner(question_file=args.questions)
    outfile = runner.run(limit=args.limit)

    if outfile:
        print(f"\n[OK] Results: {outfile}")
    else:
        print("\n[FAIL] Benchmark failed to complete")
        sys.exit(1)
