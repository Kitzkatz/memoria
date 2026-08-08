"""
benchmark_analyzer.py

Reads benchmark flight recorder files.

Does not run queries.
Does not touch memory system.

Only analyzes results.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, Dict, Any, List

from core.logger import debug, info, error


# -----------------------------------------
# SETTINGS
# -----------------------------------------

DEFAULT_RESULTS_DIR = "benchmark_output/results"
MMR_DISPLAY_LIMIT = 10


# -----------------------------------------
# ANALYZER
# -----------------------------------------

class BenchmarkAnalyzer:

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.data = None

    # -------------------------------------
    # LOAD
    # -------------------------------------

    def load(self) -> bool:
        """Load benchmark data from file."""
        try:
            with open(self.filepath, "r", encoding="utf8") as f:
                self.data = json.load(f)
            info(f"[Analyzer] Loaded {len(self.records())} records from {self.filepath}", category="benchmark")
            return True
        except FileNotFoundError:
            error(f"[Analyzer] File not found: {self.filepath}", category="benchmark")
            return False
        except json.JSONDecodeError as e:
            error(f"[Analyzer] Invalid JSON: {e}", category="benchmark")
            return False

    def records(self) -> list:
        """Return list of records from data."""
        return self.data.get("records", []) if self.data else []

    # -------------------------------------
    # ANALYZE
    # -------------------------------------

    def analyze(self, show_mmr_details: bool = False) -> dict:
        """
        Analyze benchmark results and print summary.

        Args:
            show_mmr_details: If True, show detailed MMR changes per query

        Returns:
            dict: Analysis results
        """
        records = self.records()
        total = len(records)

        if total == 0:
            info("[Analyzer] No records to analyze", category="benchmark")
            return {"total": 0}

        # Initialize metrics
        metrics = {
            "total": total,
            "retrieved": 0,
            "top1": 0,
            "top3": 0,
            "recall_counts": {1: 0, 3: 0, 5: 0, 10: 0},
            "rank_distribution": Counter(),
            "candidate_counts": [],
            "runtime": [],
            "component_totals": defaultdict(list),
            "top3_scores": [],
            "not_top3_scores": [],
            "final_scores": [],
            "mmr_changed": 0,
            "mmr_total": 0,
            "mmr_moves": [],
        }

        for record in records:
            rank = record.get("expected_rank")
            if rank:
                metrics["retrieved"] += 1
                metrics["rank_distribution"][rank] += 1
                if rank == 1:
                    metrics["top1"] += 1
                if rank <= 3:
                    metrics["top3"] += 1
                for k in metrics["recall_counts"]:
                    if rank <= k:
                        metrics["recall_counts"][k] += 1

            # Diagnostics
            diagnostics = record.get("diagnostics", {})

            # Runtime
            runtime_ms = record.get("runtime_ms")
            if runtime_ms:
                metrics["runtime"].append(runtime_ms)

            # Component timings
            for key in ("embedding_ms", "faiss_ms", "database_ms", "ranking_ms",
                        "formatting_ms", "total_query_ms"):
                value = diagnostics.get(key)
                if value is not None:
                    metrics["component_totals"][key].append(value)

            # Candidate counts
            candidates = record.get("candidates", [])
            metrics["candidate_counts"].append(record.get("candidate_count", len(candidates)))

            # Scores
            if candidates:
                best = candidates[0]
                final_score = best.get("final_score", 0)
                metrics["final_scores"].append(final_score)
                score = best.get("score", 0)
                if rank and rank <= 3:
                    metrics["top3_scores"].append(score)
                else:
                    metrics["not_top3_scores"].append(score)

            # MMR tracking
            before_mmr = diagnostics.get("before_mmr", [])
            after_mmr = diagnostics.get("after_mmr", [])
            if before_mmr and after_mmr:
                metrics["mmr_total"] += 1
                if diagnostics.get("mmr_changed", False):
                    metrics["mmr_changed"] += 1
                metrics["mmr_moves"].append(diagnostics.get("mmr_moves", 0))

                if show_mmr_details:
                    self._print_mmr_details(record, before_mmr, after_mmr)

        # Print results
        self._print_analysis(metrics)

        return {
            "metrics": metrics,
            "summary": self._build_summary(metrics),
        }

    # -------------------------------------
    # PRINT ANALYSIS
    # -------------------------------------

    def _print_analysis(self, metrics: dict):
        """Print formatted analysis results."""
        total = metrics["total"]
        retrieved = metrics["retrieved"]
        rank_distribution = metrics["rank_distribution"]

        print()
        print("=" * 60)
        print("[BENCHMARK ANALYSIS]")
        print("=" * 60)

        print()
        print(f"Questions: {total}")
        print(f"Failed:    {total - retrieved}")

        print()
        print(f"Retrieved: {retrieved} ({retrieved/max(total,1)*100:.2f}%)")
        print(f"Top 1:     {metrics['top1']} ({metrics['top1']/max(total,1)*100:.2f}%)")
        print(f"Top 3:     {metrics['top3']} ({metrics['top3']/max(total,1)*100:.2f}%)")

        if metrics["top3_scores"]:
            avg = sum(metrics["top3_scores"]) / len(metrics["top3_scores"])
            print(f"Top3 avg final score: {avg:.4f}")

        print()
        print("[FINAL SCORE]")
        if metrics["final_scores"]:
            scores = metrics["final_scores"]
            print(f"Average: {sum(scores)/len(scores):.4f}")
            print(f"Max:     {max(scores):.4f}")
            print(f"Min:     {min(scores):.4f}")
        else:
            print("Average: N/A")

        print()
        print("[RANK DISTRIBUTION]")
        for rank, count in sorted(rank_distribution.items()):
            print(f"Rank {rank}: {count}")

        print()
        print("[RANK BUCKETS]")
        buckets = {"1": 0, "2-3": 0, "4-5": 0, "6-10": 0, "11+": 0}
        for rank, count in rank_distribution.items():
            if rank == 1:
                buckets["1"] += count
            elif rank <= 3:
                buckets["2-3"] += count
            elif rank <= 5:
                buckets["4-5"] += count
            elif rank <= 10:
                buckets["6-10"] += count
            else:
                buckets["11+"] += count
        for name, count in buckets.items():
            pct = count / max(retrieved, 1) * 100 if retrieved > 0 else 0
            print(f"{name}: {count} ({pct:.2f}%)")

        print()
        print("[SCORE ANALYSIS]")
        avg_candidates = sum(metrics["candidate_counts"]) / max(len(metrics["candidate_counts"]), 1)
        print(f"Average Candidates Returned: {avg_candidates:.2f}")

        if metrics["top3_scores"]:
            avg = sum(metrics["top3_scores"]) / len(metrics["top3_scores"])
            print(f"Successful avg score: {avg:.4f}")
        if metrics["not_top3_scores"]:
            avg = sum(metrics["not_top3_scores"]) / len(metrics["not_top3_scores"])
            print(f"Failure avg score: {avg:.4f}")

        print()
        print("[TIMING]")
        for key, values in metrics["component_totals"].items():
            if values:
                print(f"{key}: {sum(values)/len(values):.3f} ms")

        print()
        print("[MMR ANALYSIS]")
        mmr_total = metrics["mmr_total"]
        mmr_changed = metrics["mmr_changed"]
        print(f"MMR tracked: {mmr_total}")
        print(f"MMR reordered: {mmr_changed} ({mmr_changed/max(mmr_total,1)*100:.2f}%)")
        if metrics["mmr_moves"]:
            avg_moves = sum(metrics["mmr_moves"]) / len(metrics["mmr_moves"])
            print(f"Average MMR moves: {avg_moves:.2f}")

        print()
        print("[RECALL@K]")
        for k, count in metrics["recall_counts"].items():
            print(f"Recall@{k}: {count} ({count/max(total,1)*100:.2f}%)")

        print()
        print("=" * 60)

    def _print_mmr_details(self, record, before_mmr, after_mmr):
        """Print MMR details for a single record."""
        print()
        print(f"Query: {record.get('query', '')[:80]}")
        print("[MMR BEFORE]")
        for mem_id in before_mmr[:MMR_DISPLAY_LIMIT]:
            print(f"  {mem_id}")
        print("[MMR AFTER]")
        for mem_id in after_mmr[:MMR_DISPLAY_LIMIT]:
            print(f"  {mem_id}")
        print("-" * 40)

    # -------------------------------------
    # BUILD SUMMARY
    # -------------------------------------

    def _build_summary(self, metrics: dict) -> dict:
        """Build a summary dictionary from metrics."""
        total = metrics["total"]
        retrieved = metrics["retrieved"]

        return {
            "total_questions": total,
            "retrieved": retrieved,
            "accuracy": round(retrieved / max(total, 1) * 100, 2),
            "top1_accuracy": round(metrics["top1"] / max(total, 1) * 100, 2),
            "top3_accuracy": round(metrics["top3"] / max(total, 1) * 100, 2),
            "avg_final_score": round(sum(metrics["final_scores"]) / max(len(metrics["final_scores"]), 1), 4),
            "avg_candidates": round(sum(metrics["candidate_counts"]) / max(len(metrics["candidate_counts"]), 1), 2),
            "mmr_reordered_pct": round(metrics["mmr_changed"] / max(metrics["mmr_total"], 1) * 100, 2),
            "recall@1": round(metrics["recall_counts"][1] / max(total, 1) * 100, 2),
            "recall@3": round(metrics["recall_counts"][3] / max(total, 1) * 100, 2),
            "recall@5": round(metrics["recall_counts"][5] / max(total, 1) * 100, 2),
            "recall@10": round(metrics["recall_counts"][10] / max(total, 1) * 100, 2),
        }

    # -------------------------------------
    # EXPLAIN FAILURES
    # -------------------------------------

    def explain_failures(self, limit: int = 20):
        """Explain failed queries in detail."""
        records = self.records()
        shown = 0
        never_found = 0
        near_misses = []

        for record in records:
            rank = record.get("expected_rank")
            candidates = record.get("candidates", [])

            if rank and rank <= 3:
                continue

            if not candidates:
                continue

            top = candidates[0]

            if rank is None:
                never_found += 1
                if shown < limit:
                    self._print_never_found(record, top)
                    shown += 1
                continue

            expected_idx = rank - 1
            if expected_idx >= len(candidates):
                continue

            expected_candidate = candidates[expected_idx]
            near_misses.append((record, top, expected_candidate))

            if shown < limit:
                self._print_near_miss(record, top, expected_candidate, rank)
                shown += 1

        print()
        print("=" * 60)
        print("[FAILURE SUMMARY]")
        print(f"Never retrieved at all: {never_found}")
        print(f"Retrieved but outranked (near misses): {len(near_misses)}")
        print("=" * 60)

        if near_misses:
            self._signal_diff_summary(near_misses)

    def _print_never_found(self, record, top):
        """Print details for a never-found failure."""
        print()
        print("-" * 60)
        print(f"Query: {record.get('query')}")
        print(f"Expected: {record.get('expected')}")
        print("Result: NEVER RETURNED (not in candidate pool)")
        print(f"Top pick instead: {top.get('text', '')[:80]}")
        print(f"  final_score={round(top.get('final_score', 0), 4)}")

    def _print_near_miss(self, record, top, expected, rank):
        """Print details for a near-miss failure."""
        print()
        print("-" * 60)
        print(f"Query: {record.get('query')}")
        print(f"Expected: {record.get('expected')}")
        print(f"Expected landed at rank: {rank}")
        print()
        self._print_signal_comparison(top, expected)

    # -------------------------------------
    # SIGNAL COMPARISON
    # -------------------------------------

    def _print_signal_comparison(self, winner: dict, expected: dict):
        """Print signal comparison between winner and expected."""
        w_ranker = winner.get("diagnostics", {}).get("ranker", {})
        e_ranker = expected.get("diagnostics", {}).get("ranker", {})

        fields = ["semantic", "importance", "recency", "token", "feedback"]

        print(f"{'signal':<18}{'winner':>10}{'expected':>10}{'delta':>10}")

        for field in fields:
            w = w_ranker.get(field, 0.0)
            e = e_ranker.get(field, 0.0)
            print(f"{field:<18}{w:>10.4f}{e:>10.4f}{(w - e):>10.4f}")

        w_attr = winner.get("diagnostics", {}).get("attribute_boost", 0.0)
        e_attr = expected.get("diagnostics", {}).get("attribute_boost", 0.0)
        print(f"{'attribute_boost':<18}{w_attr:>10.4f}{e_attr:>10.4f}{(w_attr - e_attr):>10.4f}")

        w_score = winner.get("score", 0)
        e_score = expected.get("score", 0)
        print(f"{'score(norm)':<18}{w_score:>10.4f}{e_score:>10.4f}{(w_score - e_score):>10.4f}")

        w_final = winner.get("final_score", 0)
        e_final = expected.get("final_score", 0)
        print(f"{'final_score':<18}{w_final:>10.4f}{e_final:>10.4f}{(w_final - e_final):>10.4f}")

        w_mmr = winner.get("mmr_score", 0)
        e_mmr = expected.get("mmr_score", 0)
        print(f"{'mmr_score':<18}{w_mmr:>10.4f}{e_mmr:>10.4f}{(w_mmr - e_mmr):>10.4f}")

    # -------------------------------------
    # SIGNAL DIFF SUMMARY
    # -------------------------------------

    def _signal_diff_summary(self, near_misses: list):
        """Print aggregate signal difference summary."""
        fields = ["semantic", "importance", "recency", "token", "feedback"]

        totals = {f: 0.0 for f in fields}
        counts = {f: 0 for f in fields}

        attr_total = 0.0
        attr_count = 0

        for _, winner, expected in near_misses:
            w_ranker = winner.get("diagnostics", {}).get("ranker", {})
            e_ranker = expected.get("diagnostics", {}).get("ranker", {})

            for f in fields:
                if f in w_ranker and f in e_ranker:
                    totals[f] += w_ranker[f] - e_ranker[f]
                    counts[f] += 1

            w_attr = winner.get("diagnostics", {}).get("attribute_boost")
            e_attr = expected.get("diagnostics", {}).get("attribute_boost")
            if w_attr is not None and e_attr is not None:
                attr_total += w_attr - e_attr
                attr_count += 1

        print()
        print("[AVG SIGNAL ADVANTAGE — winner minus expected, across all near misses]")

        for f in fields:
            if counts[f]:
                avg = totals[f] / counts[f]
                print(f"{f:<18}{avg:>10.4f}")

        if attr_count:
            print(f"{'attribute_boost':<18}{attr_total/attr_count:>10.4f}")

    # -------------------------------------
    # EXPORT SUMMARY
    # -------------------------------------

    def export_summary(self, output_path: str):
        """Export analysis summary to JSON."""
        analysis = self.analyze()
        summary = analysis.get("summary", {})

        with open(output_path, "w", encoding="utf8") as f:
            json.dump({
                "file": str(self.filepath),
                "summary": summary,
                "metrics": analysis.get("metrics", {})
            }, f, indent=2, default=str)

        info(f"[Analyzer] Summary exported to {output_path}", category="benchmark")


# -----------------------------------------
# ANALYZE ALL FILES IN DIRECTORY
# -----------------------------------------

def analyze_all(results_dir: str = DEFAULT_RESULTS_DIR) -> dict:
    """Analyze all benchmark files in a directory."""
    dir_path = Path(results_dir)
    if not dir_path.exists():
        error(f"[Analyzer] Directory not found: {dir_path}", category="benchmark")
        return {}

    results = {}
    for filepath in sorted(dir_path.glob("*.json")):
        info(f"[Analyzer] Analyzing: {filepath.name}", category="benchmark")
        analyzer = BenchmarkAnalyzer(str(filepath))
        if analyzer.load():
            analysis = analyzer.analyze()
            results[filepath.name] = analysis.get("summary", {})

    return results


# -----------------------------------------
# MAIN
# -----------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python benchmark_analyzer.py file.json [--explain N] [--export summary.json]")
        print("  python benchmark_analyzer.py --all [--export summary.json]")
        sys.exit(1)

    # --all flag
    if sys.argv[1] == "--all":
        results = analyze_all()
        print(json.dumps(results, indent=2))
        sys.exit(0)

    analyzer = BenchmarkAnalyzer(sys.argv[1])
    if not analyzer.load():
        sys.exit(1)

    analyzer.analyze()

    if "--explain" in sys.argv:
        idx = sys.argv.index("--explain")
        limit = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 20
        analyzer.explain_failures(limit=limit)

    if "--export" in sys.argv:
        idx = sys.argv.index("--export")
        outfile = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "summary.json"
        analyzer.export_summary(outfile)
