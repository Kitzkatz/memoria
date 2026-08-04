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


# -----------------------------------------
# SETTINGS
# -----------------------------------------

DEFAULT_RESULTS_DIR = (
    "benchmark_output/results"
)


# -----------------------------------------
# ANALYZER
# -----------------------------------------

class BenchmarkAnalyzer:


    def __init__(self, filepath):

        self.filepath = Path(filepath)

        self.data = None


    # -------------------------------------
    # LOAD
    # -------------------------------------

    def load(self):

        with open(
            self.filepath,
            "r",
            encoding="utf8"
        ) as f:

            self.data = json.load(f)



    # -------------------------------------
    # HELPERS
    # -------------------------------------

    def records(self):

        return self.data.get(
            "records",
            []
        )



    # -------------------------------------
    # ANALYZE
    # -------------------------------------

    def analyze(self):

        records = self.records()

        total = len(records)

        retrieved = 0
        top1 = 0
        top3 = 0
        recall_counts = {1: 0, 3: 0, 5: 0, 10: 0}
        SHOW_MMR_DETAILS = False

        rank_distribution = Counter()
        candidate_counts = []
        runtime = []
        component_totals = defaultdict(list)
        top3_scores = []
        not_top3_scores = []
        final_scores = []

        for record in records:
            rank = record.get("expected_rank")
            if rank:
                retrieved += 1
                rank_distribution[rank] += 1
                if rank == 1:
                    top1 += 1
                if rank <= 3:
                    top3 += 1
                for k in recall_counts:
                    if rank <= k:
                        recall_counts[k] += 1

            diagnostics = record.get("diagnostics", {})
            runtime_ms = record.get("runtime_ms")
            if runtime_ms:
                runtime.append(runtime_ms)

            for key in ("embedding_ms", "faiss_ms", "database_ms", "ranking_ms", "formatting_ms", "total_query_ms"):
                value = diagnostics.get(key)
                if value is not None:
                    component_totals[key].append(value)

            timings = diagnostics.get("timings", {})
            for key, value in timings.items():
                component_totals[key].append(value)

            candidates = record.get("candidates", [])
            candidate_counts.append(record.get("candidate_count", len(candidates)))

            if candidates:
                best = candidates[0]
                final_score = best.get("final_score", 0)
                final_scores.append(final_score)
                score = best.get("score", 0)
                if rank and rank <= 3:
                    top3_scores.append(score)
                else:
                    not_top3_scores.append(score)

        print()
        print("=" * 60)
        print("[BENCHMARK ANALYSIS]")
        print("=" * 60)

        print()
        print("Questions:", total)
        failed = total - retrieved
        print("Failed: ", failed)

        print()
        print("Retrieved:", retrieved, f"({retrieved/max(total,1)*100:.2f}%)")

        if top3_scores:
            print("Top3 avg final score:", round(sum(top3_scores) / len(top3_scores), 4))
        else:
            print("Top3 avg final score: N/A")

        print()
        print("[FINAL SCORE]")
        if final_scores:
            print("Average:", round(sum(final_scores) / len(final_scores), 4))
            print("Max:", round(max(final_scores), 4))
            print("Min:", round(min(final_scores), 4))
        else:
            print("Average: N/A")
            print("Max: N/A")
            print("Min: N/A")

        print()
        print("Top 1:", top1, f"({top1/max(total,1)*100:.2f}%)")
        print("Top 3:", top3, f"({top3/max(total,1)*100:.2f}%)")

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
            print(name, count, f"({count/max(retrieved,1)*100:.2f}%)")

        print()
        print("[SCORE ANALYSIS]")
        print()
        print("Average Candidates Returned: ", round(sum(candidate_counts) / max(len(candidate_counts), 1), 2))
        print()

        if top3_scores:
            print("Successful avg score:", round(sum(top3_scores) / len(top3_scores), 4))
        else:
            print("Successful avg score: N/A")

        if not_top3_scores:
            print("Failure avg score:", round(sum(not_top3_scores) / len(not_top3_scores), 4))
        else:
            print("Failure avg score: N/A")

        print()
        print("[TIMING]")
        for key, values in component_totals.items():
            if values:
                print(key, round(sum(values) / len(values), 3), "ms")

        print()
        print("[MMR ANALYSIS]")
        print()
        mmr_changed_count = 0
        mmr_total = 0
        mmr_moves = []

        for record in records:
            diagnostics = record.get("diagnostics", {})
            before_mmr = diagnostics.get("before_mmr", [])
            after_mmr = diagnostics.get("after_mmr", [])
            changed = diagnostics.get("mmr_changed", False)
            moves = diagnostics.get("mmr_moves", 0)

            if before_mmr and after_mmr:
                mmr_total += 1
                if changed:
                    mmr_changed_count += 1
                mmr_moves.append(moves)

                if SHOW_MMR_DETAILS:
                    print()
                    print("Query:", record.get("query"))
                    print("[MMR BEFORE]")
                    for mem_id in before_mmr[:MMR_DISPLAY_LIMIT]:
                        print(mem_id)
                    print()
                    print("[MMR AFTER]")
                    for mem_id in after_mmr[:MMR_DISPLAY_LIMIT]:
                        print(mem_id)
                    print("Changed:", changed)
                    print("Moves:", moves)
                    print("-" * 40)

        print()
        print("MMR tracked:", mmr_total)
        print("MMR reordered:", mmr_changed_count, f"({mmr_changed_count/max(mmr_total,1)*100:.2f}%)")
        if mmr_moves:
            print("Average MMR moves:", round(sum(mmr_moves) / len(mmr_moves), 2))

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
            print(name, count, f"({count/max(total,1)*100:.2f}%)")

        print()
        print("[Recall@K]")
        for k, count in recall_counts.items():
            print(f"Recall@{k}: ", count, f"({count/max(total,1)*100:.2f}%)")

        print()
        print("=" * 60)

    # -------------------------------------
    # EXPLAIN FAILURES
    # -------------------------------------

    def explain_failures(self, limit=20):

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
                    print()
                    print("-" * 60)
                    print("Query:", record.get("query"))
                    print("Expected:", record.get("expected"))
                    print("Result: NEVER RETURNED (not in candidate pool)")
                    print(
                        "Top pick instead:",
                        top.get("text", "")[:80],
                        "final_score=", round(top.get("final_score", 0), 4)
                    )
                    shown += 1

                continue

            expected_idx = rank - 1

            if expected_idx >= len(candidates):
                continue

            expected_candidate = candidates[expected_idx]

            near_misses.append((record, top, expected_candidate))

            if shown < limit:
                print()
                print("-" * 60)
                print("Query:", record.get("query"))
                print("Expected:", record.get("expected"))
                print("Expected landed at rank:", rank)
                print()

                self._print_signal_comparison(top, expected_candidate)

                shown += 1

        print()
        print("=" * 60)
        print("[FAILURE SUMMARY]")
        print("Never retrieved at all:", never_found)
        print("Retrieved but outranked (near misses):", len(near_misses))
        print("=" * 60)

        if near_misses:
            self._signal_diff_summary(near_misses)

    # -------------------------------------
    # SIGNAL COMPARISON (single query)
    # -------------------------------------

    def _print_signal_comparison(self, winner, expected):

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
    # SIGNAL DIFF SUMMARY (aggregate)
    # -------------------------------------

    def _signal_diff_summary(self, near_misses):

        fields = ["semantic", "importance", "recency", "token", "feedback"]

        totals = {f: 0.0 for f in fields}
        counts = {f: 0 for f in fields}

        attr_total = 0.0
        attr_count = 0

        for record, winner, expected in near_misses:

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

# -----------------------------------------
# MAIN
# -----------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python benchmark_analyzer.py file.json [--explain N]")
        sys.exit(1)

    analyzer = BenchmarkAnalyzer(sys.argv[1])
    analyzer.load()
    analyzer.analyze()

    if "--explain" in sys.argv:
        idx = sys.argv.index("--explain")
        limit = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 20
        analyzer.explain_failures(limit=limit)
