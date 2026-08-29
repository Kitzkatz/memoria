"""
V4 benchmark result analyzer.

Reads benchmark JSON files only.
Does not execute queries or touch the memory system.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

from core.logger import info, error

DEFAULT_RESULTS_DIR = "benchmark_output/results"
MMR_DISPLAY_LIMIT = 10


class BenchmarkAnalyzer:
    def __init__(self, filepath: str, questions_path: str = None, plugin_manager=None):
        self.filepath = Path(filepath)
        self.questions_path = Path(questions_path) if questions_path else None
        self.plugin_manager = plugin_manager
        self.data = None
        self.questions_data = None

    def load(self) -> bool:
        try:
            with open(self.filepath, "r", encoding="utf8") as f:
                self.data = json.load(f)

            # If questions file provided, load it
            if self.questions_path:
                if not self.questions_path.exists():
                    error(f"[Analyzer] Questions file not found: {self.questions_path}", category="benchmark")
                else:
                    with open(self.questions_path, "r", encoding="utf8") as qf:
                        qlist = json.load(qf)
                        # Build a dict keyed by question_id (or fallback to order)
                        self.questions_data = {}
                        for idx, q in enumerate(qlist):
                            qid = q.get("question_id", f"q_{idx}")
                            self.questions_data[qid] = q

            # If data has 'results' but not 'records', convert to internal format
            if "results" in self.data and "records" not in self.data:
                self._convert_results_to_records()

            info(
                f"[Analyzer] Loaded {len(self.records())} records from {self.filepath}",
                category="benchmark",
            )
            return True

        except FileNotFoundError:
            error(f"[Analyzer] File not found: {self.filepath}", category="benchmark")
        except json.JSONDecodeError as e:
            error(f"[Analyzer] Invalid JSON: {e}", category="benchmark")

        return False

    def records(self) -> list:
        return self.data.get("records", []) if self.data else []

    def _convert_results_to_records(self):
        """Convert adapter's 'results' format to internal 'records' format."""
        results = self.data.get("results", [])
        if not results:
            return

        records = []
        for idx, entry in enumerate(results):
            # Try to get question_id from entry, else use order
            qid = entry.get("question_id", f"q_{idx}")
            rank = entry.get("rank")
            retrieved = entry.get("retrieved", False)
            expected_rank = rank if retrieved else None

            # Get expected text and ids from questions data if available
            expected_text = ""
            expected_ids = []
            if self.questions_data and qid in self.questions_data:
                q_info = self.questions_data[qid]
                expected_text = q_info.get("expected", "")
                expected_ids = q_info.get("expected_ids", [])

            # Build a minimal candidate list (for score analysis, we fake one if rank exists)
            candidates = []
            candidate_count = entry.get("candidate_count", 0)
            if expected_rank is not None and candidate_count > 0:
                # Create a dummy candidate at the correct rank (only for analysis)
                candidates.append({
                    "rank": expected_rank,
                    "score": 1.0,
                    "final_score": 1.0,
                    "text": "dummy",
                    "metadata": {}
                })

            record = {
                "query": entry.get("question", entry.get("query", "")),
                "expected": expected_text,
                "expected_ids": expected_ids,
                "expected_rank": expected_rank,
                "retrieved": retrieved,
                "candidates": candidates,
                "candidate_count": candidate_count,
                "runtime_ms": entry.get("query_time", 0.0) * 1000,  # convert seconds to ms
                "diagnostics": {}  # no diagnostics
            }
            records.append(record)

        self.data["records"] = records

    # ------------------------------------------------------------------
    # All methods below are unchanged from the original analyzer.
    # They remain exactly as you had them.
    # ------------------------------------------------------------------

    def analyze(self, show_mmr_details: bool = False) -> dict:
        records = self.records()
        total = len(records)

        if not total:
            info("[Analyzer] No records to analyze", category="benchmark")
            return {"total": 0}

        # ---- Plugin hook: pre-analysis ----
        if self.plugin_manager:
            try:
                self.plugin_manager.memoria_analysis_pre(records, {})
            except Exception as e:
                error(f"[Plugin] analysis_pre error: {e}", category="benchmark")

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
            "policy_names": Counter(),
            "finish_reasons": Counter(),
            "submitted_sources": Counter(),
            "completed_sources": Counter(),
            "pending_sources": Counter(),
            "failed_sources": Counter(),
            "source_combinations": Counter(),
            "retrieval_wait_times": [],
            "retrieval_times": [],
            "never_retrieved_records": [],
        }

        timing_keys = (
            "query_process_ms", "embedding_ms", "retrieval_ms",
            "faiss_ms", "database_ms", "ranking_ms",
            "response_ms", "feedback_ms", "formatting_ms",
            "total_query_ms", "retrieval_wait_ms",
        )

        for record in records:
            self._analyze_record(record, metrics, timing_keys, show_mmr_details)

        self._print_analysis(metrics)
        self._print_official_metrics() 

        summary = self._build_summary(metrics)

        # ---- Plugin hook: post-analysis ----
        if self.plugin_manager:
            try:
                self.plugin_manager.memoria_analysis_post(metrics, summary)
            except Exception as e:
                error(f"[Plugin] analysis_post error: {e}", category="benchmark")

        return {
            "metrics": metrics,
            "summary": summary,
        }

    def _analyze_record(self, record, metrics, timing_keys, show_mmr_details):
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
        else:
            metrics["never_retrieved_records"].append(record)

        diagnostics = record.get("diagnostics", {})

        runtime = record.get("runtime_ms")
        if runtime is not None:
            metrics["runtime"].append(runtime)

        for key in timing_keys:
            value = diagnostics.get(key)
            if value is not None:
                metrics["component_totals"][key].append(float(value))

        retrieval = diagnostics.get("retrieval_ms")
        if retrieval is not None:
            metrics["retrieval_times"].append(float(retrieval))

        wait = diagnostics.get("retrieval_wait_ms")
        if wait is not None:
            metrics["retrieval_wait_times"].append(float(wait))

        policy = diagnostics.get("retrieval_policy")
        if policy:
            metrics["policy_names"][policy] += 1

        reason = diagnostics.get("retrieval_finish_reason")
        if reason:
            metrics["finish_reasons"][reason] += 1

        sources = {
            "submitted_sources": diagnostics.get("retrieval_submitted_sources", []),
            "completed_sources": diagnostics.get("retrieval_completed_sources", []),
            "pending_sources": diagnostics.get("retrieval_pending_sources", []),
            "failed_sources": diagnostics.get("retrieval_failed_sources", []),
        }

        for key, values in sources.items():
            for source in values:
                metrics[key][source] += 1

        submitted = sources["submitted_sources"]
        if submitted:
            metrics["source_combinations"][tuple(sorted(submitted))] += 1

        candidates = record.get("candidates", [])
        metrics["candidate_counts"].append(
            record.get("candidate_count", len(candidates))
        )

        if candidates:
            best = candidates[0]
            metrics["final_scores"].append(best.get("final_score", 0))

            score = best.get("score", 0)
            if rank and rank <= 3:
                metrics["top3_scores"].append(score)
            else:
                metrics["not_top3_scores"].append(score)

        before = diagnostics.get("before_mmr", [])
        after = diagnostics.get("after_mmr", [])

        if before and after:
            metrics["mmr_total"] += 1

            if diagnostics.get("mmr_changed", False):
                metrics["mmr_changed"] += 1

            metrics["mmr_moves"].append(diagnostics.get("mmr_moves", 0))

            if show_mmr_details:
                self._print_mmr_details(record, before, after)

    @staticmethod
    def _avg(metrics, key):
        values = metrics["component_totals"].get(key, [])
        return sum(values) / len(values) if values else None

    @staticmethod
    def _pct(value, total):
        return value / max(total, 1) * 100

    def _print_analysis(self, metrics):
        total = metrics["total"]
        retrieved = metrics["retrieved"]

        print("\n" + "=" * 60)
        print("[BENCHMARK ANALYSIS]")
        print("=" * 60)

        print(f"\nQuestions: {total}")
        print(f"Failed:    {total - retrieved}")
        print(f"\nRetrieved: {retrieved} ({self._pct(retrieved, total):.2f}%)")
        print(f"Top 1:     {metrics['top1']} ({self._pct(metrics['top1'], total):.2f}%)")
        print(f"Top 3:     {metrics['top3']} ({self._pct(metrics['top3'], total):.2f}%)")

        if metrics["top3_scores"]:
            print(
                f"Top3 avg final score: "
                f"{sum(metrics['top3_scores']) / len(metrics['top3_scores']):.4f}"
            )

        print("\n[FINAL SCORE]")
        scores = metrics["final_scores"]
        if scores:
            print(f"Average: {sum(scores) / len(scores):.4f}")
            print(f"Max:     {max(scores):.4f}")
            print(f"Min:     {min(scores):.4f}")
        else:
            print("Average: N/A")

        print("\n[RANK DISTRIBUTION]")
        for rank, count in sorted(metrics["rank_distribution"].items()):
            print(f"Rank {rank}: {count}")

        print("\n[RANK BUCKETS]")
        buckets = {"1": 0, "2-3": 0, "4-5": 0, "6-10": 0, "11+": 0}

        for rank, count in metrics["rank_distribution"].items():
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
            print(f"{name}: {count} ({self._pct(count, retrieved):.2f}%)")

        print("\n[SCORE ANALYSIS]")
        candidates = metrics["candidate_counts"]
        avg_candidates = sum(candidates) / max(len(candidates), 1)
        print(f"Average Candidates Returned: {avg_candidates:.2f}")

        if metrics["top3_scores"]:
            scores = metrics["top3_scores"]
            print(f"Successful avg score: {sum(scores) / len(scores):.4f}")

        if metrics["not_top3_scores"]:
            scores = metrics["not_top3_scores"]
            print(f"Non-top3 avg score: {sum(scores) / len(scores):.4f}")

        print("\n[TIMING]")
        timing_order = (
            "query_process_ms", "embedding_ms", "retrieval_ms",
            "retrieval_wait_ms", "database_ms", "ranking_ms",
            "response_ms", "feedback_ms", "formatting_ms",
            "faiss_ms", "total_query_ms",
        )

        for key in timing_order:
            value = self._avg(metrics, key)
            if value is not None:
                print(f"{key}: {value:.3f} ms")

        self._print_timing_accounting(metrics)
        self._print_retrieval_diagnostics(metrics)

        print("\n[MMR ANALYSIS]")
        mmr_total = metrics["mmr_total"]
        mmr_changed = metrics["mmr_changed"]
        print(f"MMR tracked: {mmr_total}")
        print(f"MMR reordered: {mmr_changed} ({self._pct(mmr_changed, mmr_total):.2f}%)")

        if metrics["mmr_moves"]:
            moves = metrics["mmr_moves"]
            print(f"Average MMR moves: {sum(moves) / len(moves):.2f}")

        print("\n[RECALL@K]")
        for k, count in metrics["recall_counts"].items():
            print(f"Recall@{k}: {count} ({self._pct(count, total):.2f}%)")

        print("\n" + "=" * 60)

    def _print_timing_accounting(self, metrics):
        total = self._avg(metrics, "total_query_ms")
        if total is None:
            return

        keys = (
            "query_process_ms",
            "embedding_ms",
            "retrieval_ms",
            "ranking_ms",
            "response_ms",
            "feedback_ms",
        )

        values = {key: self._avg(metrics, key) or 0.0 for key in keys}
        accounted = sum(values.values())
        unaccounted = total - accounted

        print("\n[TIMING ACCOUNTING]")
        labels = {
            "query_process_ms": "Query processing",
            "embedding_ms": "Embedding",
            "retrieval_ms": "Retrieval",
            "ranking_ms": "Ranking",
            "response_ms": "Response",
            "feedback_ms": "Feedback",
        }

        for key in keys:
            print(f"{labels[key] + ':':18} {values[key]:.3f} ms")

        print(f"{'Accounted:':18} {accounted:.3f} ms")
        print(f"{'Total:':18} {total:.3f} ms")
        print(f"{'Unaccounted:':18} {unaccounted:.3f} ms")
        print(f"{'Unaccounted %:':18} {unaccounted / total * 100:.2f}%")

    def _print_retrieval_diagnostics(self, metrics):
        print("\n[V4 RETRIEVAL POLICY]")

        if metrics["policy_names"]:
            print("Policies:")
            for name, count in metrics["policy_names"].most_common():
                print(f"  {name}: {count}")

        if metrics["finish_reasons"]:
            print("\nFinish reasons:")
            for reason, count in metrics["finish_reasons"].most_common():
                print(f"  {reason}: {count}")

        print("\n[RETRIEVAL SOURCES]")

        labels = (
            ("submitted_sources", "Submitted"),
            ("completed_sources", "Completed"),
            ("pending_sources", "Pending"),
            ("failed_sources", "Failed"),
        )

        for key, label in labels:
            if metrics[key]:
                print(f"{label}:")
                for source, count in metrics[key].most_common():
                    print(f"  {source}: {count}")

        if metrics["source_combinations"]:
            print("\nSubmitted source combinations:")
            for sources, count in metrics["source_combinations"].most_common():
                print(f"  {'+'.join(sources)}: {count}")

        waits = metrics["retrieval_wait_times"]
        if waits:
            print("\n[SCHEDULER WAIT]")
            print(f"Average: {sum(waits) / len(waits):.3f} ms")
            print(f"Max:     {max(waits):.3f} ms")
            print(f"Min:     {min(waits):.3f} ms")

    def _print_mmr_details(self, record, before, after):
        print(f"\nQuery: {record.get('query', '')[:80]}")
        print("[MMR BEFORE]")
        for mem_id in before[:MMR_DISPLAY_LIMIT]:
            print(f"  {mem_id}")

        print("[MMR AFTER]")
        for mem_id in after[:MMR_DISPLAY_LIMIT]:
            print(f"  {mem_id}")

        print("-" * 40)

    def _build_summary(self, metrics):
        total = metrics["total"]
        retrieved = metrics["retrieved"]

        timing = {
            key: self._avg(metrics, key)
            for key in (
                "query_process_ms",
                "embedding_ms",
                "retrieval_ms",
                "ranking_ms",
                "response_ms",
                "feedback_ms",
                "total_query_ms",
            )
        }

        accounted = None
        unaccounted = None

        if timing["total_query_ms"] is not None:
            accounted = sum(
                timing[key] or 0.0
                for key in (
                    "query_process_ms",
                    "embedding_ms",
                    "retrieval_ms",
                    "ranking_ms",
                    "response_ms",
                    "feedback_ms",
                )
            )
            unaccounted = timing["total_query_ms"] - accounted

        def rounded(value):
            return round(value, 3) if value is not None else None

        scores = metrics["final_scores"]
        candidates = metrics["candidate_counts"]

        return {
            "total_questions": total,
            "retrieved": retrieved,
            "accuracy": round(self._pct(retrieved, total), 2),
            "top1_accuracy": round(self._pct(metrics["top1"], total), 2),
            "top3_accuracy": round(self._pct(metrics["top3"], total), 2),
            "avg_final_score": round(sum(scores) / max(len(scores), 1), 4),
            "avg_candidates": round(sum(candidates) / max(len(candidates), 1), 2),
            "mmr_reordered_pct": round(
                self._pct(metrics["mmr_changed"], metrics["mmr_total"]), 2
            ),
            "recall@1": round(self._pct(metrics["recall_counts"][1], total), 2),
            "recall@3": round(self._pct(metrics["recall_counts"][3], total), 2),
            "recall@5": round(self._pct(metrics["recall_counts"][5], total), 2),
            "recall@10": round(self._pct(metrics["recall_counts"][10], total), 2),
            **{key: rounded(value) for key, value in timing.items()},
            "unaccounted_ms": rounded(unaccounted),
            "retrieval_policies": dict(metrics["policy_names"]),
            "retrieval_finish_reasons": dict(metrics["finish_reasons"]),
            "submitted_sources": dict(metrics["submitted_sources"]),
            "completed_sources": dict(metrics["completed_sources"]),
            "pending_sources": dict(metrics["pending_sources"]),
            "failed_sources": dict(metrics["failed_sources"]),
        }

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

            if not candidates or rank is None:
                never_found += 1

                if shown < limit:
                    self._print_never_found(record, candidates[0] if candidates else None)
                    shown += 1

                continue

            expected_idx = rank - 1
            if expected_idx >= len(candidates):
                continue

            expected = candidates[expected_idx]
            top = candidates[0]

            near_misses.append((record, top, expected))

            if shown < limit:
                self._print_near_miss(record, top, expected, rank)
                shown += 1

        print("\n" + "=" * 60)
        print("[FAILURE SUMMARY]")
        print(f"Never retrieved at all: {never_found}")
        print(f"Retrieved but outranked (near misses): {len(near_misses)}")
        print("=" * 60)

        if near_misses:
            self._signal_diff_summary(near_misses)

    def _print_never_found(self, record, top=None):
        print("\n" + "-" * 60)
        print(f"Query: {record.get('query')}")
        print(f"Expected: {record.get('expected')}")
        print("Result: NEVER RETURNED (not in candidate pool)")

        if top:
            print(f"Top pick instead: {top.get('text', '')[:80]}")
            print(f"  final_score={top.get('final_score', 0):.4f}")

        d = record.get("diagnostics", {})
        print(f"\nPolicy: {d.get('retrieval_policy')}")
        print(f"Finish reason: {d.get('retrieval_finish_reason')}")
        print(f"Submitted: {d.get('retrieval_submitted_sources', [])}")
        print(f"Completed: {d.get('retrieval_completed_sources', [])}")
        print(f"Pending: {d.get('retrieval_pending_sources', [])}")
        print(f"Failed: {d.get('retrieval_failed_sources', [])}")

    def _print_near_miss(self, record, top, expected, rank):
        print("\n" + "-" * 60)
        print(f"Query: {record.get('query')}")
        print(f"Expected: {record.get('expected')}")
        print(f"Expected landed at rank: {rank}\n")
        self._print_signal_comparison(top, expected)

    def _ranker_signals(self, candidate):
        return candidate.get("diagnostics", {}).get("ranker", {})

    def _print_signal_comparison(self, winner, expected):
        fields = ("semantic", "importance", "recency", "token", "feedback")

        print(f"{'signal':<18}{'winner':>10}{'expected':>10}{'delta':>10}")

        for field in fields:
            w = self._ranker_signals(winner).get(field, 0.0)
            e = self._ranker_signals(expected).get(field, 0.0)
            print(f"{field:<18}{w:>10.4f}{e:>10.4f}{w - e:>10.4f}")

        for field in ("attribute_boost", "score", "final_score", "mmr_score"):
            w = winner.get(field, winner.get("diagnostics", {}).get(field, 0.0))
            e = expected.get(field, expected.get("diagnostics", {}).get(field, 0.0))
            print(f"{field:<18}{w:>10.4f}{e:>10.4f}{w - e:>10.4f}")

    def _signal_diff_summary(self, near_misses):
        fields = ("semantic", "importance", "recency", "token", "feedback")
        totals = Counter()
        counts = Counter()
        attr_total = 0.0
        attr_count = 0

        for _, winner, expected in near_misses:
            w_ranker = self._ranker_signals(winner)
            e_ranker = self._ranker_signals(expected)

            for field in fields:
                if field in w_ranker and field in e_ranker:
                    totals[field] += w_ranker[field] - e_ranker[field]
                    counts[field] += 1

            w_attr = winner.get("diagnostics", {}).get("attribute_boost")
            e_attr = expected.get("diagnostics", {}).get("attribute_boost")

            if w_attr is not None and e_attr is not None:
                attr_total += w_attr - e_attr
                attr_count += 1

        print("\n[AVG SIGNAL ADVANTAGE — winner minus expected]")

        for field in fields:
            if counts[field]:
                print(f"{field:<18}{totals[field] / counts[field]:>10.4f}")

        if attr_count:
            print(f"{'attribute_boost':<18}{attr_total / attr_count:>10.4f}")

    def export_summary(self, output_path):
        analysis = self.analyze()

        with open(output_path, "w", encoding="utf8") as f:
            json.dump(
                {
                    "file": str(self.filepath),
                    "summary": analysis.get("summary", {}),
                    "metrics": analysis.get("metrics", {}),
                },
                f,
                indent=2,
                default=str,
            )

        info(
            f"[Analyzer] Summary exported to {output_path}",
            category="benchmark",
        )
    def _print_official_metrics(self):
        """Print official session/turn metrics from the adapter's 'metrics' field."""
        records = self.records()
        if not records:
            return

        has_metrics = any("metrics" in rec for rec in records)
        if not has_metrics:
            return

        print("\n[OFFICIAL SESSION/TURN METRICS]")

        k_values = (1, 3, 5, 10, 30, 50)
        session_recall_any = {k: 0 for k in k_values}
        session_recall_all = {k: 0 for k in k_values}
        session_ndcg_any = {k: 0.0 for k in k_values}
        turn_recall_any = {k: 0 for k in k_values}
        turn_recall_all = {k: 0 for k in k_values}
        turn_ndcg_any = {k: 0.0 for k in k_values}
        evaluable_count = 0

        for rec in records:
            if rec.get("abstention", False):
                continue
            metrics = rec.get("metrics", {})
            if not metrics:
                continue

            evaluable_count += 1

            sess = metrics.get("session", {})
            turn = metrics.get("turn", {})

            for k in k_values:
                # Handle both int and string keys
                session_recall_any[k] += 1 if (
                    sess.get("recall_any", {}).get(k, False) or 
                    sess.get("recall_any", {}).get(str(k), False)
                ) else 0
                session_recall_all[k] += 1 if (
                    sess.get("recall_all", {}).get(k, False) or 
                    sess.get("recall_all", {}).get(str(k), False)
                ) else 0
                session_ndcg_any[k] += (
                    sess.get("ndcg_any", {}).get(k, 0.0) or 
                    sess.get("ndcg_any", {}).get(str(k), 0.0) or 
                    0.0
                )
                turn_recall_any[k] += 1 if (
                    turn.get("recall_any", {}).get(k, False) or 
                    turn.get("recall_any", {}).get(str(k), False)
                ) else 0
                turn_recall_all[k] += 1 if (
                    turn.get("recall_all", {}).get(k, False) or 
                    turn.get("recall_all", {}).get(str(k), False)
                ) else 0
                turn_ndcg_any[k] += (
                    turn.get("ndcg_any", {}).get(k, 0.0) or 
                    turn.get("ndcg_any", {}).get(str(k), 0.0) or 
                    0.0
                )

        if evaluable_count == 0:
            print("  No evaluable records found.")
            return

        print(f"  Evaluable questions: {evaluable_count}")

        print("\n  SESSION-LEVEL:")
        for k in k_values:
            print(
                f"    K={k:>2}: recall_any={session_recall_any[k]/evaluable_count*100:>5.1f}%  "
                f"recall_all={session_recall_all[k]/evaluable_count*100:>5.1f}%  "
                f"ndcg_any={session_ndcg_any[k]/evaluable_count:.4f}"
            )

        print("\n  TURN-LEVEL:")
        for k in k_values:
            print(
                f"    K={k:>2}: recall_any={turn_recall_any[k]/evaluable_count*100:>5.1f}%  "
                f"recall_all={turn_recall_all[k]/evaluable_count*100:>5.1f}%  "
                f"ndcg_any={turn_ndcg_any[k]/evaluable_count:.4f}"
            )
    @staticmethod
    def compare_ablations(filepaths: list):
        """Compare multiple ablation runs (dense, bm25, raw, fusion, full)."""
        print("\n" + "=" * 70)
        print("[ABLATION COMPARISON]")
        print("=" * 70)
        print(f"{'Mode':<20} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'NDCG@10':>10}")
        print("-" * 70)

        results = {}
        for fp in filepaths:
            analyzer = BenchmarkAnalyzer(fp)
            if not analyzer.load():
                continue
            records = analyzer.records()
            
            k_values = (1, 3, 5, 10)
            recall_any = {k: 0 for k in k_values}
            ndcg_any = {k: 0.0 for k in k_values}
            evaluable = 0
            
            for rec in records:
                if rec.get("abstention", False):
                    continue
                metrics = rec.get("metrics", {})
                if not metrics:
                    continue
                evaluable += 1
                sess = metrics.get("session", {})
                recall_dict = sess.get("recall_any", {})
                ndcg_dict = sess.get("ndcg_any", {})
                for k in k_values:
                    # Try int key, then string key
                    recall_any[k] += 1 if (
                        recall_dict.get(k, False) or recall_dict.get(str(k), False)
                    ) else 0
                    ndcg_any[k] += ndcg_dict.get(k, 0.0) or ndcg_dict.get(str(k), 0.0)

            if evaluable:
                mode = Path(fp).stem
                results[mode] = {
                    "r1": recall_any[1]/evaluable*100,
                    "r3": recall_any[3]/evaluable*100,
                    "r5": recall_any[5]/evaluable*100,
                    "r10": recall_any[10]/evaluable*100,
                    "ndcg10": ndcg_any[10]/evaluable,
                }

        print()
        for mode, vals in results.items():
            print(
                f"{mode:<20} {vals['r1']:>7.1f}% {vals['r3']:>7.1f}% "
                f"{vals['r5']:>7.1f}% {vals['r10']:>7.1f}% {vals['ndcg10']:>9.4f}"
            )

def analyze_all(results_dir=DEFAULT_RESULTS_DIR):
    dir_path = Path(results_dir)

    if not dir_path.exists():
        error(f"[Analyzer] Directory not found: {dir_path}", category="benchmark")
        return {}

    results = {}

    for filepath in sorted(dir_path.glob("*.json")):
        info(
            f"[Analyzer] Analyzing: {filepath.name}",
            category="benchmark",
        )

        analyzer = BenchmarkAnalyzer(str(filepath))

        if analyzer.load():
            results[filepath.name] = analyzer.analyze().get("summary", {})

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python benchmark_analyzer.py file.json [--questions questions.json] [--explain N] [--export summary.json]")
        print("  python benchmark_analyzer.py --all [--export summary.json]")
        print("  python benchmark_analyzer.py --compare file1.json file2.json ...")
        sys.exit(1)

    if sys.argv[1] == "--all":
        results = analyze_all()
        print(json.dumps(results, indent=2))
        sys.exit(0)

    # NEW: Compare mode
    if sys.argv[1] == "--compare":
        files = sys.argv[2:]
        if not files:
            print("Error: --compare requires at least one file")
            sys.exit(1)
        BenchmarkAnalyzer.compare_ablations(files)
        sys.exit(0)

    # Parse arguments manually for single file analysis
    filepath = sys.argv[1]
    questions_path = None
    explain_limit = None
    export_path = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--questions" and i+1 < len(sys.argv):
            questions_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "--explain" and i+1 < len(sys.argv):
            explain_limit = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--export" and i+1 < len(sys.argv):
            export_path = sys.argv[i+1]
            i += 2
        else:
            i += 1

    analyzer = BenchmarkAnalyzer(filepath, questions_path)

    if not analyzer.load():
        sys.exit(1)

    analyzer.analyze()

    if explain_limit is not None:
        analyzer.explain_failures(explain_limit)

    if export_path:
        analyzer.export_summary(export_path)
