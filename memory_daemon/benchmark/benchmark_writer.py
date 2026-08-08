"""
benchmark_writer.py
Stores complete benchmark flight logs.
One record == one benchmark question.
Nothing is analyzed here.
This file ONLY records data.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from core.logger import debug, info, error


class BenchmarkWriter:
    def __init__(self, output_dir: str = "benchmark_output/results", include_embeddings: bool = False):
        """
        Initialize benchmark writer.

        Args:
            output_dir: Directory to write results
            include_embeddings: If True, include embeddings in candidates (makes files large)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.started = datetime.now().isoformat()
        self.include_embeddings = include_embeddings

        info(f"[BenchmarkWriter] Writing to {self.output_dir}", category="benchmark")
        if not include_embeddings:
            debug("[BenchmarkWriter] Embeddings excluded from output (file size optimization)", category="benchmark")

    # --------------------------------------------------
    # RECORD ONE QUESTION
    # --------------------------------------------------

    def record(
        self,
        *,
        query: str,
        expected: str,
        expected_rank: Optional[int],
        retrieved: bool,
        candidates: list,
        runtime_ms: float,
        diagnostics: Optional[Dict[str, Any]] = None
    ):
        """
        Record a single benchmark question result.

        Args:
            query: The query string
            expected: Expected answer text
            expected_rank: Rank of expected result (1-indexed) or None
            retrieved: Whether expected was retrieved
            candidates: List of candidate results
            runtime_ms: Query runtime in milliseconds
            diagnostics: Additional diagnostic data
        """
        # Truncate candidates to reduce file size (keep top 20)
        truncated_candidates = candidates[:20]

        # Optionally remove embeddings to reduce file size
        if not self.include_embeddings:
            for c in truncated_candidates:
                if isinstance(c, dict):
                    c.pop("embedding", None)
                    if "diagnostics" in c and isinstance(c["diagnostics"], dict):
                        c["diagnostics"].pop("embedding", None)

        self.records.append({
            "query": query,
            "expected": expected,
            "expected_rank": expected_rank,
            "retrieved": retrieved,
            "candidate_count": len(candidates),
            "runtime_ms": round(runtime_ms, 3),
            "diagnostics": dict(diagnostics) if diagnostics else {},
            "candidates": truncated_candidates
        })

        debug(f"[BenchmarkWriter] Recorded: {query[:50]}...", category="benchmark")

    # --------------------------------------------------
    # WRITE TO DISK
    # --------------------------------------------------

    def write(self, filename: Optional[str] = None) -> Path:
        """
        Write benchmark results to disk.

        Args:
            filename: Optional custom filename (without path)

        Returns:
            Path: Path to output file
        """
        if filename is None:
            filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        outfile = self.output_dir / filename

        payload = {
            "started": self.started,
            "finished": datetime.now().isoformat(),
            "question_count": len(self.records),
            "include_embeddings": self.include_embeddings,
            "records": self.records
        }

        try:
            with open(outfile, "w", encoding="utf8") as f:
                json.dump(
                    payload,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=str
                )

            info(f"[BenchmarkWriter] Saved {len(self.records)} questions to {outfile}", category="benchmark")
            return outfile

        except Exception as e:
            error(f"[BenchmarkWriter] Failed to write: {e}", category="benchmark")
            raise

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    def summary(self) -> dict:
        """Return summary statistics of recorded results."""
        if not self.records:
            return {"total": 0, "retrieved": 0, "accuracy": 0.0}

        total = len(self.records)
        retrieved = sum(1 for r in self.records if r.get("retrieved", False))
        retrieved_ranks = [r.get("expected_rank") for r in self.records if r.get("expected_rank") is not None]

        return {
            "total": total,
            "retrieved": retrieved,
            "accuracy": round(retrieved / total * 100, 2) if total > 0 else 0.0,
            "rank_1": sum(1 for r in retrieved_ranks if r == 1),
            "rank_3": sum(1 for r in retrieved_ranks if r <= 3),
            "rank_5": sum(1 for r in retrieved_ranks if r <= 5),
            "avg_runtime_ms": round(sum(r.get("runtime_ms", 0) for r in self.records) / total, 2),
        }

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    def reset(self):
        """Clear all records (start fresh)."""
        self.records = []
        self.started = datetime.now().isoformat()
        debug("[BenchmarkWriter] Reset", category="benchmark")
