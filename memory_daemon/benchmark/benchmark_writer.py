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


class BenchmarkWriter:
    def __init__(self, output_dir="benchmark_output/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.started = datetime.now().isoformat()

    # --------------------------------------------------
    # RECORD ONE QUESTION
    # --------------------------------------------------
    def record(
        self,
        *,
        query,
        expected,
        expected_rank,
        retrieved,
        candidates,
        runtime_ms,
        diagnostics=None
    ):
        self.records.append({
            "query": query,
            "expected": expected,
            "expected_rank": expected_rank,
            "retrieved": retrieved,
            "candidate_count": len(candidates),
            "runtime_ms": runtime_ms,
            "diagnostics": dict(diagnostics) if diagnostics else {},
            "candidates": candidates
        })

    # --------------------------------------------------
    # WRITE TO DISK
    # --------------------------------------------------
    def write(self):
        filename = (
            "benchmark_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".json"
        )
        outfile = self.output_dir / filename

        payload = {
            "started": self.started,
            "finished": datetime.now().isoformat(),
            "question_count": len(self.records),
            "records": self.records
        }

        with open(outfile, "w", encoding="utf8") as f:
            json.dump(
                payload,
                f,
                indent=4,
                ensure_ascii=False,
                default=str
            )

        print()
        print("=" * 60)
        print("[BENCHMARK WRITER]")
        print()
        print("Saved")
        print(outfile)
        print()
        print("Questions")
        print(len(self.records))
        print("=" * 60)

        return outfile

