"""
Result formatter for benchmark adapters.
Produces output compatible with the normal benchmark runner/writer format.
"""

import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional


def format_candidate(candidate: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """
    Convert a raw candidate dict (from controller.recall) to the standard format.
    """
    return {
        "rank": rank,
        "id": candidate.get("id"),
        "text": candidate.get("text", ""),
        "normalized_text": candidate.get("normalized_text", ""),
        "metadata": candidate.get("metadata", {}),
        "score": candidate.get("score"),
        "final_score": candidate.get("final_score"),
        "diagnostics": candidate.get("diagnostics", {})
    }


def build_record(
    query: str,
    expected: str = "",
    expected_ids: Optional[List[str]] = None,
    expected_rank: Optional[int] = None,
    retrieved: bool = False,
    candidates: Optional[List[Dict[str, Any]]] = None,
    runtime_ms: float = 0.0,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a single record in the standard format.
    """
    if candidates is None:
        candidates = []
    if diagnostics is None:
        diagnostics = {}

    formatted_candidates = [
        format_candidate(c, idx + 1) for idx, c in enumerate(candidates)
    ]

    return {
        "query": query,
        "expected": expected,
        "expected_ids": expected_ids or [],
        "expected_rank": expected_rank,
        "retrieved": retrieved,
        "candidates": formatted_candidates,
        "runtime_ms": runtime_ms,
        "diagnostics": diagnostics
    }


def build_output(
    records: List[Dict[str, Any]],
    question_count: int = None,
    started: Optional[str] = None,
    finished: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assemble the final output dict, compatible with benchmark_writer.
    """
    if started is None:
        started = datetime.now(timezone.utc).isoformat()
    if finished is None:
        finished = datetime.now(timezone.utc).isoformat()
    if question_count is None:
        question_count = len(records)

    return {
        "started": started,
        "finished": finished,
        "question_count": question_count,
        "records": records
    }


def write_output(output_dict: Dict[str, Any], filepath: str):
    """Write the output dict to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(output_dict, f, indent=2)
