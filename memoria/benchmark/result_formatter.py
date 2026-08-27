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
    # NEW: official evaluation fields
    gold_session_ids: Optional[List[str]] = None,
    gold_turn_ids: Optional[List[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    abstention: bool = False,
    question_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a single record in the standard format.

    Args:
        query: The query text.
        expected: The expected answer text.
        expected_ids: List of expected session/turn IDs.
        expected_rank: The rank of the first expected candidate, or None.
        retrieved: Whether any expected candidate was retrieved.
        candidates: List of raw candidates from the system.
        runtime_ms: Query runtime in milliseconds.
        diagnostics: Additional diagnostics from the system.
        gold_session_ids: Official gold session IDs for evaluation.
        gold_turn_ids: Official gold turn IDs for evaluation.
        metrics: Precomputed per‑question metrics (session/turn recall/NDCG).
        abstention: Whether this is an abstention question.
        question_id: Optional question ID for traceability.
    """
    if candidates is None:
        candidates = []
    if diagnostics is None:
        diagnostics = {}
    if expected_ids is None:
        expected_ids = []
    if gold_session_ids is None:
        gold_session_ids = []
    if gold_turn_ids is None:
        gold_turn_ids = []

    formatted_candidates = [
        format_candidate(c, idx + 1) for idx, c in enumerate(candidates)
    ]

    record = {
        "query": query,
        "expected": expected,
        "expected_ids": expected_ids,
        "expected_rank": expected_rank,
        "retrieved": retrieved,
        "candidates": formatted_candidates,
        "runtime_ms": runtime_ms,
        "diagnostics": diagnostics,
        # NEW fields
        "gold_session_ids": gold_session_ids,
        "gold_turn_ids": gold_turn_ids,
        "metrics": metrics,
        "abstention": abstention,
    }

    if question_id is not None:
        record["question_id"] = question_id

    return record


def build_output(
    records: List[Dict[str, Any]],
    question_count: int = None,
    started: Optional[str] = None,
    finished: Optional[str] = None,
    # NEW: optional metadata fields
    dataset_hash: Optional[str] = None,
    embedder: Optional[str] = None,
    retrieval_mode: Optional[str] = None,
    commit: Optional[str] = None,
    settings_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble the final output dict, compatible with benchmark_writer.

    Now includes optional metadata for reproducibility.
    """
    if started is None:
        started = datetime.now(timezone.utc).isoformat()
    if finished is None:
        finished = datetime.now(timezone.utc).isoformat()
    if question_count is None:
        question_count = len(records)

    output = {
        "started": started,
        "finished": finished,
        "question_count": question_count,
        "records": records
    }

    # Add optional metadata if provided
    if dataset_hash is not None:
        output["dataset_hash"] = dataset_hash
    if embedder is not None:
        output["embedder"] = embedder
    if retrieval_mode is not None:
        output["retrieval_mode"] = retrieval_mode
    if commit is not None:
        output["commit"] = commit
    if settings_snapshot is not None:
        output["settings"] = settings_snapshot

    return output


def write_output(output_dict: Dict[str, Any], filepath: str):
    """Write the output dict to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(output_dict, f, indent=2)
