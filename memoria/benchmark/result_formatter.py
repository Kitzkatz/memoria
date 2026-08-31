
"""
Result formatter for benchmark adapters.

Produces output compatible with the normal benchmark runner/writer format
while preserving candidate-level retrieval evidence, including temporal
post-processing fields.

The formatter is intentionally evaluation-neutral:
    - it does not rank candidates
    - it does not calculate retrieval scores
    - it does not modify candidate evidence
    - it serializes evidence already attached by the retrieval pipeline
"""

import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional


def format_candidate(
    candidate: Dict[str, Any],
    rank: int,
) -> Dict[str, Any]:
    """
    Convert a raw candidate dict from controller.recall() into the
    standard benchmark candidate format.

    Candidate-level evidence is preserved, including temporal fields
    produced by TemporalWorker.score_candidates().
    """

    return {
        # ------------------------------------------------------------
        # Identity / content
        # ------------------------------------------------------------
        "rank": rank,
        "id": candidate.get("id"),
        "text": candidate.get("text", ""),
        "normalized_text": candidate.get("normalized_text", ""),
        "metadata": candidate.get("metadata", {}),

        # ------------------------------------------------------------
        # Retrieval / ranking scores
        # ------------------------------------------------------------
        "score": candidate.get("score"),
        "final_score": candidate.get("final_score"),

        # ------------------------------------------------------------
        # Temporal evidence
        #
        # These are attached by TemporalWorker and must survive into
        # benchmark output so temporal retrieval can be analyzed.
        # ------------------------------------------------------------
        "temporal_score": candidate.get("temporal_score", 0.0),
        "temporal_matches": candidate.get("temporal_matches", []),

        # ------------------------------------------------------------
        # Candidate diagnostics
        # ------------------------------------------------------------
        "diagnostics": candidate.get("diagnostics", {}),
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
    # ------------------------------------------------------------
    # Official evaluation fields
    # ------------------------------------------------------------
    gold_session_ids: Optional[List[str]] = None,
    gold_turn_ids: Optional[List[str]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    abstention: bool = False,
    question_id: Optional[str] = None,
    # ------------------------------------------------------------
    # Temporal evaluation metadata
    # ------------------------------------------------------------
    temporal_query: bool = False,
    temporal_constraints: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Build a single benchmark record.

    Temporal information is stored both:
        1. at record level, describing the query
        2. at candidate level, describing temporal evidence attached
           by TemporalWorker

    The formatter does not calculate temporal metrics.
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

    if temporal_constraints is None:
        temporal_constraints = []

    formatted_candidates = [
        format_candidate(candidate, idx + 1)
        for idx, candidate in enumerate(candidates)
    ]

    record = {
        # ------------------------------------------------------------
        # Query / ground truth
        # ------------------------------------------------------------
        "query": query,
        "expected": expected,
        "expected_ids": expected_ids,
        "expected_rank": expected_rank,
        "retrieved": retrieved,

        # ------------------------------------------------------------
        # Candidates
        # ------------------------------------------------------------
        "candidates": formatted_candidates,

        # ------------------------------------------------------------
        # Runtime
        # ------------------------------------------------------------
        "runtime_ms": runtime_ms,

        # ------------------------------------------------------------
        # Query diagnostics
        # ------------------------------------------------------------
        "diagnostics": diagnostics,

        # ------------------------------------------------------------
        # Official evaluation fields
        # ------------------------------------------------------------
        "gold_session_ids": gold_session_ids,
        "gold_turn_ids": gold_turn_ids,
        "metrics": metrics,
        "abstention": abstention,

        # ------------------------------------------------------------
        # Temporal query metadata
        # ------------------------------------------------------------
        "temporal_query": temporal_query,
        "temporal_constraints": temporal_constraints,
    }

    if question_id is not None:
        record["question_id"] = question_id

    return record


def build_output(
    records: List[Dict[str, Any]],
    question_count: int = None,
    started: Optional[str] = None,
    finished: Optional[str] = None,
    # ------------------------------------------------------------
    # Reproducibility metadata
    # ------------------------------------------------------------
    dataset_hash: Optional[str] = None,
    embedder: Optional[str] = None,
    retrieval_mode: Optional[str] = None,
    commit: Optional[str] = None,
    settings_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble the final benchmark output.

    Optional metadata is included only when explicitly supplied.
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
        "records": records,
    }

    # ------------------------------------------------------------
    # Optional reproducibility metadata
    # ------------------------------------------------------------

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


def write_output(
    output_dict: Dict[str, Any],
    filepath: str,
):
    """
    Write benchmark output as formatted JSON.
    """

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            output_dict,
            f,
            indent=2,
            ensure_ascii=False,
        )

