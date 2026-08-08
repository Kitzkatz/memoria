"""
Response builder for memory system queries.
Shared between V4 blackboard path and V3 fallback path.

OPTIMIZED: Only builds full response for top N candidates.
"""

from core.logger import debug

# How many results to return to the user
DEFAULT_RESULTS_LIMIT = 10


def _clean_value(value):
    """
    Convert numpy types to Python types for JSON serialization.
    Handles nested dicts and lists recursively.
    """
    if value is None:
        return None
    # Import numpy here to avoid dependency if not used
    try:
        import numpy as np
        if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(value)
        if isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.bool_):
            return bool(value)
    except ImportError:
        pass
    
    if isinstance(value, dict):
        return {_clean_value(k): _clean_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_value(v) for v in value]
    return value


def build_response(results, limit: int = DEFAULT_RESULTS_LIMIT):
    """
    Build response list from ranked candidates.
    Only builds full dicts for top N results.

    Args:
        results: List of ranked CandidateRecord objects
        limit: Maximum number of results to return

    Returns:
        List of dicts with memory fields (top N only)
    """
    response = []

    # Only iterate over what we need
    for rank, candidate in enumerate(results[:limit], start=1):
        response.append({
            "rank": rank,
            "id": _clean_value(candidate.memory.id),
            "text": candidate.memory.text,
            "normalized_text": candidate.memory.normalized_text,
            "memory_type": candidate.memory.memory_type,
            "importance": _clean_value(candidate.memory.importance),
            "distance": _clean_value(candidate.distance),
            "score": _clean_value(candidate.normalized_score),
            "final_score": _clean_value(candidate.final_score),
            "created_at": candidate.memory.created_at.isoformat(),
            "last_accessed": candidate.memory.last_accessed.isoformat(),
            "token_count": _clean_value(candidate.memory.token_count),
            "tokens": candidate.memory.tokens or [],
            "metadata": candidate.memory.metadata or {},
            "entities": candidate.memory.entities or [],
            "relationships": candidate.memory.relationships or [],
            "graph_hit": bool(candidate.graph_hit),
            "diagnostics": _clean_value(candidate.diagnostics),
            "mmr_score": _clean_value(candidate.mmr_score),
            "diversity": _clean_value(candidate.diversity_score),
        })

    return response


def build_diagnostics_v4(candidates, response, embedding_ms, faiss_ms, ranking_diag, total_ms):
    """Build diagnostics dict for V4 blackboard path."""
    return _clean_value({
        "candidate_count": len(candidates),
        "returned_count": len(response),
        "embedding_ms": round(embedding_ms, 3),
        "faiss_ms": round(faiss_ms, 3),
        "database_ms": 0.0,
        "ranking_ms": round(ranking_diag.get("ranking_ms", 0), 3),
        "before_mmr": ranking_diag.get("before_mmr"),
        "after_mmr": ranking_diag.get("after_mmr"),
        "mmr_changed": ranking_diag.get("mmr_changed", False),
        "mmr_moves": ranking_diag.get("mmr_moves", 0),
        "formatting_ms": 0.0,
        "total_query_ms": round(total_ms, 3),
    })


def build_diagnostics_v3(candidates, response, embedding_ms, faiss_ms, database_ms, ranking_ms, formatting_ms, total_ms):
    """Build diagnostics dict for V3 fallback path."""
    return _clean_value({
        "candidate_count": len(candidates),
        "returned_count": len(response),
        "embedding_ms": round(embedding_ms, 3),
        "faiss_ms": round(faiss_ms, 3),
        "database_ms": round(database_ms, 3),
        "ranking_ms": round(ranking_ms, 3),
        "formatting_ms": round(formatting_ms, 3),
        "total_query_ms": round(total_ms, 3),
    })
