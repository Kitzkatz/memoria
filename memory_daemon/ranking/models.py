from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any

from memory.models import MemoryRecord


class CandidateRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Immutable Memory
    memory: MemoryRecord

    # Retrieval Signals
    distance: float
    embedding: Optional[List[float]] = None
    graph_hit: bool = False

    # Ranking Signals
    # Raw Ranking Signals
    semantic_score: float = 0.0
    importance_score: float = 0.0
    recency_score: float = 0.0
    token_score: float = 0.0
    graph_distance_score: float = 0.0
    tfidf_score: float = 0.0
    feedback_score: float = 0.0
    

    # Pipeline Scores
    base_score: float = 0.0
    normalized_score: float = 0.0
    attribute_score: float = 0.0
    context_score: float = 0.0
    graph_score: float = 0.0
    diversity_score: float = 0.0
    mmr_score: float = 0.0
    final_score: float = 0.0

    # BM25 score (added)
    bm25_score: float = 0.0

    # Diagnostics
    from pydantic import Field

    diagnostics: Dict[str, Any] = Field(default_factory=dict)
