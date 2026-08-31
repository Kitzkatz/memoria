from pydantic import BaseModel, ConfigDict, Field
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
    semantic_score: float = 0.0
    importance_score: float = 0.0
    recency_score: float = 0.0
    token_score: float = 0.0
    entity_score: float = 0.0           # ← ADD THIS
    graph_distance_score: float = 0.0
    tfidf_score: float = 0.0

    # Pipeline Scores
    base_score: float = 0.0
    normalized_score: float = 0.0
    attribute_score: float = 0.0
    context_score: float = 0.0
    graph_score: float = 0.0
    diversity_score: float = 0.0
    mmr_score: float = 0.0
    final_score: float = 0.0

    temporal_score: float = 0.0
    temporal_matches: List[str] = Field(default_factory=list)

    # BM25 score
    bm25_score: float = 0.0

    # Diagnostics
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
