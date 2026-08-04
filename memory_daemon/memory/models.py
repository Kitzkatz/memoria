
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class MemoryRecord(BaseModel):

    # -------------------------
    # Identity
    # -------------------------

    id: Optional[int] = None

    # -------------------------
    # Raw Memory
    # -------------------------

    text: str

    memory_type: str = "semantic"

    importance: float = 0.5

    metadata: Dict[str, Any] = Field(default_factory=dict)

    # -------------------------
    # Knowledge Features
    # -------------------------

    entities: List[str] = Field(
        default_factory=list
    )

    relationships: List[Dict[str, Any]] = Field(
        default_factory=list
    )
    # -------------------------
    # Preprocessed Features
    # -------------------------

    normalized_text: str = ""

    tokens: List[str] = Field(default_factory=list)

    token_count: int = 0

    # -------------------------
    # Timestamps
    # -------------------------

    created_at: datetime = Field(default_factory=datetime.utcnow)

    last_accessed: datetime = Field(default_factory=datetime.utcnow)





class GoalRecord(BaseModel):
    id: Optional[int] = None
    goal: str
    progress: str = "started"
    status: str = "active"
