from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class GraphRecord(BaseModel):
    id: Optional[int] = None
    memory_id: Optional[int] = None
    source: int
    relation: str
    target: int
    weight: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EntityRecord(BaseModel):
    id: Optional[int] = None
    name: str
    entity_type: str = "unknown"
    aliases: List[str] = Field(default_factory=list)
