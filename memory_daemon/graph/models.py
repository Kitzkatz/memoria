from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timezone


class GraphRecord(BaseModel):
    id: Optional[int] = None
    memory_id: Optional[int] = None
    source: int
    relation: str
    target: int
    weight: float = 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = self.model_dump()
        data["created_at"] = data["created_at"].isoformat()
        return data

    def __repr__(self) -> str:
        return f"GraphRecord({self.source} --{self.relation}--> {self.target}, memory={self.memory_id})"


class EntityRecord(BaseModel):
    id: Optional[int] = None
    name: str
    entity_type: str = "unknown"
    aliases: List[str] = Field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    def __repr__(self) -> str:
        return f"EntityRecord(id={self.id}, name='{self.name}', type='{self.entity_type}')"
