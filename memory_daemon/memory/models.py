"""
Memory models for the Memory Daemon system.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone


class MemoryRecord(BaseModel):
    """
    Represents a single memory in the system.

    Fields:
        id: Unique identifier (auto-generated)
        text: Raw text of the memory
        memory_type: Type classification (semantic, episodic, procedural, code, science)
        importance: Float score from 0.0 to 1.0
        metadata: Additional context and attributes
        entities: Named entities extracted from text
        relationships: Relations between entities
        normalized_text: Case-folded, normalized version
        tokens: Tokenized words
        token_count: Number of tokens (convenience)
        created_at: When the memory was created
        last_accessed: When the memory was last retrieved
    """

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

    entities: List[str] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)

    # -------------------------
    # Preprocessed Features
    # -------------------------

    normalized_text: str = ""
    tokens: List[str] = Field(default_factory=list)
    token_count: int = 0

    # -------------------------
    # Timestamps
    # -------------------------

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # -------------------------
    # Methods
    # -------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = self.model_dump()
        data["created_at"] = data["created_at"].isoformat()
        data["last_accessed"] = data["last_accessed"].isoformat()
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)

    def __repr__(self) -> str:
        return f"MemoryRecord(id={self.id}, type='{self.memory_type}', importance={self.importance:.2f}, tokens={self.token_count})"

    def __str__(self) -> str:
        return self.text[:200] + ("..." if len(self.text) > 200 else "")


class GoalRecord(BaseModel):
    """
    Represents a goal tracked by the system.

    Fields:
        id: Unique identifier (auto-generated)
        goal: The goal description
        progress: Current progress status (e.g., "started", "in_progress", "complete")
        status: Lifecycle status (e.g., "active", "archived")
    """

    id: Optional[int] = None
    goal: str
    progress: str = "started"
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    def __repr__(self) -> str:
        return f"GoalRecord(id={self.id}, goal='{self.goal[:50]}...', status='{self.status}')"
