from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class QueryRecord:
    """
    Represents a processed user query with all extracted features.

    Fields:
        text: Raw query string
        normalized_text: Case-folded, normalized version
        tokens: Tokenized words from normalized_text
        token_count: Number of tokens (convenience)
        entities: Named entities extracted from query
        keywords: Important keywords (stopwords filtered)
        metadata: Additional context (type hint, attributes, etc.)
    """

    # -------------------------
    # Raw Query
    # -------------------------

    text: str

    # -------------------------
    # Preprocessed
    # -------------------------

    normalized_text: str = ""
    tokens: List[str] = field(default_factory=list)
    token_count: int = 0

    # -------------------------
    # NLP Features
    # -------------------------

    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # -------------------------
    # Routing / Filters
    # -------------------------

    metadata: Dict[str, Any] = field(default_factory=dict)

    # -------------------------
    # Convenience Properties
    # -------------------------

    @property
    def memory_type_hint(self) -> Optional[str]:
        """Get the memory type hint from metadata."""
        return self.metadata.get("memory_type_hint")

    @property
    def subject(self) -> Optional[str]:
        """Get the subject from metadata."""
        return self.metadata.get("subject")

    @property
    def attribute(self) -> Optional[str]:
        """Get the attribute from metadata."""
        return self.metadata.get("attribute")

    @property
    def phrases(self) -> List[List[str]]:
        """Get extracted phrases from metadata."""
        return self.metadata.get("phrases", [])

    @property
    def has_entities(self) -> bool:
        """Return True if entities are present."""
        return bool(self.entities)

    @property
    def is_empty(self) -> bool:
        """Return True if the query is empty."""
        return not self.text or self.token_count == 0

    @property
    def routing_signals(self) -> Dict[str, float]:
        """Get routing signals from metadata (if present)."""
        return self.metadata.get("routing_signals", {})

    # -------------------------
    # Utility Methods
    # -------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "tokens": self.tokens,
            "token_count": self.token_count,
            "entities": self.entities,
            "keywords": self.keywords,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """Short representation for debugging."""
        return f"QueryRecord(text='{self.text[:50]}...', tokens={self.token_count}, entities={len(self.entities)})"
