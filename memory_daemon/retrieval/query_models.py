from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class QueryRecord:

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
    # Future NLP
    # -------------------------

    entities: List[str] = field(default_factory=list)

    keywords: List[str] = field(default_factory=list)

    # -------------------------
    # Router / Filters
    # -------------------------

    metadata: Dict[str, Any] = field(default_factory=dict)
