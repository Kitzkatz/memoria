from core.logger import debug
import re
from typing import List, Optional, Set

from retrieval.case_folding import fold_case
from ranking.attribute_map import ATTRIBUTE_MAP
from retrieval.query_models import QueryRecord
from memory.type_router import TypeRouter   # <-- new import


class QueryProcessor:
    """
    Query preprocessing pipeline with optional LLM enrichment.

    Handles:
    - Text normalization
    - Tokenization
    - Keyword extraction (stopword-aware)
    - Entity extraction (single + multi-word)
    - Optional LLM-based entity extraction
    """

    def __init__(self, llm=None):
        self.llm = llm

        # Stopwords to filter from keywords and entities
        self.stopwords = {
            "the", "a", "an", "of", "for", "on", "at", "to", "in", "with",
            "without", "by", "from", "up", "down", "off", "over", "under",
            "and", "or", "but", "so", "for", "nor", "yet", "as", "than",
            "that", "this", "these", "those", "then", "now", "than", "when",
            "where", "which", "who", "whom", "whose", "will", "would", "could",
            "should", "may", "might", "must", "shall"
        }

        # Type router instance
        self.type_router = TypeRouter()   # <-- new

    # ... (normalize, tokenize, etc unchanged)

    # -------------------------
    # Process Query
    # -------------------------

    def process(self, text: str, use_llm: bool = False) -> QueryRecord:
        """
        Process query text into QueryRecord.

        Args:
            text: Raw query string
            use_llm: If True and LLM available, use LLM for entity extraction

        Returns:
            QueryRecord with normalized text, tokens, keywords, entities, and phrases
        """
        normalized = self.normalize(text)
        tokens = self.tokenize(normalized)
        keywords = self.extract_keywords(tokens)

        # Entity extraction (rule-based or LLM)
        if use_llm and self.llm:
            entities = self.extract_entities_llm(text)
        else:
            entities = self.extract_entities(text)

        attribute = None
        for canonical, cfg in ATTRIBUTE_MAP.items():
            for alias in cfg["aliases"]:
                if alias in normalized:
                    attribute = canonical
                    break
            if attribute:
                break

        subject = entities[0] if entities else None

        # Extract quoted phrases
        phrases = self.extract_phrases(text)

        # Determine memory type hint using the router
        memory_type_hint = self.type_router.route(text, tokens)   # <-- new

        metadata = {
            "use_llm": use_llm,
            "entity_count": len(entities),
            "subject": subject,
            "attribute": attribute,
            "phrases": phrases,
            "memory_type_hint": memory_type_hint,   # <-- added
        }

        debug(f"\n[QUERY PROCESSOR DEBUG]")
        debug(f"  Raw text: {text}")
        debug(f"  use_llm: {use_llm}")
        debug(f"  Entities extracted: {entities}")
        debug(f"  Phrases extracted: {phrases}")
        debug(f"  Memory type hint: {memory_type_hint}")   # <-- new debug
        debug(f"  Normalized: {normalized}")
        debug(f"  Tokens: {tokens[:10]}...")
        debug(f"  Keywords: {keywords[:10]}...")
        debug(f"  Subject: {subject}")
        debug(f"  Attribute: {attribute}")
        debug("-" * 40)

        return QueryRecord(
            text=text,
            normalized_text=normalized,
            tokens=tokens,
            token_count=len(tokens),
            keywords=keywords,
            entities=entities,
            metadata=metadata
        )

    # ... (rest unchanged)
