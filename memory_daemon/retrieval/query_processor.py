from core.logger import debug
import re
from typing import List, Optional, Set

from retrieval.case_folding import fold_case
from ranking.attribute_map import ATTRIBUTE_MAP
from retrieval.query_models import QueryRecord


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

    # -------------------------
    # Normalization
    # -------------------------

    def normalize(self, text: str) -> str:
        """Lowercase, collapse whitespace, strip."""
        text = fold_case(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # -------------------------
    # Tokenization
    # -------------------------

    def tokenize(self, text: str) -> List[str]:
        """Split into alphanumeric tokens."""
        return re.findall(r"\w+", text)

    # -------------------------
    # Keyword Extraction
    # -------------------------

    def extract_keywords(self, tokens: List[str]) -> List[str]:
        """
        Extract keywords by filtering stopwords and deduplicating.
        """
        keywords = [t for t in tokens if t not in self.stopwords]
        return list(dict.fromkeys(keywords))

    # -------------------------
    # Entity Extraction (Rule-Based)
    # -------------------------

    def extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities using rule-based heuristics.

        Handles:
        - Single capitalized words (Kevin, Seattle)
        - Multi-word capitalized phrases (New York, San Francisco)
        - Entities with internal punctuation (Dr. Smith, J.R.R. Tolkien)
        """
        entities: List[str] = []
        current: List[str] = []
        
        words = text.split()
        
        for word in words:
            stripped = word.strip(".,!?()[]{}\"'")
            
            # Check if it starts with uppercase
            if stripped and stripped[0].isupper():
                current.append(stripped)
            else:
                if current:
                    entity = " ".join(current)
                    if len(entity) > 1 and entity.lower() not in self.stopwords:
                        entities.append(entity)
                    current = []
        
        # Flush remaining entity
        if current:
            entity = " ".join(current)
            if len(entity) > 1 and entity.lower() not in self.stopwords:
                entities.append(entity)
        
        return sorted(set(entities))

    # -------------------------
    # Entity Extraction (LLM-Based)
    # -------------------------

    def extract_entities_llm(self, text: str) -> List[str]:
        """
        Extract named entities using LLM.

        Falls back to rule-based extraction if LLM is not available.
        """
        if not self.llm:
            return self.extract_entities(text)

        prompt = f"""
        Extract all named entities from the following text.

        Rules:
        - Return ONLY a JSON list of entity names
        - Include people, places, organizations, technologies, projects
        - Combine multi-word entities (e.g., "New York" not "New" and "York")
        - Do NOT include common words (the, a, an, etc.)
        - Return an empty list if no entities found

        Text:
        {text}

        JSON List:
        """

        try:
            response = self.llm.chat(prompt)
            # Clean response - extract JSON array
            import json
            entities = json.loads(response.strip())
            if isinstance(entities, list):
                return sorted(set(e for e in entities if isinstance(e, str)))
            else:
                return self.extract_entities(text)
        except Exception as e:
            debug(f"[QueryProcessor] LLM fallback: {e}")
            return self.extract_entities(text)

    # -------------------------
    # Phrase Extraction (NEW)
    # -------------------------

    def extract_phrases(self, text: str) -> List[List[str]]:
        """
        Extract quoted phrases from text, return tokenized lists.
        Example: 'query "Kevin Johnson" lives in "New York"' -> [["Kevin", "Johnson"], ["New", "York"]]
        """
        import re
        quoted = re.findall(r'"([^"]*)"', text)
        phrases = []
        for phrase in quoted:
            tokens = self.tokenize(phrase)
            if tokens:
                phrases.append(tokens)
        return phrases

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

        metadata = {
            "use_llm": use_llm,
            "entity_count": len(entities),
            "subject": subject,
            "attribute": attribute,
            "phrases": phrases,           # <-- added
        }

        debug(f"\n[QUERY PROCESSOR DEBUG]")
        debug(f"  Raw text: {text}")
        debug(f"  use_llm: {use_llm}")
        debug(f"  Entities extracted: {entities}")
        debug(f"  Phrases extracted: {phrases}")
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

    # -------------------------
    # Debug / Test
    # -------------------------

    def __repr__(self) -> str:
        return f"QueryProcessor(llm={'available' if self.llm else 'None'})"
