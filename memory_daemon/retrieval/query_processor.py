from core.logger import debug
import re
import json
from typing import List, Optional, Set, Dict, Any
from functools import lru_cache

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
    - Phrase extraction (quoted text)
    - Memory type routing
    """

    def __init__(self, llm=None):
        self.llm = llm
        self._attribute_index = self._build_attribute_index()

        # Stopwords to filter from keywords and entities
        self.stopwords = {
            "the", "a", "an", "of", "for", "on", "at", "to", "in", "with",
            "without", "by", "from", "up", "down", "off", "over", "under",
            "and", "or", "but", "so", "for", "nor", "yet", "as", "than",
            "that", "this", "these", "those", "then", "now", "than", "when",
            "where", "which", "who", "whom", "whose", "will", "would", "could",
            "should", "may", "might", "must", "shall"
        }

        # Built-in type routing rules
        self.type_rules = {
            "semantic": ["what is", "define", "means", "called", "fact", "property"],
            "episodic": ["when", "what happened", "event", "time", "yesterday", "last week", "date"],
            "procedural": ["how to", "steps", "guide", "procedure", "method", "instructions"],
            "code": ["def ", "function ", "class ", "import ", "return ", "snippet", "api"],
            "science": ["formula", "equation", "chemical", "physics", "temperature", "pressure", "mass"]
        }

        # Compile regex patterns for type routing (performance)
        self._type_patterns = {
            mem_type: [re.compile(rf'\b{kw}\b', re.IGNORECASE) for kw in keywords]
            for mem_type, keywords in self.type_rules.items()
        }

    # -------------------------
    # Attribute Index
    # -------------------------

    def _build_attribute_index(self) -> Dict[str, str]:
        """Build a fast lookup from alias -> canonical attribute."""
        index = {}
        for canonical, config in ATTRIBUTE_MAP.items():
            for alias in config.get("aliases", []):
                index[alias.lower()] = canonical
        return index

    # -------------------------
    # Normalization
    # -------------------------

    def normalize(self, text: str) -> str:
        """Lowercase, collapse whitespace, strip."""
        if not text:
            return ""
        text = fold_case(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # -------------------------
    # Tokenization
    # -------------------------

    def tokenize(self, text: str) -> List[str]:
        """Split into alphanumeric tokens."""
        if not text:
            return []
        return re.findall(r"\w+", text)

    # -------------------------
    # Keyword Extraction
    # -------------------------

    def extract_keywords(self, tokens: List[str]) -> List[str]:
        """Extract keywords by filtering stopwords and deduplicating."""
        if not tokens:
            return []
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
        - Acronyms (NASA, AI)
        """
        if not text:
            return []

        entities: List[str] = []
        current: List[str] = []

        # Split on whitespace, but preserve punctuation
        words = text.split()

        for word in words:
            # Strip surrounding punctuation but keep internal
            stripped = word.strip(".,!?()[]{}\"'")
            if not stripped:
                continue

            # Check if it starts with uppercase OR is all uppercase (acronym)
            is_acronym = stripped.isupper() and len(stripped) >= 2
            is_capitalized = stripped[0].isupper() and not stripped.isupper()

            if is_capitalized or is_acronym:
                current.append(stripped)
            else:
                if current:
                    entity = " ".join(current)
                    # Filter out stopwords and single-letter entities
                    if len(entity) > 1 and entity.lower() not in self.stopwords:
                        entities.append(entity)
                    current = []

        # Flush remaining entity
        if current:
            entity = " ".join(current)
            if len(entity) > 1 and entity.lower() not in self.stopwords:
                entities.append(entity)

        # Deduplicate while preserving order
        seen = set()
        result = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                result.append(e)
        return result

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
            # Clean response — extract JSON array
            cleaned = response.strip()
            # Try to find JSON array in the response
            match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if match:
                entities = json.loads(match.group())
                if isinstance(entities, list):
                    return sorted(set(e for e in entities if isinstance(e, str) and e))
            return self.extract_entities(text)
        except Exception as e:
            debug(f"[QueryProcessor] LLM fallback: {e}")
            return self.extract_entities(text)

    # -------------------------
    # Phrase Extraction
    # -------------------------

    def extract_phrases(self, text: str) -> List[List[str]]:
        """
        Extract quoted phrases from text, return tokenized lists.
        Example: 'query "Kevin Johnson" lives in "New York"' -> [["Kevin", "Johnson"], ["New", "York"]]
        """
        if not text:
            return []

        quoted = re.findall(r'"([^"]*)"', text)
        phrases = []
        for phrase in quoted:
            tokens = self.tokenize(phrase)
            if tokens:
                phrases.append(tokens)
        return phrases

    # -------------------------
    # Determine memory type
    # -------------------------

    def _route_type(self, text: str, tokens: List[str]) -> str:
        """
        Determine memory type based on query text and tokens.
        Uses pre-compiled regex patterns for performance.
        """
        if not text:
            return "general"

        text_lower = text.lower()
        scores = {t: 0 for t in self.type_rules}

        # Check patterns
        for mem_type, patterns in self._type_patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    scores[mem_type] += 1

        # Boost for code patterns in tokens
        code_keywords = {"def", "class", "import", "return", "function", "lambda", "async", "await"}
        if any(t in code_keywords for t in tokens):
            scores["code"] += 2

        # Boost for science patterns
        science_keywords = {"formula", "equation", "chemical", "physics", "temperature", "pressure", "mass"}
        if any(t in science_keywords for t in tokens):
            scores["science"] += 1

        best_type = max(scores, key=scores.get)
        if scores[best_type] == 0:
            return "general"
        return best_type

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
        if not text:
            return QueryRecord(
                text="",
                normalized_text="",
                tokens=[],
                token_count=0,
                keywords=[],
                entities=[],
                metadata={"error": "empty_query"}
            )

        normalized = self.normalize(text)
        tokens = self.tokenize(normalized)
        keywords = self.extract_keywords(tokens)

        # Entity extraction (rule-based or LLM)
        if use_llm and self.llm:
            entities = self.extract_entities_llm(text)
        else:
            entities = self.extract_entities(text)

        # Attribute detection using fast index
        attribute = None
        text_lower = normalized.lower()
        for alias_lower, canonical in self._attribute_index.items():
            if alias_lower in text_lower:
                attribute = canonical
                break

        subject = entities[0] if entities else None

        # Extract quoted phrases
        phrases = self.extract_phrases(text)

        # Determine memory type hint
        memory_type_hint = self._route_type(text, tokens)

        metadata = {
            "use_llm": use_llm,
            "entity_count": len(entities),
            "subject": subject,
            "attribute": attribute,
            "phrases": phrases,
            "memory_type_hint": memory_type_hint,
        }

        debug(f"\n[QUERY PROCESSOR DEBUG]")
        debug(f"  Raw text: {text[:100]}{'...' if len(text) > 100 else ''}")
        debug(f"  use_llm: {use_llm}")
        debug(f"  Entities extracted: {entities}")
        debug(f"  Phrases extracted: {phrases}")
        debug(f"  Memory type hint: {memory_type_hint}")
        debug(f"  Normalized: {normalized[:100]}{'...' if len(normalized) > 100 else ''}")
        debug(f"  Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        debug(f"  Keywords: {keywords[:10]}{'...' if len(keywords) > 10 else ''}")
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
