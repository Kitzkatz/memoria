from core.logger import debug
import json
import re

from retrieval.case_folding import fold_case

from ingestion.attribute_extractor import AttributeExtractor
from memory.models import MemoryRecord


class MemoryExtractor:

    def __init__(self, llm=None):
        self.attribute_extractor = AttributeExtractor()
        self.llm = llm

        # Pre-compiled regex patterns for performance
        self._url_pattern = re.compile(r"https?://|www\.")
        self._email_pattern = re.compile(r"\S+@\S+")
        self._code_pattern = re.compile(r"[{};]|def\s+|class\s+|return\s+|import\s+")
        self._number_pattern = re.compile(r"\d")

    # ---------------------------------
    # Preprocessing Helpers
    # ---------------------------------

    def normalize_text(self, text: str) -> str:
        """
        Lightweight normalization.

        Lowercase
        Remove duplicate whitespace
        Strip leading/trailing whitespace
        """
        if not text:
            return ""
        text = fold_case(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def tokenize(self, normalized_text: str) -> list:
        """
        Simple tokenizer.

        Later this becomes:
            entity extraction
            stemming
            buckets
            etc.

        Keeping it intentionally simple for V1.
        """
        if not normalized_text:
            return []
        return normalized_text.split()

    # ---------------------------------
    # Base Extract
    # ---------------------------------

    def base_extract(self, text: str) -> MemoryRecord:
        if not text:
            return MemoryRecord(
                text="",
                memory_type="general",
                metadata={},
                entities=[],
                relationships=[],
                importance=0.0,
                normalized_text="",
                tokens=[],
                token_count=0
            )

        normalized = self.normalize_text(text)
        tokens = self.tokenize(normalized)
        entities = self.extract_entities(text)
        metadata = self.extract_metadata(text)

        memory_type, attribute_metadata, typed_relationships = (
            self.attribute_extractor.extract(text)
        )
        metadata.update(attribute_metadata)

        relationships = self.extract_relationships(text, entities)
        relationships.extend(typed_relationships)

        return MemoryRecord(
            text=text,
            memory_type=memory_type,
            metadata=metadata,
            entities=entities,
            relationships=relationships,
            importance=0.5,
            normalized_text=normalized,
            tokens=tokens,
            token_count=len(tokens)
        )

    # ---------------------------------
    # LLM Enrichment
    # ---------------------------------

    def llm_extract(self, text: str) -> dict:
        prompt = f"""
            Extract structured memory JSON.

            Return ONLY valid JSON.

            Required keys:

            - memory_type
            - text
            - importance
            - metadata
            - entities
            - relationships


            Rules:

            metadata:
            Extra useful information about the memory.

            entities:
            Important people, places, objects, technologies,
            projects, concepts, or named things.

            relationships:
            Connections between entities.

            Format relationships as:

            [
                {{
                    "source": "entity",
                    "relation": "relationship",
                    "target": "entity"
                }}
            ]


            Text:

            {text}
            """

        try:
            out = self.llm.chat(prompt)

            # Clean response — try to find JSON
            cleaned = out.strip()
            # Remove markdown code blocks if present
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
            else:
                debug("[LLM] Response was not a dict")
                return {}
        except json.JSONDecodeError as e:
            debug(f"[LLM] JSON decode error: {e}")
            return {}
        except Exception as e:
            debug(f"[LLM] Extraction error: {e}")
            return {}

    # ---------------------------------
    # Public Entry (with metadata support)
    # ---------------------------------

    def extract(self, text: str, metadata: dict = None) -> MemoryRecord:
        """
        Extract a memory record from text with optional metadata.

        Args:
            text: The text to extract from
            metadata: Optional metadata dict to merge into the record

        Returns:
            MemoryRecord
        """
        record = self.base_extract(text)

        # Merge provided metadata
        if metadata:
            record.metadata.update(metadata)

        # LLM enrichment (if available)
        if self.llm:
            try:
                data = self.llm_extract(text)
                if data:
                    if data.get("entities"):
                        record.entities = data["entities"]
                    if data.get("relationships"):
                        record.relationships = data["relationships"]
                    if data.get("metadata"):
                        record.metadata.update(data["metadata"])
                    if data.get("memory_type"):
                        record.memory_type = data["memory_type"]
                    if data.get("importance"):
                        record.importance = float(data["importance"])
            except Exception as e:
                debug(f"[Extractor] LLM fallback: {e}")

        return record

    # ---------------------------------
    # Entity Extraction
    # ---------------------------------

    def extract_entities(self, text: str) -> list:
        """
        Extract named entities using rule-based heuristics.

        Handles:
        - Single capitalized words (Kevin, Seattle)
        - Multi-word capitalized phrases (New York, San Francisco)
        - Acronyms (NASA, AI)
        """
        if not text:
            return []

        entities = []
        current = []

        # Split on whitespace, preserve punctuation for later
        words = text.split()

        for word in words:
            # Strip surrounding punctuation but keep internal
            stripped = word.strip(".,!?()[]{}\"'")
            if not stripped:
                continue

            is_acronym = stripped.isupper() and len(stripped) >= 2
            is_capitalized = stripped[0].isupper() and not stripped.isupper()

            if is_capitalized or is_acronym:
                current.append(stripped)
            else:
                if current:
                    entity = " ".join(current)
                    if len(entity) > 1:
                        entities.append(entity)
                    current = []

        if current:
            entity = " ".join(current)
            if len(entity) > 1:
                entities.append(entity)

        # Deduplicate preserving order
        seen = set()
        result = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                result.append(e)

        return result

    # ---------------------------------
    # Metadata Extraction
    # ---------------------------------

    def extract_metadata(self, text: str) -> dict:
        """Extract metadata from text using pre-compiled patterns."""
        if not text:
            return {}

        normalized = self.normalize_text(text)

        return {
            "length": len(text),
            "word_count": len(normalized.split()),
            "contains_number": bool(self._number_pattern.search(text)),
            "contains_url": bool(self._url_pattern.search(text)),
            "contains_email": bool(self._email_pattern.search(text)),
            "contains_question": "?" in text,
            "contains_code": bool(self._code_pattern.search(text)),
        }

    # ---------------------------------
    # Relationship Extraction
    # ---------------------------------

    def extract_relationships(self, text: str, entities: list) -> list:
        """
        Extract relationships between entities.

        Current implementation:
        - Connects consecutive entities with "related_to"
        - Looks for explicit relationship cues (likes, works_at, etc.)

        Future improvements:
        - Dependency parsing
        - LLM-based extraction
        """
        relationships = []

        if len(entities) < 2:
            return relationships

        # Check for explicit relationship cues
        relation_cues = {
            "likes": "likes",
            "loves": "likes",
            "hates": "dislikes",
            "dislikes": "dislikes",
            "works at": "works_at",
            "works for": "works_for",
            "lives in": "lives_in",
            "born in": "born_in",
            "created": "created",
            "developed": "developed",
            "built": "built",
            "wrote": "wrote",
            "says": "said",
        }

        text_lower = text.lower()

        # Try to find explicit relationships
        for cue, relation in relation_cues.items():
            if cue in text_lower:
                # Find source and target around the cue
                # Simple heuristic: entities before and after the cue
                for i, entity in enumerate(entities):
                    entity_lower = entity.lower()
                    if entity_lower in text_lower:
                        pos = text_lower.find(entity_lower)
                        cue_pos = text_lower.find(cue)
                        if cue_pos > pos:
                            # Entity before cue -> source
                            source = entity
                            # Find target (next entity after cue)
                            for j in range(i + 1, len(entities)):
                                if entities[j].lower() in text_lower and text_lower.find(entities[j].lower()) > cue_pos:
                                    relationships.append({
                                        "source": source,
                                        "relation": relation,
                                        "target": entities[j]
                                    })
                                    break

        # If no explicit relationships found, connect consecutive entities
        if not relationships and len(entities) >= 2:
            for i in range(len(entities) - 1):
                relationships.append({
                    "source": entities[i],
                    "relation": "related_to",
                    "target": entities[i + 1]
                })

        return relationships

    # ---------------------------------
    # Debug
    # ---------------------------------

    def __repr__(self) -> str:
        return f"MemoryExtractor(llm={'available' if self.llm else 'None'})"
