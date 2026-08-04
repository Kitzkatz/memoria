from core.logger import debug
import json
import re

from ingestion.attribute_extractor import AttributeExtractor
from memory.models import MemoryRecord


class MemoryExtractor:

    def __init__(self, llm=None):
        self.attribute_extractor = AttributeExtractor()
        self.llm = llm


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

        text = text.lower()
        text = re.sub(r"\s+", " ", text)

        return text.strip()


    def tokenize(self, normalized_text: str):
        """
        Simple tokenizer.

        Later this becomes:
            entity extraction
            stemming
            buckets
            etc.

        Keeping it intentionally simple for V1.
        """

        return normalized_text.split()


    # ---------------------------------
    # Base Extract
    # ---------------------------------

    def base_extract(self, text: str):

        normalized = self.normalize_text(text)

        tokens = self.tokenize(normalized)

        entities=self.extract_entities(text)

        metadata=self.extract_metadata(text)
        memory_type, attribute_metadata, typed_relationships = (
            self.attribute_extractor.extract(text)
        )

        metadata.update(attribute_metadata)

        relationships = self.extract_relationships(
            text,
            entities
        )

        relationships.extend(typed_relationships)

        return MemoryRecord(

            text=text,

            memory_type=memory_type,

            metadata= metadata,

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

    def llm_extract(self, text: str):

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

        out = self.llm.chat(prompt)

        data = json.loads(out)

        record = self.base_extract(text)

        if isinstance(data, dict):

            if "memory_type" in data:
                record.memory_type = data["memory_type"]

            if "metadata" in data and isinstance(data["metadata"], dict):

                record.metadata = data["metadata"]


            if "entities" in data and isinstance(data["entities"], list):

                record.entities = data["entities"]


            if "relationships" in data and isinstance(data["relationships"], list):

                record.relationships = data["relationships"]


            if "importance" in data:

                record.importance = float(data["importance"])

        return record


    # ---------------------------------
    # Public Entry
    # ---------------------------------

    def extract(self, text: str):
        record = self.base_extract(text)
        
        if self.llm:
            try:
                llm_record = self.llm_extract(text)
                
                # Only override if LLM actually returned something
                if llm_record.entities:
                    record.entities = llm_record.entities
                if llm_record.relationships:
                    record.relationships = llm_record.relationships
                record.metadata.update(llm_record.metadata)
                if llm_record.memory_type:
                    record.memory_type = llm_record.memory_type
                    
            except Exception as e:
                debug("[Extractor LLM fallback]", e)
        
        return record



    def extract_entities(self, text):

        entities = []
        current = []

        for word in text.split():

            word = word.strip(".,!?()[]{}\"'")

            if word and word[0].isupper():
                current.append(word)
            else:
                if current:
                    entities.append(" ".join(current))
                    current = []

        if current:
            entities.append(" ".join(current))

        return sorted(set(entities))


    def extract_metadata(self, text):

        normalized = self.normalize_text(text)

        metadata = {}

        metadata["length"] = len(text)

        metadata["word_count"] = len(normalized.split())

        metadata["contains_number"] = bool(
            re.search(r"\d", text)
        )

        metadata["contains_url"] = bool(
            re.search(r"https?://|www\.", text)
        )

        metadata["contains_email"] = bool(
            re.search(r"\S+@\S+", text)
        )

        metadata["contains_question"] = "?" in text

        metadata["contains_code"] = any(
            token in text
            for token in (
                "{", "}", ";",
                "def ", "class ",
                "return ", "import "
            )
        )

        return metadata


    def extract_relationships(self, text, entities):

        relationships = []

        if len(entities) < 2:
            return relationships

        #
        # Placeholder:
        # connect consecutive entities
        #

        for i in range(len(entities) - 1):

            relationships.append({

                "source": entities[i],

                "relation": "related_to",

                "target": entities[i + 1]

            })

        return relationships
