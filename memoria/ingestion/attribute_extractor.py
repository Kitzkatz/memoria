import re

from ranking.attribute_map import ATTRIBUTE_MAP


# Map extracted attributes to actual memory types
ATTRIBUTE_TYPE_MAPPING = {
    "likes": "semantic",
    "vehicle": "semantic",
    "city": "semantic",
    "career": "semantic",
    "pet": "semantic",
    "education": "semantic",
    "goal": "general",
}


class AttributeExtractor:
    """
    Deterministic structured fact extractor.

    Produces:
        subject
        attribute
        value
        typed relationships

    No LLM. Uses regex patterns to extract subject-attribute-value triples.
    """

    def __init__(self):
        # Patterns ordered by priority — first match wins
        self.patterns = [
            ("likes",
             re.compile(r"^(.*?)\s+likes\s+(.*?)[\.\!]?$", re.I)),

            ("vehicle",
             re.compile(r"^(.*?)\s+drives\s+(.*?)[\.\!]?$", re.I)),

            ("city",
             re.compile(r"^(.*?)\s+lives\s+(?:in|at)\s+(.*?)[\.\!]?$", re.I)),

            ("career",
             re.compile(r"^(.*?)\s+works\s+(?:at|for)\s+(.*?)[\.\!]?$", re.I)),

            ("pet",
             re.compile(r"^(.*?)\s+owns\s+(.*?)[\.\!]?$", re.I)),

            ("education",
             re.compile(r"^(.*?)\s+studies\s+(.*?)[\.\!]?$", re.I)),

            ("goal",
             re.compile(r"^(.*?)\s+wants\s+(.*?)[\.\!]?$", re.I))
        ]

    def extract(self, text: str):
        """
        Extract structured facts from text.

        Returns:
            tuple: (memory_type, metadata, relationships)

        Example:
            text = "Alex likes coffee"
            returns: ("semantic", {"subject": "Alex", "attribute": "likes", "value": "coffee"}, [{"source": "Alex", "relation": "likes", "target": "coffee"}])
        """
        if not text:
            return "semantic", {}, []

        text_stripped = text.strip()
        metadata = {}
        relationships = []
        memory_type = "semantic"

        for attribute, pattern in self.patterns:
            match = pattern.match(text_stripped)
            if not match:
                continue

            subject = self._clean_value(match.group(1).strip())
            value = self._clean_value(match.group(2).strip())

            # Skip if either is empty
            if not subject or not value:
                continue

            # Map attribute to actual memory type
            memory_type = ATTRIBUTE_TYPE_MAPPING.get(attribute, "semantic")
            metadata["subject"] = subject
            metadata["attribute"] = attribute
            metadata["value"] = value

            relationships.append({
                "source": subject,
                "relation": attribute,
                "target": value
            })

            # First match wins (priority order)
            break

        return memory_type, metadata, relationships

    def _clean_value(self, value: str) -> str:
        """
        Clean extracted values by stripping:
        - Leading/trailing whitespace
        - Trailing punctuation (.,!?)
        - Leading articles (a, an, the) if present
        """
        if not value:
            return ""

        # Strip whitespace
        value = value.strip()

        # Strip trailing punctuation
        value = re.sub(r'[.,!?;:]+$', '', value)

        # Strip leading articles (optional)
        value = re.sub(r'^(?:a|an|the)\s+', '', value, flags=re.I)

        return value

    def extract_from_text(self, text: str) -> dict:
        """
        Convenience method that returns a structured dict instead of tuple.
        """
        memory_type, metadata, relationships = self.extract(text)
        return {
            "memory_type": memory_type,
            "metadata": metadata,
            "relationships": relationships
        }
