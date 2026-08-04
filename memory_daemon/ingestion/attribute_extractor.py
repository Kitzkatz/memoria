import re

from ranking.attribute_map import ATTRIBUTE_MAP


class AttributeExtractor:
    """
    Deterministic structured fact extractor.

    Produces:

        subject
        attribute
        value
        typed relationships

    No LLM.
    """

    def __init__(self):
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

    def extract(self, text):

        metadata = {}

        relationships = []

        memory_type = "semantic"

        for attribute, pattern in self.patterns:

            match = pattern.match(text.strip())

            if not match:
                continue

            subject = match.group(1).strip()

            value = match.group(2).strip()

            memory_type = attribute

            metadata["subject"] = subject
            metadata["attribute"] = attribute
            metadata["value"] = value

            relationships.append({

                "source": subject,

                "relation": attribute,

                "target": value

            })

            break

        return memory_type, metadata, relationships
