"""
Canonical attribute vocabulary.

Every canonical attribute contains:
- aliases: List of words/phrases that indicate this attribute
- field: Canonical attribute name (matches the key)
- boost: Score multiplier when matched (0.0 to 1.0)

This file contains NO logic — it is a pure data definition.
Used by: AttributeExtractor, AttributeBooster, QueryProcessor
"""

ATTRIBUTE_MAP = {

    "likes": {
        "field": "likes",
        "boost": 0.18,
        "aliases": [
            "like",
            "likes",
            "love",
            "enjoy",
            "favorite",
            "prefer",
            "adore",
            "interested",
            "interest",
            "fan"
        ]
    },

    "vehicle": {
        "field": "vehicle",
        "boost": 0.18,
        "aliases": [
            "car",
            "cars",
            "truck",
            "vehicle",
            "drive",
            "drives",
            "driving",
            "ride",
            "rides",
            "motorcycle",
            "bike"
        ]
    },

    "city": {
        "field": "city",
        "boost": 0.20,
        "aliases": [
            "city",
            "town",
            "state",
            "country",
            "live",
            "lives",
            "living",
            "location",
            "home",
            "resides"
        ]
    },

    "family": {
        "field": "family",
        "boost": 0.22,
        "aliases": [
            "wife",
            "husband",
            "mom",
            "mother",
            "dad",
            "father",
            "brother",
            "sister",
            "daughter",
            "son",
            "child",
            "children",
            "kids",
            "family",
            "parent"
        ]
    },

    "career": {
        "field": "career",
        "boost": 0.17,
        "aliases": [
            "job",
            "work",
            "career",
            "profession",
            "occupation",
            "employment",
            "company",
            "office",
            "boss"
        ]
    },

    "education": {
        "field": "education",
        "boost": 0.15,
        "aliases": [
            "school",
            "college",
            "university",
            "degree",
            "major",
            "study",
            "studying",
            "education",
            "class"
        ]
    },

    "pet": {
        "field": "pet",
        "boost": 0.20,
        "aliases": [
            "dog",
            "dogs",
            "cat",
            "cats",
            "pet",
            "pets",
            "animal",
            "puppy",
            "kitten"
        ]
    },

    "food": {
        "field": "food",
        "boost": 0.14,
        "aliases": [
            "food",
            "eat",
            "eats",
            "meal",
            "restaurant",
            "pizza",
            "burger",
            "coffee",
            "tea"
        ]
    },

    "programming": {
        "field": "programming",
        "boost": 0.16,
        "aliases": [
            "python",
            "java",
            "c",
            "cpp",
            "c++",
            "javascript",
            "coding",
            "programming",
            "developer",
            "software",
            "algorithm"
        ]
    },

    "project": {
        "field": "project",
        "boost": 0.16,
        "aliases": [
            "project",
            "build",
            "building",
            "prototype",
            "design",
            "develop",
            "developing"
        ]
    },

    "goal": {
        "field": "goal",
        "boost": 0.23,
        "aliases": [
            "goal",
            "objective",
            "plan",
            "future",
            "target",
            "dream",
            "mission"
        ]
    },

    "emotion": {
        "field": "emotion",
        "boost": 0.10,
        "aliases": [
            "happy",
            "sad",
            "angry",
            "excited",
            "worried",
            "frustrated",
            "depressed",
            "anxious",
            "feeling",
            "feel"
        ]
    }

}


# -------------------------
# Helper functions (convenience, still no logic)
# -------------------------

def get_attribute_boost(attribute: str) -> float:
    """Get the boost value for a canonical attribute."""
    config = ATTRIBUTE_MAP.get(attribute, {})
    return config.get("boost", 0.10)


def get_attribute_aliases(attribute: str) -> list:
    """Get all aliases for a canonical attribute."""
    config = ATTRIBUTE_MAP.get(attribute, {})
    return config.get("aliases", [])


def get_all_attributes() -> list:
    """Return all canonical attribute names."""
    return list(ATTRIBUTE_MAP.keys())


def get_all_aliases() -> list:
    """Return all alias phrases across all attributes."""
    aliases = []
    for config in ATTRIBUTE_MAP.values():
        aliases.extend(config.get("aliases", []))
    return aliases


def get_attribute_for_alias(alias: str) -> str:
    """
    Find the canonical attribute for a given alias.
    Returns None if no match found.
    """
    alias_lower = alias.lower()
    for attr_name, config in ATTRIBUTE_MAP.items():
        for alias_str in config.get("aliases", []):
            if alias_str.lower() == alias_lower:
                return attr_name
    return None


__all__ = [
    "ATTRIBUTE_MAP",
    "get_attribute_boost",
    "get_attribute_aliases",
    "get_all_attributes",
    "get_all_aliases",
    "get_attribute_for_alias",
]
