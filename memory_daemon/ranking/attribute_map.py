"""
Canonical attribute vocabulary.

Every canonical attribute contains:

- aliases
- memory field
- boost
- optional future metadata

This file should contain NO logic.
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
