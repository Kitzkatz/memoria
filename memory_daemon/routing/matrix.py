"""
Routing Matrix — Declarative routing for memory types, signals, and workers.
"""

# Signal definitions — what each signal actually means
SIGNAL_DEFINITIONS = {
    "semantic": "Cosine similarity from FAISS embeddings",
    "entity": "Entity match score from EntityStore",
    "subject": "Subject match score from EntityStore",
    "attribute": "Attribute match score from EntityStore",
    "token": "Token overlap ratio between query and memory",
    "tfidf": "TF-IDF similarity from inverted index",
    "importance": "Importance score from ImportanceScorer",
    "recency": "Time decay factor (newer = higher)",
    "temporal": "Temporal proximity score for episodic memories",
    "graph_distance": "Graph shortest path distance between entities"
}

# Valid workers — used for validation
VALID_WORKERS = ["faiss", "bm25", "graph", "phrase", "attribute"]

# Valid pools — used for validation
VALID_POOLS = ["memories", "memories_semantic", "memories_episodic", 
               "memories_procedural", "memories_code", "memories_science", 
               "memories_relevance"]

ROUTING_MATRIX = {
    "semantic": {
        "pool": "memories_semantic",
        "signals": {
            "semantic": 0.30,      # tuned on benchmark 2026-08-04
            "entity": 0.25,        # tuned on benchmark 2026-08-05
            "subject": 0.20,       # default, not yet tuned
            "attribute": 0.15,     # default, not yet tuned
            "recency": 0.00,
            "token": 0.00,
            "importance": 0.10,    # boosted from 0.08 based on feedback
        },
        "workers": ["faiss", "attribute", "graph"],
        "graph_depth": 0,
        "fallback_pools": ["memories_relevance", "memories_semantic"],
        "entity_required": False,
        "temporal_weight": 0.0,
        "description": "Factual knowledge, definitions, properties",
        # Added: detection hints for query processor
        "detection": {
            "keywords": ["what", "is", "definition", "means", "describe", "explain", "called"],
            "exclude": ["when", "where", "how to", "code", "function", "event", "happened"],
            "min_confidence": 0.5
        }
    },
    "episodic": {
        "pool": "memories_episodic",
        "signals": {
            "recency": 0.35,       # tuned on benchmark 2026-08-06
            "temporal": 0.20,      # tuned on benchmark 2026-08-06
            "token": 0.10,         # default
            "semantic": 0.10,      # default
            "entity": 0.10,        # boosted from 0.05 based on feedback
            "importance": 0.10,    # default
            "subject": 0.03,       # low weight for episodic
            "attribute": 0.02,     # low weight for episodic
        },
        "workers": ["faiss", "graph"],
        "graph_depth": 1,
        "fallback_pools": ["memories_relevance", "memories_semantic"],
        "entity_required": True,
        "temporal_weight": 0.20,
        "description": "Events, timestamps, personal experiences",
        "detection": {
            "keywords": ["when", "where", "event", "happened", "occurred", "during", "after", "before"],
            "exclude": ["what is", "define", "how to", "code", "formula"],
            "min_confidence": 0.6
        }
    },
    "procedural": {
        "pool": "memories_procedural",
        "signals": {
            "token": 0.35,         # tuned on benchmark 2026-08-05
            "semantic": 0.20,      # default
            "attribute": 0.15,     # default
            "tfidf": 0.15,         # default
            "importance": 0.10,    # reduced from 0.15, procedural less importance-sensitive
            "entity": 0.05,        # low weight for procedural
            "recency": 0.00,
            "subject": 0.00,
        },
        "workers": ["bm25", "attribute", "faiss"],
        "graph_depth": 0,
        "fallback_pools": ["memories_relevance", "memories_semantic"],
        "entity_required": False,
        "temporal_weight": 0.0,
        "description": "How-to guides, instructions, processes",
        "detection": {
            "keywords": ["how to", "steps", "guide", "tutorial", "instructions", "process", "method", "way to"],
            "exclude": ["what is", "define", "code", "function"],
            "min_confidence": 0.6
        }
    },
    "code": {
        "pool": "memories_code",
        "signals": {
            "token": 0.30,         # tuned on benchmark 2026-08-05
            "entity": 0.25,        # tuned on benchmark 2026-08-06
            "tfidf": 0.20,         # default
            "semantic": 0.10,      # default
            "importance": 0.10,    # reduced from 0.15
            "subject": 0.03,       # low weight
            "attribute": 0.02,     # low weight
            "recency": 0.00,
        },
        "workers": ["bm25", "faiss"],
        "graph_depth": 0,
        "fallback_pools": ["memories_relevance", "memories_semantic"],
        "entity_required": False,
        "temporal_weight": 0.0,
        "description": "Code, functions, classes, symbols",
        "detection": {
            "keywords": ["code", "function", "class", "method", "api", "library", "import", "def", "return"],
            "exclude": ["what is", "define", "how to", "when"],
            "min_confidence": 0.7
        }
    },
    "science": {
        "pool": "memories_science",
        "signals": {
            "entity": 0.30,        # tuned on benchmark 2026-08-06
            "attribute": 0.25,     # tuned on benchmark 2026-08-06
            "semantic": 0.20,      # default
            "importance": 0.15,    # default
            "token": 0.05,         # low weight for science
            "subject": 0.05,       # low weight
            "recency": 0.00,
        },
        "workers": ["faiss", "attribute", "graph"],
        "graph_depth": 0,
        "fallback_pools": ["memories_relevance", "memories_semantic"],
        "entity_required": True,
        "temporal_weight": 0.0,
        "description": "Formulas, equations, scientific facts",
        "detection": {
            "keywords": ["formula", "equation", "theory", "hypothesis", "experiment", "data", "measurement", "scientific"],
            "exclude": ["how to", "code", "when", "event"],
            "min_confidence": 0.6
        }
    },
    "general": {
        "pool": "memories",
        "signals": {
            "semantic": 0.35,      # default
            "entity": 0.20,        # default
            "subject": 0.15,       # default
            "attribute": 0.15,     # default
            "token": 0.0,
            "importance": 0.08,
            "recency": 0.05,
            "tfidf": 0.02,
        },
        "workers": ["faiss", "bm25", "graph", "attribute"],
        "graph_depth": 1,
        "fallback_pools": [],
        "entity_required": False,
        "temporal_weight": 0.0,
        "description": "Fallback for unknown or mixed types",
        "detection": {
            "keywords": [],        # catch-all, no specific keywords
            "exclude": [],
            "min_confidence": 0.0  # always matches
        }
    },
}

# ========================
# Validation
# ========================

def validate_matrix(matrix):
    """
    Validate the routing matrix.
    Raises ValueError if anything is invalid.
    """
    for type_name, config in matrix.items():
        # Check required fields
        required_fields = ["pool", "signals", "workers", "graph_depth", 
                          "fallback_pools", "entity_required", "temporal_weight", 
                          "description"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in {type_name}")
        
        # Check signals sum to 1.0 (within tolerance)
        total = sum(config["signals"].values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"{type_name} signals sum to {total:.4f}, not 1.0")
        
        # Check workers are valid
        for worker in config["workers"]:
            if worker not in VALID_WORKERS:
                raise ValueError(f"Unknown worker '{worker}' in {type_name}")
        
        # Check pools are valid
        if config["pool"] not in VALID_POOLS:
            raise ValueError(f"Unknown pool '{config['pool']}' in {type_name}")
        
        for pool in config["fallback_pools"]:
            if pool not in VALID_POOLS:
                raise ValueError(f"Unknown fallback pool '{pool}' in {type_name}")
        
        # Check graph_depth is non-negative
        if config["graph_depth"] < 0:
            raise ValueError(f"graph_depth must be >= 0 in {type_name}")
        
        # Check temporal_weight is between 0 and 1
        if not 0 <= config["temporal_weight"] <= 1:
            raise ValueError(f"temporal_weight must be between 0 and 1 in {type_name}")
        
        # Check entity_required is bool
        if not isinstance(config["entity_required"], bool):
            raise ValueError(f"entity_required must be boolean in {type_name}")
        
        # Check detection field exists and has required keys
        if "detection" not in config:
            raise ValueError(f"Missing 'detection' field in {type_name}")
        detection = config["detection"]
        if "keywords" not in detection:
            raise ValueError(f"Missing 'keywords' in detection for {type_name}")
        if "exclude" not in detection:
            raise ValueError(f"Missing 'exclude' in detection for {type_name}")
        if "min_confidence" not in detection:
            raise ValueError(f"Missing 'min_confidence' in detection for {type_name}")

# Validate on import
validate_matrix(ROUTING_MATRIX)

# ========================
# Helper Functions
# ========================

def get_type_config(type_name: str):
    """Get the routing config for a type, falling back to general if not found."""
    return ROUTING_MATRIX.get(type_name, ROUTING_MATRIX["general"])

def get_signal_definitions(type_name: str):
    """Get signal definitions for a type."""
    config = get_type_config(type_name)
    signals = config.get("signals", {})
    return {signal: SIGNAL_DEFINITIONS.get(signal, f"Unknown signal: {signal}") 
            for signal in signals.keys()}
