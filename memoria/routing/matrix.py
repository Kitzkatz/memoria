"""
Routing Matrix — Declarative routing for memory types, signals, and workers.
"""

from cache.config import settings  # <-- ADD THIS IMPORT

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

# Valid workers — includes fusion now
VALID_WORKERS = ["faiss", "bm25", "graph", "phrase", "attribute", "fusion"]

# Valid pools — used for validation
VALID_POOLS = ["memories", "memories_semantic", "memories_episodic", 
               "memories_procedural", "memories_code", "memories_science", 
               "memories_relevance"]

# Detection scoring weights
DETECTION_WEIGHTS = {
    "keyword_match": 1.0,
    "exclude_match": -2.0,
    "entity_required_boost": 2.0,
    "attribute_required_boost": 1.5,
    "phrase_match_boost": 1.5,
}

ROUTING_MATRIX = {
    "semantic": {
        "pool": "memories_semantic",
        "signals": {
            "semantic": 0.30,
            "entity": 0.25,
            "subject": 0.20,
            "attribute": 0.15,
            "recency": 0.00,
            "token": 0.00,
            "importance": 0.10,
        },
        "workers": ["faiss", "attribute", "graph"],
        "fusion_enabled": True,   # <-- NEW
        "graph_depth": 0,
        "fallback_pools": ["memories_relevance", "memories_semantic"],
        "entity_required": False,
        "attribute_required": False,
        "temporal_weight": 0.0,
        "description": "Factual knowledge, definitions, properties",
        "detection": {
            "keywords": ["what", "is", "definition", "means", "describe", "explain", "called"],
            "exclude": ["when", "where", "how to", "code", "function", "event", "happened"],
            "min_confidence": 0.5,
            "exclude_penalty": 2.0,
            "boost": {
                "entity_required": 0.2,
                "attribute_required": 0.15,
            }
        }
    },
    "episodic": {
        "pool": "memories_episodic",
        "signals": {
            "recency": 0.35,
            "temporal": 0.20,
            "token": 0.10,
            "semantic": 0.10,
            "entity": 0.10,
            "importance": 0.10,
            "subject": 0.03,
            "attribute": 0.02,
        },
        "workers": ["faiss", "graph"],
        "fusion_enabled": False,   # <-- NEW
        "graph_depth": 1,
        "fallback_pools": ["memories_relevance", "memories_semantic"],
        "entity_required": True,
        "attribute_required": False,
        "temporal_weight": 0.20,
        "description": "Events, timestamps, personal experiences",
        "detection": {
            "keywords": ["when", "where", "event", "happened", "occurred", "during", "after", "before"],
            "exclude": ["what is", "define", "how to", "code", "formula"],
            "min_confidence": 0.6,
            "exclude_penalty": 2.5,
            "boost": {
                "entity_required": 0.3,
                "attribute_required": 0.1,
            }
        }
    },
    "procedural": {
        "pool": "memories_procedural",
        "signals": {
            "token": 0.35,
            "semantic": 0.20,
            "attribute": 0.15,
            "tfidf": 0.15,
            "importance": 0.10,
            "entity": 0.05,
            "recency": 0.00,
            "subject": 0.00,
        },
        "workers": ["bm25", "attribute", "faiss"],
        "fusion_enabled": False,   # <-- NEW
        "graph_depth": 0,
        "fallback_pools": ["memories_relevance", "memories_semantic"],
        "entity_required": False,
        "attribute_required": True,
        "temporal_weight": 0.0,
        "description": "How-to guides, instructions, processes",
        "detection": {
            "keywords": ["how to", "steps", "guide", "tutorial", "instructions", "process", "method", "way to"],
            "exclude": ["what is", "define", "code", "function"],
            "min_confidence": 0.6,
            "exclude_penalty": 2.0,
            "boost": {
                "entity_required": 0.1,
                "attribute_required": 0.25,
            }
        }
    },
    "code": {
        "pool": "memories_code",
        "signals": {
            "token": 0.30,
            "entity": 0.25,
            "tfidf": 0.20,
            "semantic": 0.10,
            "importance": 0.10,
            "subject": 0.03,
            "attribute": 0.02,
            "recency": 0.00,
        },
        "workers": ["bm25", "faiss"],
        "fusion_enabled": False,   # <-- NEW
        "graph_depth": 0,
        "fallback_pools": ["memories_relevance", "memories_semantic"],
        "entity_required": False,
        "attribute_required": False,
        "temporal_weight": 0.0,
        "description": "Code, functions, classes, symbols",
        "detection": {
            "keywords": ["code", "function", "class", "method", "api", "library", "import", "def", "return"],
            "exclude": ["what is", "define", "how to", "when"],
            "min_confidence": 0.7,
            "exclude_penalty": 3.0,
            "boost": {
                "entity_required": 0.1,
                "attribute_required": 0.1,
            }
        }
    },
    "science": {
        "pool": "memories_science",
        "signals": {
            "entity": 0.30,
            "attribute": 0.25,
            "semantic": 0.20,
            "importance": 0.15,
            "token": 0.05,
            "subject": 0.05,
            "recency": 0.00,
        },
        "workers": ["faiss", "attribute", "graph"],
        "fusion_enabled": False,   # <-- NEW
        "graph_depth": 0,
        "fallback_pools": ["memories_relevance", "memories_semantic"],
        "entity_required": True,
        "attribute_required": True,
        "temporal_weight": 0.0,
        "description": "Formulas, equations, scientific facts",
        "detection": {
            "keywords": ["formula", "equation", "theory", "hypothesis", "experiment", "data", "measurement", "scientific"],
            "exclude": ["how to", "code", "when", "event"],
            "min_confidence": 0.6,
            "exclude_penalty": 2.0,
            "boost": {
                "entity_required": 0.3,
                "attribute_required": 0.3,
            }
        }
    },
    "general": {
        "pool": "memories",
        "signals": {
            "semantic": 0.35,
            "entity": 0.20,
            "subject": 0.15,
            "attribute": 0.15,
            "token": 0.0,
            "importance": 0.08,
            "recency": 0.05,
            "tfidf": 0.02,
        },
        "workers": ["faiss", "bm25", "graph", "attribute"],
        "fusion_enabled": True,   # <-- NEW
        "graph_depth": 1,
        "fallback_pools": [],
        "entity_required": False,
        "attribute_required": False,
        "temporal_weight": 0.0,
        "description": "Fallback for unknown or mixed types",
        "detection": {
            "keywords": [],
            "exclude": [],
            "min_confidence": 0.0,
            "exclude_penalty": 0.0,
            "boost": {}
        }
    },
}


# ========================
# Detection Helper Functions
# ========================

def get_detection_weights(type_name: str) -> dict:
    """Get detection weights for a type, with fallback to defaults."""
    config = ROUTING_MATRIX.get(type_name, ROUTING_MATRIX.get("general", {}))
    detection = config.get("detection", {})
    
    return {
        "keyword_weight": detection.get("keyword_weight", DETECTION_WEIGHTS["keyword_match"]),
        "exclude_penalty": detection.get("exclude_penalty", DETECTION_WEIGHTS["exclude_match"]),
        "entity_boost": detection.get("boost", {}).get("entity_required", 0.2),
        "attribute_boost": detection.get("boost", {}).get("attribute_required", 0.15),
        "min_confidence": detection.get("min_confidence", 0.5),
    }


def compute_detection_score(type_name: str, query: str, entities: list = None, attributes: list = None) -> tuple:
    """
    Compute weighted detection score for a type.
    
    Returns:
        tuple: (score, confidence, matched_keywords, matched_excludes)
    """
    config = ROUTING_MATRIX.get(type_name, ROUTING_MATRIX.get("general", {}))
    detection = config.get("detection", {})
    weights = get_detection_weights(type_name)
    
    query_lower = query.lower()
    matched_keywords = []
    matched_excludes = []
    
    for kw in detection.get("keywords", []):
        if kw in query_lower:
            matched_keywords.append(kw)
    
    for ex in detection.get("exclude", []):
        if ex in query_lower:
            matched_excludes.append(ex)
    
    score = len(matched_keywords) * weights["keyword_weight"]
    score += len(matched_excludes) * weights["exclude_penalty"]
    
    if entities and config.get("entity_required", False):
        score += weights["entity_boost"]
    
    if attributes and config.get("attribute_required", False):
        score += weights["attribute_boost"]
    
    raw_confidence = (score + 10) / 30
    confidence = max(0.0, min(1.0, raw_confidence))
    
    return score, confidence, matched_keywords, matched_excludes


# ========================
# Worker helper
# ========================

def get_workers_for_type(type_name: str) -> list:
    """
    Return the list of workers for a type, with fusion applied if:
    - USE_FUSION is True in config
    - fusion_enabled is True for this type
    - both faiss and bm25 are present in the original worker list
    """
    config = ROUTING_MATRIX.get(type_name, ROUTING_MATRIX["general"])
    workers = config.get("workers", ["faiss", "bm25", "graph", "attribute"])

    use_fusion = getattr(settings, "USE_FUSION", False)
    fusion_enabled = config.get("fusion_enabled", False)

    if use_fusion and fusion_enabled and "faiss" in workers and "bm25" in workers:
        workers = ["fusion" if w in ("faiss", "bm25") else w for w in workers]
        seen = set()
        workers = [w for w in workers if not (w in seen or seen.add(w))]

    return workers


# ========================
# Validation
# ========================

def validate_matrix(matrix):
    """
    Validate the routing matrix.
    """
    for type_name, config in matrix.items():
        required_fields = ["pool", "signals", "workers", "graph_depth", 
                          "fallback_pools", "entity_required", "temporal_weight", 
                          "description", "fusion_enabled"]   # <-- ADDED
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in {type_name}")
        
        total = sum(config["signals"].values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"{type_name} signals sum to {total:.4f}, not 1.0")
        
        for worker in config["workers"]:
            if worker not in VALID_WORKERS:
                raise ValueError(f"Unknown worker '{worker}' in {type_name}")
        
        if config["pool"] not in VALID_POOLS:
            raise ValueError(f"Unknown pool '{config['pool']}' in {type_name}")
        
        for pool in config["fallback_pools"]:
            if pool not in VALID_POOLS:
                raise ValueError(f"Unknown fallback pool '{pool}' in {type_name}")
        
        if config["graph_depth"] < 0:
            raise ValueError(f"graph_depth must be >= 0 in {type_name}")
        
        if not 0 <= config["temporal_weight"] <= 1:
            raise ValueError(f"temporal_weight must be between 0 and 1 in {type_name}")
        
        if not isinstance(config["entity_required"], bool):
            raise ValueError(f"entity_required must be boolean in {type_name}")
        
        if "attribute_required" not in config:
            config["attribute_required"] = False
        
        if "detection" not in config:
            raise ValueError(f"Missing 'detection' field in {type_name}")
        detection = config["detection"]
        if "keywords" not in detection:
            raise ValueError(f"Missing 'keywords' in detection for {type_name}")
        if "exclude" not in detection:
            raise ValueError(f"Missing 'exclude' in detection for {type_name}")
        if "min_confidence" not in detection:
            raise ValueError(f"Missing 'min_confidence' in detection for {type_name}")
        if "exclude_penalty" not in detection:
            detection["exclude_penalty"] = 2.0
        if "boost" not in detection:
            detection["boost"] = {}

validate_matrix(ROUTING_MATRIX)


def get_type_config(type_name: str):
    return ROUTING_MATRIX.get(type_name, ROUTING_MATRIX["general"])


def get_signal_definitions(type_name: str):
    config = get_type_config(type_name)
    signals = config.get("signals", {})
    return {signal: SIGNAL_DEFINITIONS.get(signal, f"Unknown signal: {signal}") 
            for signal in signals.keys()}
