"""
Configuration for the Memory Daemon system.

All settings are defined in the Settings class, which uses Pydantic for validation.
Environment variables can override settings by prefixing with "MEMORY_".
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import os


class Settings(BaseModel):
    """Main configuration class for the Memory Daemon."""

    # -------------------------
    # Paths
    # -------------------------

    DB_PATH: str = "memory.db"
    VECTOR_INDEX_PATH: str = "memory.index"
    CACHE_PATH: str = "cache/embedding_cache.pkl"

    # -------------------------
    # Models
    # -------------------------

    EMBEDDING_MODEL: str = "memory/models/all-MiniLM-L6-v2"
    CHAT_TEMPLATE_DIR: str = "chat_templates"
    CHAT_TEMPLATE_FILE: str = "llama3.txt"
    VECTOR_DIM: int = 384
    CHAT_MODEL: str = "mistral"
    LLM_URL: str = "http://localhost:8080"
    LLM_ENDPOINT: str = "/v1/completions"
    LLM_MAX_TOKENS: int = 256
    LLM_TEMPERATURE: float = 0.7
    LLM_TIMEOUT: int = 600
    LLM_STOP_TOKENS: List[str] = Field(default_factory=lambda: ["<|eot_id|>"])

    # -------------------------
    # Paths
    # -------------------------

    DB_PATH: str = "memory.db"
    VECTOR_INDEX_PATH: str = "memory.index"
    CACHE_PATH: str = "cache/embedding_cache.pkl"

    # -------------------------
    # Models
    # -------------------------

    EMBEDDING_MODEL: str = "memory/models/all-MiniLM-L6-v2"
    CHAT_TEMPLATE_DIR: str = "chat_templates"
    CHAT_TEMPLATE_FILE: str = "llama3.txt"
    VECTOR_DIM: int = 384
    CHAT_MODEL: str = "mistral"
    LLM_URL: str = "http://localhost:8080"
    LLM_ENDPOINT: str = "/v1/completions"
    LLM_MAX_TOKENS: int = 256
    LLM_TEMPERATURE: float = 0.7
    LLM_TIMEOUT: int = 600
    LLM_STOP_TOKENS: List[str] = Field(default_factory=lambda: ["<|eot_id|>"])

    # -------------------------
    # Retrieval
    # -------------------------

    TOP_K: int = 5000
    TOP_N: int = 550
    GRAPH_TOP_K: int = 50
    GRAPH_SEARCH_LIMIT: int = 200
    GRAPH_DEPTH: int = 3
    GRAPH_SEARCH_DEPTH: int = 1

    USE_QUERY_EXPANSION: bool = False
    SYNONYM_PATH: str = "retrieval/synonyms.json"

    # -------------------------
    # Indexing
    # -------------------------

    USE_INVERTED_INDEX: bool = True
    USE_PHRASE_SEARCH: bool = True
    USE_BM25: bool = True
    USE_FUSION: bool = True
    # FIX (2026-08-24): Rolled back to neutral 0.5. Your upstream retriever was starving.
    FUSION_SEMANTIC_WEIGHT: float = 0.5
    # CRITICAL FIX: The benchmark IGNORED this flag last time and forced MMR ON.
    # To guarantee it's off, you MUST pass MEMORY_MMR_ENABLED=False as an ENV var.
    RRF_K: int = 10

    MMR_ENABLED: bool = False
    USE_BLACKBOARD: bool = True
    USE_CASE_FOLDING: bool = True

    # -------------------------
    # Retrieval Workers
    # -------------------------

    WORKERS_TO_USE: List[str] = ["fusion"]  # exclude "graph", "attribute"

    # -------------------------
    # Cross-Encoder
    # -------------------------

    USE_CROSS_ENCODER: bool = False
    CROSS_ENCODER_MODEL_PATH: str = "memory/models/cross-encoder"
    CROSS_ENCODER_TOP_K: int = 10

    # -------------------------
    # Routing
    # -------------------------

    USE_ROUTING: bool = True
    ROUTING_MATRIX_OVERRIDE: bool = True
    ROUTING_FALLBACK_ENABLED: bool = True

    RETRIEVAL_MIN_CANDIDATES: int = 550
    MIN_RETRIEVAL_SOURCES: int = 2
    RETRIEVAL_DEADLINE: float = 0.050

    # -------------------------
    # Ranking
    # -------------------------
    

    RANKING_ENABLED: bool = False   # Set to False to skip ranking and use raw retrieval scores
    CONTEXT_MAX_MEMORIES: int = 50
    CONTEXT_MIN_SCORE: float = 0.15
    CONTEXT_TOKEN_BUDGET: int = 10000

    # -------------------------
    # Memory
    # -------------------------

    MEMORY_DECAY_DAYS: int = 30
    MEMORY_DECAY_RATE: float = 0.001
    IMPORTANCE_DELTA: float = 0.01

    # -------------------------
    # Sharding
    # -------------------------

    NUM_SHARDS: int = 5
    USE_SHARDING: bool = False
    TOP_K_PER_SHARD: int = 200

    # -------------------------
    # Finalizer
    # -------------------------

    FINALIZER_USE_SIGMOID: bool = False
    # FIX (2026-08-24): Scale was 0.015 (brick wall). Changed to 3.0.
    # Since your scores average ~4.7, sigmoid(4.7/3.0) = sigmoid(1.57) = 0.82.
    # This gives a healthy spread between #1 and #5 without collapsing to 1.0.
    FINALIZER_SIGMOID_SCALE: float = 0.5

    # -------------------------
    # Embedding Cache
    # -------------------------

    SKIP_EMBEDDING: bool = False
    EMBEDDING_CACHE_MAX_SIZE: int = 100000

    # -------------------------
    # Debug
    # -------------------------

    DEBUG: bool = False
    RANKER_DIAGNOSTICS: bool = False

    # -------------------------
    # Boosting
    # -------------------------

    ENTITY_BOOST: float = 0.50

    # -------------------------
    # Adaptive Weighter
    # -------------------------

    AUTO_STORE_MEMORIES: bool = False
    AUTO_STORE_THRESHOLD: float = 0.7
    AUTO_STORE_MAX_PER_SESSION: int = 10
    AUTO_STORE_TYPES: List[str] = ["general", "chat"]

    USE_ADAPTIVE_WEIGHTS: bool = True
    ADAPTIVE_WEIGHT_STEP: float = 0.02
    ADAPTIVE_WEIGHT_MAX: float = 0.40
    ADAPTIVE_WEIGHT_MIN: float = 0.01

    # -------------------------
    # Ranking Weights (from backup)
    # -------------------------
    SIGNAL_REGISTRY_PATH: str = "ranking/signal_registry.json"
    ENABLE_SIGNAL_REGISTRY: bool = True

    RANKING_SEMANTIC: float = 0.40
    RANKING_IMPORTANCE: float = 0.01
    RANKING_RECENCY: float = 0.01
    RANKING_TOKEN: float = 0.15
    RANKING_FEEDBACK: float = 0.01
    RANKING_ENTITY: float = 0.01
    RANKING_SUBJECT: float = 0.10
    RANKING_ATTRIBUTE: float = 0.01
    RANKING_TFIDF: float = 0.15
    RANKING_BM25: float = 0.15


    # -------------------------
    # Score Normalizer
    # -------------------------

    SCORE_NORMALIZER_METHOD: str = "zscore"  # Options: "zscore" or "minmax"

    # -------------------------
    # Finalizer Weights (REBALANCED)
    # -------------------------

    # FIX (2026-08-24): Restored Attribute to 0.40 and BM25 to 0.10.
    # Your Attribute booster was holding the ranking together.
    FINALIZER_RELEVANCE: float = 1.0
    FINALIZER_IMPORTANCE: float = 0.0
    FINALIZER_RECENCY: float = 0.0
    FINALIZER_DIVERSITY: float = 0.0
    FINALIZER_ATTRIBUTE: float = 0.0
    FINALIZER_BM25: float = 0.0

    # -------------------------
    # Consolidator
    # -------------------------

    CONSOLIDATE_THRESHOLD: float = 0.85
    CONSOLIDATE_BATCH_SIZE: int = 500
    CONSOLIDATE_AUTO: bool = False
    CONSOLIDATE_INTERVAL: int = 3600

    # -------------------------
    # Feedback Loop
    # -------------------------

    FEEDBACK_WEIGHT: float = 0.08
    FEEDBACK_PERSIST_PATH: str = "feedback_data.json"
    QUERY_HISTORY_PERSIST_PATH: str = "query_history.json"
    QUERY_HISTORY_MAX: int = 1000

    # -------------------------
    # Memory Pruner
    # -------------------------

    PRUNE_THRESHOLD: float = 0.1
    PRUNE_MAX_AGE_DAYS: int = 365
    PRUNE_BATCH_SIZE: int = 100
    PRUNE_INTERVAL_SECONDS: int = 3600
    PRUNE_AUTO_START: bool = False

    CHAT_TEMPLATE: str = "llama3"

    # -------------------------
    # CLI
    # -------------------------

    CLI_DEFAULT_LIMIT: int = 3
    CLI_OUTPUT_FORMAT: str = "table"
    CLI_HISTORY_FILE: str = ".memory_history"
    CLI_SHOW_SCORES: bool = True
    CLI_TABLE_WIDTH: int = 80

    # ---- Plugin System ----
    PLUGIN_ENABLED: bool = True
    PLUGIN_DIR: str = "plugins"
    PLUGIN_AUTO_LOAD: bool = True

    # -------------------------
    # Validation
    # -------------------------

    @field_validator("RANKING_SEMANTIC", "RANKING_IMPORTANCE", "RANKING_RECENCY",
                     "RANKING_TOKEN", "RANKING_FEEDBACK", "RANKING_ENTITY",
                     "RANKING_SUBJECT", "RANKING_ATTRIBUTE", "RANKING_TFIDF")
    @classmethod
    def validate_positive_weights(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Weight must be >= 0, got {v}")
        return v

    def validate_ranking_weights(self) -> bool:
        weights = [
            self.RANKING_SEMANTIC,
            self.RANKING_IMPORTANCE,
            self.RANKING_RECENCY,
            self.RANKING_TOKEN,
            self.RANKING_FEEDBACK,
            self.RANKING_ENTITY,
            self.RANKING_SUBJECT,
            self.RANKING_ATTRIBUTE,
            self.RANKING_TFIDF,
            self.RANKING_BM25,
        ]
        total = sum(weights)
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Ranking weights sum to {total}, expected ~1.0")
        return True

    def model_post_init(self, __context):
        self.validate_ranking_weights()


# -------------------------
# Settings Instance
# -------------------------

settings = Settings()


# -------------------------
# Safety Guard
# -------------------------

class Safety:
    @staticmethod
    def assert_test_mode():
        if os.getenv("ENV") != "test":
            raise RuntimeError(
                "[SAFETY] Refusing destructive operation outside test mode. "
                "Set ENV=test to override."
            )

    @staticmethod
    def assert_not_production():
        if os.getenv("ENV") == "production":
            raise RuntimeError(
                "[SAFETY] Refusing destructive operation in production mode."
            )

    @staticmethod
    def is_test_mode() -> bool:
        return os.getenv("ENV") == "test"

    @staticmethod
    def is_production() -> bool:
        return os.getenv("ENV") == "production"


# -------------------------
# Environment Variable Support
# -------------------------

def load_from_env():
    for key, value in os.environ.items():
        if key.startswith("MEMORY_"):
            setting_name = key[7:]
            if hasattr(settings, setting_name):
                current = getattr(settings, setting_name)
                if isinstance(current, bool):
                    setattr(settings, setting_name, value.lower() in ("true", "1", "yes"))
                elif isinstance(current, int):
                    setattr(settings, setting_name, int(value))
                elif isinstance(current, float):
                    setattr(settings, setting_name, float(value))
                else:
                    setattr(settings, setting_name, value)


# load_from_env()
