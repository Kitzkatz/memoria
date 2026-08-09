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

    TOP_K: int = 500
    TOP_N: int = 3
    GRAPH_TOP_K: int = 50
    GRAPH_SEARCH_LIMIT: int = 200
    GRAPH_DEPTH: int = 3
    GRAPH_SEARCH_DEPTH: int = 1  # For retrieval_engine

    # -------------------------
    # Indexing
    # -------------------------

    USE_INVERTED_INDEX: bool = True
    USE_PHRASE_SEARCH: bool = True
    USE_BM25: bool = True
    # ---- MMR ----
    MMR_ENABLED: bool = False  # Set to False to disable MMR entirely
    USE_BLACKBOARD: bool = True
    USE_CASE_FOLDING: bool = True

    # -------------------------
    # Routing
    # -------------------------

    USE_ROUTING: bool = True
    ROUTING_MATRIX_OVERRIDE: bool = False
    ROUTING_FALLBACK_ENABLED: bool = True

    # -------------------------
    # Ranking
    # -------------------------

    CONTEXT_MAX_MEMORIES: int = 50
    CONTEXT_MIN_SCORE: float = 0.15
    CONTEXT_TOKEN_BUDGET: int = 800

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
    TOP_K_PER_SHARD: int = 150

    # -------------------------
    # Debug
    # -------------------------

    DEBUG: bool = False


    # ---- Ranking Diagnostics ----
    # Set to False to skip expensive diagnostics in the ranker

    
    RANKER_DIAGNOSTICS: bool = False  # Default: off for performance


    # -------------------------
    # Boosting
    # -------------------------

    ENTITY_BOOST: float = 0.50

    # -------------------------
    # Adaptive Weighter
    # -------------------------

    USE_ADAPTIVE_WEIGHTS: bool = True
    ADAPTIVE_WEIGHT_STEP: float = 0.02
    ADAPTIVE_WEIGHT_MAX: float = 0.40
    ADAPTIVE_WEIGHT_MIN: float = 0.01

    # -------------------------
    # Ranking Weights (must sum to ~1.0)
    # -------------------------


    RANKING_SEMANTIC: float = 0.2100
    RANKING_IMPORTANCE: float = 0.0480
    RANKING_RECENCY: float = 0.0260
    RANKING_TOKEN: float = 0.0720
    RANKING_FEEDBACK: float = 0.0260
    RANKING_ENTITY: float = 0.1840
    RANKING_SUBJECT: float = 0.1600
    RANKING_ATTRIBUTE: float = 0.1200
    RANKING_TFIDF: float = 0.0640
    RANKING_BM25: float = 0.0900
    # Total: 1.0000

    # -------------------------
    # Finalizer Weights
    # -------------------------

    FINALIZER_RELEVANCE: float = 0.25
    FINALIZER_IMPORTANCE: float = 0.10
    FINALIZER_RECENCY: float = 0.10
    FINALIZER_DIVERSITY: float = 0.05
    FINALIZER_ATTRIBUTE: float = 0.50
    FINALIZER_BM25: float = 0.10

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


    # Chat template: "llama3", "chatml", "simple"
    CHAT_TEMPLATE: str = "llama3"

    # -------------------------
    # CLI
    # -------------------------

    CLI_DEFAULT_LIMIT: int = 3
    CLI_OUTPUT_FORMAT: str = "table"  # table, json, raw
    CLI_HISTORY_FILE: str = ".memory_history"
    CLI_SHOW_SCORES: bool = True
    CLI_TABLE_WIDTH: int = 80

    # -------------------------
    # Validation
    # -------------------------

    @field_validator("RANKING_SEMANTIC", "RANKING_IMPORTANCE", "RANKING_RECENCY",
                     "RANKING_TOKEN", "RANKING_FEEDBACK", "RANKING_ENTITY",
                     "RANKING_SUBJECT", "RANKING_ATTRIBUTE", "RANKING_TFIDF")
    @classmethod
    def validate_positive_weights(cls, v: float) -> float:
        """Ensure weights are positive."""
        if v < 0:
            raise ValueError(f"Weight must be >= 0, got {v}")
        return v

    @field_validator("RANKING_SEMANTIC", "RANKING_IMPORTANCE", "RANKING_RECENCY",
                     "RANKING_TOKEN", "RANKING_FEEDBACK", "RANKING_ENTITY",
                     "RANKING_SUBJECT", "RANKING_ATTRIBUTE", "RANKING_TFIDF")
    @classmethod
    def validate_weights_sum(cls, values: dict) -> dict:
        """Ensure ranking weights sum to approximately 1.0."""
        # Note: This validator runs on individual fields, not the whole model.
        # We'll validate in a separate method.
        return values

    def validate_ranking_weights(self) -> bool:
        """Check that ranking weights sum to ~1.0."""
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
        """Run validation after initialization."""
        self.validate_ranking_weights()


# -------------------------
# Settings Instance
# -------------------------

settings = Settings()


# -------------------------
# Safety Guard
# -------------------------

class Safety:
    """Safety utilities for preventing destructive operations in production."""

    @staticmethod
    def assert_test_mode():
        """Raise an error if not in test mode."""
        if os.getenv("ENV") != "test":
            raise RuntimeError(
                "[SAFETY] Refusing destructive operation outside test mode. "
                "Set ENV=test to override."
            )

    @staticmethod
    def assert_not_production():
        """Raise an error if in production mode."""
        if os.getenv("ENV") == "production":
            raise RuntimeError(
                "[SAFETY] Refusing destructive operation in production mode."
            )

    @staticmethod
    def is_test_mode() -> bool:
        """Return True if in test mode."""
        return os.getenv("ENV") == "test"

    @staticmethod
    def is_production() -> bool:
        """Return True if in production mode."""
        return os.getenv("ENV") == "production"


# -------------------------
# Environment Variable Support
# -------------------------

def load_from_env():
    """
    Override settings from environment variables.
    Prefix with "MEMORY_" (e.g., MEMORY_TOP_K=100).
    """
    for key, value in os.environ.items():
        if key.startswith("MEMORY_"):
            setting_name = key[7:]  # Remove "MEMORY_"
            if hasattr(settings, setting_name):
                # Try to convert to appropriate type
                current = getattr(settings, setting_name)
                if isinstance(current, bool):
                    setattr(settings, setting_name, value.lower() in ("true", "1", "yes"))
                elif isinstance(current, int):
                    setattr(settings, setting_name, int(value))
                elif isinstance(current, float):
                    setattr(settings, setting_name, float(value))
                else:
                    setattr(settings, setting_name, value)


# Auto-load from environment if desired
# Uncomment to enable:
# load_from_env()
