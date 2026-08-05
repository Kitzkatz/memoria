from pydantic import BaseModel
import os

class Settings(BaseModel):
    # Paths
    DB_PATH: str = "memory.db"
    VECTOR_INDEX_PATH: str = "memory.index"
    CACHE_PATH: str = "cache/embedding_cache.pkl"
    

    # Models
    EMBEDDING_MODEL: str = "memory/models/all-MiniLM-L6-v2"
    VECTOR_DIM: int = 384
    CHAT_MODEL: str = "mistral"
    LLM_URL: str = "http://localhost:8080"
    LLM_ENDPOINT: str = "/v1/completions"
    LLM_MAX_TOKENS: int = 256
    LLM_TEMPERATURE: float = 0.7
    LLM_TIMEOUT: int = 600
    

    # Retrieval
    TOP_K: int = 300
    TOP_N: int = 3
    GRAPH_TOP_K: int = 50
    GRAPH_SEARCH_LIMIT: int = 200

    # After existing fields
    USE_INVERTED_INDEX: bool = True
    USE_PHRASE_SEARCH: bool = True
    USE_BM25: bool = True
    USE_BLACKBOARD: bool = True
    USE_CASE_FOLDING: bool = True

    # Ranking
    CONTEXT_MAX_MEMORIES: int = 50
    CONTEXT_MIN_SCORE: float = 0.15
    CONTEXT_TOKEN_BUDGET: int = 800

    # Memory
    MEMORY_DECAY_DAYS: int = 30
    MEMORY_DECAY_RATE: float = 0.001
    IMPORTANCE_DELTA: float = 0.01

    # Debug
    DEBUG: bool = False

    # Add to Settings class
    ENTITY_BOOST: float = 0.50

    # Ranking Weights
    RANKING_SEMANTIC: float = 0.20
    RANKING_IMPORTANCE: float = 0.08
    RANKING_RECENCY: float = 0.05
    RANKING_TOKEN: float = 0.07
    RANKING_FEEDBACK: float = 0.02
    RANKING_ENTITY: float = 0.23
    RANKING_SUBJECT: float = 0.20
    RANKING_ATTRIBUTE: float = 0.15

    # Finalizer Weights
    FINALIZER_RELEVANCE: float = 0.25
    FINALIZER_IMPORTANCE: float = 0.10
    FINALIZER_RECENCY: float = 0.10
    FINALIZER_DIVERSITY: float = 0.05
    FINALIZER_ATTRIBUTE: float = 0.50

    # Add these to the existing Settings class
    CLI_DEFAULT_LIMIT: int = 3
    CLI_OUTPUT_FORMAT: str = "table"   # options: "table", "json", "raw"
    CLI_HISTORY_FILE: str = ".memory_history"
    CLI_SHOW_SCORES: bool = True
    CLI_TABLE_WIDTH: int = 80
settings = Settings()





class Safety:
    @staticmethod
    def assert_test_mode():
        if os.getenv("ENV") != "test":
            raise RuntimeError(
                "[SAFETY] Refusing destructive operation outside test mode"
            )
