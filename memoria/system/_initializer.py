"""
MemorySystem initialization logic.
All the messy dependency setup, extracted from __init__.
"""

import numpy as np

from cache.config import settings
from core.logger import debug

from memory.feedback import FeedbackLoop
from memory.query_history import QueryHistory
from memory.memory_pruner import MemoryPruner
from memory.relevance_manager import RelevanceManager

from ingestion.memory_extractor import MemoryExtractor
from ingestion.code_worker import CodeWorker
from ingestion.pdf_worker import PDFWorker

from ranking.importance_scorer import ImportanceScorer
from ranking.attribute_map import ATTRIBUTE_MAP
from ranking.ranking_pipeline import RankingPipeline
from ranking.bm25_ranker import BM25
from ranking.tfidf import TFIDF

from retrieval.retrieval_engine import RetrievalEngine
from retrieval.query_processor import QueryProcessor
from retrieval.shard_manager import ShardManager
from retrieval.inverted_index import InvertedIndex
from retrieval.query_expander import QueryExpander   # <-- NEW IMPORT
from temporality.temporal_index import TemporalIndex
from blackboard.temporal_worker import TemporalWorker

from graph.search import GraphSearch
from graph.edge_store import EdgeStore
from graph.numpy_graph import NumpyGraph
from graph.entity_resolver import EntityResolver
from graph.relationship_builder import RelationshipBuilder

from cache.embedding_cache import EmbeddingCache

from blackboard.consolidator import Consolidator
from blackboard.core import Blackboard
from blackboard.scheduler import Scheduler
from blackboard.workers import (
    FAISSWorker, BM25Worker, GraphWorker, PhraseWorker, AttributeWorker, FusionWorker,
)

from routing import Router


def initialize_components(system, db, vector_store, embedder, entity_store, llm=None):
    """
    Initialize all components for the MemorySystem.
    Attaches them directly to the system instance.
    """
    # ---- Core components ----
    system.attribute_map = ATTRIBUTE_MAP
    # Pass plugin_manager to MemoryExtractor
    system.extractor = MemoryExtractor(llm, plugin_manager=getattr(system, 'plugin_manager', None))
    system.scorer = ImportanceScorer()
    system.embedding_cache = EmbeddingCache()
    system.query_processor = QueryProcessor()
    system.query_expander = QueryExpander()   # <-- NEW: query expansion
    system.router = Router()
    debug(f"Router initialized with {len(system.router.list_types())} memory types")

    # ---- Graph components ----
    system.entity_store = entity_store
    system.edge_store = EdgeStore(db)
    system.entity_resolver = EntityResolver(entity_store)
    system.relationship_builder = RelationshipBuilder(system.edge_store, system.entity_store)
    system.graph_search = GraphSearch(system.edge_store, entity_store)
    system.retrieval = RetrievalEngine(
        db, vector_store, system.embedding_cache, system.graph_search
    )

    # ---- Pruner ----
    system.pruner = MemoryPruner(
        db=db,
        vector_store=vector_store,
        embedding_cache=system.embedding_cache,
        threshold=getattr(settings, "PRUNE_THRESHOLD", 0.1),
        max_age_days=getattr(settings, "PRUNE_MAX_AGE_DAYS", 365),
        batch_size=getattr(settings, "PRUNE_BATCH_SIZE", 100),
        interval_seconds=getattr(settings, "PRUNE_INTERVAL_SECONDS", 3600),
        auto_start=getattr(settings, "PRUNE_AUTO_START", False),
    )
    debug(f"MemoryPruner initialized (threshold={system.pruner.threshold}, max_age={system.pruner.max_age_days}d)")

    # ---- PDF and Code workers ----
    system.pdf_worker = PDFWorker(system)
    system.code_worker = CodeWorker(system)

    # ---- Sharding ----
    system.shard_manager = ShardManager(
        num_shards=getattr(settings, "NUM_SHARDS", 5)
    )
    debug(f"Shard manager initialized: {system.shard_manager.num_shards} shards (type-based)")

    # ---- Numpy Graph ----
    system.numpy_graph = NumpyGraph(db)
    debug(f"Numpy graph built: {len(system.numpy_graph.entities)} entities, {np.count_nonzero(system.numpy_graph.adj_matrix)} edges")

    # ---- Relevance Manager ----
    system.relevance_manager = RelevanceManager(db, persist_path="relevance_data.json")

    # ---- Feedback Loop ----
    system.feedback = FeedbackLoop(
        db,
        persist_path=getattr(settings, "FEEDBACK_PERSIST_PATH", "feedback_data.json"),
        plugin_manager=getattr(system, 'plugin_manager', None),
    )
    system.query_history = QueryHistory(
        max_history=getattr(settings, "QUERY_HISTORY_MAX", 1000),
        persist_path=getattr(settings, "QUERY_HISTORY_PERSIST_PATH", "query_history.json")
    )

    # ---- TF/IDF ----
    system.tfidf = _build_tfidf(db)

    # ---- Blackboard ----
    _init_blackboard(system, db, vector_store)

    # ---- Ranking Pipeline ----
    system.pipeline = RankingPipeline(
        attribute_map=ATTRIBUTE_MAP,
        db=db,
        bm25_ranker=system.bm25_ranker,
        tfidf_ranker=system.tfidf,
        numpy_graph=system.numpy_graph,
        feedback_loop=system.feedback,
        ranker=None,
        normalizer=None,
        booster=None,
        context_builder=None,
        mmr=None,
        finalizer=None,
        plugin_manager=getattr(system, 'plugin_manager', None),
    )

    # ---- Propagate plugin manager to router and scheduler ----
    if hasattr(system, 'plugin_manager') and system.plugin_manager:
        if hasattr(system, 'router') and system.router:
            system.router.plugin_manager = system.plugin_manager
        if hasattr(system, 'scheduler') and system.scheduler:
            system.scheduler.plugin_manager = system.plugin_manager


def _build_tfidf(db):
    """Build TF/IDF from existing memories."""
    tfidf = TFIDF()
    corpus_tokens = []
    mem_types = ["semantic", "episodic", "procedural", "code", "science"]
    for mem_type in mem_types:
        rows = db.fetch_many_by_type(mem_type, limit=5000)
        for row in rows:
            tokens = row.get("tokens", [])
            if tokens:
                corpus_tokens.append(tokens)
    if corpus_tokens:
        tfidf.build(corpus_tokens)
        debug(f"TF/IDF built on {len(corpus_tokens)} memories")
        return tfidf
    else:
        debug("TF/IDF: No memories found, skipping")
        return None


def _init_blackboard(system, db, vector_store):
    """Initialize blackboard, scheduler, and workers."""
    use_blackboard = getattr(settings, "USE_BLACKBOARD", False)
    system.use_blackboard = use_blackboard
    system.bm25_ranker = None
    system.inverted_index = None
    system.blackboard = None
    system.scheduler = None
    system.consolidator = Consolidator(db, vector_store, system.embedding_cache)

    if not use_blackboard:
        return

    blackboard = Blackboard()
    scheduler = Scheduler(blackboard)

    # Build BM25
    bm25_ranker = None
    if getattr(settings, "USE_BM25", False):
        bm25_ranker = BM25()
        memories = db.fetch_all()
        corpus = [m["tokens"] for m in memories if m.get("tokens")]
        bm25_ranker.build(corpus)
        debug(f"BM25 built on {len(corpus)} memories")

    # Build inverted index
    inverted_index = None
    if getattr(settings, "USE_INVERTED_INDEX", False):
        inverted_index = InvertedIndex(db)
        inverted_index.build()

    # Create base workers
    faiss_worker = FAISSWorker(vector_store)
    bm25_worker = BM25Worker(bm25_ranker, inverted_index) if bm25_ranker else None
    graph_worker = GraphWorker(system.numpy_graph)
    attribute_worker = AttributeWorker(db)

    # Register base workers
    scheduler.register_worker("attribute", attribute_worker.process)
    scheduler.register_worker("faiss", faiss_worker.process)
    if bm25_worker:
        scheduler.register_worker("bm25", bm25_worker.process)
    scheduler.register_worker("graph", graph_worker.process)

    if getattr(settings, "USE_INVERTED_INDEX", False) and inverted_index:
        phrase_worker = PhraseWorker(inverted_index)
        scheduler.register_worker("phrase", phrase_worker.process)

    # ---- Fusion worker (if enabled) ----
    use_fusion = getattr(settings, "USE_FUSION", False)
    if use_fusion and bm25_worker:
        semantic_weight = getattr(settings, "FUSION_SEMANTIC_WEIGHT", 0.5)
        fusion_worker = FusionWorker(faiss_worker, bm25_worker, semantic_weight)
        scheduler.register_worker("fusion", fusion_worker.process)
        debug(f"FusionWorker registered (semantic_weight={semantic_weight})")
    elif use_fusion:
        debug("FusionWorker skipped: BM25 is not available")

    # ---- Temporal Worker (NEW) ----
    if getattr(settings, "USE_TEMPORAL_WORKER", False):
        

        # Initialize temporal index
        temporal_index = TemporalIndex(
            db,
            cache_path=getattr(settings, "TEMPORAL_INDEX_PATH", "cache/temporal_index.json")
        )
        temporal_index.load()
        temporal_index.build()

        # Create and register temporal worker
        temporal_worker = TemporalWorker(db, temporal_index)
        scheduler.register_worker("temporal", temporal_worker.process)
        debug("TemporalWorker registered")

        # Store on system for cleanup/saving
        system.temporal_index = temporal_index

    system.blackboard = blackboard
    system.scheduler = scheduler
    system.bm25_ranker = bm25_ranker
    system.inverted_index = inverted_index
