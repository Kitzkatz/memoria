from ingestion.memory_extractor import MemoryExtractor
from ranking.importance_scorer import ImportanceScorer
from retrieval.retrieval_engine import RetrievalEngine
from ranking.memory_ranker import MemoryRanker
from core.logger import debug
from core.context_builder import ContextBuilder
from cache.embedding_cache import EmbeddingCache
from ranking.mmr_reranker import MMRReranker
from ranking.attribute_booster import AttributeBooster
from ranking.score_normalizer import ScoreNormalizer
from retrieval.query_processor import QueryProcessor
from ranking.attribute_map import ATTRIBUTE_MAP
from ranking.ranking_pipeline import RankingPipeline
from graph.search import GraphSearch
from graph.edge_store import EdgeStore
from cache.config import settings
from ranking.models import CandidateRecord

from graph.entity_resolver import EntityResolver
from graph.relationship_builder import RelationshipBuilder
import time

# V4 blackboard imports
from blackboard.core import Blackboard, BlackboardEntry
from blackboard.scheduler import Scheduler
from blackboard.workers import (
    FAISSWorker, BM25Worker, GraphWorker, RankingWorker,
    PhraseWorker   # <--- added
)

class MemorySystem:

    def __init__(self, db, vector_store, embedder, entity_store, llm=None):

        self.db = db
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm
        
        self.attribute_map = ATTRIBUTE_MAP

        self.extractor = MemoryExtractor(llm)
        self.scorer = ImportanceScorer()

        self.embedding_cache = EmbeddingCache()

        # -------------------------
        # V4 Blackboard Integration
        # -------------------------
        self.use_blackboard = getattr(settings, "USE_BLACKBOARD", False)
        self.bm25_ranker = None
        self.inverted_index = None

        if self.use_blackboard:
            # Initialize blackboard and scheduler
            self.blackboard = Blackboard()
            self.scheduler = Scheduler(self.blackboard)

            # Build BM25 ranker and inverted index (if toggles enabled)
            if getattr(settings, "USE_BM25", False):
                from ranking.bm25_ranker import BM25
                self.bm25_ranker = BM25()
                memories = self.db.fetch_all()
                corpus = [m["tokens"] for m in memories if m.get("tokens")]
                self.bm25_ranker.build(corpus)
                debug(f"BM25 built on {len(corpus)} memories")

            if getattr(settings, "USE_INVERTED_INDEX", False):
                from retrieval.inverted_index import InvertedIndex
                self.inverted_index = InvertedIndex(self.db)
                self.inverted_index.build()

            # Create workers
            faiss_worker = FAISSWorker(self.blackboard, self.vector_store)
            bm25_worker = BM25Worker(self.blackboard, self.bm25_ranker, self.inverted_index) if self.bm25_ranker else None
            graph_worker = GraphWorker(self.blackboard, self.graph_search, self.entity_store)
            ranking_worker = RankingWorker(self.blackboard, None, self.bm25_ranker)  # pipeline passed later

            # Register workers
            self.scheduler.register_worker("faiss", faiss_worker)
            if bm25_worker:
                self.scheduler.register_worker("bm25", bm25_worker)
            self.scheduler.register_worker("graph", graph_worker)
            self.scheduler.register_worker("ranking", ranking_worker)

            # Register PhraseWorker if inverted index is available
            if getattr(settings, "USE_INVERTED_INDEX", False) and self.inverted_index:
                phrase_worker = PhraseWorker(self.blackboard, self.inverted_index)
                self.scheduler.register_worker("phrase", phrase_worker)

            # Start scheduler
            self.scheduler.start()
        else:
            self.blackboard = None
            self.scheduler = None

        # Initialize ranking pipeline (always needed)
        self.pipeline = RankingPipeline(
            attribute_map=self.attribute_map,
            db=self.db,
            bm25_ranker=self.bm25_ranker,
            ranker=None,
            normalizer=None,
            booster=None,
            context_builder=None,
            mmr=None,
            finalizer=None,
        )

        # Other components (kept outside blackboard branch)
        self.query_processor = QueryProcessor()
        self.entity_store = entity_store
        self.edge_store = EdgeStore(db)
        self.entity_resolver = EntityResolver(entity_store)
        self.relationship_builder = RelationshipBuilder(self.edge_store, self.entity_store)
        self.graph_search = GraphSearch(self.edge_store, entity_store)
        self.retrieval = RetrievalEngine(
            db,
            vector_store,
            self.embedding_cache,
            self.graph_search
        )

    # -------------------------------------
    # STORE (unchanged)
    # -------------------------------------

    def store(self, text):
        import time
        t0 = time.perf_counter()
        record = self.extractor.extract(text)
        debug("extract:", time.perf_counter() - t0)

        debug("\n[TEST EXTRACTOR OUTPUT]")
        debug("TEXT:", record.text)
        debug("TYPE:", record.memory_type)
        debug("META:", record.metadata)
        debug("IMPORTANCE:", record.importance)
        debug("TEXT:", record.text)
        debug("NORMALIZED:", record.normalized_text)
        debug("TOKENS:", record.tokens)
        debug("TOKEN COUNT:", record.token_count)
        t0 = time.perf_counter()

        record.importance = self.scorer.score(
            record.text,
            record.metadata
        )
        debug("score:", time.perf_counter() - t0)

        t0 = time.perf_counter()
        debug(id(self.vector_store))
        vec = self.embedder.embed(record.normalized_text)
        debug("embed:", time.perf_counter() - t0)

        t0 = time.perf_counter()
        mem_id = self.db.insert(record)
        self.relationship_builder.build(
            mem_id,
            record.relationships
        )
        debug("\n[TEST DB INSERT]")
        row = self.db.fetch(mem_id)
        debug(f"[CACHE] {self.embedding_cache.count()} vectors")
        debug(row)
        debug("db:", time.perf_counter() - t0)

        self.register_embedding(mem_id, vec)

        t0 = time.perf_counter()
        debug("\n[TEST FAISS SYNC]")
        debug("FAISS COUNT:", self.vector_store.count())
        debug("DB COUNT:", self.db.count())
        debug("faiss:", time.perf_counter() - t0)

        debug(f"Stored memory {mem_id}")
        return mem_id

    # -------------------------------------
    # STORE MANY (unchanged)
    # -------------------------------------

    def store_many(self, texts):
        import time
        overall_start = time.perf_counter()
        total = len(texts)

        debug()
        debug("=" * 60)
        debug("[STORE MANY]")
        debug("=" * 60)
        debug(f"Loading {total} memories")

        records = []
        vectors = []

        for text in texts:
            record = self.extractor.extract(text)
            record.importance = self.scorer.score(
                record.text,
                record.metadata
            )
            vector = self.embedder.embed(
                record.normalized_text
            )
            records.append(record)
            vectors.append(vector)
        debug(f"[READY] {len(records)} records")

        ids = self.db.insert_many(records)
        for mem_id, record in zip(ids, records):
            self.relationship_builder.build(
                mem_id,
                record.relationships
            )

        self.embedding_cache.add_many(
            ids,
            vectors
        )
        self.vector_store.add_many(
            ids,
            vectors,
            persist=False
        )
        debug("[DB] Insert complete")
        self.vector_store.save()

        runtime = time.perf_counter() - overall_start
        debug(
            f"[COMPLETE] {len(ids)} memories "
            f"in {runtime:.2f}s"
        )
        return ids

    # -------------------------------------
    # QUERY — with Blackboard + Phrase support
    # -------------------------------------

    def query(self, text):
        overall_start = time.perf_counter()
        query = self.query_processor.process(text)

        # Embedding
        t0_embed = time.perf_counter()
        vec = self.embedder.embed(query.normalized_text)
        embedding_ms = (time.perf_counter() - t0_embed) * 1000

        if self.use_blackboard and self.scheduler:
            # ---------- Blackboard Path ----------
            t0_faiss = time.perf_counter()
            self.scheduler.submit("faiss", {"vector": vec, "top_k": settings.TOP_K})
            if self.bm25_ranker:
                self.scheduler.submit("bm25", {"tokens": query.tokens})
            self.scheduler.submit("graph", {"entities": query.entities})

            # Submit phrase search if phrases exist
            phrases = query.metadata.get("phrases", [])
            if phrases and self.inverted_index:
                self.scheduler.submit("phrase", {"phrases": phrases})

            # Allow workers to process (simple polling)
            time.sleep(0.1)

            # Collect all candidates from blackboard
            entries = self.blackboard.get_entries(type="candidates")
            candidates = []
            existing_ids = set()

            for entry in entries:
                source = entry.content.get("source")
                cand_list = entry.content.get("candidates", [])
                if source == "faiss":
                    for mem_id, dist in cand_list:
                        if mem_id not in existing_ids:
                            row = self.db.fetch(mem_id)
                            if row:
                                memory = self.retrieval._build_memory_record(row)
                                embedding = self.embedding_cache.get(mem_id) or self.vector_store.get(mem_id)
                                candidates.append(
                                    CandidateRecord(
                                        memory=memory,
                                        distance=float(dist),
                                        embedding=embedding,
                                        graph_hit=False
                                    )
                                )
                                existing_ids.add(mem_id)
                elif source == "bm25":
                    for doc_id, score in cand_list:
                        if doc_id not in existing_ids:
                            row = self.db.fetch(doc_id)
                            if row:
                                memory = self.retrieval._build_memory_record(row)
                                pseudo_dist = 1.0 / (score + 1e-6)
                                embedding = self.embedding_cache.get(doc_id) or self.vector_store.get(doc_id)
                                candidates.append(
                                    CandidateRecord(
                                        memory=memory,
                                        distance=pseudo_dist,
                                        embedding=embedding,
                                        graph_hit=False
                                    )
                                )
                                existing_ids.add(doc_id)
                elif source == "graph":
                    for mem_id in cand_list:
                        if mem_id not in existing_ids:
                            row = self.db.fetch(mem_id)
                            if row:
                                memory = self.retrieval._build_memory_record(row)
                                embedding = self.embedding_cache.get(mem_id) or self.vector_store.get(mem_id)
                                candidates.append(
                                    CandidateRecord(
                                        memory=memory,
                                        distance=0.0,
                                        embedding=embedding,
                                        graph_hit=True
                                    )
                                )
                                existing_ids.add(mem_id)
                elif source == "phrase":
                    for doc_id, score in cand_list:
                        if doc_id not in existing_ids:
                            row = self.db.fetch(doc_id)
                            if row:
                                memory = self.retrieval._build_memory_record(row)
                                embedding = self.embedding_cache.get(doc_id) or self.vector_store.get(doc_id)
                                candidates.append(
                                    CandidateRecord(
                                        memory=memory,
                                        distance=0.0,
                                        embedding=embedding,
                                        graph_hit=False
                                    )
                                )
                                existing_ids.add(doc_id)

            faiss_ms = (time.perf_counter() - t0_faiss) * 1000
            database_ms = faiss_ms  # approximate, since we don't separate DB fetch time

            t0_rank = time.perf_counter()
            results, ranking_diag = self.pipeline.run(query, candidates)
            ranking_ms = (time.perf_counter() - t0_rank) * 1000

            response = []
            for rank, candidate in enumerate(results, start=1):
                response.append({
                    "rank": rank,
                    "id": candidate.memory.id,
                    "text": candidate.memory.text,
                    "normalized_text": candidate.memory.normalized_text,
                    "memory_type": candidate.memory.memory_type,
                    "importance": candidate.memory.importance,
                    "distance": candidate.distance,
                    "score": candidate.normalized_score,
                    "final_score": candidate.final_score,
                    "created_at": candidate.memory.created_at.isoformat(),
                    "last_accessed": candidate.memory.last_accessed.isoformat(),
                    "token_count": candidate.memory.token_count,
                    "tokens": candidate.memory.tokens,
                    "metadata": candidate.memory.metadata,
                    "entities": candidate.memory.entities,
                    "relationships": candidate.memory.relationships,
                    "graph_hit": candidate.graph_hit,
                    "diagnostics": candidate.diagnostics,
                    "mmr_score": candidate.mmr_score,
                    "diversity": candidate.diversity_score,
                })

            formatting_ms = (time.perf_counter() - t0_rank) * 1000
            total_query_ms = (time.perf_counter() - overall_start) * 1000

            return {
                "results": response,
                "diagnostics": {
                    "candidate_count": len(candidates),
                    "returned_count": len(response),
                    "embedding_ms": round(embedding_ms, 3),
                    "faiss_ms": round(faiss_ms, 3),
                    "database_ms": round(database_ms, 3),
                    "ranking_ms": round(ranking_ms, 3),
                    "before_mmr": ranking_diag.get("before_mmr"),
                    "after_mmr": ranking_diag.get("after_mmr"),
                    "mmr_changed": ranking_diag.get("mmr_changed", False),
                    "mmr_moves": ranking_diag.get("mmr_moves", 0),
                    "formatting_ms": round(formatting_ms, 3),
                    "total_query_ms": round(total_query_ms, 3)
                }
            }

        else:
            # ---------- V3 Fallback Path ----------
            t0_faiss = time.perf_counter()
            ids, distances = self.vector_store.search(vec)
            faiss_ms = (time.perf_counter() - t0_faiss) * 1000

            t0_db = time.perf_counter()
            candidates = self.retrieval.retrieve(query, ids, distances)
            database_ms = (time.perf_counter() - t0_db) * 1000

            debug("passing to pipeline:", len(candidates))

            t0_rank = time.perf_counter()
            results, ranking_diag = self.pipeline.run(query, candidates)
            ranking_ms = (time.perf_counter() - t0_rank) * 1000

            response = []
            for rank, candidate in enumerate(results, start=1):
                response.append({
                    "rank": rank,
                    "id": candidate.memory.id,
                    "text": candidate.memory.text,
                    "normalized_text": candidate.memory.normalized_text,
                    "memory_type": candidate.memory.memory_type,
                    "importance": candidate.memory.importance,
                    "distance": candidate.distance,
                    "score": candidate.normalized_score,
                    "final_score": candidate.final_score,
                    "created_at": candidate.memory.created_at.isoformat(),
                    "last_accessed": candidate.memory.last_accessed.isoformat(),
                    "token_count": candidate.memory.token_count,
                    "tokens": candidate.memory.tokens,
                    "metadata": candidate.memory.metadata,
                    "entities": candidate.memory.entities,
                    "relationships": candidate.memory.relationships,
                    "graph_hit": candidate.graph_hit,
                    "diagnostics": candidate.diagnostics,
                    "mmr_score": candidate.mmr_score,
                    "diversity": candidate.diversity_score,
                })

            formatting_ms = (time.perf_counter() - t0_rank) * 1000
            total_query_ms = (time.perf_counter() - overall_start) * 1000

            return {
                "results": response,
                "diagnostics": {
                    "candidate_count": len(candidates),
                    "returned_count": len(response),
                    "embedding_ms": round(embedding_ms, 3),
                    "faiss_ms": round(faiss_ms, 3),
                    "database_ms": round(database_ms, 3),
                    "ranking_ms": round(ranking_ms, 3),
                    "before_mmr": ranking_diag.get("before_mmr"),
                    "after_mmr": ranking_diag.get("after_mmr"),
                    "mmr_changed": ranking_diag.get("mmr_changed", False),
                    "mmr_moves": ranking_diag.get("mmr_moves", 0),
                    "formatting_ms": round(formatting_ms, 3),
                    "total_query_ms": round(total_query_ms, 3)
                }
            }

    # -------------------------------------
    # EMBEDDING REGISTRATION (unchanged)
    # -------------------------------------

    def register_embedding(self, mem_id, vector):
        debug("\nREGISTER EMBEDDING")
        debug(mem_id)
        debug(len(vector))
        debug(vector[:5])
        self.embedding_cache.add(mem_id, vector)
        self.vector_store.add(mem_id, vector, persist=True)
