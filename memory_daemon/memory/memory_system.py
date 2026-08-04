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


from graph.entity_resolver import EntityResolver
from graph.relationship_builder import RelationshipBuilder
import time



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

        self.pipeline = RankingPipeline(
            attribute_map=self.attribute_map
        )
        self.query_processor = QueryProcessor()

        self.entity_store = entity_store
        self.edge_store = EdgeStore(db)
        
        self.entity_resolver = EntityResolver(entity_store)
        self.relationship_builder = RelationshipBuilder(

            self.edge_store,

            self.entity_store,

          

        )
        self.graph_search = GraphSearch(
            self.edge_store,
            
            entity_store
        )

        self.retrieval = RetrievalEngine(
            db,
            vector_store,
            self.embedding_cache,
            self.graph_search
        )

    # -------------------------------------
    # STORE
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
    # STORE MANY
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

        # -----------------------------
        # Extract + Score + Embed
        # -----------------------------

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

        # -----------------------------
        # Insert
        # -----------------------------

        #
        # Insert everything
        #

        ids = self.db.insert_many(records)

        for mem_id, record in zip(ids, records):

            self.relationship_builder.build(

                mem_id,

                record.relationships

            )

        #
        # Cache everything
        #

        self.embedding_cache.add_many(
            ids,
            vectors
        )

        #
        # FAISS everything
        #

        self.vector_store.add_many(
            ids,
            vectors,
            persist=False
        )
        debug("[DB] Insert complete")

        # -----------------------------
        # Save FAISS ONCE
        # -----------------------------

        self.vector_store.save()

        runtime = (
            time.perf_counter()
            - overall_start
        )

        debug(
            f"[COMPLETE] {len(ids)} memories "
            f"in {runtime:.2f}s"
        )

        return ids
    # -------------------------------------
    # QUERY (FIXED INDENTATION)
    # -------------------------------------

    def query(self, text):

        overall_start = time.perf_counter()

        query = self.query_processor.process(text)

        #
        # Embedding
        #

        t0 = time.perf_counter()

        vec = self.embedder.embed(query.normalized_text)

        embedding_ms = (
            time.perf_counter() - t0
        ) * 1000

        #
        # FAISS
        #

        t0 = time.perf_counter()

        ids, distances = self.vector_store.search(vec)

        faiss_ms = (
            time.perf_counter() - t0
        ) * 1000

        #
        # Retrieval (FAISS + graph candidate assembly)
        #

        t0 = time.perf_counter()

        candidates = self.retrieval.retrieve(query, ids, distances)

        database_ms = (
            time.perf_counter() - t0
        ) * 1000

        debug("passing to pipeline:", len(candidates))

        t0 = time.perf_counter()

        results, ranking_diag = self.pipeline.run(
            query,
            candidates
        )

        ranking_ms = (
            time.perf_counter() - t0
        ) * 1000

        t0 = time.perf_counter()

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

        formatting_ms = (
            time.perf_counter() - t0
        ) * 1000

        total_query_ms = (
            time.perf_counter() - overall_start
        ) * 1000

        return {
            "results": response,
            "diagnostics": {
                "candidate_count": len(candidates),
                "returned_count": len(response),
                "embedding_ms": round(embedding_ms, 3),
                "faiss_ms": round(faiss_ms, 3),
                "database_ms": round(database_ms, 3),
                "ranking_ms": round(ranking_ms, 3),
                "before_mmr": ranking_diag.get("before_mmr") if ranking_diag else None,
                "after_mmr": ranking_diag.get("after_mmr") if ranking_diag else None,
                "mmr_changed": ranking_diag.get("mmr_changed") if ranking_diag else False,
                "mmr_moves": ranking_diag.get("mmr_moves") if ranking_diag else 0,
                "formatting_ms": round(formatting_ms, 3),
                "total_query_ms": round(total_query_ms, 3)
            }
        }
    # -------------------------------------
    # EMBEDDING REGISTRATION
    # -------------------------------------

    def register_embedding(self, mem_id, vector):
        debug("\nREGISTER EMBEDDING")
        debug(mem_id)
        debug(len(vector))
        debug(vector[:5])
        self.embedding_cache.add(mem_id, vector)

        self.vector_store.add(mem_id, vector, persist=True)
