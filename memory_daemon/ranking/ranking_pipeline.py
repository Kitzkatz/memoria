from core.logger import debug
from ranking.memory_ranker import MemoryRanker
from ranking.score_normalizer import ScoreNormalizer
from ranking.attribute_booster import AttributeBooster
from ranking.mmr_reranker import MMRReranker
from core.context_builder import ContextBuilder
from ranking.score_finalizer import ScoreFinalizer
from cache.config import settings
from ranking.bm25_ranker import BM25
import time


class RankingPipeline:
    """
    Ranking Orchestrator
    This class contains ZERO retrieval logic.
    It assumes candidate memories already exist.
    Every stage receives and returns the same list
    of CandidateRecord objects.
    """

    def __init__(
        self,
        attribute_map,
        db=None,
        bm25_ranker=None,
        tfidf_ranker=None,
        numpy_graph=None,
        feedback_loop=None,
        ranker=None,
        normalizer=None,
        booster=None,
        context_builder=None,
        mmr=None,
        finalizer=None,
    ):
        self.ranker = ranker or MemoryRanker(
            tfidf_ranker=tfidf_ranker,
            feedback_loop=feedback_loop,
            numpy_graph=numpy_graph
        )
        self.normalizer = normalizer or ScoreNormalizer()

        if booster is None:
            self.booster = AttributeBooster(
                attribute_map=attribute_map,
                boost_value=0.15,
                entity_boost=getattr(settings, "ENTITY_BOOST", 0.50)
            )
        else:
            self.booster = booster

        self.context_builder = context_builder or ContextBuilder()
        self.mmr = mmr or MMRReranker(lambda_param=0.5)
        self.finalizer = finalizer or ScoreFinalizer()

        # --- BM25 Setup ---
        self.bm25 = None
        self.id_to_idx = {}

        if bm25_ranker is not None:
            self.bm25 = bm25_ranker
            self._build_id_to_idx(db)
            debug(f"Using provided BM25 ranker with {len(self.id_to_idx)} memories")
        elif getattr(settings, "USE_BM25", False) and db is not None:
            all_memories = db.fetch_all()
            if all_memories:
                corpus_tokens = [m["tokens"] for m in all_memories if m.get("tokens")]
                if corpus_tokens:
                    self.bm25 = BM25()
                    self.bm25.build(corpus_tokens)
                    self._build_id_to_idx(db)
                    debug(f"BM25 built on {len(corpus_tokens)} memories")
                else:
                    debug("BM25: No tokenized memories found")
            else:
                debug("BM25: No memories found in DB")

    def _build_id_to_idx(self, db):
        """Build mapping from memory ID to BM25 index position."""
        if db is None:
            return
        all_memories = db.fetch_all()
        bm25_rows = [m for m in all_memories if m.get("tokens")]
        self.id_to_idx = {m["id"]: idx for idx, m in enumerate(bm25_rows)}

    # ----------------------------------------------------
    # Ranking Pipeline
    # ----------------------------------------------------

    def run(self, query, candidates):
        diagnostics = {}
        t_total_start = time.perf_counter()

        # --- Ranker ---
        t0 = time.perf_counter()
        candidates = self.ranker.rank(candidates, query)
        t_rank = (time.perf_counter() - t0) * 1000
        debug(f"[Pipeline] rank: {t_rank:.2f}ms, {len(candidates)} candidates")
        diagnostics["rank_count"] = len(candidates)
        diagnostics["rank_ms"] = round(t_rank, 2)

        # --- Normalizer ---
        t0 = time.perf_counter()
        candidates = self.normalizer.normalize(candidates)
        t_normalize = (time.perf_counter() - t0) * 1000
        debug(f"[Pipeline] normalize: {t_normalize:.2f}ms")

        # --- Booster ---
        t0 = time.perf_counter()
        candidates = self.booster.boost(query, candidates)
        t_boost = (time.perf_counter() - t0) * 1000
        debug(f"[Pipeline] booster: {t_boost:.2f}ms")

        # --- BM25 Scoring (Optimized: only score candidates) ---
        t0 = time.perf_counter()
        if self.bm25 is not None and getattr(settings, "USE_BM25", False):
            query_tokens = query.tokens
            if query_tokens and candidates:
                candidate_ids = [c.memory.id for c in candidates]
                scores = self.bm25.score_ids(query_tokens, candidate_ids)
                for candidate in candidates:
                    candidate.bm25_score = scores.get(candidate.memory.id, 0.0)
                    candidate.diagnostics["bm25_score"] = candidate.bm25_score
            else:
                for candidate in candidates:
                    candidate.bm25_score = 0.0
                    candidate.diagnostics["bm25_score"] = 0.0
        t_bm25 = (time.perf_counter() - t0) * 1000
        debug(f"[Pipeline] bm25: {t_bm25:.2f}ms")

        diagnostics["boosted_top"] = [
            {"id": c.memory.id, "score": c.normalized_score}
            for c in candidates[:5]
        ]

        # --- Finalizer ---
        t0 = time.perf_counter()
        candidates = self.finalizer.finalize(candidates)
        t_finalize = (time.perf_counter() - t0) * 1000
        debug(f"[Pipeline] finalizer: {t_finalize:.2f}ms")

        # --- Context Builder ---
        t0 = time.perf_counter()
        candidates = self.context_builder.build(candidates)
        t_context = (time.perf_counter() - t0) * 1000
        debug(f"[Pipeline] context_builder: {t_context:.2f}ms, {len(candidates)} candidates")

        # --- MMR ---
        t0 = time.perf_counter()
        before_mmr = [c.memory.id for c in candidates]
        candidates = self.mmr.rerank(
            candidates,
            k=self.context_builder.max_memories
        )
        t_mmr = (time.perf_counter() - t0) * 1000
        debug(f"[Pipeline] mmr: {t_mmr:.2f}ms")

        after_mmr = [c.memory.id for c in candidates]

        diagnostics["before_mmr"] = before_mmr
        diagnostics["after_mmr"] = after_mmr
        diagnostics["mmr_changed"] = before_mmr != after_mmr
        diagnostics["mmr_moves"] = sum(
            1 for a, b in zip(before_mmr, after_mmr) if a != b
        )
        diagnostics["mmr_top"] = [
            {"id": c.memory.id, "score": c.final_score}
            for c in candidates[:5]
        ]

        t_total = (time.perf_counter() - t_total_start) * 1000
        debug(f"[Pipeline] TOTAL pipeline time: {t_total:.2f}ms")
        diagnostics["pipeline_total_ms"] = round(t_total, 2)

        return candidates, diagnostics
