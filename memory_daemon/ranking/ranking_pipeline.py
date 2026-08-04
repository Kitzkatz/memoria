from core.logger import debug
from ranking.memory_ranker import MemoryRanker
from ranking.score_normalizer import ScoreNormalizer
from ranking.attribute_booster import AttributeBooster
from ranking.mmr_reranker import MMRReranker
from core.context_builder import ContextBuilder
from ranking.score_finalizer import ScoreFinalizer
from cache.config import settings
from ranking.bm25_ranker import BM25   # ← new import


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
        db=None,                     # ← new param
        ranker=None,
        normalizer=None,
        booster=None,
        context_builder=None,
        mmr=None,
        finalizer=None,
    ):
        self.ranker = ranker or MemoryRanker()
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
        if getattr(settings, "USE_BM25", False) and db is not None:
            all_memories = db.fetch_all()
            if all_memories:
                corpus_tokens = [m["tokens"] for m in all_memories if m.get("tokens")]
                if corpus_tokens:
                    self.bm25 = BM25()
                    self.bm25.build(corpus_tokens)
                    self.id_to_idx = {m["id"]: idx for idx, m in enumerate(all_memories)}
                    debug(f"BM25 built on {len(corpus_tokens)} memories")
                else:
                    debug("BM25: No tokenized memories found")
            else:
                debug("BM25: No memories found in DB")

    # ----------------------------------------------------
    # Ranking Pipeline
    # ----------------------------------------------------

    def run(self, query, candidates):

        diagnostics = {}

        debug("After fetch:", len(candidates))

        candidates = self.ranker.rank(candidates, query)
        debug("After rank:", len(candidates))
        diagnostics["rank_count"] = len(candidates)

        candidates = self.normalizer.normalize(candidates)
        debug("After normalize:", len(candidates))

        candidates = self.booster.boost(query, candidates)
        debug("After booster:", len(candidates))

        # --- BM25 Scoring (after booster, before context) ---
        if self.bm25 is not None and getattr(settings, "USE_BM25", False):
            query_tokens = query.tokens
            if query_tokens:
                scores = self.bm25.get_scores(query_tokens)
                for candidate in candidates:
                    idx = self.id_to_idx.get(candidate.memory.id)
                    if idx is not None and idx < len(scores):
                        candidate.bm25_score = scores[idx]
                    else:
                        candidate.bm25_score = 0.0
                    candidate.diagnostics["bm25_score"] = candidate.bm25_score
            else:
                for candidate in candidates:
                    candidate.bm25_score = 0.0
                    candidate.diagnostics["bm25_score"] = 0.0

        diagnostics["boosted_top"] = [
            {"id": c.memory.id, "score": c.normalized_score}
            for c in candidates[:5]
        ]

        candidates = self.context_builder.build(candidates)

        before_mmr = [c.memory.id for c in candidates]

        candidates = self.mmr.rerank(
            candidates,
            k=self.context_builder.max_memories
        )

        candidates = self.finalizer.finalize(candidates)
        debug("After finalizer:", len(candidates))

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

        debug("After mmr:", len(candidates))

        return candidates, diagnostics
