from datetime import datetime
from collections import defaultdict
import math
from cache.config import settings


class MemoryRanker:

    def __init__(self):

        self.rank_feedback = defaultdict(float)

        

        # In __init__:
        self.weights = {
            "semantic": getattr(settings, "RANKING_SEMANTIC", 0.20),
            "importance": getattr(settings, "RANKING_IMPORTANCE", 0.08),
            "recency": getattr(settings, "RANKING_RECENCY", 0.05),
            "token": getattr(settings, "RANKING_TOKEN", 0.07),
            "feedback": getattr(settings, "RANKING_FEEDBACK", 0.02),
            "entity": getattr(settings, "RANKING_ENTITY", 0.23),
            "subject": getattr(settings, "RANKING_SUBJECT", 0.20),
            "attribute": getattr(settings, "RANKING_ATTRIBUTE", 0.15)
        }

    


    # ---------------------------------
    # Recency
    # ---------------------------------

    def recency_score(
        self,
        created_at,
        decay_days=30
    ):

        try:

            created = datetime.fromisoformat(created_at)

            age = max(
                0,
                (datetime.utcnow() - created).days
            )

            return math.exp(
                -age / decay_days
            )

        except Exception:

            return 0.5


    # ---------------------------------
    # Token similarity
    # ---------------------------------

    def token_overlap(
        self,
        query_tokens,
        memory_tokens
    ):

        if not query_tokens or not memory_tokens:

            return 0.0


        q = set(query_tokens)

        m = set(memory_tokens)


        intersection = len(q & m)

        union = len(q | m)


        return intersection / max(union, 1)


    def entity_overlap(self, query_entities, memory_entities):

        if not query_entities or not memory_entities:
            return 0.0

        q = {e.lower() for e in query_entities}
        m = {e.lower() for e in memory_entities}

        return len(q & m) / max(len(q), 1)


    def subject_score(self, query):

        subject = query.metadata.get("subject")

        if not subject:
            return 0.0

        mem_subject = (
            candidate.memory.metadata.get("subject")
            if candidate.memory.metadata else None
        )

        if not mem_subject:
            return 0.0

        return 1.0 if subject.lower() == mem_subject.lower() else 0.0


    def attribute_score(self, query):

        attr = query.metadata.get("attribute")

        if not attr:
            return 0.0

        mem_attr = (
            candidate.memory.metadata.get("attribute")
            if candidate.memory.metadata else None
        )

        if not mem_attr:
            return 0.0

        return 1.0 if attr == mem_attr else 0.0
    # ---------------------------------
    # Semantic score
    # ---------------------------------

    def semantic_score(
        self,
        distance
    ):

        return 1.0 / (
            1.0 + float(distance)
        )


    # ---------------------------------
    # Compute one candidate score
    # ---------------------------------

    def compute_score(
        self,
        candidate,
        query
    ):

        semantic = self.semantic_score(
            candidate.distance
        )


        importance = max(
            0.0,
            min(
                float(candidate.memory.importance),
                1.0
            )
        )


        recency = self.recency_score(
            candidate.memory.created_at
        )


        token = self.token_overlap(
            query.tokens,
            candidate.memory.tokens
        )


        feedback = max(
            -1.0,
            min(
                self.rank_feedback[candidate.memory.id],
                1.0
            )
        )

        entity = self.entity_overlap(
            query.entities,
            candidate.memory.entities
        )

        subject = 0.0

        attribute = 0.0

        if candidate.memory.metadata:

            mem_subject = candidate.memory.metadata.get("subject")
            mem_attribute = candidate.memory.metadata.get("attribute")

            query_subject = query.metadata.get("subject")
            query_attribute = query.metadata.get("attribute")

            if mem_subject and query_subject:

                if mem_subject.lower() == query_subject.lower():
                    subject = 1.0

            if mem_attribute and query_attribute:

                if mem_attribute == query_attribute:
                    attribute = 1.0


        score = (

            semantic * self.weights["semantic"]

            + importance * self.weights["importance"]

            + recency * self.weights["recency"]

            + token * self.weights["token"]

            + feedback * self.weights["feedback"]

            + entity * self.weights["entity"]

            + subject * self.weights["subject"]

            + attribute * self.weights["attribute"]

        )


        

        candidate.semantic_score = semantic
        candidate.importance_score = importance
        candidate.recency_score = recency
        candidate.token_score = token
        candidate.feedback_score = feedback

        candidate.base_score = score

        candidate.diagnostics["ranker"] = {

            "semantic": semantic,

            "importance": importance,

            "recency": recency,

            "token": token,

            "feedback": feedback,

            "entity": entity,

            "subject": subject,

            "attribute": attribute

        }

        return candidate


    # ---------------------------------
    # Pipeline entry
    # ---------------------------------

    def rank(self, candidates, query):

        updated = []
        for candidate in candidates:
            updated.append(
                self.compute_score(candidate, query)
            )

        updated.sort(
            key=lambda x: x.base_score,
            reverse=True
        )

        return updated
    



    # ---------------------------------
    # Feedback
    # ---------------------------------

    def reinforce(
        self,
        mem_id,
        amount=0.05
    ):

        self.rank_feedback[mem_id] += amount


    def punish(
        self,
        mem_id,
        amount=0.05
    ):

        self.rank_feedback[mem_id] -= amount


