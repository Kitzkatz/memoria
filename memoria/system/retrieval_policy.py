from blackboard.scheduler import CompletionPolicy


class RetrievalCompletionPolicy(CompletionPolicy):
    """
    Decide whether a set of retrieval workers has produced enough
    evidence to proceed to expensive ranking.

    This policy intentionally knows about retrieval results but does
    not know anything about the Scheduler's implementation.
    """

    name = "retrieval_sufficient"

    def __init__(
        self,
        min_candidates,
        min_sources=1,
        required_sources=None,
    ):
        if min_candidates < 1:
            raise ValueError(
                "min_candidates must be >= 1"
            )

        if min_sources < 1:
            raise ValueError(
                "min_sources must be >= 1"
            )

        self.min_candidates = min_candidates
        self.min_sources = min_sources
        self.required_sources = set(
            required_sources or []
        )

    def should_finish(self, state):
        completed_results = state.get(
            "completed_results",
            {}
        )

        candidate_ids = set()
        completed_sources = set()

        for result in completed_results.values():

            if not result:
                continue

            source = result.get("source")

            if source:
                completed_sources.add(source)

            candidates = result.get(
                "candidates",
                []
            )

            for candidate in candidates:

                if isinstance(candidate, (tuple, list)):
                    if not candidate:
                        continue

                    candidate_ids.add(
                        candidate[0]
                    )

                else:
                    candidate_ids.add(candidate)

        # Required retrieval modalities must have
        # successfully completed.
        if not self.required_sources.issubset(
            completed_sources
        ):
            return False

        # Require diversity across retrieval mechanisms.
        if len(completed_sources) < self.min_sources:
            return False

        # Require enough unique memories to justify
        # the expensive ranking stage.
        if len(candidate_ids) < self.min_candidates:
            return False

        return True
