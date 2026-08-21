from core.plugin_manager import hookspec


@hookspec
def memoria_register_ranking_signal():
    """
    Register a custom ranking signal.

    Should return a dict with:
        name: str
        score_func: callable(memory, query) -> float
        weight: float (optional)
    """
    pass


@hookspec
def memoria_register_reranker():
    """
    Register a custom reranker.

    Should return a dict with:
        name: str
        reranker: callable(candidates, query) -> list
    """
    pass


@hookspec
def memoria_ranking_pre(query, candidates):
    """Called before ranking begins."""
    pass


@hookspec
def memoria_ranking_post(query, candidates, scored_candidates):
    """Called after ranking completes, before final selection."""
    pass
