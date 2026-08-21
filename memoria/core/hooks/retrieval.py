from core.plugin_manager import hookspec


@hookspec
def memoria_register_retriever():
    """
    Register a custom retriever.

    Should return a dict with:
        name: str
        retriever: callable or object with a `retrieve(query, limit)` method
    """
    pass


@hookspec
def memoria_retrieval_pre(query, candidates):
    """
    Hook called before retrieval sources are executed.
    Can modify the query or filter candidates.
    """
    pass


@hookspec
def memoria_retrieval_post(query, candidates, results):
    """
    Hook called after retrieval sources have completed.
    Can modify the final candidate pool.
    """
    pass
