from core.plugin_manager import hookspec


@hookspec
def memoria_register_query_processor():
    """
    Register a custom query processor.
    Should return a callable that processes text into a QueryRecord.
    """
    pass


@hookspec
def memoria_query_process_pre(text):
    """Called before query processing."""
    pass


@hookspec
def memoria_query_process_post(query_record):
    """Called after query processing, can modify the QueryRecord."""
    pass
