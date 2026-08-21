from core.plugin_manager import hookspec


@hookspec
def memoria_on_startup():
    """Called when the memory system starts up."""
    pass


@hookspec
def memoria_on_shutdown():
    """Called when the memory system shuts down."""
    pass


@hookspec
def memoria_pre_query(text):
    """Called before any query is executed."""
    pass


@hookspec
def memoria_post_query(text, response):
    """Called after a query is executed, can modify response."""
    pass


@hookspec
def memoria_pre_store(text, metadata):
    """Called before storing a memory (global)."""
    pass


@hookspec
def memoria_post_store(mem_id, text, metadata):
    """Called after storing a memory (global)."""
    pass
