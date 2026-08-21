from core.plugin_manager import hookspec


@hookspec
def memoria_register_database_backend():
    """
    Register a custom database backend.
    Should return a subclass of DBConnection or a compatible object.
    """
    pass


@hookspec
def memoria_register_vector_store():
    """
    Register a custom vector store.
    Should return an object with search/add/save/load methods.
    """
    pass


@hookspec
def memoria_storage_pre(text, metadata):
    """Called before storing a memory."""
    pass


@hookspec
def memoria_storage_post(mem_id, text, metadata):
    """Called after storing a memory."""
    pass
