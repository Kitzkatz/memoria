from core.plugin_manager import hookspec


@hookspec
def memoria_register_extractor():
    """
    Register a custom memory extractor.
    Should return a callable that takes text and returns a MemoryRecord.
    """
    pass


@hookspec
def memoria_register_entity_recognizer():
    """
    Register a custom entity recognizer.
    Should return a callable that takes text and returns a list of entities.
    """
    pass


@hookspec
def memoria_ingestion_pre(text, metadata):
    """Called before extraction begins."""
    pass


@hookspec
def memoria_ingestion_post(record):
    """Called after extraction is complete, can modify the record."""
    pass
