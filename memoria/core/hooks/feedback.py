from core.plugin_manager import hookspec


@hookspec
def memoria_register_feedback_recorder():
    """
    Register a custom feedback recorder.
    Should return a callable that records feedback (click, etc.).
    """
    pass


@hookspec
def memoria_feedback_pre(query, result_id):
    """Called before recording feedback."""
    pass


@hookspec
def memoria_feedback_post(query, result_id, success):
    """Called after recording feedback."""
    pass
