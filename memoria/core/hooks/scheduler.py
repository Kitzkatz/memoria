from core.plugin_manager import hookspec


@hookspec
def memoria_register_worker():
    """
    Register a custom scheduler worker.
    Should return a dict with:
        name: str
        worker: callable(task) -> result
    """
    pass


@hookspec
def memoria_register_scheduler_policy():
    """
    Register a custom scheduler completion policy.
    Should return a callable that takes submitted sources and returns a policy.
    """
    pass


@hookspec
def memoria_scheduler_pre(task_ids):
    """Called before scheduler executes tasks."""
    pass


@hookspec
def memoria_scheduler_post(execution_result):
    """Called after scheduler execution completes."""
    pass
