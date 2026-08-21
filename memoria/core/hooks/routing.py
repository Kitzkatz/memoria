from core.plugin_manager import hookspec


@hookspec
def memoria_register_type_router():
    """
    Register a custom type router.
    Should return a callable that takes memory_type_hint and returns a route dict.
    """
    pass


@hookspec
def memoria_register_signal_router():
    """
    Register a custom signal router.
    Should return a callable that takes signals and returns modified signals.
    """
    pass


@hookspec
def memoria_routing_pre(query, memory_type_hint):
    """Called before routing decision."""
    pass


@hookspec
def memoria_routing_post(route):
    """Called after routing decision, can modify the route."""
    pass
