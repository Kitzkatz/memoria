from core.plugin_manager import hookspec


@hookspec
def memoria_register_benchmark_adapter():
    """
    Register a custom benchmark adapter.
    Should return a class or function that adapts a dataset to the benchmark format.
    """
    pass


@hookspec
def memoria_register_analyzer():
    """
    Register a custom analyzer.
    Should return a callable that analyzes benchmark results.
    """
    pass
