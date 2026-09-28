from core.plugin_manager import hookspec


@hookspec
def memoria_analysis_pre(records, context):
    pass


@hookspec
def memoria_analysis_post(metrics, summary):
    pass
