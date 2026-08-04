from fastapi import FastAPI

from routes.memory import router as memory_router
from routes.chat import router as chat_router
from routes.debug import router as debug_router
from routes.maintenance import router as maintenance_router
from routes.benchmark import router as benchmark_router

app = FastAPI(
    title="Memory Daemon",
    version="3.0"
)

app.include_router(memory_router)
app.include_router(chat_router)
app.include_router(debug_router)
app.include_router(maintenance_router)
app.include_router(benchmark_router)
