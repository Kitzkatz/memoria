"""
Memory Daemon — FastAPI application.

Routes:
- /memory: Store and retrieve memories
- /chat: Chat with the memory system
- /debug: Debug endpoints
- /maintenance: System maintenance
- /benchmark: Benchmark endpoints
"""

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pathlib import Path

from routes.memory import router as memory_router
from routes.chat import router as chat_router
from routes.debug import router as debug_router
from routes.maintenance import router as maintenance_router
from routes.benchmark import router as benchmark_router

from core.logger import info, debug
from cache.config import settings


# ─────────────────────────────────────────────
# Numpy JSON Encoder Fix
# ─────────────────────────────────────────────

def convert_numpy(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {convert_numpy(k): convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    return obj


def jsonable_encoder_numpy(obj, *args, **kwargs):
    """Wrapper for jsonable_encoder that handles numpy types."""
    return jsonable_encoder(convert_numpy(obj), *args, **kwargs)


# Patch FastAPI's jsonable_encoder to handle numpy types
import fastapi.encoders
fastapi.encoders.jsonable_encoder = jsonable_encoder_numpy


# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Memory Daemon",
    version="4.0",
    description="Local-first, LLM-agnostic memory system with feedback learning",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ─────────────────────────────────────────────
# CORS (optional — enable for web clients)
# ─────────────────────────────────────────────

if getattr(settings, "ENABLE_CORS", False):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    info("[App] CORS enabled", category="app")


# ─────────────────────────────────────────────
# Static Files
# ─────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    debug("[App] Static directory not found", category="app")


# ─────────────────────────────────────────────
# Templates
# ─────────────────────────────────────────────

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

app.include_router(memory_router)
app.include_router(chat_router)
app.include_router(debug_router)
app.include_router(maintenance_router)
app.include_router(benchmark_router)


# ─────────────────────────────────────────────
# Root Endpoint
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def gui(request: Request):
    """Serve the main GUI interface."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    try:
        with open(html_path, "r") as f:
            html = f.read()
        return HTMLResponse(html)
    except FileNotFoundError:
        return HTMLResponse(
            "<html><body><h1>Memory Daemon V4</h1>"
            "<p>Running. No GUI found. Use /docs for API.</p>"
            f"<p>Version: 4.0</p></body></html>"
        )


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        "version": "4.0",
        "service": "Memory Daemon"
    }


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────

info("========================================", category="app")
info("      Memory Daemon V4", category="app")
info("========================================", category="app")
info(f"   Docs: http://localhost:8000/docs", category="app")
info(f"   GUI:  http://localhost:8000/", category="app")
info("========================================", category="app")
