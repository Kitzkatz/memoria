#!/usr/bin/env python3
"""
Memoria GUI — web-based interface with FastAPI.
Runs on port 5000 by default.
"""

import os
from pathlib import Path
from datetime import date

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from shared.memory_interface import MemoryInterface
from core.logger import info, debug, error

# Signal registry (optional)
try:
    from ranking.signal_registry import get_registry
    from ranking.signal_router import SignalRouter
    HAS_SIGNAL_REGISTRY = True
except ImportError:
    HAS_SIGNAL_REGISTRY = False

# Query history (optional)
try:
    from memory.query_history import QueryHistory
    HAS_QUERY_HISTORY = True
except ImportError:
    HAS_QUERY_HISTORY = False

# Use absolute paths relative to this file
BASE_DIR = Path(__file__).parent.absolute()
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(
    title="Memoria GUI",
    version="1.0",
    description="Web interface for Memoria"
)

# Direct interface
memory = None
query_history = None

@app.on_event("startup")
async def startup():
    global memory, query_history

    memory = MemoryInterface()

    if HAS_QUERY_HISTORY:
        query_history = QueryHistory()

    info("[GUI] Memory interface initialized", category="gui")
# --------------------------------------------------
# Root / GUI
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def gui(request: Request):
    """Serve the main GUI interface."""
    html_path = TEMPLATES_DIR / "index.html"
    try:
        with open(html_path, "r") as f:
            html = f.read()
        return HTMLResponse(html)
    except FileNotFoundError:
        error("[GUI] index.html not found", category="gui")
        return HTMLResponse(
            f"""
            <html>
            <body>
                <h1>Memoria GUI</h1>
                <p>GUI template not found. Please ensure templates/index.html exists.</p>
                <p>Looking in: {TEMPLATES_DIR}</p>
                <p>Status: Running</p>
            </body>
            </html>
            """
        )


# --------------------------------------------------
# API Endpoints
# --------------------------------------------------

@app.post("/query")
async def query(request: Request):
    """Query the memory system."""
    try:
        data = await request.json()
        query_text = data.get("query", "")
        if not query_text:
            return JSONResponse({"error": "No query provided"}, status_code=400)

        debug(f"[GUI] Query: {query_text[:50]}...", category="gui")
        results = memory.recall(query_text)
        return results

    except Exception as e:
        error(f"[GUI] Query error: {e}", category="gui")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/store")
async def store(request: Request):
    """Store a single memory."""
    try:
        data = await request.json()
        text = data.get("text", "")
        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)

        debug(f"[GUI] Store: {text[:50]}...", category="gui")
        mem_id = memory.remember(text)
        return {"id": mem_id, "message": "Memory stored"}

    except Exception as e:
        error(f"[GUI] Store error: {e}", category="gui")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/store_many")
async def store_many(request: Request):
    """Store multiple memories from a JSON list."""
    try:
        data = await request.json()
        texts = data.get("texts", [])
        if not texts or not isinstance(texts, list):
            return JSONResponse({"error": "No texts provided or invalid format"}, status_code=400)

        debug(f"[GUI] Store many: {len(texts)} texts", category="gui")
        ids = memory.remember_many(texts)
        return {"ids": ids, "message": f"Stored {len(ids)} memories"}

    except Exception as e:
        error(f"[GUI] Store many error: {e}", category="gui")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/chat")
async def chat(request: Request):
    """Chat with the memory system using LLM."""
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        auto_store = data.get("auto_store")  # Optional override

        if not prompt:
            return JSONResponse({"error": "No prompt provided"}, status_code=400)

        debug(f"[GUI] Chat: {prompt[:50]}...", category="gui")
        response = memory.chat(prompt, auto_store=auto_store)
        return {"response": response}

    except Exception as e:
        error(f"[GUI] Chat error: {e}", category="gui")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/ingest_code")
async def ingest_code(request: Request):
    """Ingest a codebase from a directory."""
    try:
        data = await request.json()
        directory = data.get("directory", "")
        max_files = data.get("max_files", 1000)

        if not directory:
            return JSONResponse({"error": "No directory provided"}, status_code=400)

        if not os.path.exists(directory):
            return JSONResponse({"error": "Directory does not exist"}, status_code=400)

        debug(f"[GUI] Ingest code: {directory}", category="gui")

        result = memory.ingest_code(directory, max_files=max_files)
        return {
            "message": f"Ingested code from {directory}",
            "files_processed": result.get("files_processed", 0),
            "symbols_ingested": result.get("symbols_ingested", 0),
        }

    except Exception as e:
        error(f"[GUI] Ingest code error: {e}", category="gui")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/ingest_pdf")
async def ingest_pdf(request: Request):
    """Ingest a PDF file."""
    try:
        data = await request.json()
        filepath = data.get("filepath", "")
        max_pages = data.get("max_pages", 100)

        if not filepath:
            return JSONResponse({"error": "No filepath provided"}, status_code=400)

        if not os.path.exists(filepath):
            return JSONResponse({"error": "File does not exist"}, status_code=400)

        debug(f"[GUI] Ingest PDF: {filepath}", category="gui")

        count = memory.ingest_pdf(filepath, max_pages=max_pages)
        return {"message": f"Ingested PDF from {filepath} ({count} pages)"}

    except Exception as e:
        error(f"[GUI] Ingest PDF error: {e}", category="gui")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/set_goal")
async def set_goal(request: Request):
    """Set a new goal."""
    try:
        data = await request.json()
        goal = data.get("goal", "")
        progress = data.get("progress", "started")

        if not goal:
            return JSONResponse({"error": "No goal provided"}, status_code=400)

        debug(f"[GUI] Set goal: {goal[:50]}...", category="gui")
        goal_id = memory.set_goal(goal, progress)
        return {"id": goal_id, "message": "Goal set"}

    except Exception as e:
        error(f"[GUI] Set goal error: {e}", category="gui")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/list_goals")
async def list_goals(status: str = None):
    """List all goals, optionally filtered by status."""
    try:
        debug(f"[GUI] List goals (status={status})", category="gui")
        goals = memory.list_goals(status=status)
        return {"goals": goals}

    except Exception as e:
        error(f"[GUI] List goals error: {e}", category="gui")
        return JSONResponse({"error": str(e)}, status_code=500)


# --------------------------------------------------
# Signals Endpoints
# --------------------------------------------------

@app.get("/signals")
async def get_signals(memory_type: str = Query("general", description="Memory type")):
    """Get active signals for a memory type."""
    if not HAS_SIGNAL_REGISTRY:
        return JSONResponse({"error": "Signal registry not available"}, status_code=501)

    try:
        registry = get_registry()
        router = SignalRouter(registry)
        signals = router.get_active_signals(memory_type)

        # Get full info for each signal
        signal_info = []
        for name, weight in signals.items():
            signal_info.append({
                "name": name,
                "weight": weight,
                "cost": registry.get_cost(name),
                "enabled": registry.is_enabled_for_type(name, memory_type),
                "description": registry.get_description(name),
                "category": registry.get_category(name),
            })

        return {
            "memory_type": memory_type,
            "signals": signal_info,
            "total_weight": sum(signals.values()),
        }

    except Exception as e:
        error(f"[GUI] Signals error: {e}", category="gui")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/signals/toggle")
async def toggle_signal(request: Request):
    """Toggle a signal on/off."""
    if not HAS_SIGNAL_REGISTRY:
        return JSONResponse({"error": "Signal registry not available"}, status_code=501)

    try:
        data = await request.json()
        name = data.get("name")
        if not name:
            return JSONResponse({"error": "No signal name provided"}, status_code=400)

        registry = get_registry()
        current = registry.is_enabled(name)

        if data.get("enable") is True:
            registry.enable(name)
            enabled = True
            status = "enabled"
        elif data.get("enable") is False:
            registry.disable(name)
            enabled = False
            status = "disabled"
        else:
            # Toggle
            if current:
                registry.disable(name)
                enabled = False
                status = "disabled"
            else:
                registry.enable(name)
                enabled = True
                status = "enabled"

        # Clear router cache
        
        router = SignalRouter(registry)
        router.clear_cache()

        return {
            "signal": name,
            "enabled": enabled,
            "status": status,
        }

    except Exception as e:
        error(f"[GUI] Toggle signal error: {e}", category="gui")
        return JSONResponse({"error": str(e)}, status_code=500)


# --------------------------------------------------
# Query History Endpoints
# --------------------------------------------------

@app.get("/history")
async def get_history(
    mode: str = Query("recent", description="History view: recent, frequent, context, previous"),
    limit: int = Query(10, description="Maximum entries to return")
):
    """Access query history using the current QueryHistory interface."""
    if not HAS_QUERY_HISTORY or query_history is None:
        return JSONResponse({"error": "Query history not available"}, status_code=501)

    try:
        if limit <= 0:
            return JSONResponse(
                {"error": "Limit must be greater than 0"},
                status_code=400
            )

        mode = mode.lower()

        if mode == "recent":
            entries = query_history.get_recent(n=limit)
            return {
                "mode": "recent",
                "count": len(entries),
                "entries": entries,
            }

        if mode == "frequent":
            queries = query_history.get_frequent(n=limit)
            return {
                "mode": "frequent",
                "count": len(queries),
                "queries": queries,
            }

        if mode == "context":
            queries = query_history.get_context()
            return {
                "mode": "context",
                "count": len(queries),
                "queries": queries,
            }

        if mode == "previous":
            previous = query_history.get_previous_query()
            return {
                "mode": "previous",
                "query": previous,
            }

        return JSONResponse(
            {
                "error": f"Unknown history mode: {mode}",
                "available": ["recent", "frequent", "context", "previous"],
            },
            status_code=400
        )

    except Exception as e:
        error(f"[GUI] History error: {e}", category="gui")
        return JSONResponse({"error": str(e)}, status_code=500)
# --------------------------------------------------
# Auto-Store Endpoints (NEW)
# --------------------------------------------------

@app.get("/settings/auto-store")
async def get_auto_store_settings():
    """Get current auto-store settings."""
    from cache.config import settings
    return {
        "auto_store": settings.AUTO_STORE_MEMORIES,
        "threshold": settings.AUTO_STORE_THRESHOLD,
        "max_per_session": settings.AUTO_STORE_MAX_PER_SESSION,
        "types": settings.AUTO_STORE_TYPES
    }


@app.post("/settings/auto-store")
async def update_auto_store_settings(request: Request):
    """Update auto-store settings."""
    from cache.config import settings
    
    try:
        data = await request.json()
        updates = {}
        
        if "auto_store" in data:
            settings.AUTO_STORE_MEMORIES = data["auto_store"]
            updates["auto_store"] = data["auto_store"]
        
        if "threshold" in data:
            threshold = data["threshold"]
            if 0.0 <= threshold <= 1.0:
                settings.AUTO_STORE_THRESHOLD = threshold
                updates["threshold"] = threshold
            else:
                return JSONResponse({"error": "Threshold must be between 0.0 and 1.0"}, status_code=400)
        
        if "max_per_session" in data:
            max_val = data["max_per_session"]
            if max_val > 0:
                settings.AUTO_STORE_MAX_PER_SESSION = max_val
                updates["max_per_session"] = max_val
            else:
                return JSONResponse({"error": "Max per session must be > 0"}, status_code=400)
        
        if "types" in data:
            types = data["types"]
            if isinstance(types, list):
                settings.AUTO_STORE_TYPES = types
                updates["types"] = types
            else:
                return JSONResponse({"error": "Types must be a list of strings"}, status_code=400)
        
        return {
            "updated": updates,
            "settings": {
                "auto_store": settings.AUTO_STORE_MEMORIES,
                "threshold": settings.AUTO_STORE_THRESHOLD,
                "max_per_session": settings.AUTO_STORE_MAX_PER_SESSION,
                "types": settings.AUTO_STORE_TYPES
            }
        }
        
    except Exception as e:
        error(f"[GUI] Auto-store settings error: {e}", category="gui")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/settings/auto-store/toggle")
async def toggle_auto_store(request: Request):
    """Toggle auto-store on/off."""
    from cache.config import settings
    
    try:
        data = await request.json()
        enable = data.get("enable")
        
        if enable is None:
            # Toggle
            settings.AUTO_STORE_MEMORIES = not settings.AUTO_STORE_MEMORIES
        else:
            settings.AUTO_STORE_MEMORIES = bool(enable)
        
        status = "enabled" if settings.AUTO_STORE_MEMORIES else "disabled"
        return {
            "auto_store": settings.AUTO_STORE_MEMORIES,
            "status": status,
            "message": f"Auto-store {status}"
        }
        
    except Exception as e:
        error(f"[GUI] Toggle auto-store error: {e}", category="gui")
        return JSONResponse({"error": str(e)}, status_code=500)


# --------------------------------------------------
# Health and Stats
# --------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        if memory is None:
            return {
                "status": "initializing",
                "version": "1.0",
                "service": "Memoria GUI",
            }
        return {
            "status": "ok",
            "version": "1.0",
            "service": "Memoria GUI",
            "memory_count": memory.controller.system.db.count(),
            "signals_available": HAS_SIGNAL_REGISTRY,
            "query_history_available": HAS_QUERY_HISTORY,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "version": "1.0",
            "error": str(e),
        }


@app.get("/stats")
async def stats():
    """Get system statistics."""
    try:
        stats_data = {
            "memory_count": memory.controller.system.db.count(),
            "version": "1.0",
            "goals": len(memory.list_goals()),
            "signals_available": HAS_SIGNAL_REGISTRY,
            "query_history_available": HAS_QUERY_HISTORY,
        }
        
        
        
        # Add auto-store status
        from cache.config import settings
        stats_data["auto_store"] = {
            "enabled": settings.AUTO_STORE_MEMORIES,
            "threshold": settings.AUTO_STORE_THRESHOLD,
            "max_per_session": settings.AUTO_STORE_MAX_PER_SESSION,
        }
        
        return stats_data
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# --------------------------------------------------
# Main Entry Point
# --------------------------------------------------

def main():
    """Start the GUI server."""
    info("========================================", category="gui")
    info("      Memoria GUI v1.0", category="gui")
    info("========================================", category="gui")
    info(f"   GUI:  http://localhost:5000", category="gui")
    info(f"   API:  http://localhost:5000/docs", category="gui")
    info("========================================", category="gui")

    uvicorn.run(
        "gui:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
