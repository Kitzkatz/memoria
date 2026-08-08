#!/usr/bin/env python3
"""
Memory Daemon GUI — web-based interface with FastAPI.
Runs on port 5000 by default.
"""

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from shared.memory_interface import MemoryInterface
from core.logger import info, debug, error

# Use absolute paths relative to this file
BASE_DIR = Path(__file__).parent.absolute()
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(
    title="Memory Daemon GUI",
    version="4.0",
    description="Web interface for Memory Daemon"
)

# Direct interface
memory = MemoryInterface()


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
                <h1>Memory Daemon GUI</h1>
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
        if not prompt:
            return JSONResponse({"error": "No prompt provided"}, status_code=400)

        debug(f"[GUI] Chat: {prompt[:50]}...", category="gui")
        response = memory.chat(prompt)
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

        # Use the interface (you added this method)
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

        # Use the interface (you added this method)
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


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        return {
            "status": "ok",
            "version": "4.0",
            "service": "Memory Daemon GUI",
            "memory_count": memory.controller.db.count(),
        }
    except Exception as e:
        return {
            "status": "degraded",
            "version": "4.0",
            "error": str(e),
        }


@app.get("/stats")
async def stats():
    """Get system statistics."""
    try:
        return {
            "memory_count": memory.controller.db.count(),
            "version": "4.0",
            "goals": len(memory.list_goals()),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# --------------------------------------------------
# Main Entry Point
# --------------------------------------------------

def main():
    """Start the GUI server."""
    info("========================================", category="gui")
    info("      Memory Daemon GUI v4.0", category="gui")
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
