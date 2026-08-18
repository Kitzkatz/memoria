#!/usr/bin/env python3
"""
Memory Daemon — Main Entry Point

Usage:
    python main.py
    python main.py --host 0.0.0.0 --port 8000
"""

import argparse
from core.logger import debug, info
import uvicorn
from app import app


def main():
    parser = argparse.ArgumentParser(description="Memory Daemon Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    args = parser.parse_args()

    info("========================================")
    info("      🧠 Memory Daemon V4")
    info("========================================")
    info(f"   Host: {args.host}")
    info(f"   Port: {args.port}")
    info(f"   Docs: http://localhost:{args.port}/docs")
    info("   LLM:  llama.cpp (external)")
    info("========================================")

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
