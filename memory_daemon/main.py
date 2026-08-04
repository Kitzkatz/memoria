from core.logger import debug
import uvicorn
from api import app

def main():
    debug("\n🧠 Memory Daemon starting...\n")
    debug("⚙️ Initializing FastAPI server")
    debug("📡 API will be available at http://localhost:8000")
    debug("🧩 LLM backend: llama.cpp (external)\n")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
