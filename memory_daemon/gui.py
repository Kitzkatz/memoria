# memory_daemon/gui.py
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from pathlib import Path

from shared.memory_interface import MemoryInterface

app = FastAPI(title="Memory Daemon GUI")

# Direct interface (server-off mode)
memory = MemoryInterface()

@app.get("/", response_class=HTMLResponse)
async def gui(request: Request):
    html_path = Path(__file__).parent / "templates" / "index.html"
    with open(html_path, "r") as f:
        html = f.read()
    return HTMLResponse(html)

@app.post("/query")
async def query(request: Request):
    try:
        data = await request.json()
        query_text = data.get("query", "")
        if not query_text:
            return JSONResponse({"error": "No query provided"}, status_code=400)
        results = memory.recall(query_text)
        return results
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/store")
async def store(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)
        mem_id = memory.remember(text)
        return {"id": mem_id, "message": "Memory stored"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt:
            return JSONResponse({"error": "No prompt provided"}, status_code=400)
        # This calls MemoryInterface.chat() which uses the LLM adapter
        response = memory.chat(prompt)
        return {"response": response}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
