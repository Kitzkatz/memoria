from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from routes.memory import router as memory_router
from routes.chat import router as chat_router
from routes.debug import router as debug_router
from routes.maintenance import router as maintenance_router
from routes.benchmark import router as benchmark_router




app = FastAPI(
    title="Memory Daemon",
    version="3.0"
)

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates directory
##templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
##templates = Jinja2Templates(directory="templates")

app.include_router(memory_router)
app.include_router(chat_router)
app.include_router(debug_router)
app.include_router(maintenance_router)
app.include_router(benchmark_router)

@app.get("/", response_class=HTMLResponse)
async def gui(request: Request):
    html_path = Path(__file__).parent / "templates" / "index.html"
    with open(html_path, "r") as f:
        html = f.read()
    return HTMLResponse(html)
