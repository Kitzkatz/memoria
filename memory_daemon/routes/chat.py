from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from memory.memory_controller import MemoryController
from core.logger import debug, info, error

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)

mc = MemoryController()


# --------------------------------------------------
# Models
# --------------------------------------------------

class ChatInput(BaseModel):
    text: str
    top_n: Optional[int] = None


class ChatResponse(BaseModel):
    input: str
    response: str
    status: str = "ok"


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@router.post("/")
@router.post("/chat")
def chat(inp: ChatInput):
    """
    Chat with the memory system.
    Retrieves relevant memories and generates a response using the LLM.
    """
    try:
        if not inp.text or not inp.text.strip():
            return ChatResponse(
                input=inp.text,
                response="Empty prompt provided.",
                status="error"
            )

        debug(f"[API] Chat: {inp.text[:50]}...", category="api")

        # Call the controller's chat method
        response = mc.chat(inp.text, top_n=inp.top_n)

        debug(f"[API] Chat response: {response[:100]}...", category="api")

        return ChatResponse(
            input=inp.text,
            response=response,
            status="ok"
        )

    except Exception as e:
        error(f"[API] Chat error: {e}", category="api")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
def chat_history():
    """
    Get chat history (if the controller supports it).
    """
    try:
        # This assumes you've added chat history tracking to MemoryController
        # If not, return an empty list
        if hasattr(mc, 'chat_history'):
            return {"history": mc.chat_history}
        else:
            return {"history": [], "message": "Chat history not enabled"}
    except Exception as e:
        error(f"[API] Chat history error: {e}", category="api")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/raw")
def raw_chat(inp: ChatInput):
    """
    Send a raw prompt directly to the LLM (no retrieval).
    """
    try:
        if not inp.text or not inp.text.strip():
            return ChatResponse(
                input=inp.text,
                response="Empty prompt provided.",
                status="error"
            )

        debug(f"[API] Raw chat: {inp.text[:50]}...", category="api")

        response = mc.raw_chat(inp.text)

        return ChatResponse(
            input=inp.text,
            response=response,
            status="ok"
        )

    except Exception as e:
        error(f"[API] Raw chat error: {e}", category="api")
        raise HTTPException(status_code=500, detail=str(e))
