from fastapi import APIRouter
from pydantic import BaseModel

from memory.memory_controller import MemoryController

router = APIRouter(

    prefix="",

    tags=["Chat"]

)

mc = MemoryController()


class ChatInput(BaseModel):

    text: str


@router.post("/chat")

def chat(inp: ChatInput):

    if not inp.text.strip():

        return {

            "status": "error",

            "response": "Empty prompt."

        }

    response = mc.chat(inp.text)

    return {

        "input": inp.text,

        "response": response

    }
