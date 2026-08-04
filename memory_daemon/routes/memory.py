from fastapi import APIRouter
from pydantic import BaseModel

from memory.memory_controller import MemoryController

router = APIRouter(

    prefix="/memory",

    tags=["Memory"]

)

mc = MemoryController()


class MemoryInput(BaseModel):

    text: str


class BatchInput(BaseModel):

    texts: list[str]


@router.post("/store")
def store(inp: MemoryInput):

    mem_id = mc.remember(inp.text)

    return {

        "status": "stored",

        "id": mem_id

    }


@router.post("/query")
def query(inp: MemoryInput):

    response = mc.recall(inp.text)

    return {

        "query": inp.text,
        "count": len(
            response["results"]
            ),
        "results": response["results"],
        "diagnostics": response["diagnostics"]

       # **payload

    }


@router.post("/reflect")
def reflect():

    return {

        "reflection": mc.reflect()

    }


@router.post("/batch_store")
def batch_store(inp: BatchInput):

    stored = 0

    for text in inp.texts:

        mc.remember(text)

        stored += 1

    mc.manager.vector_store.save()

    return {

        "stored": stored

    }


@router.post("/test_store")
def test_store(inp: MemoryInput):

    mem_id = mc.remember(inp.text)

    return {

        "id": mem_id

    }
