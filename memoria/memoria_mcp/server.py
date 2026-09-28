"""
Memoria MCP server.

Thin MCP adapter over the existing MemoryController.
Uses STDIO transport for local MCP clients.
"""

from typing import Any, Optional

from mcp.server import MCPServer

from core.logger import debug
from memory.memory_controller import MemoryController


mcp = MCPServer("Memoria")

_controller: Optional[MemoryController] = None


def _get_controller() -> MemoryController:
    global _controller

    if _controller is None:
        _controller = MemoryController()

    return _controller


@mcp.tool()
def memory_search(query: str) -> dict[str, Any]:
    """
    Search Memoria using its normal retrieval pipeline.

    Args:
        query: Natural-language query to search the memory system.
    """
    return _get_controller().recall(query)


@mcp.tool()
def memory_store(
    text: str,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Store a single memory through Memoria's normal store pipeline.

    Args:
        text: Memory text to store.
        metadata: Optional metadata dictionary.
    """
    mem_id = _get_controller().remember(
        text,
        metadata=metadata,
    )

    return {
        "id": mem_id,
        "text": text,
    }


@mcp.tool()
def memory_store_many(
    texts: list[str],
    metadatas: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Store multiple memories through Memoria's normal batch store pipeline.

    Args:
        texts: Memory texts to store.
        metadatas: Optional metadata dictionaries corresponding to the texts.
    """
    controller = _get_controller()

    ids = controller.remember_many(
        texts,
        metadatas=metadatas,
    )

    return {
        "ids": ids,
        "count": len(ids),
    }


@mcp.tool()
def memory_fetch(mem_id: int) -> Optional[dict[str, Any]]:
    """
    Fetch a single memory by ID.

    Args:
        mem_id: Memory ID.
    """
    return _get_controller().fetch(mem_id)


@mcp.tool()
def memory_update(
    mem_id: int,
    text: Optional[str] = None,
    normalized_text: Optional[str] = None,
    tokens: Optional[list[str]] = None,
    token_count: Optional[int] = None,
    importance: Optional[float] = None,
    memory_type: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    entities: Optional[list[Any]] = None,
    relationships: Optional[list[Any]] = None,
    last_accessed: Optional[str] = None,
) -> dict[str, Any]:
    """
    Update fields on an existing memory.

    Only fields supplied by the caller are changed.

    Args:
        mem_id: Memory ID.
        text: New memory text.
        normalized_text: New normalized text.
        tokens: New token list.
        token_count: New token count.
        importance: New importance value.
        memory_type: New memory type.
        metadata: New metadata dictionary.
        entities: New entities.
        relationships: New relationships.
        last_accessed: New last-accessed timestamp.
    """
    fields = {
        "text": text,
        "normalized_text": normalized_text,
        "tokens": tokens,
        "token_count": token_count,
        "importance": importance,
        "memory_type": memory_type,
        "metadata": metadata,
        "entities": entities,
        "relationships": relationships,
        "last_accessed": last_accessed,
    }

    fields = {
        key: value
        for key, value in fields.items()
        if value is not None
    }

    if not fields:
        return {
            "id": mem_id,
            "updated": False,
            "reason": "No fields supplied",
        }

    _get_controller().update(
        mem_id,
        **fields,
    )

    return {
        "id": mem_id,
        "updated": True,
        "fields": list(fields.keys()),
    }


@mcp.tool()
def memory_delete(mem_id: int) -> dict[str, Any]:
    """
    Soft-delete a memory by ID.

    Args:
        mem_id: Memory ID.
    """
    _get_controller().delete(mem_id)

    return {
        "id": mem_id,
        "deleted": True,
    }


def main() -> None:
    """Run the Memoria MCP server over STDIO."""
    debug("Starting Memoria MCP server")
    mcp.run()


if __name__ == "__main__":
    main()
