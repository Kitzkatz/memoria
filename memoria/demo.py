
"""
Memoria quick-start demo.

Demonstrates:
    - storing memories
    - semantic/hybrid retrieval
    - temporal retrieval
    - fetching a memory by ID
"""

from memory.memory_controller import MemoryController


def main():
    print("=" * 60)
    print("MEMORIA QUICK DEMO")
    print("=" * 60)
    print()

    controller = MemoryController()

    print("Storing memories...")
    memories = [
        "Memoria is a local-first memory system with hybrid retrieval.",
        "The project uses FAISS and BM25 together for retrieval.",
        "I started reworking the Temporal retrieval worker on September 27, 2026.",
        "On September 27, 2026, I tested the Temporal worker with exact dates.",
        "The Memoria MCP server exposes memory search and storage to AI agents.",
        "Obsidian notes can now be ingested into Memoria for retrieval, Yay!.",
    ]

    ids = controller.remember_many(memories)

    print(f"Stored {len(ids)} memories.")
    print()

    print("-" * 60)
    print("HYBRID SEARCH")
    print("-" * 60)

    query = "How does Memoria retrieve information?"
    print(f"Query: {query}")
    results = controller.recall(query)
    print(results)
    print()

    print("-" * 60)
    print("TEMPORAL SEARCH")
    print("-" * 60)

    query = "What happened on September 27, 2026?"
    print(f"Query: {query}")
    results = controller.recall(query)
    print(results)
    print()

    print("-" * 60)
    print("FETCH BY ID")
    print("-" * 60)

    mem_id = ids[0]
    print(f"ID: {mem_id}")
    result = controller.fetch(mem_id)
    print(result)
    print()

    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

