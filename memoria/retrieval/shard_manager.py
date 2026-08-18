from typing import List, Optional
from core.logger import debug


class ShardManager:
    """
    Maps memory types to shard IDs for parallel retrieval.

    Each shard can be processed independently, allowing for parallel
    execution across workers.

    Shard mapping:
    - semantic → 0
    - episodic → 1
    - procedural → 2
    - code → 3
    - science → 4
    - relevance → 0 (shares with semantic)
    - general → (queries all shards)
    """

    # Type to shard mapping
    TYPE_TO_SHARD = {
        "semantic": 0,
        "episodic": 1,
        "procedural": 2,
        "code": 3,
        "science": 4,
        "relevance": 0,  # Relevance shares shard 0 with semantic
    }

    def __init__(self, num_shards: int = 5, type_to_shard: Optional[dict] = None):
        """
        Args:
            num_shards: Total number of shards (default 5)
            type_to_shard: Optional custom mapping of type -> shard ID
        """
        self.num_shards = num_shards

        # Merge custom mapping with defaults if provided
        if type_to_shard:
            self.type_to_shard = {**self.TYPE_TO_SHARD, **type_to_shard}
        else:
            self.type_to_shard = self.TYPE_TO_SHARD

    def shard_id_for_type(self, mem_type: str) -> int:
        """
        Return shard ID for a given memory type.
        Returns 0 for unknown types (safe fallback).
        """
        shard_id = self.type_to_shard.get(mem_type)
        if shard_id is None:
            # Fallback: hash the type to a shard if not in mapping
            shard_id = hash(mem_type) % self.num_shards if mem_type else 0
            debug(f"[ShardManager] Unknown type '{mem_type}' → hashed to shard {shard_id}")
            return shard_id
        return shard_id

    def get_shards_for_query(self, query, memory_type_hint: Optional[str] = None) -> List[int]:
        """
        Return shard IDs to query based on memory type hint.
        If no hint, query all shards.

        Args:
            query: QueryRecord object (unused but kept for API consistency)
            memory_type_hint: Optional type hint to narrow search

        Returns:
            List of shard IDs to query
        """
        if memory_type_hint and memory_type_hint != "general":
            shards = [self.shard_id_for_type(memory_type_hint)]
            debug(f"[ShardManager] type_hint='{memory_type_hint}' → shards={shards}")
            return shards

        # Query all shards for general or no hint
        shards = list(range(self.num_shards))
        debug(f"[ShardManager] No type hint (or 'general') → all shards={shards}")
        return shards

    def get_all_shards(self) -> List[int]:
        """Return all shard IDs."""
        return list(range(self.num_shards))

    def shard_for_memory_id(self, mem_id: int) -> int:
        """
        Return the shard ID for a given memory ID.
        Uses simple modulo distribution.
        """
        return mem_id % self.num_shards

    def get_shards_for_types(self, mem_types: List[str]) -> List[int]:
        """
        Return shard IDs for a list of memory types.
        Deduplicated and sorted.
        """
        shards = set()
        for mem_type in mem_types:
            shards.add(self.shard_id_for_type(mem_type))
        return sorted(shards)

    def get_stats(self) -> dict:
        """Return shard distribution statistics."""
        type_counts = {}
        for mem_type, shard_id in self.type_to_shard.items():
            if shard_id not in type_counts:
                type_counts[shard_id] = []
            type_counts[shard_id].append(mem_type)
        return {
            "num_shards": self.num_shards,
            "type_mapping": type_counts,
            "total_types": len(self.type_to_shard),
        }
