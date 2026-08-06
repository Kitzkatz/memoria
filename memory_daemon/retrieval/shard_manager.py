# retrieval/shard_manager.py
from typing import List, Optional
from core.logger import debug

class ShardManager:
    # Map memory type to shard ID
    TYPE_TO_SHARD = {
        "semantic": 0,
        "episodic": 1,
        "procedural": 2,
        "code": 3,
        "science": 4,
        "general": 0,  # fallback
    }

    def __init__(self, num_shards: int = 5):
        self.num_shards = num_shards

    def shard_id_for_type(self, mem_type: str) -> int:
        """Return shard ID for a given memory type."""
        return self.TYPE_TO_SHARD.get(mem_type, 0)

    def get_shards_for_query(self, query, memory_type_hint: Optional[str] = None) -> List[int]:
        """
        Return shard IDs to query based on memory type hint.
        If no hint, query all shards.
        """
        if memory_type_hint and memory_type_hint != "general":
            return [self.shard_id_for_type(memory_type_hint)]
        return list(range(self.num_shards))

    def get_shards_for_query(self, query, memory_type_hint: Optional[str] = None) -> List[int]:
        """
        Return shard IDs to query based on memory type hint.
        If no hint, query all shards.
        """
        if memory_type_hint and memory_type_hint != "general":
            shards = [self.shard_id_for_type(memory_type_hint)]
        else:
            shards = list(range(self.num_shards))
        debug(f"ShardManager: type_hint='{memory_type_hint}' → shards={shards}")
        return shards
