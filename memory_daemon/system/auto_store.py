# system/auto_store.py - NEW FILE

"""
Auto-store memories from chat interactions.
Off by default for benchmarks.
"""

from cache.config import settings
from core.logger import debug
from typing import List, Dict, Any, Optional
import time

class AutoStore:
    """Auto-store memories from chat interactions."""
    
    def __init__(self, memory_system):
        self.system = memory_system
        self.session_count = 0
        self.session_id = None
    
    def should_auto_store(self, query_type: str = "chat") -> bool:
        """Check if auto-store is enabled and applicable."""
        if not settings.AUTO_STORE_MEMORIES:
            return False
        
        if query_type not in settings.AUTO_STORE_TYPES:
            return False
        
        if self.session_count >= settings.AUTO_STORE_MAX_PER_SESSION:
            return False
        
        return True
    
    def process_results(self, query: str, results: List[Dict[str, Any]], 
                        query_type: str = "chat") -> int:
        """Process results and auto-store qualifying memories."""
        if not self.should_auto_store(query_type):
            return 0
        
        threshold = settings.AUTO_STORE_THRESHOLD
        max_store = settings.AUTO_STORE_MAX_PER_SESSION - self.session_count
        
        # Filter results above threshold
        candidates = [r for r in results if r.get("score", 0) >= threshold]
        candidates = candidates[:max_store]
        
        if not candidates:
            return 0
        
        stored = 0
        for candidate in candidates:
            try:
                # Build memory data
                memory_data = {
                    "text": candidate.get("text", ""),
                    "source": "auto_store",
                    "source_query": query[:200],
                    "score": candidate.get("score", 0),
                    "memory_type": candidate.get("memory_type", query_type),
                    "timestamp": time.time(),
                    "auto_stored": True
                }
                
                # Store using the memory system
                if hasattr(self.system, "store"):
                    self.system.store(memory_data)
                elif hasattr(self.system, "memory_store") and hasattr(self.system.memory_store, "save"):
                    self.system.memory_store.save(memory_data)
                else:
                    # Fallback: just log
                    debug(f"[AutoStore] Would store: {memory_data['text'][:50]}...")
                    continue
                
                stored += 1
                self.session_count += 1
                
            except Exception as e:
                debug(f"[AutoStore] Failed to store: {e}")
        
        if stored > 0:
            debug(f"[AutoStore] Stored {stored} memories from query: {query[:50]}...")
        
        return stored
    
    def reset_session(self):
        """Reset session counter for new conversation."""
        self.session_count = 0
        self.session_id = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-store statistics."""
        return {
            "enabled": settings.AUTO_STORE_MEMORIES,
            "threshold": settings.AUTO_STORE_THRESHOLD,
            "max_per_session": settings.AUTO_STORE_MAX_PER_SESSION,
            "session_count": self.session_count,
            "types": settings.AUTO_STORE_TYPES
        }
