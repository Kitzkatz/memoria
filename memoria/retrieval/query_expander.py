"""
Query expansion — adds synonyms to query tokens to improve recall.
Uses a configurable synonym dictionary.
"""

import json
from pathlib import Path
from typing import List, Dict, Set

from cache.config import settings
from core.logger import debug


class QueryExpander:
    """
    Expands query tokens using a synonym dictionary.
    """

    def __init__(self, synonym_path: str = None):
        self.synonym_path = Path(synonym_path or getattr(settings, "SYNONYM_PATH", "retrieval/synonyms.json"))
        self.synonym_map: Dict[str, Set[str]] = {}
        self._load_synonyms()

    def _load_synonyms(self):
        """Load synonym dictionary from JSON file."""
        if not self.synonym_path.exists():
            debug(f"[QueryExpander] Synonym file not found: {self.synonym_path}, using empty map.")
            return
        try:
            with open(self.synonym_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for term, synonyms in data.items():
                self.synonym_map[term.lower()] = set(synonyms)
            debug(f"[QueryExpander] Loaded {len(self.synonym_map)} synonym groups from {self.synonym_path}")
        except Exception as e:
            debug(f"[QueryExpander] Failed to load synonyms: {e}")

    def expand(self, tokens: List[str]) -> List[str]:
        """
        Expand a list of tokens by adding synonyms.
        Returns the original tokens plus any synonyms that aren't already present.
        """
        expanded = list(tokens)
        for token in tokens:
            token_lower = token.lower()
            if token_lower in self.synonym_map:
                for syn in self.synonym_map[token_lower]:
                    if syn not in expanded:
                        expanded.append(syn)
        return expanded

    def expand_query(self, query):
        """Expand the tokens in a QueryRecord and update metadata."""
        if not query.tokens:
            return query
        original_tokens = query.tokens.copy()
        expanded_tokens = self.expand(original_tokens)
        if expanded_tokens != original_tokens:
            query.tokens = expanded_tokens
            query.metadata["original_tokens"] = original_tokens
            query.metadata["expanded_tokens"] = expanded_tokens
            query.metadata["expansion_applied"] = True
        return query
