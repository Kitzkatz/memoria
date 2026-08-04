from collections import defaultdict
from typing import List, Dict, Set, Optional, Tuple
import json

class InvertedIndex:
    """
    Inverted index mapping terms to memory IDs.
    Also supports positional indexing for phrase queries.
    """
    def __init__(self, db, tokenizer=None):
        self.db = db
        self.tokenizer = tokenizer or (lambda text: text.split())
        self.index: Dict[str, List[int]] = {}          # term -> [mem_id, ...]
        self.positional_index: Dict[str, Dict[int, List[int]]] = {}  # term -> {mem_id: [pos1, pos2, ...]}
        self.document_frequency: Dict[str, int] = {}    # term -> doc count

    def build(self, memory_ids: Optional[List[int]] = None):
        """
        Build the inverted index from the database.
        If memory_ids is given, only index those.
        """
        if memory_ids is None:
            rows = self.db.fetch_all()
            memory_ids = [row["id"] for row in rows]
        else:
            rows = [self.db.fetch(mid) for mid in memory_ids if self.db.fetch(mid)]

        for row in rows:
            mem_id = row["id"]
            text = row.get("normalized_text", row["text"])
            tokens = self.tokenizer(text)
            for position, token in enumerate(tokens):
                # Add to standard index
                if token not in self.index:
                    self.index[token] = []
                self.index[token].append(mem_id)

                # Add to positional index
                if token not in self.positional_index:
                    self.positional_index[token] = {}
                if mem_id not in self.positional_index[token]:
                    self.positional_index[token][mem_id] = []
                self.positional_index[token][mem_id].append(position)

        # Compute document frequencies
        self.document_frequency = {term: len(set(docs)) for term, docs in self.index.items()}

    def search(self, term: str) -> List[int]:
        """Return list of memory IDs containing the term."""
        return self.index.get(term, [])

    def phrase_search(self, phrase_tokens: List[str]) -> List[int]:
        """
        Return memory IDs where the phrase appears in order.
        """
        if not phrase_tokens:
            return []

        # Start with candidates from first term
        candidates = set(self.search(phrase_tokens[0]))

        for i, token in enumerate(phrase_tokens[1:], start=1):
            # Only consider documents that contain all terms
            candidates &= set(self.search(token))

            # Check positions
            if not candidates:
                return []

            # For each candidate, check if positions are consecutive
            valid_candidates = set()
            for mem_id in candidates:
                pos_lists = []
                for token in phrase_tokens:
                    pos_lists.append(self.positional_index.get(token, {}).get(mem_id, []))
                if all(pos_lists) and self._has_consecutive(pos_lists):
                    valid_candidates.add(mem_id)
            candidates = valid_candidates

        return list(candidates)

    def _has_consecutive(self, pos_lists: List[List[int]]) -> bool:
        """
        Check if there exists a sequence of positions that are consecutive
        across the token lists (i.e., positions 1,2,3,...).
        """
        if not pos_lists:
            return False
        # For each position of first token, check if next token appears at pos+1, etc.
        for pos in pos_lists[0]:
            valid = True
            for i in range(1, len(pos_lists)):
                if pos + i not in pos_lists[i]:
                    valid = False
                    break
            if valid:
                return True
        return False

    def and_query(self, terms: List[str]) -> List[int]:
        """Boolean AND query on terms."""
        if not terms:
            return []
        result = set(self.search(terms[0]))
        for term in terms[1:]:
            result &= set(self.search(term))
        return list(result)

    def save(self, filepath: str):
        """Save the index to disk (JSON)."""
        data = {
            'index': self.index,
            'positional_index': self.positional_index,
            'document_frequency': self.document_frequency
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath: str):
        """Load the index from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.index = data['index']
        self.positional_index = data['positional_index']
        self.document_frequency = data['document_frequency']
