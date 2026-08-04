# retrieval/inverted_index.py
from collections import defaultdict
from typing import List, Dict, Set, Optional
import json

class InvertedIndex:
    def __init__(self, db, tokenizer=None):
        self.db = db
        self.tokenizer = tokenizer or (lambda text: text.split())
        self.index: Dict[str, List[int]] = {}
        self.positional_index: Dict[str, Dict[int, List[int]]] = {}
        self.document_frequency: Dict[str, int] = {}

    def build(self, memory_ids: Optional[List[int]] = None):
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
                # Inverted index
                if token not in self.index:
                    self.index[token] = []
                self.index[token].append(mem_id)

                # Positional index
                if token not in self.positional_index:
                    self.positional_index[token] = {}
                if mem_id not in self.positional_index[token]:
                    self.positional_index[token][mem_id] = []
                self.positional_index[token][mem_id].append(position)

        self.document_frequency = {
            term: len(set(docs)) for term, docs in self.index.items()
        }

    def search(self, term: str) -> List[int]:
        return self.index.get(term, [])

    def phrase_search(self, phrase_tokens: List[str]) -> List[int]:
        if not phrase_tokens:
            return []

        candidates = set(self.search(phrase_tokens[0]))
        for i, token in enumerate(phrase_tokens[1:], start=1):
            candidates &= set(self.search(token))
            if not candidates:
                return []

            # Check positions for consecutive order
            valid = set()
            for mem_id in candidates:
                pos_lists = []
                for token in phrase_tokens:
                    pos_lists.append(self.positional_index.get(token, {}).get(mem_id, []))
                if all(pos_lists) and self._has_consecutive(pos_lists):
                    valid.add(mem_id)
            candidates = valid
        return list(candidates)

    def _has_consecutive(self, pos_lists: List[List[int]]) -> bool:
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
        if not terms:
            return []
        result = set(self.search(terms[0]))
        for term in terms[1:]:
            result &= set(self.search(term))
        return list(result)

    def save(self, filepath: str):
        data = {
            'index': self.index,
            'positional_index': self.positional_index,
            'document_frequency': self.document_frequency
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.index = data['index']
        self.positional_index = data['positional_index']
        self.document_frequency = data['document_frequency']
        # Add to InvertedIndex class

    def build_skips(self, skip_interval: int = 4):
        """Build skip pointers for all posting lists."""
        self.skip_lists = {}
        for term, posting in self.index.items():
            skips = []
            for i in range(0, len(posting), skip_interval):
                # Store (doc_id, index_position) to skip ahead
                skips.append((posting[i], i))
            self.skip_lists[term] = skips

    def and_query_skip(self, terms: List[str]) -> List[int]:
        """
        Boolean AND query using skip pointers for faster intersection.
        """
        if not terms:
            return []

        # Get posting lists (deduplicated)
        posting_lists = [list(set(self.index.get(term, []))) for term in terms]
        posting_lists = [pl for pl in posting_lists if pl]  # remove empty
        if not posting_lists:
            return []

        # Sort posting lists by length (shortest first) for efficiency
        posting_lists.sort(key=len)

        # Build skip lists if not already built
        if not hasattr(self, 'skip_lists') or not self.skip_lists:
            self.build_skips()

        # For simplicity in this phase, we'll use the standard merge with skip jumps
        # Standard merge on the shortest list
        result = []
        # We'll use the first (shortest) list as base
        base_list = posting_lists[0]
        other_lists = posting_lists[1:]

        for doc_id in base_list:
            found = True
            for other in other_lists:
                # Use binary search or skip pointers to find doc_id
                # Since skip pointers are complex to integrate here quickly,
                # we'll use binary search from the 'bisect' module for O(log n) per check
                import bisect
                idx = bisect.bisect_left(other, doc_id)
                if idx == len(other) or other[idx] != doc_id:
                    found = False
                    break
            if found:
                result.append(doc_id)

        return result
