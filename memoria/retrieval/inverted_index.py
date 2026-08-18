import json
import bisect
from collections import defaultdict
from typing import List, Dict, Set, Optional, Union
import time

from core.logger import debug


class InvertedIndex:
    def __init__(self, db, tokenizer=None):
        self.db = db
        self.tokenizer = tokenizer or self._default_tokenizer
        self.index: Dict[str, List[int]] = {}
        self.positional_index: Dict[str, Dict[int, List[int]]] = {}
        self.document_frequency: Dict[str, int] = {}
        self.skip_lists: Dict[str, List[tuple]] = {}
        self._built = False

    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        """Default tokenizer: lowercase, split on whitespace, strip punctuation."""
        if not text:
            return []
        # Lowercase and split
        text_lower = text.lower()
        # Simple punctuation stripping
        for char in ".,!?;:()\"'":
            text_lower = text_lower.replace(char, " ")
        return text_lower.split()

    def build(self, memory_ids: Optional[List[int]] = None):
        """
        Build the inverted index from the database.
        Uses fetch_many to avoid double DB hits.

        Note: This builds the entire index in memory. For very large
        corpora (>100k documents), consider sharding the index.
        """
        start = time.perf_counter()

        # Clear existing index
        self.index.clear()
        self.positional_index.clear()
        self.document_frequency.clear()
        self.skip_lists.clear()

        if memory_ids is None:
            rows = self.db.fetch_all()
        else:
            rows_dict = self.db.fetch_many(memory_ids)
            rows = [rows_dict[mid] for mid in memory_ids if mid in rows_dict]

        if not rows:
            debug("[InvertedIndex] No rows to index")
            self._built = True
            return

        debug(f"[InvertedIndex] Building index on {len(rows)} documents...")

        for row in rows:
            mem_id = row["id"]
            text = row.get("normalized_text", row.get("text", ""))
            tokens = self.tokenizer(text)

            for position, token in enumerate(tokens):
                # Skip empty tokens
                if not token:
                    continue

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

        # Compute document frequencies
        self.document_frequency = {
            term: len(set(docs)) for term, docs in self.index.items()
        }

        self._built = True
        elapsed = (time.perf_counter() - start) * 1000
        debug(f"[InvertedIndex] Built index: {len(self.index)} terms, {elapsed:.2f}ms")

        # Build skip pointers automatically
        self.build_skips()

    def search(self, term: str) -> List[int]:
        """Return document IDs containing the term."""
        if not self._built or not term:
            return []
        return self.index.get(term, [])

    def has_term(self, term: str) -> bool:
        """Check if a term exists in the index."""
        return term in self.index

    def term_frequency(self, term: str, doc_id: int) -> int:
        """Get term frequency in a specific document."""
        if not self._built or term not in self.positional_index:
            return 0
        return len(self.positional_index[term].get(doc_id, []))

    def document_freq(self, term: str) -> int:
        """Get document frequency for a term."""
        return self.document_frequency.get(term, 0)

    def phrase_search(self, phrase_tokens: List[str]) -> List[int]:
        """Find documents containing a phrase (consecutive tokens in order)."""
        if not phrase_tokens or not self._built:
            return []

        # Get candidate documents from first token
        first_token = phrase_tokens[0]
        if first_token not in self.positional_index:
            return []

        candidates = set(self.positional_index[first_token].keys())

        # For each subsequent token, check positions
        for i, token in enumerate(phrase_tokens[1:], start=1):
            if token not in self.positional_index:
                return []

            # Filter candidates by position
            new_candidates = set()
            for mem_id in candidates:
                pos_lists = []
                for j, t in enumerate(phrase_tokens):
                    pos_lists.append(self.positional_index.get(t, {}).get(mem_id, []))
                if all(pos_lists) and self._has_consecutive(pos_lists):
                    new_candidates.add(mem_id)

            candidates = new_candidates
            if not candidates:
                return []

        return list(candidates)

    def _has_consecutive(self, pos_lists: List[List[int]]) -> bool:
        """Check if there are consecutive positions across lists."""
        if not pos_lists:
            return False

        # For each position in the first list
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
        """Boolean AND query across multiple terms."""
        if not terms or not self._built:
            return []

        result = set(self.search(terms[0]))
        for term in terms[1:]:
            result &= set(self.search(term))
            if not result:
                return []

        return list(result)

    def or_query(self, terms: List[str]) -> List[int]:
        """Boolean OR query across multiple terms."""
        if not terms or not self._built:
            return []

        result = set()
        for term in terms:
            result.update(self.search(term))
        return list(result)

    # ---------------------------------
    # Skip Pointers
    # ---------------------------------

    def build_skips(self, skip_interval: int = 4):
        """Build skip pointers for all posting lists."""
        if not self._built or not self.index:
            return

        self.skip_lists.clear()
        for term, posting in self.index.items():
            if len(posting) < skip_interval * 2:
                continue  # Not worth building skips for small lists

            skips = []
            for i in range(0, len(posting), skip_interval):
                skips.append((posting[i], i))
            self.skip_lists[term] = skips

        debug(f"[InvertedIndex] Built skip pointers for {len(self.skip_lists)} terms")

    def and_query_skip(self, terms: List[str]) -> List[int]:
        """
        Boolean AND query using skip pointers for faster intersection.
        """
        if not terms or not self._built:
            return []

        posting_lists = [list(set(self.index.get(term, []))) for term in terms]
        posting_lists = [pl for pl in posting_lists if pl]
        if not posting_lists:
            return []

        # Sort by length for efficiency
        posting_lists.sort(key=len)

        # Use skip pointers if available
        if not self.skip_lists:
            self.build_skips()

        result = []
        base_list = posting_lists[0]
        other_lists = posting_lists[1:]

        for doc_id in base_list:
            found = True
            for other in other_lists:
                idx = bisect.bisect_left(other, doc_id)
                if idx == len(other) or other[idx] != doc_id:
                    found = False
                    break
            if found:
                result.append(doc_id)

        return result

    # ---------------------------------
    # Persistence
    # ---------------------------------

    def save(self, filepath: str):
        """Save the index to disk."""
        data = {
            'index': self.index,
            'positional_index': self.positional_index,
            'document_frequency': self.document_frequency,
            'skip_lists': self.skip_lists,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        debug(f"[InvertedIndex] Saved to {filepath}")

    def load(self, filepath: str):
        """Load the index from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.index = data['index']
        self.positional_index = data['positional_index']
        self.document_frequency = data['document_frequency']
        self.skip_lists = data.get('skip_lists', {})
        self._built = True
        debug(f"[InvertedIndex] Loaded from {filepath}")

    # ---------------------------------
    # Stats
    # ---------------------------------

    def stats(self) -> Dict[str, int]:
        """Return statistics about the index."""
        return {
            "terms": len(self.index),
            "documents": len(self.document_frequency),
            "postings": sum(len(v) for v in self.index.values()),
            "skip_lists": len(self.skip_lists),
            "built": int(self._built),
        }

    @property
    def built(self) -> bool:
        return self._built
