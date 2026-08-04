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
