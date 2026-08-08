from core.logger import debug
"""
mmr_reranker.py

Vector-space Maximal Marginal Relevance
"""

import math
import numpy as np
from typing import Dict, List, Optional


class MMRReranker:

    def __init__(
        self,
        lambda_param: float = 0.50,
        debug: bool = False,
        adaptive_lambda: bool = True
    ):
        self.lambda_param = lambda_param
        self.debug = debug
        self.adaptive_lambda = adaptive_lambda
        self.last_diagnostics = {}

    # ---------------------------------
    # Cosine Similarity (Vectorized)
    # ---------------------------------

    def _cosine_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix using numpy."""
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        normalized = embeddings / norms
        # Cosine similarity = dot product of normalized vectors
        return np.dot(normalized, normalized.T)

    # ---------------------------------
    # Diagnostics
    # ---------------------------------

    def _build_diagnostics(self, before_ids, after, lambda_used):
        after_ids = [c.memory.id for c in after]

        moves = 0
        for idx, item in enumerate(after_ids):
            if item in before_ids:
                old = before_ids.index(item)
                moves += abs(old - idx)

        return {
            "candidate_count": len(before_ids),
            "returned_count": len(after),
            "reordered": before_ids != after_ids,
            "total_moves": moves,
            "lambda": lambda_used,
            "avg_mmr_score": round(
                sum(c.mmr_score for c in after) / max(len(after), 1),
                4
            ),
            "avg_diversity": round(
                sum(c.diversity_score for c in after) / max(len(after), 1),
                4
            ),
        }

    # ---------------------------------
    # Maximal Marginal Relevance (Vectorized)
    # ---------------------------------

    def rerank(self, candidates, k=20):
        if not candidates:
            return []

        # Copy candidates so we never mutate the caller's list
        working = [c.model_copy() for c in candidates]
        working.sort(key=lambda c: c.normalized_score, reverse=True)

        # Cap at 50 for MMR efficiency (increased from 25 for better results)
        working = working[:50]
        before_ids = [c.memory.id for c in working]

        k = min(k, len(working))

        # ---- Extract embeddings ----
        embeddings = []
        for c in working:
            emb = c.embedding
            if emb is None:
                emb = np.zeros(384)  # fallback
            elif not isinstance(emb, np.ndarray):
                emb = np.array(emb)
            embeddings.append(emb)

        embeddings = np.array(embeddings, dtype=np.float32)

        # ---- Compute similarity matrix once ----
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        normalized = embeddings / norms
        sim_matrix = np.dot(normalized, normalized.T)

        # ---- Adaptive Lambda ----
        current_lambda = self.lambda_param
        if self.adaptive_lambda:
            candidate_count = len(working)
            if candidate_count <= 10:
                current_lambda = 0.95
            elif candidate_count <= 30:
                current_lambda = 0.90
            elif candidate_count <= 75:
                current_lambda = 0.85
            else:
                current_lambda = 0.80

        # ---- Seed with highest relevance ----
        selected_indices = []
        remaining_indices = list(range(len(working)))

        # Select first candidate (highest relevance)
        first_idx = max(remaining_indices, key=lambda i: working[i].normalized_score)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # ---- Greedy MMR Selection ----
        effective_lambda = min(1.0, current_lambda + 0.10)

        while remaining_indices and len(selected_indices) < k:
            best_idx = -1
            best_score = float("-inf")

            # Get similarity to selected set for all remaining
            sim_to_selected = sim_matrix[remaining_indices][:, selected_indices]

            # For each remaining candidate, compute max similarity to selected
            if len(selected_indices) == 1:
                # If only one selected, max is the similarity to that one
                max_similarities = sim_to_selected.flatten()
            else:
                max_similarities = np.max(sim_to_selected, axis=1)

            # Scores: relevance (normalized_score) - diversity (max similarity)
            for idx_in_batch, remaining_idx in enumerate(remaining_indices):
                relevance = working[remaining_idx].normalized_score
                diversity_penalty = max_similarities[idx_in_batch]

                mmr_score = effective_lambda * relevance - (1.0 - effective_lambda) * diversity_penalty

                working[remaining_idx].mmr_score = mmr_score
                working[remaining_idx].diversity_score = diversity_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = remaining_idx

            if best_idx == -1:
                break

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Build result in original order
        selected = [working[i] for i in selected_indices]

        self.last_diagnostics = self._build_diagnostics(
            before_ids, selected, current_lambda
        )

        return selected
