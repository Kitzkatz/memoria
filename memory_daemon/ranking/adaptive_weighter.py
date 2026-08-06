# ranking/adaptive_weighter.py
import json
from typing import Dict, List
from cache.config import settings

def compute_signal_deltas(benchmark_file: str) -> Dict[str, float]:
    """
    Read benchmark results from the JSON file written by BenchmarkWriter.
    Compute average signal deltas from near-misses (expected_rank > 3).
    """
    with open(benchmark_file, 'r') as f:
        data = json.load(f)

    records = data.get("records", [])
    deltas = {
        "semantic": 0.0,
        "importance": 0.0,
        "recency": 0.0,
        "token": 0.0,
        "feedback": 0.0,
        "attribute_boost": 0.0,
    }

    count = 0
    for record in records:
        expected_rank = record.get("expected_rank")
        if expected_rank is None or expected_rank <= 3:
            continue  # skip successful and never retrieved

        candidates = record.get("candidates", [])
        if not candidates or expected_rank >= len(candidates):
            continue

        winner = candidates[0]
        expected = candidates[expected_rank - 1]

        w_diag = winner.get("diagnostics", {}).get("ranker", {})
        e_diag = expected.get("diagnostics", {}).get("ranker", {})

        for signal in deltas:
            w = w_diag.get(signal, 0.0)
            e = e_diag.get(signal, 0.0)
            deltas[signal] += w - e
            count += 1

    if count > 0:
        for signal in deltas:
            deltas[signal] /= count

    return deltas


def adjust_weights(deltas: Dict[str, float]) -> Dict[str, float]:
    """
    Adjust ranking weights based on signal deltas.
    Signals that helped winners more get their weights increased.
    """
    weights = {
        "semantic": getattr(settings, "RANKING_SEMANTIC", 0.20),
        "importance": getattr(settings, "RANKING_IMPORTANCE", 0.08),
        "recency": getattr(settings, "RANKING_RECENCY", 0.05),
        "token": getattr(settings, "RANKING_TOKEN", 0.07),
        "feedback": getattr(settings, "RANKING_FEEDBACK", 0.02),
        "entity": getattr(settings, "RANKING_ENTITY", 0.23),
        "subject": getattr(settings, "RANKING_SUBJECT", 0.20),
        "attribute": getattr(settings, "RANKING_ATTRIBUTE", 0.15),
        "tfidf": getattr(settings, "RANKING_TFIDF", 0.08),
    }

    avg_delta = sum(deltas.values()) / max(len(deltas), 1)

    for signal in weights:
        if signal in deltas:
            if deltas[signal] > avg_delta:
                weights[signal] = min(weights[signal] + 0.02, 0.40)
            elif deltas[signal] < avg_delta:
                weights[signal] = max(weights[signal] - 0.02, 0.01)

    # Normalize to sum to 1
    total = sum(weights.values())
    for signal in weights:
        weights[signal] /= total

    return weights


def print_weight_comparison(old_weights: Dict[str, float], new_weights: Dict[str, float]):
    """Pretty print the weight changes."""
    print("\n[WEIGHT ADJUSTMENT]")
    print(f"{'Signal':<15} {'Old':<8} {'New':<8} {'Δ':<8}")
    print("-" * 40)
    for signal in old_weights:
        old = old_weights.get(signal, 0.0)
        new = new_weights.get(signal, 0.0)
        delta = new - old
        print(f"{signal:<15} {old:<8.4f} {new:<8.4f} {delta:<+8.4f}")

def save_weights_to_config(new_weights: Dict[str, float]):
    """Update config.py with new weights."""
    import re
    with open('cache/config.py', 'r') as f:
        content = f.read()
    
    for signal, value in new_weights.items():
        pattern = f"RANKING_{signal.upper()}: float = [0-9.]+"
        replacement = f"RANKING_{signal.upper()}: float = {value:.4f}"
        content = re.sub(pattern, replacement, content)
    
    with open('cache/config.py', 'w') as f:
        f.write(content)
    print(f"[AdaptiveWeighter] Updated config.py with new weights")
