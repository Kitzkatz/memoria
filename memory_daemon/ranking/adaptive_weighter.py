import json
import re
import os
from typing import Dict, List, Optional
from pathlib import Path

from cache.config import settings
from core.logger import debug


def compute_signal_deltas(benchmark_file: str) -> Dict[str, float]:
    """
    Read benchmark results from the JSON file written by BenchmarkWriter.
    Compute average signal deltas from near-misses (expected_rank > 3).

    Returns:
        Dict mapping signal names to average delta values.
        Signals: semantic, importance, recency, token, feedback,
                 entity, subject, attribute, tfidf, bm25
    """
    if not os.path.exists(benchmark_file):
        debug(f"[AdaptiveWeighter] Benchmark file not found: {benchmark_file}")
        return {}

    try:
        with open(benchmark_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        debug(f"[AdaptiveWeighter] Invalid JSON in {benchmark_file}: {e}")
        return {}

    records = data.get("records", [])
    if not records:
        debug("[AdaptiveWeighter] No records found in benchmark file")
        return {}

    deltas = {
        "semantic": 0.0,
        "importance": 0.0,
        "recency": 0.0,
        "token": 0.0,
        "feedback": 0.0,
        "entity": 0.0,
        "subject": 0.0,
        "attribute": 0.0,
        "tfidf": 0.0,
        "bm25": 0.0,
    }

    count = 0
    for record in records:
        expected_rank = record.get("expected_rank")
        if expected_rank is None or expected_rank <= 3:
            continue

        candidates = record.get("candidates", [])
        if not candidates or expected_rank >= len(candidates):
            continue

        winner = candidates[0]
        expected = candidates[expected_rank - 1]

        w_diag = winner.get("diagnostics", {}).get("ranker", {})
        e_diag = expected.get("diagnostics", {}).get("ranker", {})

        for signal in deltas:
            w_val = w_diag.get(signal, 0.0)
            e_val = e_diag.get(signal, 0.0)
            deltas[signal] += w_val - e_val

        count += 1

    if count > 0:
        for signal in deltas:
            deltas[signal] /= count

    return deltas


def adjust_weights(deltas: Dict[str, float], step_size: float = 0.02) -> Dict[str, float]:
    """
    Adjust ranking weights based on signal deltas.

    CORRECT LOGIC:
    - If delta is NEGATIVE (expected wins on this signal), INCREASE the weight
    - If delta is POSITIVE (winner wins on this signal), DECREASE the weight
    """
    if not deltas:
        debug("[AdaptiveWeighter] No deltas provided, returning default weights")
        return get_default_weights()

    # Get current weights from settings
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
        "bm25": getattr(settings, "RANKING_BM25", 0.10),
    }

    valid_signals = [s for s in weights if s in deltas]
    if not valid_signals:
        debug("[AdaptiveWeighter] No matching signals found, returning default weights")
        return weights

    # Adjust weights based on signal deltas
    for signal in valid_signals:
        delta = deltas[signal]

        # CORRECTED LOGIC:
        # Negative delta = expected wins = need MORE weight
        # Positive delta = winner wins = need LESS weight
        # Adjustment direction is inverted from before
        adjustment = -step_size * (delta / (abs(delta) + 0.001))
        adjustment = max(-step_size, min(step_size, adjustment))

        weights[signal] = max(0.01, min(0.50, weights[signal] + adjustment))

    # Normalize to sum to 1.0
    total = sum(weights.values())
    for signal in weights:
        weights[signal] /= total

    # Clamp to reasonable bounds
    for signal in weights:
        weights[signal] = max(0.01, min(0.50, weights[signal]))

    return weights


def get_default_weights() -> Dict[str, float]:
    """Get default weights from settings."""
    return {
        "semantic": getattr(settings, "RANKING_SEMANTIC", 0.20),
        "importance": getattr(settings, "RANKING_IMPORTANCE", 0.08),
        "recency": getattr(settings, "RANKING_RECENCY", 0.05),
        "token": getattr(settings, "RANKING_TOKEN", 0.07),
        "feedback": getattr(settings, "RANKING_FEEDBACK", 0.02),
        "entity": getattr(settings, "RANKING_ENTITY", 0.23),
        "subject": getattr(settings, "RANKING_SUBJECT", 0.20),
        "attribute": getattr(settings, "RANKING_ATTRIBUTE", 0.15),
        "tfidf": getattr(settings, "RANKING_TFIDF", 0.08),
        "bm25": getattr(settings, "RANKING_BM25", 0.10),
    }


def print_weight_comparison(old_weights: Dict[str, float], new_weights: Dict[str, float]):
    """Pretty print the weight changes."""
    print("\n[WEIGHT ADJUSTMENT]")
    print(f"{'Signal':<15} {'Old':<8} {'New':<8} {'Δ':<8}")
    print("-" * 40)

    all_signals = set(old_weights.keys()) | set(new_weights.keys())

    for signal in sorted(all_signals):
        old = old_weights.get(signal, 0.0)
        new = new_weights.get(signal, 0.0)
        delta = new - old
        print(f"{signal:<15} {old:<8.4f} {new:<8.4f} {delta:<+8.4f}")


def save_weights_to_config(new_weights: Dict[str, float], config_path: Optional[str] = None):
    """
    Update config.py with new weights.
    """
    if config_path is None:
        config_path = os.path.join("cache", "config.py")

    if not os.path.exists(config_path):
        debug(f"[AdaptiveWeighter] Config file not found: {config_path}")
        return

    try:
        with open(config_path, 'r') as f:
            content = f.read()

        backup_path = config_path + ".backup"
        with open(backup_path, 'w') as f:
            f.write(content)
        debug(f"[AdaptiveWeighter] Backed up config to {backup_path}")

        for signal, value in new_weights.items():
            pattern = f"RANKING_{signal.upper()}: float = [0-9.]+"
            replacement = f"RANKING_{signal.upper()}: float = {value:.4f}"
            content = re.sub(pattern, replacement, content)

            pattern2 = f"RANKING_{signal.upper()}: float=[0-9.]+"
            replacement2 = f"RANKING_{signal.upper()}: float={value:.4f}"
            content = re.sub(pattern2, replacement2, content)

        with open(config_path, 'w') as f:
            f.write(content)

        debug(f"[AdaptiveWeighter] Updated {config_path} with new weights")

    except Exception as e:
        debug(f"[AdaptiveWeighter] Error saving weights: {e}")


def adaptive_weighter_pipeline(benchmark_file: str, dry_run: bool = False, step_size: float = 0.02):
    """
    Run the full adaptive weighting pipeline.
    """
    debug("[AdaptiveWeighter] Starting pipeline...")

    deltas = compute_signal_deltas(benchmark_file)
    if not deltas:
        debug("[AdaptiveWeighter] No deltas computed, skipping")
        return None

    old_weights = get_default_weights()
    new_weights = adjust_weights(deltas, step_size=step_size)

    print_weight_comparison(old_weights, new_weights)

    if not dry_run:
        save_weights_to_config(new_weights)
        debug("[AdaptiveWeighter] Pipeline complete")
    else:
        debug("[AdaptiveWeighter] Dry run complete (no changes saved)")

    return {
        "old_weights": old_weights,
        "new_weights": new_weights,
        "deltas": deltas,
    }
