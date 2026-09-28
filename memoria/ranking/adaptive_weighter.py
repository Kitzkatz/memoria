
import json
import re
import os
from typing import Dict, Optional

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
        debug(
            f"[AdaptiveWeighter] Benchmark file not found: "
            f"{benchmark_file}"
        )
        return {}

    try:
        with open(benchmark_file, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        debug(
            f"[AdaptiveWeighter] Invalid JSON in "
            f"{benchmark_file}: {e}"
        )
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


def _normalize_bounded_weights(
    weights: Dict[str, float],
    minimum: float = 0.01,
    maximum: float = 0.50,
) -> Dict[str, float]:
    """
    Normalize weights so they sum to 1.0 while respecting bounds.

    Values outside the bounds are fixed at the relevant boundary and
    the remaining residual is redistributed among the unfixed values.
    A final correction is applied to eliminate accumulated floating-
    point drift as far as Python floating-point arithmetic permits.
    """
    if not weights:
        return {}

    result = {
        signal: max(minimum, min(maximum, float(value)))
        for signal, value in weights.items()
    }

    fixed = set()

    # Iteratively redistribute the residual while respecting bounds.
    for _ in range(len(result) + 1):
        total = sum(result.values())
        residual = 1.0 - total

        if abs(residual) <= 1e-15:
            break

        adjustable = [
            signal
            for signal in result
            if signal not in fixed
        ]

        if not adjustable:
            break

        share = residual / len(adjustable)
        changed = False

        for signal in adjustable:
            new_value = result[signal] + share

            if new_value < minimum:
                result[signal] = minimum
                fixed.add(signal)
                changed = True

            elif new_value > maximum:
                result[signal] = maximum
                fixed.add(signal)
                changed = True

            else:
                result[signal] = new_value

        if not changed:
            break

    # Final correction. Apply the remaining tiny residual to a weight
    # with enough room to absorb it without violating the bounds.
    total = sum(result.values())
    residual = 1.0 - total

    if abs(residual) > 0.0:
        candidates = sorted(
            result,
            key=lambda signal: (
                result[signal]
                if residual < 0
                else -result[signal]
            ),
        )

        for signal in candidates:
            new_value = result[signal] + residual

            if minimum <= new_value <= maximum:
                result[signal] = new_value
                break

    # Force the final floating-point correction onto the largest
    # available component. This keeps the returned sum at 1.0 to
    # normal floating-point precision without changing the bounds.
    total = sum(result.values())
    residual = 1.0 - total

    if residual != 0.0:
        for signal in sorted(
            result,
            key=lambda s: result[s],
            reverse=(residual > 0),
        ):
            new_value = result[signal] + residual

            if minimum <= new_value <= maximum:
                result[signal] = new_value
                break

    return result


def adjust_weights(
    deltas: Dict[str, float],
    step_size: float = 0.02,
) -> Dict[str, float]:
    """
    Adjust ranking weights based on signal deltas.

    CORRECT LOGIC:
    - If delta is NEGATIVE (expected wins on this signal), INCREASE
      the weight.
    - If delta is POSITIVE (winner wins on this signal), DECREASE
      the weight.
    """
    if not deltas:
        debug(
            "[AdaptiveWeighter] No deltas provided, "
            "returning default weights"
        )
        return get_default_weights()

    # Get current weights from settings.
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

    valid_signals = [
        signal
        for signal in weights
        if signal in deltas
    ]

    if not valid_signals:
        debug(
            "[AdaptiveWeighter] No matching signals found, "
            "returning default weights"
        )
        return weights

    # Adjust weights based on signal deltas.
    for signal in valid_signals:
        delta = deltas[signal]

        # Negative delta = expected wins = need MORE weight.
        # Positive delta = winner wins = need LESS weight.
        adjustment = -step_size * (
            delta / (abs(delta) + 0.001)
        )

        adjustment = max(
            -step_size,
            min(step_size, adjustment),
        )

        weights[signal] += adjustment

    # Apply bounds and normalize together so the final result:
    #   1. stays within [0.01, 0.50]
    #   2. sums to 1.0
    weights = _normalize_bounded_weights(weights)

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


def print_weight_comparison(
    old_weights: Dict[str, float],
    new_weights: Dict[str, float],
):
    """Pretty print the weight changes."""
    print("\n[WEIGHT ADJUSTMENT]")
    print(f"{'Signal':<15} {'Old':<8} {'New':<8} {'Δ':<8}")
    print("-" * 40)

    all_signals = set(old_weights.keys()) | set(new_weights.keys())

    for signal in sorted(all_signals):
        old = old_weights.get(signal, 0.0)
        new = new_weights.get(signal, 0.0)
        delta = new - old

        print(
            f"{signal:<15} "
            f"{old:<8.4f} "
            f"{new:<8.4f} "
            f"{delta:<+8.4f}"
        )


def save_weights_to_config(
    new_weights: Dict[str, float],
    config_path: Optional[str] = None,
):
    """
    Update config.py with new weights.
    """
    if config_path is None:
        config_path = os.path.join("cache", "config.py")

    if not os.path.exists(config_path):
        debug(
            f"[AdaptiveWeighter] Config file not found: "
            f"{config_path}"
        )
        return

    try:
        with open(config_path, "r") as f:
            content = f.read()

        backup_path = config_path + ".backup"

        with open(backup_path, "w") as f:
            f.write(content)

        debug(
            f"[AdaptiveWeighter] Backed up config to {backup_path}"
        )

        for signal, value in new_weights.items():
            pattern = (
                f"RANKING_{signal.upper()}: float = [0-9.]+"
            )
            replacement = (
                f"RANKING_{signal.upper()}: float = {value:.4f}"
            )
            content = re.sub(
                pattern,
                replacement,
                content,
            )

            pattern2 = (
                f"RANKING_{signal.upper()}: float=[0-9.]+"
            )
            replacement2 = (
                f"RANKING_{signal.upper()}: float={value:.4f}"
            )
            content = re.sub(
                pattern2,
                replacement2,
                content,
            )

        with open(config_path, "w") as f:
            f.write(content)

        debug(
            f"[AdaptiveWeighter] Updated {config_path} "
            f"with new weights"
        )

    except Exception as e:
        debug(
            f"[AdaptiveWeighter] Error saving weights: {e}"
        )


def adaptive_weighter_pipeline(
    benchmark_file: str,
    dry_run: bool = False,
    step_size: float = 0.02,
):
    """
    Run the full adaptive weighting pipeline.
    """
    debug("[AdaptiveWeighter] Starting pipeline...")

    deltas = compute_signal_deltas(benchmark_file)

    if not deltas:
        debug(
            "[AdaptiveWeighter] No deltas computed, skipping"
        )
        return None

    old_weights = get_default_weights()
    new_weights = adjust_weights(
        deltas,
        step_size=step_size,
    )

    print_weight_comparison(
        old_weights,
        new_weights,
    )

    if not dry_run:
        save_weights_to_config(new_weights)
        debug("[AdaptiveWeighter] Pipeline complete")
    else:
        debug(
            "[AdaptiveWeighter] Dry run complete "
            "(no changes saved)"
        )

    return {
        "old_weights": old_weights,
        "new_weights": new_weights,
        "deltas": deltas,
    }
