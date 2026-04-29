"""
Utility helpers: reproducible seeds, CSV export, graph persistence, and insights text.
"""

from __future__ import annotations

import pickle
import random
from typing import Any

import numpy as np
import pandas as pd


def set_global_seed(seed: int) -> None:
    """Set NumPy and Python `random` seeds for reproducible runs."""
    np.random.seed(seed)
    random.seed(seed)


def export_results_csv(df: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame to CSV (UTF-8)."""
    df.to_csv(filename, index=False, encoding="utf-8")


def save_graph(graph: Any, filepath: str) -> None:
    """Pickle a NetworkX graph to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(filepath: str) -> Any:
    """Load a pickled NetworkX graph."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def get_topology_insights(results_dict: dict[str, pd.DataFrame]) -> str:
    """
    Produce a short human-readable comparison of topology Monte Carlo results.

    Parameters
    ----------
    results_dict
        Keys: topology names (e.g. "Tree", "Looped", "Segmented").
        Values: DataFrames from MonteCarloRunner with at least `safety_score`
        and `mean_pressure` columns.
    """
    if not results_dict:
        return "No Monte Carlo results available to summarize."

    means: dict[str, dict[str, float]] = {}
    for name, df in results_dict.items():
        if df is None or df.empty:
            continue
        row: dict[str, float] = {}
        if "safety_score" in df.columns:
            row["safety_score"] = float(df["safety_score"].mean())
        if "mean_pressure" in df.columns:
            row["mean_pressure"] = float(df["mean_pressure"].mean())
        if "redundancy_score" in df.columns:
            row["redundancy_score"] = float(df["redundancy_score"].mean())
        means[name] = row

    if not means:
        return "Monte Carlo frames were empty; run simulations to populate insights."

    # Prefer comparing against Tree as baseline when present
    baseline_name = "Tree" if "Tree" in means else next(iter(means))
    baseline = means[baseline_name]
    parts: list[str] = []

    for other in sorted(means.keys()):
        if other == baseline_name:
            continue
        o = means[other]
        if "safety_score" in baseline and "safety_score" in o and baseline["safety_score"] > 0:
            pct = (o["safety_score"] - baseline["safety_score"]) / baseline["safety_score"] * 100.0
            parts.append(
                f"{other} topology changes composite safety by about {pct:+.1f}% "
                f"versus {baseline_name} (mean safety score)."
            )
        if "mean_pressure" in baseline and "mean_pressure" in o and baseline["mean_pressure"] > 0:
            mp_pct = (o["mean_pressure"] - baseline["mean_pressure"]) / baseline["mean_pressure"] * 100.0
            parts.append(f"Mean actuator pressure differs by about {mp_pct:+.1f}% ({other} vs {baseline_name}).")
        if "redundancy_score" in baseline and "redundancy_score" in o and baseline["redundancy_score"] > 1e-6:
            ratio = o["redundancy_score"] / baseline["redundancy_score"]
            parts.append(f"Redundancy score ratio ({other} / {baseline_name}) is about {ratio:.2f}x.")

    if not parts:
        best = max(means.items(), key=lambda kv: kv[1].get("safety_score", 0.0))
        return (
            f"Highest mean composite safety among available topologies: {best[0]} "
            f"(mean safety score ≈ {best[1].get('safety_score', float('nan')):.3f})."
        )

    return " ".join(parts)
