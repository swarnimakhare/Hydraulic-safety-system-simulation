"""
Monte Carlo batch runner over random failure realizations.
"""

from __future__ import annotations

from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd

from failure_engine import FailureEngine
from metrics import SafetyMetrics
from simulation import HydraulicSimulator
from utils import set_global_seed


class MonteCarloRunner:
    """
    Repeatedly builds graphs (optional), injects failures, simulates, and records metrics.
    """

    def __init__(self, seed: int | None = 42) -> None:
        self.seed = seed

    def run(
        self,
        graph_builder_fn: Callable[[], nx.DiGraph],
        n_iterations: int,
        edge_failure_rate: float,
        node_failure_rate: float,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Run ``n_iterations`` scenarios.

        Each iteration:
        - Fresh graph from ``graph_builder_fn``
        - Random failures + isolation
        - Steady-state simulation
        - Safety metrics snapshot
        """
        rng_seed = int(seed if seed is not None else (self.seed or 0))
        rows: list[dict] = []

        sim = HydraulicSimulator()
        metrics = SafetyMetrics()

        for it in range(int(n_iterations)):
            set_global_seed(rng_seed + it)

            G = graph_builder_fn().copy()
            fe = FailureEngine(seed=rng_seed + it)
            fe.inject_random_failures(G, edge_failure_rate, node_failure_rate, seed=rng_seed + it)
            fe.apply_isolation_logic(G)
            sim.run_steady_state(G)

            func_ratio = metrics.functional_actuator_ratio(G, sim)
            ps = metrics.pressure_stats(G, sim)
            red = metrics.redundancy_score(G)
            safety = metrics.composite_safety_score(G, sim)
            sizes = metrics.disconnected_subgraph_sizes(G)
            largest = sizes[0] if sizes else 0

            rows.append(
                {
                    "iteration": it,
                    "functional_ratio": func_ratio,
                    "mean_pressure": ps["mean"],
                    "variance_pressure": ps["variance"],
                    "redundancy_score": red,
                    "safety_score": safety,
                    "largest_component_size": largest,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def summary_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Mean and std for each numeric metric column (skips ``iteration``)."""
        out: dict[str, dict[str, float]] = {}
        for col in df.columns:
            if col == "iteration":
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            out[col] = {"mean": float(s.mean()), "std": float(s.std(ddof=0))}
        return out
