"""
Safety metrics for hydraulic network resilience analysis.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from simulation import HydraulicSimulator


class SafetyMetrics:
    """
    Computes functional ratios, pressure statistics, topology redundancy, and composite scores.
    """

    def __init__(self, pressure_threshold: float = 1500.0, top_n_critical: int = 8) -> None:
        self.pressure_threshold = float(pressure_threshold)
        self.top_n_critical = int(top_n_critical)

    def functional_actuator_ratio(self, graph: nx.DiGraph, simulator: HydraulicSimulator) -> float:
        """Fraction of actuators whose pressure is at or above the threshold."""
        acts = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "actuator"]
        if not acts:
            return 0.0
        ok = 0
        for n in acts:
            p = float(graph.nodes[n].get("pressure", 0.0))
            if simulator.is_actuator_functional(p, self.pressure_threshold):
                ok += 1
        return ok / len(acts)

    def pressure_stats(self, graph: nx.DiGraph, simulator: HydraulicSimulator) -> dict[str, float]:
        """Mean, variance, min, max of actuator pressures (empty => zeros)."""
        vals = [
            float(graph.nodes[n].get("pressure", 0.0))
            for n, d in graph.nodes(data=True)
            if d.get("node_type") == "actuator"
        ]
        if not vals:
            return {"mean": 0.0, "variance": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(vals, dtype=float)
        return {
            "mean": float(arr.mean()),
            "variance": float(arr.var()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    def disconnected_subgraph_sizes(self, graph: nx.DiGraph) -> list[int]:
        """Sizes of weakly connected components (counts all nodes in each)."""
        if graph.number_of_nodes() == 0:
            return []
        comps = list(nx.weakly_connected_components(graph))
        return sorted([len(c) for c in comps], reverse=True)

    def redundancy_score(self, graph: nx.DiGraph) -> float:
        """
        Average node connectivity between pumps and actuators (undirected simplification).

        Pairs in different components contribute 0. Uses ``networkx.node_connectivity`` which
        counts node-independent paths to some extent via Menger's theorem.
        """
        pumps = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "pump"]
        acts = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "actuator"]
        if not pumps or not acts:
            return 0.0

        U = graph.to_undirected()
        scores: list[float] = []
        for p in pumps:
            for a in acts:
                if p == a:
                    continue
                if not nx.has_path(U, p, a):
                    scores.append(0.0)
                    continue
                try:
                    k = float(nx.node_connectivity(U, p, a))
                except Exception:
                    k = 0.0
                scores.append(k)
        return float(np.mean(scores)) if scores else 0.0

    def critical_nodes(self, graph: nx.DiGraph) -> list[tuple[Any, float]]:
        """Top-N nodes by betweenness centrality excluding pumps and reservoirs."""
        if graph.number_of_nodes() == 0:
            return []
        U = graph.to_undirected()
        bc = nx.betweenness_centrality(U, normalized=True)
        ranked: list[tuple[Any, float]] = []
        for n, c in bc.items():
            nt = graph.nodes[n].get("node_type", "")
            if nt in ("pump", "reservoir"):
                continue
            ranked.append((n, float(c)))
        ranked.sort(key=lambda t: t[1], reverse=True)
        return ranked[: self.top_n_critical]

    def composite_safety_score(self, graph: nx.DiGraph, simulator: HydraulicSimulator) -> float:
        """
        Weighted composite:

        0.4 * functional_ratio
        + 0.3 * normalized_mean_pressure
        + 0.2 * normalized_redundancy
        + 0.1 * (1 - largest_disconnected_ratio)
        """
        func = self.functional_actuator_ratio(graph, simulator)
        ps = self.pressure_stats(graph, simulator)
        mean_p = ps["mean"]
        norm_p = float(np.clip(mean_p / max(simulator.pump_pressure_psi, 1e-6), 0.0, 1.0))

        red = self.redundancy_score(graph)
        norm_red = float(np.clip(red / 5.0, 0.0, 1.0))  # typical small integers

        sizes = self.disconnected_subgraph_sizes(graph)
        n = graph.number_of_nodes()
        largest = sizes[0] if sizes else 0
        largest_ratio = float(largest / max(n, 1))
        frag = 1.0 - largest_ratio

        score = 0.4 * func + 0.3 * norm_p + 0.2 * norm_red + 0.1 * frag
        return float(np.clip(score, 0.0, 1.0))
