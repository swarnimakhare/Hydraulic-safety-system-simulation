"""
Random failure injection and isolation valve logic.
"""

from __future__ import annotations

import random
from typing import Any

import networkx as nx
import numpy as np


class FailureEngine:
    """
    Injects random failures into hydraulic graphs and applies valve isolation rules.

    Failure marking uses ``is_failed`` on edges/nodes and ``is_active`` on nodes (False when failed).
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def inject_random_failures(
        self,
        graph: nx.DiGraph,
        edge_failure_rate: float,
        node_failure_rate: float,
        seed: int | None = None,
    ) -> nx.DiGraph:
        """
        Stochastically fail edges and nodes.

        - Each edge fails independently with probability ``edge_failure_rate * edge.failure_probability``.
        - Each eligible node (not pump/reservoir) fails with probability ``node_failure_rate``.
        """
        np_rng = np.random.default_rng(seed) if seed is not None else self._np_rng

        efr = float(np.clip(edge_failure_rate, 0.0, 1.0))
        nfr = float(np.clip(node_failure_rate, 0.0, 1.0))

        for _, _, d in graph.edges(data=True):
            p = efr * float(d.get("failure_probability", 0.0))
            if np_rng.random() < p:
                d["is_failed"] = True

        for n, d in graph.nodes(data=True):
            nt = d.get("node_type", "")
            if nt in ("pump", "reservoir"):
                continue
            if np_rng.random() < nfr:
                d["is_failed"] = True
                d["is_active"] = False

        # Edge failure: mark adjacent non-reservoir nodes? Spec only marks edges.
        # If an edge fails, it is skipped in simulation; optionally deactivate endpoints — skip.

        return graph

    def apply_isolation_logic(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        For each failed node, close all incident valve edges (``is_valve=True``).

        Failed nodes are treated as inactive and excluded from the active subgraph for metrics.
        """
        failed_nodes = [n for n, d in graph.nodes(data=True) if d.get("is_failed", False)]

        for n in failed_nodes:
            # Close valves adjacent to failed node
            for u, v, d in list(graph.edges(data=True)):
                if not d.get("is_valve", False):
                    continue
                if u == n or v == n:
                    d["is_failed"] = True
            graph.nodes[n]["is_active"] = False

        # Edges incident to inactive nodes are effectively unusable
        for u, v, d in graph.edges(data=True):
            if not graph.nodes[u].get("is_active", True) or not graph.nodes[v].get("is_active", True):
                d["is_failed"] = True

        return graph

    def reset_failures(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Restore default failure flags and node activity."""
        for _, _, d in graph.edges(data=True):
            d["is_failed"] = False
        for _, d in graph.nodes(data=True):
            d["is_failed"] = False
            d["is_active"] = True
        return graph
