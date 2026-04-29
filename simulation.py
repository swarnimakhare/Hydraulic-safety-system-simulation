"""
Steady-state hydraulic pressure propagation on a directed graph.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


class HydraulicSimulator:
    """
    Simplified hydraulic steady-state model:

    - Pumps impose a fixed source pressure (PSI).
    - Along each active edge, pressure drops by ``flow_nominal * resistance`` where
      ``flow_nominal`` is a nominal per-edge flow (L/min scale) derived from capacity.
    - At junctions, the simulator keeps the **maximum** arriving pressure (best-path analogy).
    - Failed / inactive nodes and failed edges are ignored.
    """

    PUMP_PRESSURE_PSI = 3000.0
    FLOW_NOMINAL_SCALE = 1.0  # scales resistance to PSI drop

    def __init__(self, pump_pressure_psi: float | None = None) -> None:
        self.pump_pressure_psi = float(pump_pressure_psi or self.PUMP_PRESSURE_PSI)

    def run_steady_state(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Compute node pressures via multi-source propagation (BFS-like relaxation).

        Disconnected nodes retain pressure 0. Unreachable actuators => 0.
        """
        G = graph
        # Initialize
        for n in G.nodes():
            G.nodes[n]["pressure"] = 0.0

        # Multi-source max-pressure relaxation (finite graph => converges in one pass per wave)
        # Use iterative approach: repeat until no change (handles diamond DAGs); cap iterations.
        pumps = [n for n, d in G.nodes(data=True) if d.get("node_type") == "pump" and d.get("is_active", True)]
        for p in pumps:
            if G.nodes[p].get("is_active", True):
                G.nodes[p]["pressure"] = max(G.nodes[p]["pressure"], self.pump_pressure_psi)

        changed = True
        it = 0
        # Relax until quiescence; long paths may need multiple sweeps on cyclic graphs.
        max_iter = max(10, 5 * G.number_of_nodes() + 10)
        while changed and it < max_iter:
            changed = False
            it += 1
            for u, v, ed in G.edges(data=True):
                if not G.nodes[u].get("is_active", True):
                    continue
                if not G.nodes[v].get("is_active", True):
                    continue
                if ed.get("is_failed", False):
                    continue
                res = float(ed.get("resistance", 0.0))
                cap = float(ed.get("flow_capacity", 1.0))
                # Nominal flow inversely related to resistance path — keep simple stable mapping
                flow_nom = self.FLOW_NOMINAL_SCALE * min(1.0, 50.0 / max(cap, 1e-6))
                drop = flow_nom * res
                pu = float(G.nodes[u]["pressure"])
                cand = max(0.0, pu - drop)
                if cand > float(G.nodes[v]["pressure"]) + 1e-9:
                    G.nodes[v]["pressure"] = cand
                    changed = True

        # Ensure non-active / failed nodes read as 0 for downstream metrics consistency
        for n, d in G.nodes(data=True):
            if not d.get("is_active", True):
                G.nodes[n]["pressure"] = 0.0

        return G

    def get_actuator_pressures(self, graph: nx.DiGraph) -> dict[int, float]:
        """Return `{node_id: pressure}` for nodes tagged as actuators."""
        out: dict[int, float] = {}
        for n, d in graph.nodes(data=True):
            if d.get("node_type") == "actuator":
                out[n] = float(d.get("pressure", 0.0))
        return out

    @staticmethod
    def is_actuator_functional(pressure: float, threshold: float = 1500.0) -> bool:
        """True if supplied pressure meets minimum operational threshold."""
        return float(pressure) >= float(threshold)
