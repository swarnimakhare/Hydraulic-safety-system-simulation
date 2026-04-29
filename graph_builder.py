"""
Hydraulic network topology construction using NetworkX.
"""

from __future__ import annotations

import random
from typing import Any

import networkx as nx
import numpy as np


def _edge_attrs(
    length_m: float,
    flow_cap: float,
    resistance: float,
    fail_p: float,
    is_valve: bool = False,
) -> dict[str, Any]:
    return {
        "length": float(length_m),
        "flow_capacity": float(flow_cap),
        "resistance": float(resistance),
        "failure_probability": float(np.clip(fail_p, 0.0, 1.0)),
        "is_valve": bool(is_valve),
        "is_failed": False,
    }


def _node_attrs(node_type: str, pressure: float = 0.0, is_active: bool = True) -> dict[str, Any]:
    return {"node_type": node_type, "pressure": float(pressure), "is_active": bool(is_active)}


class GraphBuilder:
    """
    Build hydraulic-style directed graphs representing pumps, reservoirs, junctions, and actuators.

    All methods accept an optional `seed` for reproducible randomness (line lengths, resistances, etc.).
    """

    def __init__(self, seed: int | None = 42) -> None:
        self.seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def _fresh_rng(self, seed: int | None) -> tuple[random.Random, np.random.Generator]:
        if seed is None:
            return self._rng, self._np_rng
        return random.Random(seed), np.random.default_rng(seed)

    def build_tree_topology(self, num_pumps: int, num_actuators: int, seed: int | None = None) -> nx.DiGraph:
        """
        Branching tree: one independent path from a pump subtree to each actuator (no redundancy).

        Each pump has a dedicated reservoir feed. Internal junctions split flow toward actuators.
        """
        _, npr = self._fresh_rng(seed)
        G = nx.DiGraph()

        pump_ids: list[int] = []
        res_ids: list[int] = []
        for i in range(num_pumps):
            pid = G.number_of_nodes()
            rid = pid + 1
            pump_ids.append(pid)
            res_ids.append(rid)
            G.add_node(pid, **_node_attrs("pump"))
            G.add_node(rid, **_node_attrs("reservoir"))
            fl = float(npr.uniform(2.0, 8.0))
            G.add_edge(
                rid,
                pid,
                **_edge_attrs(length_m=fl, flow_cap=80.0, resistance=0.02 * fl, fail_p=0.01),
            )

        # Fan-out junction chain per pump to distribute actuators across pumps
        actuators_per = max(1, num_actuators // num_pumps)
        rem = num_actuators % num_pumps
        act_count = 0
        for pi, pump in enumerate(pump_ids):
            n_here = actuators_per + (1 if pi < rem else 0)
            prev = pump
            for j in range(max(0, n_here - 1)):
                jid = G.number_of_nodes()
                jtype = "junction_T" if j % 2 == 0 else "junction_X"
                G.add_node(jid, **_node_attrs(jtype))
                fl = float(npr.uniform(1.5, 6.0))
                G.add_edge(
                    prev,
                    jid,
                    **_edge_attrs(length_m=fl, flow_cap=60.0, resistance=0.04 * fl, fail_p=0.03),
                )
                prev = jid
            for _ in range(n_here):
                if act_count >= num_actuators:
                    break
                aid = G.number_of_nodes()
                G.add_node(aid, **_node_attrs("actuator"))
                fl = float(npr.uniform(1.0, 5.0))
                G.add_edge(
                    prev,
                    aid,
                    **_edge_attrs(length_m=fl, flow_cap=40.0, resistance=0.05 * fl, fail_p=0.04),
                )
                act_count += 1

        # If rounding left actuators unassigned, attach to last junction/pump chain
        while act_count < num_actuators:
            aid = G.number_of_nodes()
            G.add_node(aid, **_node_attrs("actuator"))
            attach = pump_ids[-1]
            fl = float(npr.uniform(1.0, 5.0))
            G.add_edge(
                attach,
                aid,
                **_edge_attrs(length_m=fl, flow_cap=40.0, resistance=0.06 * fl, fail_p=0.04),
            )
            act_count += 1

        return G

    def build_looped_topology(
        self,
        num_pumps: int,
        num_actuators: int,
        redundancy_level: int,
        seed: int | None = None,
    ) -> nx.DiGraph:
        """
        Ring-bus style layout: pumps and junctions form loops; actuators tap the ring.

        Ensures each actuator has at least ``redundancy_level`` edge-disjoint routes to *some* pump
        by adding parallel ring chords and multiple taps (approximation of independent paths).
        """
        _, npr = self._fresh_rng(seed)
        G = nx.DiGraph()
        redundancy_level = max(1, int(redundancy_level))

        # Backbone ring nodes (junctions) — one per pump at minimum
        ring_size = max(num_pumps * 2, num_actuators // 2 + 2, 4)
        ring_nodes: list[int] = []
        for _ in range(ring_size):
            nid = G.number_of_nodes()
            ring_nodes.append(nid)
            G.add_node(nid, **_node_attrs("junction_X" if nid % 2 == 0 else "junction_T"))

        # Connect directed ring (both directions for redundancy)
        for i in range(ring_size):
            a, b = ring_nodes[i], ring_nodes[(i + 1) % ring_size]
            fl = float(npr.uniform(2.0, 5.0))
            G.add_edge(a, b, **_edge_attrs(length_m=fl, flow_cap=120.0, resistance=0.03 * fl, fail_p=0.025))
            fl2 = float(npr.uniform(2.0, 5.0))
            G.add_edge(b, a, **_edge_attrs(length_m=fl2, flow_cap=120.0, resistance=0.03 * fl2, fail_p=0.025))

        # Add chord edges for higher redundancy (creates multiple cycles)
        for k in range(redundancy_level - 1):
            step = max(1, ring_size // (k + 3))
            for i in range(0, ring_size, step):
                a, b = ring_nodes[i], ring_nodes[(i + 2 * step) % ring_size]
                if not G.has_edge(a, b):
                    fl = float(npr.uniform(1.5, 4.0))
                    G.add_edge(a, b, **_edge_attrs(length_m=fl, flow_cap=100.0, resistance=0.035 * fl, fail_p=0.03))
                if not G.has_edge(b, a):
                    fl = float(npr.uniform(1.5, 4.0))
                    G.add_edge(b, a, **_edge_attrs(length_m=fl, flow_cap=100.0, resistance=0.035 * fl, fail_p=0.03))

        # Pumps + reservoirs around the ring
        pump_ids: list[int] = []
        for i in range(num_pumps):
            pid = G.number_of_nodes()
            rid = pid + 1
            pump_ids.append(pid)
            G.add_node(pid, **_node_attrs("pump"))
            G.add_node(rid, **_node_attrs("reservoir"))
            fl = float(npr.uniform(2.0, 6.0))
            G.add_edge(rid, pid, **_edge_attrs(length_m=fl, flow_cap=90.0, resistance=0.02 * fl, fail_p=0.01))
            rn = ring_nodes[i % ring_size]
            fl2 = float(npr.uniform(1.5, 4.0))
            G.add_edge(pid, rn, **_edge_attrs(length_m=fl2, flow_cap=100.0, resistance=0.03 * fl2, fail_p=0.02))

        # Actuators: tap multiple ring nodes for path diversity
        for a in range(num_actuators):
            aid = G.number_of_nodes()
            G.add_node(aid, **_node_attrs("actuator"))
            taps = {a % ring_size, (a * redundancy_level) % ring_size, (a + ring_size // 2) % ring_size}
            for t in taps:
                src = ring_nodes[t % ring_size]
                fl = float(npr.uniform(1.0, 4.0))
                G.add_edge(
                    src,
                    aid,
                    **_edge_attrs(length_m=fl, flow_cap=50.0, resistance=0.045 * fl, fail_p=0.035),
                )

        return G

    def build_segmented_topology(
        self,
        num_pumps: int,
        num_actuators: int,
        num_segments: int,
        seed: int | None = None,
    ) -> nx.DiGraph:
        """
        Segmented layout: linear segments separated by isolation valves on inter-segment links.

        ``num_segments`` controls how many isolated sections exist along the main feed.
        """
        _, npr = self._fresh_rng(seed)
        G = nx.DiGraph()
        num_segments = max(2, int(num_segments))

        # One reservoir + pump feeds segment 0
        pid = G.number_of_nodes()
        rid = pid + 1
        G.add_node(pid, **_node_attrs("pump"))
        G.add_node(rid, **_node_attrs("reservoir"))
        fl = float(npr.uniform(2.0, 7.0))
        G.add_edge(rid, pid, **_edge_attrs(length_m=fl, flow_cap=85.0, resistance=0.02 * fl, fail_p=0.01))

        prev = pid
        seg_heads: list[int] = [pid]

        for seg in range(num_segments):
            j = G.number_of_nodes()
            G.add_node(j, **_node_attrs("junction_T" if seg % 2 == 0 else "junction_X"))
            fl = float(npr.uniform(1.5, 5.0))
            is_valve = seg > 0  # valve between segments
            G.add_edge(
                prev,
                j,
                **_edge_attrs(
                    length_m=fl,
                    flow_cap=70.0,
                    resistance=0.04 * fl,
                    fail_p=0.03,
                    is_valve=is_valve,
                ),
            )
            prev = j
            seg_heads.append(j)

        # Distribute actuators across segments
        base = num_actuators // num_segments
        extra = num_actuators % num_segments
        ai = 0
        for seg in range(num_segments):
            head = seg_heads[min(seg + 1, len(seg_heads) - 1)]
            n_here = base + (1 if seg < extra else 0)
            chain = head
            for _ in range(max(0, n_here - 1)):
                jj = G.number_of_nodes()
                G.add_node(jj, **_node_attrs("junction_T"))
                fl = float(npr.uniform(1.0, 4.0))
                G.add_edge(
                    chain,
                    jj,
                    **_edge_attrs(length_m=fl, flow_cap=55.0, resistance=0.05 * fl, fail_p=0.035),
                )
                chain = jj
            for _ in range(n_here):
                if ai >= num_actuators:
                    break
                ax = G.number_of_nodes()
                G.add_node(ax, **_node_attrs("actuator"))
                fl = float(npr.uniform(1.0, 4.0))
                G.add_edge(
                    chain,
                    ax,
                    **_edge_attrs(length_m=fl, flow_cap=45.0, resistance=0.05 * fl, fail_p=0.04),
                )
                ai += 1

        while ai < num_actuators:
            ax = G.number_of_nodes()
            G.add_node(ax, **_node_attrs("actuator"))
            fl = float(npr.uniform(1.0, 4.0))
            G.add_edge(prev, ax, **_edge_attrs(length_m=fl, flow_cap=45.0, resistance=0.06 * fl, fail_p=0.04))
            ai += 1

        # Additional pumps feed downstream segments (optional diversity)
        for p in range(1, num_pumps):
            pid2 = G.number_of_nodes()
            rid2 = pid2 + 1
            G.add_node(pid2, **_node_attrs("pump"))
            G.add_node(rid2, **_node_attrs("reservoir"))
            fl = float(npr.uniform(2.0, 6.0))
            G.add_edge(rid2, pid2, **_edge_attrs(length_m=fl, flow_cap=80.0, resistance=0.02 * fl, fail_p=0.01))
            attach = seg_heads[min(p, len(seg_heads) - 1)]
            fl2 = float(npr.uniform(1.5, 4.0))
            G.add_edge(
                pid2,
                attach,
                **_edge_attrs(length_m=fl2, flow_cap=75.0, resistance=0.035 * fl2, fail_p=0.025),
            )

        return G
