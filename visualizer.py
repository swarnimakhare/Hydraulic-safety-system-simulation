"""
Plotly/Matplotlib visualization helpers for hydraulic networks and analytics.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import pandas as pd

from simulation import HydraulicSimulator


class Visualizer:
    """Builds Plotly figures for graph layout, pressures, Monte Carlo panels, and GA convergence."""

    NODE_COLORS = {
        "pump": "#1f77b4",
        "reservoir": "#17becf",
        "junction_T": "#7f7f7f",
        "junction_X": "#9467bd",
        "actuator": "#2ca02c",
    }

    def draw_network(self, graph: nx.DiGraph, title: str = "Hydraulic network") -> go.Figure:
        """
        Spring layout; node color encodes type / failure; edge style encodes failure state.
        Node size scales with pressure magnitude.
        """
        if graph.number_of_nodes() == 0:
            fig = go.Figure()
            fig.update_layout(title=title, annotations=[dict(text="Empty graph", showarrow=False)])
            return fig

        pos = nx.spring_layout(graph, seed=42)
        edge_traces: list[Any] = []
        for u, v, d in graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            failed = bool(d.get("is_failed", False))
            color = "red" if failed else "black"
            dash = "dash" if failed else "solid"
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=1.5, color=color, dash=dash),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

        node_x: list[float] = []
        node_y: list[float] = []
        colors: list[str] = []
        sizes: list[float] = []
        texts: list[str] = []

        pmax = max(
            (float(graph.nodes[n].get("pressure", 0.0)) for n in graph.nodes()),
            default=1.0,
        )

        for n, d in graph.nodes(data=True):
            x, y = pos[n]
            node_x.append(float(x))
            node_y.append(float(y))
            nt = d.get("node_type", "")
            if d.get("is_failed", False):
                col = "red"
            elif not d.get("is_active", True):
                col = "orange"
            else:
                col = self.NODE_COLORS.get(nt, "#888888")
            colors.append(col)
            p = float(d.get("pressure", 0.0))
            sz = 10.0 + 28.0 * (p / max(pmax, 1.0))
            sizes.append(sz)
            texts.append(f"node {n}<br>type={nt}<br>p={p:.1f} psi")

        nodes_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=sizes, color=colors, line=dict(width=0.5, color="#333")),
            text=texts,
            hoverinfo="text",
            name="nodes",
        )

        fig = go.Figure(data=edge_traces + [nodes_trace])
        fig.update_layout(
            title=title,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="white",
        )
        return fig

    def plot_pressure_distribution(self, graph: nx.DiGraph, simulator: HydraulicSimulator, threshold: float = 1500.0) -> go.Figure:
        """Bar chart of actuator pressures with pass/fail coloring."""
        labels: list[str] = []
        pressures: list[float] = []
        colors: list[str] = []
        for n, d in graph.nodes(data=True):
            if d.get("node_type") != "actuator":
                continue
            p = float(d.get("pressure", 0.0))
            labels.append(f"A{n}")
            pressures.append(p)
            ok = simulator.is_actuator_functional(p, threshold)
            colors.append("#2ca02c" if ok else "#d62728")

        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=pressures,
                    marker_color=colors,
                    text=[f"{v:.0f}" for v in pressures],
                    textposition="auto",
                )
            ]
        )
        fig.add_hline(y=threshold, line_dash="dash", line_color="#555", annotation_text="threshold")
        fig.update_layout(title="Actuator pressures (PSI)", yaxis_title="PSI", xaxis_title="Actuator")
        return fig

    def plot_safety_comparison(self, results_dict: dict[str, pd.DataFrame]) -> go.Figure:
        """Grouped-style bar chart of mean safety score per topology name."""
        names = list(results_dict.keys())
        means = []
        for k in names:
            df = results_dict[k]
            if df is None or df.empty or "safety_score" not in df.columns:
                means.append(0.0)
            else:
                means.append(float(df["safety_score"].mean()))

        fig = go.Figure(data=[go.Bar(x=names, y=means, marker_color="#393b79", text=[f"{m:.3f}" for m in means], textposition="auto")])
        fig.update_layout(title="Mean safety score by topology", yaxis_title="score", xaxis_title="topology")
        return fig

    def plot_failure_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Heatmap of metrics across Monte Carlo iterations."""
        if df.empty:
            return go.Figure()
        metric_cols = [c for c in df.columns if c != "iteration"]
        z = df[metric_cols].to_numpy().T
        # Normalize columns for display
        z_norm = np.zeros_like(z, dtype=float)
        for i in range(z.shape[0]):
            col = z[i, :]
            m = np.nanmax(np.abs(col)) or 1.0
            z_norm[i, :] = col / m

        fig = go.Figure(
            data=go.Heatmap(
                z=z_norm,
                x=list(df["iteration"]),
                y=metric_cols,
                colorscale="Viridis",
                colorbar=dict(title="norm"),
            )
        )
        fig.update_layout(title="Monte Carlo metrics heatmap (column-normalized)", xaxis_title="iteration")
        return fig

    def plot_ga_convergence(self, fitness_history: list[float]) -> go.Figure:
        """Line chart of best fitness per generation."""
        gens = list(range(len(fitness_history)))
        fig = go.Figure(data=[go.Scatter(x=gens, y=fitness_history, mode="lines+markers", line=dict(color="#ff7f0e"))])
        fig.update_layout(title="GA convergence (best fitness)", xaxis_title="generation", yaxis_title="fitness")
        return fig
