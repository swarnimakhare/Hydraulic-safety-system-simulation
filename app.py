"""
Streamlit entry point: hydraulic system safety simulator and analytics dashboard.
"""

from __future__ import annotations

import io
import pickle

import networkx as nx
import pandas as pd
import streamlit as st

from failure_engine import FailureEngine
from graph_builder import GraphBuilder
from metrics import SafetyMetrics
from monte_carlo import MonteCarloRunner
from optimizer import TopologyOptimizer
from simulation import HydraulicSimulator
from utils import export_results_csv, get_topology_insights, set_global_seed
from visualizer import Visualizer


def _run_single_scenario(
    graph: nx.DiGraph,
    edge_failure_rate: float,
    node_failure_rate: float,
    seed: int,
) -> nx.DiGraph:
    """Copy graph, inject failures, isolate, and simulate pressures."""
    G = graph.copy()
    fe = FailureEngine(seed=seed)
    fe.inject_random_failures(G, edge_failure_rate, node_failure_rate, seed=seed)
    fe.apply_isolation_logic(G)
    sim = HydraulicSimulator()
    sim.run_steady_state(G)
    return G


def _run_all_topologies_mc(
    num_pumps: int,
    num_actuators: int,
    redundancy_level: int,
    edge_failure_rate: float,
    node_failure_rate: float,
    n_iterations: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Monte Carlo batches for Tree, Looped, and Segmented topologies."""
    out: dict[str, pd.DataFrame] = {}
    for name in ("Tree", "Looped", "Segmented"):
        set_global_seed(seed)
        gb = GraphBuilder(seed=seed)
        # Bind `gb` by default arg so closures do not all reference the final loop instance.
        if name == "Tree":
            fn = lambda gb=gb: gb.build_tree_topology(num_pumps, num_actuators, seed=seed)
        elif name == "Looped":
            fn = lambda gb=gb: gb.build_looped_topology(num_pumps, num_actuators, redundancy_level, seed=seed)
        else:
            fn = lambda gb=gb: gb.build_segmented_topology(
                num_pumps, num_actuators, max(2, redundancy_level), seed=seed
            )

        mc = MonteCarloRunner(seed=seed)
        out[name] = mc.run(fn, n_iterations, edge_failure_rate, node_failure_rate, seed=seed)
    return out


def main() -> None:
    st.set_page_config(page_title="Hydraulic Safety Simulator", layout="wide")
    st.title("Aircraft hydraulic system — safety under failures")
    st.caption(
        "Steady-state pressure propagation, stochastic failures, isolation valves, and Monte Carlo resilience metrics."
    )

    with st.sidebar:
        st.header("Controls")
        topology = st.selectbox(
            "Topology",
            ["Tree", "Looped", "Segmented", "Run Optimizer"],
            index=1,
            help="Select baseline topology for single-run views, or launch the GA optimizer.",
        )
        num_pumps = st.slider("Number of pumps", 1, 4, 2)
        num_actuators = st.slider("Number of actuators", 2, 10, 6)
        redundancy_level = st.slider("Redundancy / segments level", 1, 4, 2)
        edge_failure_rate = st.slider("Edge failure rate multiplier", 0.0, 1.0, 0.2, 0.05)
        node_failure_rate = st.slider("Node failure rate", 0.0, 0.5, 0.1, 0.05)
        mc_iters = st.slider("Monte Carlo iterations", 10, 500, 100, 10)
        random_seed = st.number_input("Random seed", value=42, step=1)

        run_sim = st.button("Run Simulation", type="primary")
        export_btn = st.button("Export CSV")

    # Default first-load scenario (Looped, 2/6/2, rates, 100 iters, seed 42)
    if "mc_results" not in st.session_state:
        with st.spinner("Running default Monte Carlo for all topologies (first load)…"):
            set_global_seed(int(random_seed))
            st.session_state.mc_results = _run_all_topologies_mc(
                num_pumps=2,
                num_actuators=6,
                redundancy_level=2,
                edge_failure_rate=0.2,
                node_failure_rate=0.1,
                n_iterations=100,
                seed=42,
            )
            gb0 = GraphBuilder(seed=42)
            st.session_state.baseline_graph = gb0.build_looped_topology(2, 6, 2, seed=42)
            st.session_state.failed_graph = _run_single_scenario(
                st.session_state.baseline_graph.copy(),
                0.2,
                0.1,
                seed=42,
            )
            st.session_state.optimizer_result = None

    if run_sim:
        set_global_seed(int(random_seed))
        with st.spinner("Running Monte Carlo across Tree / Looped / Segmented…"):
            st.session_state.mc_results = _run_all_topologies_mc(
                num_pumps,
                num_actuators,
                redundancy_level,
                edge_failure_rate,
                node_failure_rate,
                mc_iters,
                int(random_seed),
            )
        gb = GraphBuilder(seed=int(random_seed))
        if topology == "Tree":
            base = gb.build_tree_topology(num_pumps, num_actuators, seed=int(random_seed))
        elif topology == "Looped":
            base = gb.build_looped_topology(num_pumps, num_actuators, redundancy_level, seed=int(random_seed))
        elif topology == "Segmented":
            base = gb.build_segmented_topology(num_pumps, num_actuators, max(2, redundancy_level), seed=int(random_seed))
        else:
            base = gb.build_looped_topology(num_pumps, num_actuators, redundancy_level, seed=int(random_seed))

        st.session_state.baseline_graph = base
        st.session_state.failed_graph = _run_single_scenario(
            base.copy(),
            edge_failure_rate,
            node_failure_rate,
            seed=int(random_seed) + 7,
        )

        if topology == "Run Optimizer":
            with st.spinner("Running genetic optimizer (this may take a minute)…"):
                opt = TopologyOptimizer(seed=int(random_seed))
                best_g, hist, _ = opt.evolve(
                    base.copy(),
                    generations=8,
                    population_size=16,
                    edge_failure_rate=edge_failure_rate,
                    node_failure_rate=node_failure_rate,
                )
                st.session_state.optimizer_result = {"graph": best_g, "history": hist, "baseline": base}
        else:
            st.session_state.optimizer_result = None

    mc_results: dict[str, pd.DataFrame] = st.session_state.mc_results
    baseline: nx.DiGraph = st.session_state.baseline_graph
    failed_g: nx.DiGraph = st.session_state.failed_graph
    sim = HydraulicSimulator()
    metrics = SafetyMetrics()
    viz = Visualizer()

    # Re-simulate baseline for pressure display (no failures)
    baseline_clean = baseline.copy()
    FailureEngine(seed=int(random_seed)).reset_failures(baseline_clean)
    for _, _, d in baseline_clean.edges(data=True):
        d["is_failed"] = False
    sim.run_steady_state(baseline_clean)

    sim_failed = HydraulicSimulator()
    sim_failed.run_steady_state(failed_g)

    merged_frames: list[pd.DataFrame] = []
    for name, df in mc_results.items():
        tdf = df.copy()
        tdf.insert(0, "topology", name)
        merged_frames.append(tdf)
    combined_mc = pd.concat(merged_frames, ignore_index=True) if merged_frames else pd.DataFrame()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Network View", "Pressure Analysis", "Monte Carlo Results", "Optimizer", "Insights"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Baseline (no failures)")
            st.plotly_chart(viz.draw_network(baseline_clean, "Baseline topology"), use_container_width=True)
        with c2:
            st.subheader("Sample post-failure state")
            st.plotly_chart(viz.draw_network(failed_g, "After failures + isolation"), use_container_width=True)
        st.markdown(
            "**Legend:** blue=pump, cyan=reservoir, gray=junction_T, purple=junction_X, green=actuator; "
            "red=failed, orange=inactive; dashed red edges=failed lines."
        )

    with tab2:
        st.plotly_chart(viz.plot_pressure_distribution(failed_g, sim_failed), use_container_width=True)
        rows = []
        for n, d in failed_g.nodes(data=True):
            p = float(d.get("pressure", 0.0))
            nt = d.get("node_type", "")
            if nt == "actuator":
                status = "functional" if sim_failed.is_actuator_functional(p) else "below threshold"
            else:
                status = "failed" if d.get("is_failed") else ("inactive" if not d.get("is_active", True) else "active")
            rows.append({"node": n, "type": nt, "pressure_psi": p, "status": status})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab3:
        st.plotly_chart(viz.plot_safety_comparison(mc_results), use_container_width=True)
        summary_rows = []
        for name, df in mc_results.items():
            ss = MonteCarloRunner.summary_stats(df)
            for col, stats in ss.items():
                summary_rows.append(
                    {
                        "topology": name,
                        "metric": col,
                        "mean": stats["mean"],
                        "std": stats["std"],
                    }
                )
        st.subheader("Summary statistics (mean ± std)")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        pick = mc_results.get("Looped")
        if pick is None or pick.empty:
            pick = next((df for df in mc_results.values() if df is not None and not df.empty), None)
        if pick is not None:
            st.plotly_chart(viz.plot_failure_heatmap(pick), use_container_width=True)

    with tab4:
        if st.session_state.optimizer_result is None:
            st.info('Select **Run Optimizer** in the sidebar and press **Run Simulation** to evolve a topology.')
        else:
            res = st.session_state.optimizer_result
            hist = res["history"]
            best = res["graph"]
            base = res["baseline"]
            st.plotly_chart(viz.plot_ga_convergence(hist), use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Best GA graph")
                bcopy = best.copy()
                FailureEngine().reset_failures(bcopy)
                sim.run_steady_state(bcopy)
                st.plotly_chart(viz.draw_network(bcopy, "Optimized topology (clean)"), use_container_width=True)
            with c2:
                st.caption("Baseline vs best — composite score (single MC eval)")
                mc_quick = MonteCarloRunner(seed=int(random_seed))
                f_base = mc_quick.run(lambda: base.copy(), 15, edge_failure_rate, node_failure_rate, seed=int(random_seed))[
                    "safety_score"
                ].mean()
                f_best = mc_quick.run(lambda: best.copy(), 15, edge_failure_rate, node_failure_rate, seed=int(random_seed) + 3)[
                    "safety_score"
                ].mean()
                st.metric("Baseline mean safety_score (15 draws)", f"{f_base:.4f}")
                st.metric("Optimized mean safety_score (15 draws)", f"{f_best:.4f}")

    with tab5:
        st.markdown(get_topology_insights(mc_results))
        st.subheader("Critical nodes (betweenness)")
        crit = metrics.critical_nodes(baseline_clean)
        st.dataframe(pd.DataFrame(crit, columns=["node", "betweenness"]), use_container_width=True)

        csv_buf = io.StringIO()
        if not combined_mc.empty:
            combined_mc.to_csv(csv_buf, index=False)
        st.download_button(
            "Download combined Monte Carlo CSV",
            data=csv_buf.getvalue(),
            file_name="hydraulic_monte_carlo.csv",
            mime="text/csv",
        )

        pbuf = io.BytesIO()
        pickle.dump(baseline_clean, pbuf, protocol=pickle.HIGHEST_PROTOCOL)
        st.download_button(
            "Download baseline graph (pickle)",
            data=pbuf.getvalue(),
            file_name="baseline_graph.pkl",
            mime="application/octet-stream",
        )

    if export_btn and not combined_mc.empty:
        export_results_csv(combined_mc, "hydraulic_export.csv")
        st.sidebar.success("Wrote hydraulic_export.csv")


if __name__ == "__main__":
    main()
