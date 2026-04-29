"""
Genetic algorithm over optional junction-to-junction connections.
"""

from __future__ import annotations

import random
from typing import Callable

import networkx as nx

from monte_carlo import MonteCarloRunner
from simulation import HydraulicSimulator


def _default_edge_data() -> dict:
    return {
        "length": 3.0,
        "flow_capacity": 70.0,
        "resistance": 0.12,
        "failure_probability": 0.03,
        "is_valve": False,
        "is_failed": False,
    }


class TopologyOptimizer:
    """
    Evolves a binary genome that toggles optional edges between junction nodes.

    The genome length equals the number of candidate junction pairs in the catalog
    (capped for runtime). Fitness is the mean composite safety score from a small
    Monte Carlo batch.
    """

    def __init__(self, seed: int | None = 42) -> None:
        self.seed = int(seed if seed is not None else 42)
        self._rng = random.Random(self.seed)

    def _junction_nodes(self, G: nx.DiGraph) -> list:
        return [
            n
            for n, d in G.nodes(data=True)
            if str(d.get("node_type", "")).startswith("junction")
        ]

    def _build_catalog(self, base_graph: nx.DiGraph, max_pairs: int = 48) -> list[tuple[int, int]]:
        """Candidate directed junction edges not present in the baseline graph."""
        juncs = self._junction_nodes(base_graph)
        catalog: list[tuple[int, int]] = []
        for i, u in enumerate(juncs):
            for v in juncs[i + 1 :]:
                if not base_graph.has_edge(u, v):
                    catalog.append((u, v))
                if len(catalog) >= max_pairs:
                    return catalog
        return catalog

    def initialize_population(self, base_graph: nx.DiGraph, population_size: int) -> tuple[list[list[bool]], list[tuple[int, int]]]:
        """
        Create random genomes and the shared catalog describing optional edges.

        Returns
        -------
        population
            List of genomes (each a list[bool]).
        catalog
            Parallel list of (u, v) junction pairs.
        """
        catalog = self._build_catalog(base_graph)
        if not catalog:
            # Degenerate case: create a dummy gene so the GA still runs
            j = self._junction_nodes(base_graph)
            if len(j) >= 2:
                catalog = [(j[0], j[1])]
            else:
                catalog = [(0, 0)]  # will be ignored; fitness uses baseline

        pop: list[list[bool]] = []
        for _ in range(int(population_size)):
            pop.append([self._rng.random() < 0.35 for _ in catalog])
        return pop, catalog

    def _apply_genome_additive(
        self, base_graph: nx.DiGraph, genome: list[bool], catalog: list[tuple[int, int]]
    ) -> nx.DiGraph:
        """Copy baseline and add catalog edges where genome is True."""
        G = base_graph.copy()
        for gene, (u, v) in zip(genome, catalog):
            if not gene or u == v:
                continue
            if not G.has_edge(u, v):
                G.add_edge(u, v, **_default_edge_data())
        return G

    def fitness(
        self,
        graph_factory: Callable[[], nx.DiGraph],
        n_eval_iterations: int = 20,
        edge_failure_rate: float = 0.2,
        node_failure_rate: float = 0.1,
    ) -> float:
        """
        Monte Carlo mean composite safety score for graphs produced by ``graph_factory``.
        """
        mc = MonteCarloRunner(seed=self.seed)
        df = mc.run(graph_factory, n_eval_iterations, edge_failure_rate, node_failure_rate, seed=self.seed)
        if df.empty:
            return 0.0
        return float(df["safety_score"].mean())

    def crossover(self, parent_a: list[bool], parent_b: list[bool]) -> list[bool]:
        """Single-point crossover on boolean genomes."""
        if not parent_a:
            return []
        pt = self._rng.randrange(1, max(len(parent_a), 2))
        return parent_a[:pt] + parent_b[pt:]

    def mutate(self, individual: list[bool], mutation_rate: float = 0.1) -> list[bool]:
        """Bit-flip mutation."""
        out = individual[:]
        for i in range(len(out)):
            if self._rng.random() < mutation_rate:
                out[i] = not out[i]
        return out

    def evolve(
        self,
        base_graph: nx.DiGraph,
        generations: int = 10,
        population_size: int = 20,
        edge_failure_rate: float = 0.2,
        node_failure_rate: float = 0.1,
    ) -> tuple[nx.DiGraph, list[float], list[bool]]:
        """
        Run a simple elitist GA and return the best graph, fitness history, and genome.

        Notes
        -----
        Fitness evaluations are intentionally small Monte Carlo samples for responsiveness.
        """
        population, catalog = self.initialize_population(base_graph, population_size)
        fitness_history: list[float] = []

        def make_factory(genome: list[bool]) -> Callable[[], nx.DiGraph]:
            def _fn() -> nx.DiGraph:
                return self._apply_genome_additive(base_graph, genome, catalog)

            return _fn

        best_genome = population[0]
        best_fit = -1.0

        for gen in range(int(generations)):
            fits: list[float] = []
            for ind in population:
                f = self.fitness(make_factory(ind), n_eval_iterations=12, edge_failure_rate=edge_failure_rate, node_failure_rate=node_failure_rate)
                fits.append(f)
                if f > best_fit:
                    best_fit = f
                    best_genome = ind[:]

            fitness_history.append(best_fit)

            # Selection + breeding
            ranked = sorted(zip(fits, population), key=lambda t: t[0], reverse=True)
            new_pop: list[list[bool]] = [best_genome[:]]  # elitism
            while len(new_pop) < int(population_size):
                p1 = ranked[self._rng.randrange(0, max(1, len(ranked) // 2))][1]
                p2 = ranked[self._rng.randrange(0, max(1, len(ranked) // 2))][1]
                child = self.crossover(p1, p2)
                child = self.mutate(child, mutation_rate=0.12)
                new_pop.append(child)
            population = new_pop

        best_graph = self._apply_genome_additive(base_graph, best_genome, catalog)
        return best_graph, fitness_history, best_genome
