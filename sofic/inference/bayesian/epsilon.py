"""Conjugate Bayesian inference for fixed unifilar generator topologies."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.special import logsumexp

from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_EMISSION, ATTR_PROB
from sofic.inference.bayesian.counts import (
    BayesianInferenceError,
    PathCountEM,
    dirichlet_multinomial_log_evidence,
    scan_unifilar_topology,
)


class DirichletDistributionEM:
    """Product-of-Dirichlets prior/posterior for a fixed unifilar topology."""

    def __init__(self, machine: MealyHMM, data: Sequence[Any] | None = None, state_path: bool = False):
        self.machine = machine
        self.alphas: dict[Hashable | tuple[Hashable, Any], float] = {}
        self.edges: list[tuple[Hashable, Any]] = []
        self.valid_edges: list[tuple[Hashable, Any]] = []
        self.nodes: list[Hashable] = list(machine.states())
        self.trace: dict[tuple[Hashable, Any], Hashable] = {}
        self.data: PathCountEM | None = None
        self.deterministic = False
        self._process_machine_topology()
        self.valid_startnodes = list(self.nodes)
        self._generate_uniform_alphas()
        if data is not None:
            self.data = PathCountEM(machine, data, state_path=state_path)
            self.valid_startnodes = self.data.get_possible_start_nodes()

    def _process_machine_topology(self) -> None:
        self.trace, self.edges = scan_unifilar_topology(self.machine)
        outgoing: dict[Hashable, list[tuple[Hashable, Any]]] = {}
        for edge in self.edges:
            outgoing.setdefault(edge[0], []).append(edge)
        for _source, edges in outgoing.items():
            if len(edges) > 1:
                self.valid_edges.extend(edges)
            else:
                edge = edges[0]
                prob = self._topology_probability(edge)
                if not np.isclose(prob, 1.0):
                    self.valid_edges.append(edge)
        self.valid_edges.sort(key=repr)
        self.deterministic = not self.valid_edges

    def _topology_probability(self, edge: tuple[Hashable, Any]) -> float:
        source, symbol = edge
        for transition in self.machine.graph.out_transitions(source):
            if transition.data.get(ATTR_EMISSION) == symbol:
                return float(transition.data.get(ATTR_PROB, 0.0))
        return 0.0

    def _generate_uniform_alphas(self) -> None:
        self.alphas.clear()
        for edge in self.valid_edges:
            source, _symbol = edge
            self.alphas[edge] = 1.0
            self.alphas[source] = self.alphas.get(source, 0.0) + 1.0

    def get_edges(self) -> list[tuple[Hashable, Any]]:
        return list(self.edges)

    def get_nodes(self) -> list[Hashable]:
        return list(self.nodes)

    def get_possible_start_nodes(self) -> list[Hashable]:
        return list(self.valid_startnodes)

    def get_edge_alpha(self, start_node: Hashable, edge: tuple[Hashable, Any]) -> float | None:
        del start_node
        return self.alphas.get(edge)

    def get_node_alpha(self, start_node: Hashable, node: Hashable) -> float | None:
        del start_node
        return self.alphas.get(node)

    def get_edge_count(self, start_node: Hashable, edge: tuple[Hashable, Any]) -> int | None:
        return None if self.data is None else self.data.get_edge_count(start_node, edge)

    def get_node_count(self, start_node: Hashable, node: Hashable) -> int | None:
        return None if self.data is None else self.data.get_node_count(start_node, node)

    def get_last_node(self, start_node: Hashable) -> Hashable | None:
        return None if self.data is None else self.data.get_last_node(start_node)

    def get_state_path(self, start_node: Hashable) -> tuple[Hashable, ...]:
        return () if self.data is None else self.data.get_state_path(start_node)

    def log_evidence_start_node(self, start_node: Hashable) -> float:
        if start_node not in self.valid_startnodes:
            return -np.inf
        rows: dict[Hashable, list[tuple[float, float]]] = {}
        for edge in self.valid_edges:
            source, _symbol = edge
            alpha = self.get_edge_alpha(start_node, edge)
            alpha_row = self.get_node_alpha(start_node, source)
            if alpha is None or alpha_row is None:
                raise BayesianInferenceError("missing Dirichlet alpha")
            count = self.get_edge_count(start_node, edge) or 0
            rows.setdefault(source, []).append((alpha, count))
        evidence = 0.0
        for source, cells in rows.items():
            alpha_row = self.get_node_alpha(start_node, source)
            count_row = self.get_node_count(start_node, source) or 0
            evidence += dirichlet_multinomial_log_evidence(alpha_row, count_row, cells)
        return float(evidence)

    def mean_edge_probability(self, start_node: Hashable, edge: tuple[Hashable, Any]) -> float | None:
        if start_node not in self.valid_startnodes:
            return None
        if edge in self.edges and edge not in self.valid_edges:
            return 1.0
        count = self.get_edge_count(start_node, edge) or 0
        row_count = self.get_node_count(start_node, edge[0]) or 0
        alpha = self.get_edge_alpha(start_node, edge)
        row_alpha = self.get_node_alpha(start_node, edge[0])
        if alpha is None or row_alpha is None:
            return None
        return float((count + alpha) / (row_count + row_alpha))

    def set_edge_alpha(self, edge: tuple[Hashable, Any], value: float) -> None:
        if edge not in self.valid_edges:
            raise BayesianInferenceError("cannot set alpha for deterministic edge")
        self.alphas[edge] = float(value)
        for node in self.nodes:
            self.alphas[node] = 0.0
        for valid_edge in self.valid_edges:
            self.alphas[valid_edge[0]] += self.alphas[valid_edge]

    def _machine_from_probabilities(
        self,
        start_node: Hashable,
        probabilities: Mapping[tuple[Hashable, Any], float],
        name: str,
        initial: Mapping[Hashable, float] | None = None,
    ) -> MealyHMM:
        machine = MealyHMM(observation_alphabet=getattr(self.machine, "observation_alphabet", frozenset()))
        machine.name = name
        machine.initial_distribution = dict(initial) if initial is not None else {start_node: 1.0}
        for node in self.nodes:
            machine.graph.add_state(node)
        for edge in self.edges:
            source, symbol = edge
            prob = probabilities[edge]
            machine.graph.add_transition(source, self.trace[edge], **{ATTR_EMISSION: symbol, ATTR_PROB: prob})
        machine.validate()
        if machine.is_unifilar():
            eps = EpsilonMachine.from_networkx(
                machine.to_networkx(),
                initial_distribution=machine.initial_distribution,
                observation_alphabet=machine.observation_alphabet,
            )
            eps.name = name
            eps.validate()
            return eps
        return machine

    def posterior_mean_machine(
        self, start_node: Hashable, initial: Mapping[Hashable, float] | None = None
    ) -> MealyHMM | None:
        """Posterior-mean machine given ``start_node``.

        The machine starts in ``start_node`` (the state before the first
        observation) unless ``initial`` overrides the initial distribution.
        """
        if start_node not in self.valid_startnodes:
            return None
        probabilities = {}
        for edge in self.edges:
            prob = self.mean_edge_probability(start_node, edge)
            if prob is None:
                raise BayesianInferenceError(f"missing probability for edge {edge!r}")
            probabilities[edge] = prob
        return self._machine_from_probabilities(
            start_node, probabilities, f"Posterior Mean Machine, Start Node: {start_node}", initial
        )

    def generate_sample(self, start_node: Hashable, rng: np.random.Generator | None = None) -> MealyHMM | None:
        if start_node not in self.valid_startnodes:
            return None
        generator = rng if rng is not None else np.random.default_rng()
        probabilities: dict[tuple[Hashable, Any], float] = {}
        grouped: dict[Hashable, list[tuple[Hashable, Any]]] = {}
        for edge in self.edges:
            if edge in self.valid_edges:
                grouped.setdefault(edge[0], []).append(edge)
            else:
                probabilities[edge] = 1.0
        for _source, edges in grouped.items():
            alpha = []
            for edge in edges:
                a = self.get_edge_alpha(start_node, edge)
                if a is None:
                    raise BayesianInferenceError("missing Dirichlet alpha")
                alpha.append(a + (self.get_edge_count(start_node, edge) or 0))
            sample = generator.dirichlet(np.asarray(alpha, dtype=float))
            for edge, prob in zip(edges, sample, strict=True):
                probabilities[edge] = float(prob)
        return self._machine_from_probabilities(start_node, probabilities, f"Sampled Machine, Start Node: {start_node}")


class EpsilonMachinePosterior:
    """Posterior over parameters and unknown start state for a topology."""

    def __init__(
        self,
        machine: MealyHMM,
        data: Sequence[Any] | None = None,
        start_dist: Mapping[Hashable, float] | None = None,
        state_path: bool = False,
    ):
        self.machine = machine
        self.dirichlet = DirichletDistributionEM(machine, data, state_path=state_path)
        if start_dist is None:
            nodes = self.dirichlet.get_nodes()
            self.start_dist = {node: 1.0 / len(nodes) for node in nodes} if nodes else {}
        else:
            self.start_dist = dict(start_dist)

    def log_evidence(self) -> float:
        terms = []
        for node in self.dirichlet.get_possible_start_nodes():
            prior = self.start_dist.get(node, 0.0)
            if prior > 0:
                terms.append(np.log(prior) + self.dirichlet.log_evidence_start_node(node))
        return float(logsumexp(terms)) if terms else -np.inf

    def probability_start_node(self, start_node: Hashable) -> float:
        if start_node not in self.dirichlet.get_possible_start_nodes():
            return 0.0
        log_norm = self.log_evidence()
        prior = self.start_dist.get(start_node, 0.0)
        if prior <= 0:
            return 0.0
        return float(np.exp(np.log(prior) + self.dirichlet.log_evidence_start_node(start_node) - log_norm))

    def start_node_probabilities(self) -> dict[Hashable, float]:
        return {node: self.probability_start_node(node) for node in self.dirichlet.get_possible_start_nodes()}

    def sample_start_node(self, rng: np.random.Generator | None = None) -> Hashable:
        generator = rng if rng is not None else np.random.default_rng()
        probs = self.start_node_probabilities()
        nodes = tuple(probs)
        weights = np.array([probs[node] for node in nodes], dtype=float)
        return nodes[int(generator.choice(len(nodes), p=weights))]

    def generate_sample(self, rng: np.random.Generator | None = None) -> tuple[Hashable, MealyHMM]:
        generator = rng if rng is not None else np.random.default_rng()
        start = self.sample_start_node(generator)
        machine = self.dirichlet.generate_sample(start, rng=generator)
        if machine is None:
            raise BayesianInferenceError("sampled impossible start node")
        return start, machine

    def posterior_mean_machine(self, start_node: Hashable | None = None) -> MealyHMM | None:
        """Posterior-mean machine.

        With ``start_node`` given, transition probabilities are posterior means
        conditioned on that start state and the machine starts there. Otherwise
        the parameters are conditioned on the MAP start state and the initial
        distribution is the posterior over start states.
        """
        if start_node is not None:
            return self.dirichlet.posterior_mean_machine(start_node)
        probs = self.start_node_probabilities()
        if not probs:
            return None
        start_node = max(probs, key=probs.get)
        initial = {node: mass for node, mass in probs.items() if mass > 0.0}
        return self.dirichlet.posterior_mean_machine(start_node, initial=initial)

    def as_pymc_model(self, start_node: Hashable | None = None) -> Any:
        from sofic.inference.bayesian.pymc_backend import epsilon_machine_model

        return epsilon_machine_model(self, start_node=start_node)


def _pooled_posterior_mean_machine(machine: MealyHMM, sequences: Sequence[Sequence[Any]]) -> MealyHMM | None:
    """Posterior-mean machine from several independent sequences of one topology.

    Each sequence is traced from its own MAP start state and the edge counts are
    pooled under a single uniform Dirichlet prior. The initial distribution is the
    average of the per-sequence start-state posteriors.
    """
    posteriors = [EpsilonMachinePosterior(machine, list(seq)) for seq in sequences]
    starts = [posterior.start_node_probabilities() for posterior in posteriors]
    if not posteriors or any(not probs for probs in starts):
        return None
    maps = [max(probs, key=probs.get) for probs in starts]
    base = posteriors[0].dirichlet
    probabilities: dict[tuple[Hashable, Any], float] = {}
    for edge in base.edges:
        if edge not in base.valid_edges:
            probabilities[edge] = 1.0
            continue
        count = sum(p.dirichlet.get_edge_count(s, edge) or 0 for p, s in zip(posteriors, maps, strict=True))
        row = sum(p.dirichlet.get_node_count(s, edge[0]) or 0 for p, s in zip(posteriors, maps, strict=True))
        alpha = base.alphas[edge]
        row_alpha = base.alphas[edge[0]]
        probabilities[edge] = float((count + alpha) / (row + row_alpha))
    initial: dict[Hashable, float] = {}
    for probs in starts:
        for node, mass in probs.items():
            initial[node] = initial.get(node, 0.0) + mass / len(starts)
    initial = {node: mass for node, mass in initial.items() if mass > 0.0}
    return base._machine_from_probabilities(maps[0], probabilities, "Pooled Posterior Mean Machine", initial)


InferEM = EpsilonMachinePosterior
