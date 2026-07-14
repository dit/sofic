"""Conjugate Bayesian inference for fixed stack-HMM topologies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from pensive.generators.stack_hmm import Configuration, HiddenMarkovStackModel
from pensive.graph import ATTR_SYMBOL
from pensive.inference.bayesian.counts import (
    BayesianInferenceError,
    dirichlet_multinomial_log_evidence,
    posterior_weights,
)
from pensive.shifts.sofic_dyck import TransitionRef, transition_ref


class PathCountStackHMM:
    """Transition counts along a sample path under a fixed stack topology."""

    def __init__(
        self,
        model: HiddenMarkovStackModel,
        data: Sequence[Any],
        *,
        max_stack_depth: int = 16,
    ) -> None:
        self.model = model
        self.max_stack_depth = int(max_stack_depth)
        self.counts: dict[Configuration, dict[TransitionRef, int]] = defaultdict(lambda: defaultdict(int))
        self.config_visits: dict[Configuration, int] = defaultdict(int)
        self._add_counts_from(data)

    def _add_counts_from(self, data: Sequence[Any]) -> None:
        data = tuple(data)
        if not data:
            return
        initial_states = [state for state, mass in self.model.initial_distribution.items() if mass > 0.0]
        if not initial_states:
            return
        configs: dict[Configuration, float] = {(state, ()): 1.0 for state in initial_states}
        for symbol in data:
            if symbol not in self.model.symbol_alphabet:
                return
            next_configs: dict[Configuration, float] = defaultdict(float)
            for config, mass in configs.items():
                if mass <= 0.0:
                    continue
                self.config_visits[config] += 1
                successors = self.model._normalized_successors(config, max_stack_depth=self.max_stack_depth)
                matching = [
                    (transition, prob, next_config)
                    for transition, prob, next_config in successors
                    if transition.data.get(ATTR_SYMBOL) == symbol
                ]
                if not matching:
                    return
                transition, prob, next_config = max(matching, key=lambda item: item[1])
                ref = transition_ref(transition)
                self.counts[config][ref] += 1
                next_configs[next_config] += mass * prob
            configs = dict(next_configs)
            if not configs:
                return

    def get_count(self, config: Configuration, edge: TransitionRef) -> int:
        return self.counts.get(config, {}).get(edge, 0)

    def get_config_count(self, config: Configuration) -> int:
        return self.config_visits.get(config, 0)


class DirichletDistributionStackHMM:
    """Product-of-Dirichlets posterior over enabled transitions per configuration."""

    def __init__(
        self,
        model: HiddenMarkovStackModel,
        data: Sequence[Any] | None = None,
        *,
        max_stack_depth: int = 16,
    ) -> None:
        self.model = model
        self.max_stack_depth = max_stack_depth
        self.configurations = model.reachable_configurations(max_stack_depth)
        self.edges_by_config: dict[Configuration, list[TransitionRef]] = {}
        self.alphas: dict[tuple[Configuration, TransitionRef], float] = {}
        self.data: PathCountStackHMM | None = None
        self._build_topology()
        self._uniform_alphas()
        if data is not None:
            self.data = PathCountStackHMM(model, data, max_stack_depth=max_stack_depth)

    def _build_topology(self) -> None:
        for config in self.configurations:
            enabled: list[TransitionRef] = []
            for transition, _prob, _next_config in self.model._normalized_successors(
                config,
                max_stack_depth=self.max_stack_depth,
            ):
                enabled.append(transition_ref(transition))
            if len(enabled) > 1:
                self.edges_by_config[config] = enabled

    def _uniform_alphas(self) -> None:
        self.alphas.clear()
        for config, edges in self.edges_by_config.items():
            for edge in edges:
                self.alphas[(config, edge)] = 1.0

    def log_evidence(self) -> float:
        if self.data is None:
            raise BayesianInferenceError("data is required for log evidence")
        evidence = 0.0
        for config, edges in self.edges_by_config.items():
            row_count = self.data.get_config_count(config)
            if row_count == 0:
                continue
            row_alpha = sum(self.alphas[(config, edge)] for edge in edges)
            cells = [(self.alphas[(config, edge)], self.data.get_count(config, edge)) for edge in edges]
            evidence += dirichlet_multinomial_log_evidence(row_alpha, row_count, cells)
        return float(evidence)

    def posterior_mean_probabilities(self) -> dict[TransitionRef, float]:
        if self.data is None:
            raise BayesianInferenceError("data is required for posterior mean")
        edge_totals: dict[TransitionRef, float] = defaultdict(float)
        edge_counts: dict[TransitionRef, float] = defaultdict(float)
        for config, edges in self.edges_by_config.items():
            row_count = self.data.get_config_count(config)
            if row_count == 0:
                continue
            row_alpha = sum(self.alphas[(config, edge)] for edge in edges)
            for edge in edges:
                alpha = self.alphas[(config, edge)]
                count = self.data.get_count(config, edge)
                mean = (count + alpha) / (row_count + row_alpha)
                edge_totals[edge] += mean
                edge_counts[edge] += 1.0
        return {edge: edge_totals[edge] / edge_counts[edge] for edge in edge_totals if edge_counts[edge] > 0}

    def posterior_mean_model(self) -> HiddenMarkovStackModel:
        from pensive.shifts.sofic_dyck import SoficDyckShift

        shift: SoficDyckShift = self.model.to_sofic_dyck_shift()
        probabilities = self.posterior_mean_probabilities()
        for transition in shift.transitions():
            ref = transition_ref(transition)
            if ref not in probabilities:
                probabilities[ref] = 1.0
        return HiddenMarkovStackModel.from_sofic_dyck_shift(
            shift,
            probabilities,
            initial_distribution=self.model.initial_distribution,
            allow_empty_stack_returns=self.model.allow_empty_stack_returns,
        )


class StackHMMPosterior:
    """Posterior over parameters for a fixed stack topology."""

    def __init__(
        self,
        model: HiddenMarkovStackModel,
        data: Sequence[Any] | None = None,
        *,
        max_stack_depth: int = 16,
    ) -> None:
        self.dirichlet = DirichletDistributionStackHMM(model, data, max_stack_depth=max_stack_depth)

    def log_evidence(self) -> float:
        return self.dirichlet.log_evidence()

    def posterior_mean_model(self) -> HiddenMarkovStackModel:
        return self.dirichlet.posterior_mean_model()


class ModelComparisonStackHMM:
    """Compare enumerated stack topologies by conjugate marginal likelihood."""

    def __init__(
        self,
        models: Sequence[HiddenMarkovStackModel],
        data: Sequence[Any],
        *,
        max_stack_depth: int = 16,
        topology_prior: str = "uniform",
    ) -> None:
        if not models:
            raise BayesianInferenceError("models must be non-empty")
        self.models = list(models)
        self.data = tuple(data)
        self.max_stack_depth = max_stack_depth
        self.topology_prior = topology_prior
        self.posteriors = [
            StackHMMPosterior(model, self.data, max_stack_depth=max_stack_depth) for model in self.models
        ]

    def log_evidences(self) -> list[float]:
        return [posterior.log_evidence() for posterior in self.posteriors]

    def model_probabilities(self) -> dict[int, float]:
        values = np.array(self.log_evidences(), dtype=float)
        if self.topology_prior == "penalty":
            values -= np.array([len(list(model.transitions())) for model in self.models], dtype=float)
        weights, _ = posterior_weights(list(range(len(values))), values)
        return weights

    def most_probable_model(self) -> HiddenMarkovStackModel:
        probs = self.model_probabilities()
        index = max(probs, key=probs.get)
        return self.posteriors[index].posterior_mean_model()
