"""Bayesian model comparison utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy.special import logsumexp

from pensive.generators.mealy import MealyHMM
from pensive.inference.bayesian.counts import BayesianInferenceError
from pensive.inference.bayesian.markov import MarkovChainPosterior


class ModelComparisonMC:
    """Compare Markov-chain orders using exact conjugate evidences."""

    def __init__(
        self,
        alphabet: Sequence[Any],
        data: Sequence[Any],
        min_order: int,
        max_order: int,
        mc_prior: str = "uniform",
        mo_prior: str = "uniform",
    ):
        if min_order < 0 or max_order < min_order:
            raise BayesianInferenceError("invalid min/max Markov orders")
        self.alphabet = tuple(alphabet)
        self.min_order = int(min_order)
        self.max_order = int(max_order)
        self.markov_chain_prior = mc_prior
        self.model_order_prior = mo_prior
        self.mc_dict = {
            order: MarkovChainPosterior(self.alphabet, data, order, prior_type=mc_prior)
            for order in range(self.min_order, self.max_order + 1)
        }

    def log_evidence(self) -> dict[int, float]:
        return {order: posterior.log_evidence() for order, posterior in sorted(self.mc_dict.items())}

    def _log_prior_penalty(self, order: int) -> float:
        if self.model_order_prior == "uniform":
            return 0.0
        if self.model_order_prior == "penalty":
            a = len(self.alphabet)
            return -float((a**order) * (a - 1))
        raise BayesianInferenceError("unknown model-order prior")

    def model_probabilities(self) -> dict[int, float]:
        orders = sorted(self.mc_dict)
        values = np.array([self.mc_dict[order].log_evidence() + self._log_prior_penalty(order) for order in orders], dtype=float)
        norm = logsumexp(values)
        return {order: float(np.exp(value - norm)) for order, value in zip(orders, values, strict=True)}

    def most_probable_model(self) -> MealyHMM:
        probs = self.model_probabilities()
        order = max(probs, key=probs.get)
        return self.mc_dict[order].generate_mealy_hmm(method="PME")


class ModelComparisonMC2(ModelComparisonMC):
    """Compare an explicit list of Markov-chain orders."""

    def __init__(
        self,
        alphabet: Sequence[Any],
        data: Sequence[Any],
        orders: int | Sequence[int],
        mc_prior: str = "uniform",
        mo_prior: str = "uniform",
    ):
        order_list = list(range(orders + 1)) if isinstance(orders, int) else sorted(int(order) for order in orders)
        if not order_list or min(order_list) < 0:
            raise BayesianInferenceError("invalid Markov orders")
        self.alphabet = tuple(alphabet)
        self.min_order = min(order_list)
        self.max_order = max(order_list)
        self.orders = order_list
        self.markov_chain_prior = mc_prior
        self.model_order_prior = mo_prior
        self.mc_dict = {order: MarkovChainPosterior(self.alphabet, data, order, prior_type=mc_prior) for order in order_list}


class ModelComparisonEM:
    """Compare candidate unifilar topologies using epsilon-machine evidences."""

    def __init__(self, machines: Iterable[MealyHMM], data: Sequence[Any] | None = None, beta: float = 0.0, state_path: bool = False):
        from pensive.inference.bayesian.epsilon import EpsilonMachinePosterior

        self.beta = float(beta)
        self.em_dict: dict[str, EpsilonMachinePosterior] = {}
        self.numMachines = 0
        self.possMachines = 0
        for index, machine in enumerate(machines):
            self.numMachines += 1
            posterior = EpsilonMachinePosterior(machine, data, state_path=state_path)
            if posterior.log_evidence() > -np.inf:
                self.possMachines += 1
                name = getattr(machine, "name", None) or f"Machine-{index}"
                self.em_dict[str(name)] = posterior
        self.evidence_dictionary: dict[str, float] = {}
        self.probs: dict[str, float] = {}
        self.set_evidence = -np.inf

    def log_evidence(self) -> dict[str, float]:
        if not self.evidence_dictionary:
            self.evidence_dictionary = {name: posterior.log_evidence() for name, posterior in self.em_dict.items()}
        return dict(self.evidence_dictionary)

    def model_probabilities(self) -> dict[str, float]:
        if self.probs:
            return dict(self.probs)
        evidence = self.log_evidence()
        names = list(evidence)
        values = np.array(
            [evidence[name] - self.beta * len(self.em_dict[name].dirichlet.nodes) for name in names],
            dtype=float,
        )
        self.set_evidence = float(logsumexp(values)) if len(values) else -np.inf
        self.probs = {name: float(np.exp(value - self.set_evidence)) for name, value in zip(names, values, strict=True)}
        return dict(self.probs)

    def generate_sample(self, rng: np.random.Generator | None = None) -> tuple[Any, MealyHMM]:
        generator = rng if rng is not None else np.random.default_rng()
        probs = self.model_probabilities()
        names = tuple(probs)
        weights = np.array([probs[name] for name in names], dtype=float)
        choice = names[int(generator.choice(len(names), p=weights))]
        return self.em_dict[choice].generate_sample(rng=generator)
