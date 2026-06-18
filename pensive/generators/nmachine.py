"""Negative machines (n-machines) with quasiprobabilities."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np

from pensive.exceptions import QuasiStochasticValidationError
from pensive.generators.base import QuasiStochasticModel
from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.graph import ATTR_EMISSION, ATTR_QUASIPROB


class NMachine(QuasiStochasticModel):
    """Mealy-type generator with signed joint quasiprobabilities on edges."""

    observation_alphabet: frozenset[Any]

    def __init__(self, observation_alphabet: frozenset[Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.observation_alphabet = observation_alphabet if observation_alphabet is not None else frozenset()

    def validate_quasistochastic(self) -> None:
        super().validate_quasistochastic()
        for state in self.states():
            outgoing = list(self.graph.out_transitions(state))
            total = sum(t.data.get(ATTR_QUASIPROB, 0.0) for t in outgoing)
            if outgoing and not np.isclose(total, 1.0):
                raise QuasiStochasticValidationError(f"quasi masses from {state!r} sum to {total}")
            marginals: dict[Any, float] = {}
            for transition in outgoing:
                emission = transition.data.get(ATTR_EMISSION)
                if emission is None:
                    continue
                marginals[emission] = marginals.get(emission, 0.0) + transition.data.get(ATTR_QUASIPROB, 0.0)
            for emission, mass in marginals.items():
                if mass < -1e-12:
                    raise QuasiStochasticValidationError(
                        f"negative marginal output probability P({emission!r}|{state!r})"
                    )
                self._require(emission in self.observation_alphabet, f"unknown emission {emission!r}")

    def is_unifilar(self) -> bool:
        """Return whether each state emits at most one edge per symbol."""
        from pensive.properties import is_unifilar_emissions

        return is_unifilar_emissions(self)

    @classmethod
    def from_epsilon_machine(
        cls,
        eps: EpsilonMachine,
        splits: Mapping[Hashable, int] | None = None,
        **kwargs: Any,
    ) -> NMachine:
        from pensive.generators.nmachine_construction import build_nmachine_from_epsilon

        return build_nmachine_from_epsilon(eps, splits)

    def coarse_grained_distribution(self) -> dict[Hashable, float]:
        from pensive.generators.nmachine_construction import coarse_grained_distribution

        eps_states = tuple(
            substate[0]
            for substate in self.states()
            if isinstance(substate, tuple) and len(substate) == 2
        )
        if not eps_states:
            eps_states = tuple(self.states())
        return coarse_grained_distribution(self, eps_states)

    def to_quasi_realization(self) -> QuasiRealization:
        from pensive.generators.quasi_realization import QuasiRealization

        return QuasiRealization.from_nmachine(self)
