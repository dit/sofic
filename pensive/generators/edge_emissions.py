"""Shared helpers for edge-labeled stochastic generators."""

from __future__ import annotations

from typing import Any

from pensive.exceptions import StochasticValidationError
from pensive.generators.prob import is_symbolic, row_sums_to_one
from pensive.graph import ATTR_EMISSION, ATTR_PROB


def validate_stochastic_edge_emissions(
    model: Any,
    *,
    alphabet: frozenset[Any],
    alphabet_name: str,
    row_mass_label: str,
    negative_probability_label: str,
) -> None:
    """Validate row-stochastic edge probabilities and emission alphabet membership."""
    for state in model.states():
        outgoing = list(model.graph.out_transitions(state))
        probs = [t.data.get(ATTR_PROB, 0.0) for t in outgoing]
        if outgoing and not row_sums_to_one(probs):
            total = sum(probs)
            raise StochasticValidationError(f"{row_mass_label} from {state!r} sum to {total}")
        for transition in outgoing:
            prob = transition.data.get(ATTR_PROB, 0.0)
            if is_symbolic(prob):
                if getattr(prob, "is_negative", None) is True:
                    raise StochasticValidationError(f"{negative_probability_label} on {transition}")
            elif prob < 0:
                raise StochasticValidationError(f"{negative_probability_label} on {transition}")
            emission = transition.data.get(ATTR_EMISSION)
            if emission is not None:
                model._require(emission in alphabet, f"emission {emission!r} not in {alphabet_name} alphabet")
