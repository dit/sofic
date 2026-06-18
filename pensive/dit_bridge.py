"""Thin adapters from pensive generators to dit information measures."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from pensive.generators.base import HiddenMarkovModel, QuasiStochasticModel
from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from pensive.generators.epsilon_machine import EpsilonMachine


def _require_dit():
    try:
        import dit
    except ImportError as exc:
        raise ImportError("dit is required for entropy measures; install with `pip install pensive[measures]`") from exc
    return dit


def joint_block_distribution(
    generator: HiddenMarkovModel,
    history_length: int = 1,
) -> Any:
    """Build a dit Distribution over (past..., present) emission blocks."""
    from pensive.graph import ATTR_EMISSION, ATTR_PROB

    dit = _require_dit()
    idx = generator.reindex()
    pi = np.array([generator.initial_distribution.get(s, 0.0) for s in idx.states], dtype=float)
    if pi.sum() <= 0.0:
        pi = generator.stationary_distribution()
    joint: dict[Any, np.ndarray] = {}
    for transition in generator.transitions():
        emission = transition.data.get(ATTR_EMISSION)
        if emission is None:
            continue
        matrix = joint.setdefault(emission, np.zeros((len(idx), len(idx)), dtype=float))
        i = idx.index(transition.source)
        j = idx.index(transition.target)
        matrix[i, j] += float(transition.data.get(ATTR_PROB, 0.0))

    if history_length <= 0:
        marginal = np.zeros(len(generator.observation_alphabet), dtype=float)
        symbol_list = sorted(generator.observation_alphabet, key=repr)
        for symbol in symbol_list:
            matrix = joint.get(symbol)
            if matrix is not None:
                marginal[symbol_list.index(symbol)] = float(pi @ matrix @ np.ones(len(idx)))
        outcomes = [(symbol,) for symbol in symbol_list]
        return dit.Distribution(outcomes, marginal)

    # history_length == 1: (past, present)
    outcomes: list[tuple[Any, ...]] = []
    probs: list[float] = []
    for past_symbol in generator.observation_alphabet:
        past_matrix = joint.get(past_symbol)
        if past_matrix is None:
            continue
        after_past = pi @ past_matrix
        for present_symbol in generator.observation_alphabet:
            present_matrix = joint.get(present_symbol)
            if present_matrix is None:
                continue
            prob = float(after_past @ present_matrix @ np.ones(len(idx)))
            if prob > 0.0:
                outcomes.append((past_symbol, present_symbol))
                probs.append(prob)
    total = sum(probs)
    if total > 0.0:
        probs = [p / total for p in probs]
    return dit.Distribution(outcomes, probs)


def entropy_rate(generator: HiddenMarkovModel | EpsilonMachine) -> float:
    """Deprecated: use ``generator.entropy_rate()`` instead."""
    warnings.warn(
        "pensive.dit_bridge.entropy_rate is deprecated; call generator.entropy_rate()",
        DeprecationWarning,
        stacklevel=2,
    )
    return generator.entropy_rate()


def statistical_complexity(eps_machine: EpsilonMachine) -> float:
    """Deprecated: use ``eps_machine.statistical_complexity()`` instead."""
    warnings.warn(
        "pensive.dit_bridge.statistical_complexity is deprecated; call eps_machine.statistical_complexity()",
        DeprecationWarning,
        stacklevel=2,
    )
    return eps_machine.statistical_complexity()


def excess_entropy_bidirectional(bidir: BidirectionalEpsilonMachine) -> float:
    """Exact excess entropy E = I[S⁺; S⁻] from the bidirectional joint distribution."""
    dit = _require_dit()
    joint = bidir.joint_distribution()
    if not joint:
        return 0.0

    pi_plus: dict[Any, float] = {}
    pi_minus: dict[Any, float] = {}
    for (alpha, gamma), mass in joint.items():
        pi_plus[alpha] = pi_plus.get(alpha, 0.0) + mass
        pi_minus[gamma] = pi_minus.get(gamma, 0.0) + mass

    plus_outcomes = list(pi_plus.keys())
    minus_outcomes = list(pi_minus.keys())
    plus_dist = dit.Distribution(plus_outcomes, [pi_plus[s] for s in plus_outcomes])
    minus_dist = dit.Distribution(minus_outcomes, [pi_minus[s] for s in minus_outcomes])
    joint_outcomes = list(joint.keys())
    joint_dist = dit.Distribution(joint_outcomes, [joint[outcome] for outcome in joint_outcomes])
    return float(
        dit.shannon.entropy(plus_dist)
        + dit.shannon.entropy(minus_dist)
        - dit.shannon.entropy(joint_dist)
    )


def bidirectional_statistical_complexity(bidir: BidirectionalEpsilonMachine) -> float:
    """C± = H[S⁺, S⁻] under the bidirectional stationary distribution."""
    dit = _require_dit()
    joint = bidir.joint_distribution()
    if not joint:
        return 0.0
    outcomes = list(joint.keys())
    probs = [joint[outcome] for outcome in outcomes]
    return float(dit.shannon.entropy(dit.Distribution(outcomes, probs)))


def crypticity(bidir: BidirectionalEpsilonMachine) -> float:
    """χ = C± − E for a bidirectional presentation."""
    return bidirectional_statistical_complexity(bidir) - excess_entropy_bidirectional(bidir)


# RV indices in :func:`~pensive.generators.bidirectional_construction.bidirectional_step_distribution`
_STEP_S_PLUS_0 = 0
_STEP_S_MINUS_0 = 1
_STEP_X_0 = 2
_STEP_S_PLUS_1 = 3
_STEP_S_MINUS_1 = 4


def _step_distribution(bidir: BidirectionalEpsilonMachine) -> Any:
    from pensive.generators.bidirectional_construction import bidirectional_step_distribution

    return bidirectional_step_distribution(bidir)


def predicted_information(bidir: BidirectionalEpsilonMachine) -> float:
    """ρ_μ = I[X₀ : S⁺₀] — predicted information rate (James et al., 2013, Eq. 2)."""
    dit = _require_dit()
    dist = _step_distribution(bidir)
    return float(
        dit.shannon.mutual_information(dist, [_STEP_X_0], [_STEP_S_PLUS_0])
    )


def bound_information(bidir: BidirectionalEpsilonMachine) -> float:
    """b_μ = H[X₀ | S⁺₀, S⁻₁] — bound information rate (James et al., 2013, supplement)."""
    dit = _require_dit()
    dist = _step_distribution(bidir)
    return float(
        dit.shannon.conditional_entropy(
            dist, [_STEP_X_0], [_STEP_S_PLUS_0, _STEP_S_MINUS_1]
        )
    )


def ephemeral_information(bidir: BidirectionalEpsilonMachine) -> float:
    """r_μ = I[X₀ : S⁻₁ | S⁺₀] — ephemeral information rate (James et al., 2013, supplement)."""
    return float(bidir.entropy_rate() - bound_information(bidir))


def information_anatomy(bidir: BidirectionalEpsilonMachine) -> dict[str, float]:
    """Return ρ_μ, b_μ, r_μ, h_μ, E, and χ for a bidirectional presentation."""
    h_mu = bidir.entropy_rate()
    b_mu = bound_information(bidir)
    return {
        "rho_mu": predicted_information(bidir),
        "bound_mu": b_mu,
        "ephemeral_mu": ephemeral_information(bidir),
        "entropy_rate": h_mu,
        "excess_entropy": excess_entropy_bidirectional(bidir),
        "crypticity": crypticity(bidir),
    }


def excess_entropy(generator: HiddenMarkovModel, max_block: int = 4) -> float:
    """E = lim H(X_{-n:0}) - n h; approximated by finite block entropies via dit."""
    dit = _require_dit()
    h = generator.entropy_rate()
    estimates: list[float] = []
    for n in range(1, max_block + 1):
        dist = joint_block_distribution(generator, history_length=n)
        estimates.append(float(dit.shannon.entropy(dist)) - n * h)
    return float(np.mean(estimates)) if estimates else 0.0


def collision_entropy(quasi_model: QuasiStochasticModel) -> float:
    """Second Renyi entropy rate H_2 from quasi transition matrices."""
    dit = _require_dit()
    matrices = quasi_model.transition_matrices()
    pi = quasi_model.stationary_quasidistribution()
    total = 0.0
    for matrix in matrices.values():
        total += float(pi @ (matrix @ matrix) @ np.ones(len(pi)))
    if total <= 0.0:
        return 0.0
    return float(-np.log(total))


def process_negativity(quasi_model: QuasiStochasticModel) -> float:
    """Negativity proxy: L1 distance of stationary quasidistribution from positive part."""
    pi = quasi_model.stationary_quasidistribution()
    positive = np.maximum(pi, 0.0)
    return float(np.sum(np.abs(pi - positive)))
