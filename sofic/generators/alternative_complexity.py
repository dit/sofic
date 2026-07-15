"""Alternative complexity measures for epsilon-machines."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from sofic.generators.stochastic import shannon_entropy
from sofic.graph import ATTR_PROB
from sofic.properties import transition_matrix

if TYPE_CHECKING:
    from sofic.generators.epsilon_machine import EpsilonMachine


def structural_information(machine: EpsilonMachine) -> float:
    """Asymptotic structural information (equals excess entropy for finite-state sources).

    For a canonical epsilon-machine, :math:`I_{\\mathrm{struct}} = \\lim_{L\\to\\infty}
    H[S_L \\mid X_{0:L}] = E` (Feldman & Crutchfield, 1998; Ellison et al., 2009).
    """
    return machine.excess_entropy()


def thermodynamic_depth(machine: EpsilonMachine) -> float:
    """Thermodynamic depth: stationary average mean first-passage time to causal states.

    Uses the hidden-state transition matrix (summing over emissions) and the
    stationary causal-state distribution (Shalizi & Crutchfield, 1999).
    """
    pi = machine.stationary_distribution()
    states = list(machine.states())
    if not states:
        return 0.0
    index = {state: i for i, state in enumerate(states)}
    n = len(states)
    transition, _ = transition_matrix(machine, attr=ATTR_PROB, states=states)

    pi_vec = np.asarray([pi[index[state]] for state in states], dtype=float)
    pi_vec = pi_vec / pi_vec.sum()

    depths = np.zeros(n, dtype=float)
    for target in range(n):
        depths[target] = _mean_first_passage_time(transition, pi_vec, target)

    return float(np.dot(pi_vec, depths))


def spectral_complexity(machine: EpsilonMachine) -> float:
    """Spectral entropy of nontrivial mixed-state transition eigenvalues.

    Builds the recurrent mixed-state presentation and summarizes the modulus
    spectrum of its transition operator (Riechers & Crutchfield, 2017).
    """
    from sofic.generators.mixed_state_construction import build_mixed_state_presentation

    msp = build_mixed_state_presentation(machine)
    states = tuple(msp.recurrent_states)
    if len(states) <= 1:
        return 0.0

    matrix, _ = transition_matrix(msp, attr=ATTR_PROB, states=states)

    eigenvalues = np.linalg.eigvals(matrix)
    moduli = np.sort(np.abs(eigenvalues))[::-1]
    nontrivial = moduli[moduli < 1.0 - 1e-9]
    if nontrivial.size == 0:
        return 0.0
    weights = nontrivial / nontrivial.sum()
    return shannon_entropy(weights)


def _mean_first_passage_time(transition: np.ndarray, start: np.ndarray, target: int) -> float:
    n = transition.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0

    # Fundamental-matrix formula for MFPT from i to j.
    pi = _stationary_vector(transition)
    if pi is None:
        pi = np.ones(n) / n
    fundamental = np.linalg.inv(np.eye(n) - transition + np.outer(np.ones(n), pi))
    mfpt = np.zeros(n, dtype=float)
    for i in range(n):
        if i == target:
            mfpt[i] = 0.0
        elif pi[target] <= 0.0:
            mfpt[i] = math.inf
        else:
            mfpt[i] = (fundamental[target, target] - fundamental[i, target]) / pi[target]
    return float(start @ mfpt)


def _stationary_vector(transition: np.ndarray) -> np.ndarray | None:
    from sofic.generators.stationary import stationary_distribution_from_transition

    try:
        return stationary_distribution_from_transition(transition)
    except Exception:
        return None
