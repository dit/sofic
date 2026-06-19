"""Information measures for stochastic generators via dit."""

from __future__ import annotations

from typing import Any

import numpy as np

from pensive.generators.base import HiddenMarkovModel, StochasticModel
from pensive.generators.markov import MarkovChain
from pensive.graph import ATTR_EMISSION, ATTR_PROB


def _require_dit():
    try:
        import dit
    except ImportError as exc:
        raise ImportError("dit is required for entropy measures; install with `pip install pensive[measures]`") from exc
    return dit


def state_distribution(model: StochasticModel) -> Any:
    """Return the stationary state law as a ``dit.Distribution``."""
    dit = _require_dit()
    idx = model.reindex()
    pi = model.stationary_distribution()
    outcomes = [(idx.state(i),) for i in range(len(idx))]
    return dit.Distribution(outcomes, pi)


def state_entropy(model: StochasticModel) -> float:
    """Shannon entropy of the stationary state distribution in bits."""
    dit = _require_dit()
    return float(dit.shannon.entropy(state_distribution(model)))


def _entropy_rate_from_transitions(
    hmm: HiddenMarkovModel,
    pi: np.ndarray,
    idx: Any,
) -> float:
    """Entropy rate from edge probabilities when symbol-labeled joint mass is absent."""
    dit = _require_dit()
    rate = 0.0
    for state in idx.states:
        i = idx.index(state)
        outgoing = list(hmm.graph.out_transitions(state))
        if not outgoing:
            continue
        targets: list[Any] = []
        probs: list[float] = []
        for transition in outgoing:
            prob = float(transition.data.get(ATTR_PROB, 0.0))
            if prob <= 0.0:
                continue
            emission = transition.data.get(ATTR_EMISSION)
            target = (transition.target, emission) if emission is not None else transition.target
            targets.append(target)
            probs.append(prob)
        if not probs:
            continue
        conditional = dit.Distribution(targets, probs)
        rate += float(pi[i] * dit.shannon.entropy(conditional))
    return rate


def entropy_rate_hmm(hmm: HiddenMarkovModel) -> float:
    """Shannon entropy rate for unifilar hidden Markov presentations."""
    from pensive.generators.hmm_inference import _emission_transition_tensors

    is_unifilar = getattr(hmm, "is_unifilar", None)
    if is_unifilar is None or not is_unifilar():
        raise NotImplementedError("entropy_rate_hmm is only exact for unifilar HMM presentations")

    dit = _require_dit()
    idx = hmm.reindex()
    pi = hmm.stationary_distribution()
    _, joint = _emission_transition_tensors(hmm)

    outcomes: list[tuple[Any, Any]] = []
    probs: list[float] = []
    for state in idx.states:
        i = idx.index(state)
        for symbol, matrix in joint.items():
            row_mass = float(pi[i] * matrix[i].sum())
            if row_mass <= 0.0:
                continue
            outcomes.append((state, symbol))
            probs.append(row_mass)

    if not probs:
        return entropy_rate_markov(hmm) if isinstance(hmm, MarkovChain) else _entropy_rate_from_transitions(hmm, pi, idx)

    total = sum(probs)
    joint_dist = dit.Distribution(outcomes, [p / total for p in probs])
    state_dist = state_distribution(hmm)
    return float(dit.shannon.entropy(joint_dist) - dit.shannon.entropy(state_dist))


def entropy_rate_markov(chain: MarkovChain) -> float:
    """Shannon entropy rate of a visible Markov chain in bits."""
    dit = _require_dit()
    idx = chain.reindex()
    pi = chain.stationary_distribution()
    rate = 0.0
    for state in idx.states:
        i = idx.index(state)
        outgoing = list(chain.graph.out_transitions(state))
        if not outgoing:
            continue
        targets: list[Any] = []
        probs: list[float] = []
        for transition in outgoing:
            prob = float(transition.data.get(ATTR_PROB, 0.0))
            if prob <= 0.0:
                continue
            targets.append(transition.target)
            probs.append(prob)
        if not probs:
            continue
        conditional = dit.Distribution(targets, probs)
        rate += float(pi[i] * dit.shannon.entropy(conditional))
    return rate
