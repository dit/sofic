"""Information measures for stochastic generators via dit."""

from __future__ import annotations

from typing import Any

import numpy as np

from sofic.generators.base import HiddenMarkovModel, QuasiStochasticModel, StochasticModel
from sofic.generators.markov import MarkovChain
from sofic.graph import ATTR_EMISSION, ATTR_PROB


def require_dit(feature: str = "entropy measures") -> Any:
    """Import and return the :mod:`dit` package, or raise a helpful error.

    ``feature`` names the capability requiring dit and is interpolated into the
    error message when the optional dependency is missing.
    """
    try:
        import dit
    except ImportError as exc:
        raise ImportError(f"dit is required for {feature}; install with `pip install dit`") from exc
    return dit


# Backwards-compatible internal alias.
_require_dit = require_dit


def dit_state_label(state: Any) -> Any:
    """Return a dit-safe single-symbol label for a machine state.

    ``dit.Distribution`` treats each outcome as a sequence of random-variable
    values, so a tuple-valued state (e.g. an edge-machine state like
    ``("A", "0", "A")``) is misread as multi-dimensional coordinate data and
    raises ``MissingDimensionsError``. Tuple states are encoded to a lossless
    string via :func:`sofic.generators.edge_machine.edge_state_label` (invert
    with ``parse_edge_state_label``); scalar states pass through unchanged.
    """
    if isinstance(state, tuple):
        from sofic.generators.edge_machine import edge_state_label

        return edge_state_label(state)
    return state


def state_distribution(model: StochasticModel) -> Any:
    """Return the stationary state law as a ``dit.Distribution``.

    States are emitted as dit-safe labels (see :func:`dit_state_label`): scalar
    states are preserved verbatim, tuple states are encoded to a lossless string.
    """
    dit = _require_dit()
    idx = model.reindex()
    pi = model.stationary_distribution()
    outcomes = [(dit_state_label(idx.state(i)),) for i in range(len(idx))]
    from sofic.generators.prob import as_prob, has_symbolic, simplify_prob

    probs = [as_prob(pi[i]) for i in range(len(idx))]
    if has_symbolic(probs):
        from dit.symbolic import symbolic_distribution

        return symbolic_distribution(outcomes, [simplify_prob(p) for p in probs])
    return dit.Distribution(outcomes, [float(p) for p in probs])


def state_entropy(model: StochasticModel) -> Any:
    """Shannon entropy of the stationary state distribution in bits."""
    dit = _require_dit()
    dist = state_distribution(model)
    value = dit.shannon.entropy(dist)
    if hasattr(dist, "is_symbolic") and dist.is_symbolic():
        return value
    return float(value)


def joint_block_distribution(
    generator: HiddenMarkovModel,
    history_length: int = 1,
) -> Any:
    """Build a ``dit.Distribution`` over observed emission blocks.

    ``history_length`` counts symbols before the present symbol, so the emitted
    block length is ``history_length + 1``.
    """
    from itertools import product

    from sofic.generators.hmm_inference import _stationary_emission_tensors

    dit = _require_dit()
    # Blocks of a stationary process are weighted by the stationary state law, not
    # the model's initial distribution (which may describe only the transient).
    pi, joint = _stationary_emission_tensors(generator)

    symbol_list = sorted(generator.observation_alphabet, key=repr)
    block_length = max(1, history_length + 1)
    ones = np.ones(len(pi), dtype=float)
    outcomes = list(product(symbol_list, repeat=block_length))
    probs = []
    for outcome in outcomes:
        mass = pi.copy()
        for symbol in outcome:
            mass = mass @ joint.get(symbol, np.zeros((len(pi), len(pi)), dtype=float))
        probs.append(float(mass @ ones))

    total = sum(probs)
    if total > 0.0:
        probs = [p / total for p in probs]
    return dit.Distribution(outcomes, probs)


def _entropy_rate_from_transitions(
    model: StochasticModel,
    pi: np.ndarray,
    idx: Any,
) -> Any:
    """Entropy rate from edge probabilities when symbol-labeled joint mass is absent.

    Accepts any :class:`StochasticModel` (visible Markov chain or hidden Markov
    model); the target state stands in as the emitted symbol when no emission is set.
    """
    dit = _require_dit()
    from sofic.generators.prob import (
        as_prob,
        has_symbolic,
        is_positive_mass,
        is_symbolic,
        simplify_prob,
    )

    symbolic = pi.dtype == object or has_symbolic(pi.ravel())
    rate: Any = 0 if symbolic else 0.0
    for state in idx.states:
        i = idx.index(state)
        outgoing = list(model.graph.out_transitions(state))
        if not outgoing:
            continue
        targets: list[Any] = []
        probs: list[Any] = []
        for transition in outgoing:
            prob = as_prob(transition.data.get(ATTR_PROB, 0.0))
            if not is_positive_mass(prob):
                continue
            emission = transition.data.get(ATTR_EMISSION)
            target = (transition.target, emission) if emission is not None else transition.target
            targets.append(dit_state_label(target))
            probs.append(prob)
        if not probs:
            continue
        if has_symbolic(probs) or symbolic:
            from dit.symbolic import symbolic_distribution

            conditional = symbolic_distribution(targets, [simplify_prob(p) for p in probs])
            contrib = as_prob(pi[i]) * dit.shannon.entropy(conditional)
            rate = simplify_prob(as_prob(rate) + as_prob(contrib))
        else:
            conditional = dit.Distribution(targets, [float(p) for p in probs])
            rate += float(pi[i] * dit.shannon.entropy(conditional))
    if is_symbolic(rate):
        return simplify_prob(rate)
    return float(rate)


def entropy_rate_hmm(hmm: HiddenMarkovModel) -> Any:
    """Shannon entropy rate for unifilar hidden Markov presentations.

    Returns a sympy :class:`~sympy.Expr` when the stationary law or emission
    tensors are symbolic; otherwise a Python ``float``.
    """
    from sofic.generators.hmm_inference import _emission_transition_tensors
    from sofic.generators.prob import (
        array_sum,
        as_prob,
        has_symbolic,
        is_positive_mass,
        is_symbolic,
        simplify_prob,
        sum_probs,
    )

    is_unifilar = getattr(hmm, "is_unifilar", None)
    if is_unifilar is None or not is_unifilar():
        raise NotImplementedError("entropy_rate_hmm is only exact for unifilar HMM presentations")

    dit = _require_dit()
    idx = hmm.reindex()
    pi = hmm.stationary_distribution()
    _, joint = _emission_transition_tensors(hmm)

    symbolic = pi.dtype == object or has_symbolic(pi.ravel())
    if not symbolic:
        symbolic = any(matrix.dtype == object or has_symbolic(matrix.ravel()) for matrix in joint.values())

    # Emit dit-safe state labels so tuple-valued states (e.g. edge-machine
    # states like ("A", "0", "A")) do not break dit.Distribution.
    outcomes: list[tuple[Any, Any]] = []
    probs: list[Any] = []
    for state in idx.states:
        i = idx.index(state)
        label = dit_state_label(state)
        for symbol, matrix in joint.items():
            row_mass = as_prob(pi[i]) * array_sum(matrix[i])
            row_mass = simplify_prob(row_mass) if symbolic or is_symbolic(row_mass) else float(row_mass)
            if not is_positive_mass(row_mass):
                continue
            outcomes.append((label, symbol))
            probs.append(row_mass)

    if not probs:
        return (
            entropy_rate_markov(hmm) if isinstance(hmm, MarkovChain) else _entropy_rate_from_transitions(hmm, pi, idx)
        )

    total = sum_probs(probs)
    state_dist = state_distribution(hmm)
    if symbolic or has_symbolic(probs) or is_symbolic(total):
        from dit.symbolic import symbolic_distribution

        joint_dist = symbolic_distribution(
            outcomes,
            [simplify_prob(as_prob(p) / as_prob(total)) for p in probs],
        )
        return simplify_prob(as_prob(dit.shannon.entropy(joint_dist)) - as_prob(dit.shannon.entropy(state_dist)))
    joint_dist = dit.Distribution(outcomes, [float(p) / float(total) for p in probs])
    return float(dit.shannon.entropy(joint_dist) - dit.shannon.entropy(state_dist))


def entropy_rate_markov(chain: MarkovChain) -> Any:
    """Shannon entropy rate of a visible Markov chain in bits.

    A visible Markov chain has no separate emissions, so its entropy rate is the
    conditional-transition entropy computed by :func:`_entropy_rate_from_transitions`
    (which treats the target state as the emitted symbol when no emission is set).
    """
    idx = chain.reindex()
    pi = chain.stationary_distribution()
    return _entropy_rate_from_transitions(chain, pi, idx)


def collision_entropy(quasi_model: QuasiStochasticModel) -> float:
    """Second Renyi entropy rate from quasi transition matrices."""
    matrices = quasi_model.transition_matrices()
    pi = quasi_model.stationary_quasidistribution()
    total = 0.0
    for matrix in matrices.values():
        total += float(pi @ (matrix @ matrix) @ np.ones(len(pi)))
    if total <= 0.0:
        return 0.0
    return float(-np.log(total))


def process_negativity(quasi_model: QuasiStochasticModel) -> float:
    """Negativity proxy from the stationary quasidistribution."""
    pi = quasi_model.stationary_quasidistribution()
    positive = np.maximum(pi, 0.0)
    return float(np.sum(np.abs(pi - positive)))
