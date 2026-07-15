"""Directional information flow on known stochastic generators."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np

from sofic.generators.base import HiddenMarkovModel, StochasticModel
from sofic.graph import ATTR_EMISSION, ATTR_PROB


def _require_dit():
    from sofic.generators.measures import require_dit

    return require_dit("directional flow")


def _pair_block_distribution(generator: HiddenMarkovModel, *, history: int) -> Any:
    """Joint law over flattened ``(x0, y0, x1, y1, ...)`` windows."""
    from sofic.generators.hmm_inference import _stationary_emission_tensors

    dit = _require_dit()
    # Directional-flow statistics describe the stationary joint process, so weight
    # the initial state by the stationary law rather than ``initial_distribution``.
    pi, joint = _stationary_emission_tensors(generator)

    block_length = history + 1
    ones = np.ones(len(pi), dtype=float)
    outcomes: list[tuple[Any, ...]] = []
    probs: list[float] = []

    def walk(
        mass: np.ndarray,
        prefix: tuple[Any, ...],
        steps_remaining: int,
    ) -> None:
        if steps_remaining == 0:
            outcomes.append(prefix)
            probs.append(float(mass @ ones))
            return
        for symbol, matrix in joint.items():
            if not isinstance(symbol, tuple) or len(symbol) != 2:
                raise TypeError("generator must emit length-2 tuple symbols")
            next_mass = mass @ matrix
            if next_mass.sum() <= 0.0:
                continue
            walk(next_mass, prefix + (symbol[0], symbol[1]), steps_remaining - 1)

    walk(pi.copy(), (), block_length)
    total = sum(probs)
    if total > 0.0:
        probs = [p / total for p in probs]
    return dit.Distribution(outcomes, probs)


def _window_distribution(
    generator: HiddenMarkovModel,
    *,
    history: int,
) -> Any:
    if history < 0:
        raise ValueError("history must be nonnegative")
    return _pair_block_distribution(generator, history=history)


def _rv_indices(history: int) -> dict[str, list[int]]:
    """Map window variables to flattened distribution indices."""
    names: dict[str, list[int]] = {}
    for t in range(history + 1):
        names[f"x_{t}"] = [2 * t]
        names[f"y_{t}"] = [2 * t + 1]
    names["x_past"] = [idx for t in range(history) for idx in names[f"x_{t}"]]
    names["y_past"] = [idx for t in range(history) for idx in names[f"y_{t}"]]
    names["x_pres"] = names[f"x_{history}"]
    names["y_pres"] = names[f"y_{history}"]
    return names


def transfer_entropy(
    generator: HiddenMarkovModel,
    *,
    source: str = "x",
    target: str = "y",
    history: int = 1,
) -> float:
    """One-step transfer entropy ``I(Y_t; X_{past} | Y_{past})`` on tuple emissions.

    The generator must emit ``(x, y)`` pairs at each step. ``source`` and
    ``target`` must be ``'x'`` or ``'y'``.
    """
    if source not in {"x", "y"} or target not in {"x", "y"}:
        raise ValueError("source and target must be 'x' or 'y'")
    _require_dit()
    from dit.multivariate import total_correlation as I

    dist = _window_distribution(generator, history=history)
    idx = _rv_indices(history)
    source_past = idx[f"{source}_past"]
    target_past = idx[f"{target}_past"]
    target_pres = idx[f"{target}_pres"]
    return float(I(dist, [target_pres, source_past], target_past))


def directed_information(
    generator: HiddenMarkovModel,
    *,
    source: str = "x",
    target: str = "y",
    length: int = 1,
) -> float:
    """Finite-length directed information ``I(X^{n} -> Y^{n})`` on tuple emissions."""
    if length < 1:
        raise ValueError("length must be positive")
    _require_dit()
    from dit.multivariate import total_correlation as I

    total = 0.0
    for t in range(length):
        dist = _window_distribution(generator, history=t)
        idx = _rv_indices(t)
        source_block = [index for k in range(t + 1) for index in idx[f"{source}_{k}"]]
        target_pres = idx[f"{target}_{t}"]
        target_past = idx[f"{target}_past"] if t > 0 else []
        total += float(I(dist, [source_block, target_pres], target_past))
    return total


def intrinsic_information_flow(
    generator: HiddenMarkovModel,
    *,
    source: str = "x",
    target: str = "y",
    history: int = 1,
) -> float:
    """Intrinsic information flow estimate at finite window length."""
    _require_dit()
    from dit.multivariate import intrinsic_total_correlation as IMI

    dist = _window_distribution(generator, history=history)
    idx = _rv_indices(history)
    source_past = idx[f"{source}_past"]
    target_past = idx[f"{target}_past"]
    target_pres = idx[f"{target}_pres"]
    return float(IMI(dist, [target_pres, source_past], target_past))


def shared_information_flow(
    generator: HiddenMarkovModel,
    *,
    source: str = "x",
    target: str = "y",
    history: int = 1,
) -> float:
    """Shared flow: time-delayed mutual information minus intrinsic flow."""
    _require_dit()
    from dit.multivariate import total_correlation as I

    dist = _window_distribution(generator, history=history)
    idx = _rv_indices(history)
    source_past = idx[f"{source}_past"]
    target_pres = idx[f"{target}_pres"]
    tdmi = float(I(dist, [source_past, target_pres]))
    intrinsic = intrinsic_information_flow(
        generator,
        source=source,
        target=target,
        history=history,
    )
    return tdmi - intrinsic


def synergistic_information_flow(
    generator: HiddenMarkovModel,
    *,
    source: str = "x",
    target: str = "y",
    history: int = 1,
) -> float:
    """Synergistic flow: transfer entropy minus intrinsic flow."""
    te = transfer_entropy(generator, source=source, target=target, history=history)
    intrinsic = intrinsic_information_flow(
        generator,
        source=source,
        target=target,
        history=history,
    )
    return te - intrinsic


def independent_pair_generator(
    left: StochasticModel,
    right: StochasticModel,
) -> HiddenMarkovModel:
    """Build an independent pair generator emitting ``(x, y)`` tuple symbols."""
    from sofic.generators.mealy import MealyHMM
    from sofic.graph import TransitionGraph

    left_states = tuple(left.states())
    right_states = tuple(right.states())
    if not left_states or not right_states:
        raise ValueError("both generators must have at least one state")

    graph = TransitionGraph()
    for left_state in left_states:
        for right_state in right_states:
            graph.add_state((left_state, right_state))

    left_idx = {state: i for i, state in enumerate(left_states)}
    right_idx = {state: i for i, state in enumerate(right_states)}
    left_pi = left.stationary_distribution()
    right_pi = right.stationary_distribution()
    initial: dict[tuple[Hashable, Hashable], float] = {}
    for left_state in left_states:
        for right_state in right_states:
            mass = float(left_pi[left_idx[left_state]] * right_pi[right_idx[right_state]])
            if mass > 0.0:
                initial[(left_state, right_state)] = mass

    alphabet: set[tuple[Any, Any]] = set()
    for left_state in left_states:
        for transition in left.graph.out_transitions(left_state):
            x = transition.data.get(ATTR_EMISSION)
            prob_left = float(transition.data.get(ATTR_PROB, 0.0))
            if prob_left <= 0.0 or x is None:
                continue
            for right_state in right_states:
                for transition_r in right.graph.out_transitions(right_state):
                    y = transition_r.data.get(ATTR_EMISSION)
                    prob_right = float(transition_r.data.get(ATTR_PROB, 0.0))
                    if prob_right <= 0.0 or y is None:
                        continue
                    alphabet.add((x, y))
                    graph.add_transition(
                        (left_state, right_state),
                        (transition.target, transition_r.target),
                        **{ATTR_PROB: prob_left * prob_right, ATTR_EMISSION: (x, y)},
                    )

    return MealyHMM(
        graph=graph,
        initial_distribution=initial,
        observation_alphabet=frozenset(alphabet),
    )
