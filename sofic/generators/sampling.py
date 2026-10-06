"""Sampling observation and hidden-state sequences from hidden Markov models."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np

from sofic.generators.base import HiddenMarkovModel
from sofic.generators.matrices import emission_tensors


def sample(
    hmm: HiddenMarkovModel,
    n: int,
    rng: np.random.Generator | None = None,
) -> tuple[list[Any], list[Hashable]]:
    generator = rng if rng is not None else np.random.default_rng()
    mealy = hmm.to_mealy()
    idx = mealy.reindex()
    pi, joint = emission_tensors(mealy)
    total = float(np.sum(pi))
    if not total > 0.0:
        raise ValueError("cannot sample: the initial state distribution has no mass")
    state = int(generator.choice(len(idx), p=pi / total))

    observations: list[Any] = []
    states: list[Hashable] = []
    for _ in range(n):
        states.append(idx.state(state))
        row_sum = sum(matrix[state].sum() for matrix in joint.values())
        if row_sum <= 0.0:
            break
        symbol_probs = np.array([joint[sym][state].sum() for sym in joint], dtype=float)
        symbol_probs /= symbol_probs.sum()
        symbol_index = int(generator.choice(len(joint), p=symbol_probs))
        symbol = list(joint.keys())[symbol_index]
        observations.append(symbol)
        matrix = joint[symbol]
        row = matrix[state]
        if row.sum() <= 0.0:
            break
        state = int(generator.choice(len(idx), p=row / row.sum()))
    return observations, states
