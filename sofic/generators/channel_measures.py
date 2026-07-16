"""Information measures for channels presented as (epsilon-)transducers.

Following Barnett & Crutchfield (J. Stat. Phys. 161:2 (2015)), a transducer's
structural quantities are defined relative to a driving input process. Each
measure here drives the transducer with a supplied input generator, forms the
joint ``(input, output)`` process via
:func:`~sofic.automata.transducer_operations.compose_tg`, and reads off the
quantity -- reusing the directional-flow estimators in
:mod:`sofic.generators.directional_flow`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sofic.automata.transducers import MealyMachine
    from sofic.generators.base import HiddenMarkovModel
    from sofic.generators.mealy import MealyHMM


def driven_joint_generator(transducer: MealyMachine, input_process: HiddenMarkovModel) -> MealyHMM:
    """Return the joint ``(input, output)`` generator induced by ``input_process``."""
    from sofic.automata.transducer_operations import compose_tg

    return compose_tg(transducer, input_process, joint=True)


def channel_statistical_complexity(transducer: MealyMachine, input_process: HiddenMarkovModel) -> float:
    """Return the channel statistical complexity ``H[S]`` under ``input_process``.

    ``S`` is the transducer's causal-state component of the driven joint process,
    weighted by its stationary occupation distribution.
    """
    joint = driven_joint_generator(transducer, input_process)
    idx = joint.reindex()
    if len(idx) == 0:
        return 0.0
    pi = np.asarray(joint.stationary_distribution(), dtype=float)
    mass: dict[Any, float] = defaultdict(float)
    for state, weight in zip(idx.states, pi, strict=True):
        transducer_state = state[1] if isinstance(state, tuple) and len(state) == 2 else state
        mass[transducer_state] += float(weight)
    probs = np.array([value for value in mass.values() if value > 0.0], dtype=float)
    if probs.size == 0:
        return 0.0
    probs = probs / probs.sum()
    return float(-(probs * np.log2(probs)).sum())


def driven_entropy_rate(transducer: MealyMachine, input_process: HiddenMarkovModel) -> float:
    """Return the entropy rate of the output process induced by ``input_process``."""
    from sofic.automata.transducer_operations import transduce_generator

    output_generator = transduce_generator(transducer, input_process)
    return output_generator.entropy_rate()


def directed_information(transducer: MealyMachine, input_process: HiddenMarkovModel, *, length: int = 1) -> float:
    """Return the directed information ``I(X^n -> Y^n)`` when driven by ``input_process``."""
    from sofic.generators.directional_flow import directed_information as _directed_information

    joint = driven_joint_generator(transducer, input_process)
    return _directed_information(joint, source="x", target="y", length=length)


def transfer_entropy(transducer: MealyMachine, input_process: HiddenMarkovModel, *, history: int = 1) -> float:
    """Return the input-to-output transfer entropy when driven by ``input_process``."""
    from sofic.generators.directional_flow import transfer_entropy as _transfer_entropy

    joint = driven_joint_generator(transducer, input_process)
    return _transfer_entropy(joint, source="x", target="y", history=history)
