"""Shared configuration-set simulation driver for stack/output machines.

VPA, NWA, and transducer simulations all advance a set of ``(state, extra)``
configurations one input step at a time, unioning the successors produced by a
per-machine step function and stopping if the set ever empties. This captures
that loop; the per-kind transition logic stays with each caller.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable


def simulate_configs[Config, Step](
    initial: set[Config],
    steps: Iterable[Step],
    step_fn: Callable[[Config, Step], Iterable[Config]],
    *,
    closure: Callable[[set[Config]], set[Config]] | None = None,
) -> set[Config]:
    """Advance ``initial`` through ``steps`` via ``step_fn``.

    ``step_fn(config, step)`` yields successor configurations for one input step.
    An optional ``closure`` is applied to the seed set and after every step (used
    by the transducer's epsilon-input closure). Returns the final configuration
    set, which is empty if the machine had no successors at some step.
    """
    current = closure(initial) if closure is not None else initial
    for step in steps:
        nxt: set[Config] = set()
        for config in current:
            nxt.update(step_fn(config, step))
        current = closure(nxt) if closure is not None else nxt
        if not current:
            return set()
    return current
