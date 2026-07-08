"""Lumping (state aggregation) of Markov chains and hidden Markov models.

Strong lumpability in the sense of Kemeny & Snell (:cite:`KemenySnell1976`,
Ch. 6): a partition of the state space is *strongly lumpable* when, for every
block, the total probability of moving into each target block is identical for
every state in the source block. That common value defines the transition law of
the coarser lumped chain, and -- because the condition constrains only the
transition matrix -- it holds for every initial distribution.

Hidden Markov models extend the condition to each emitted symbol so that the
lumped model generates the same observed process:

* :class:`~pensive.generators.mealy.MealyHMM` -- the joint block-and-symbol mass
  ``sum_{t in B_j} P(t, o | s)`` must be constant across ``s`` in a block.
* :class:`~pensive.generators.moore.MooreHMM` -- additionally the state emission
  law ``P(o | s)`` must be identical across a block.

The public entry points are :func:`is_lumpable` (predicate) and :func:`lump`
(constructor). ``lump`` raises :class:`~pensive.exceptions.LumpabilityError` when
the partition is not strongly lumpable unless ``check=False``.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from pensive.exceptions import LumpabilityError
from pensive.graph import ATTR_EMISSION, ATTR_EMISSION_DIST, ATTR_PROB, TransitionGraph

if TYPE_CHECKING:
    from pensive.base import StateMachine
    from pensive.generators.markov import MarkovChain
    from pensive.generators.mealy import MealyHMM
    from pensive.generators.moore import MooreHMM

PartitionLike = Iterable[Iterable[Hashable]] | Mapping[Hashable, Hashable]
LabelsLike = Mapping[frozenset[Hashable], Hashable] | Callable[[frozenset[Hashable]], Hashable]


def normalize_partition(model: StateMachine, partition: PartitionLike) -> list[frozenset[Hashable]]:
    """Return ``partition`` as an ordered list of disjoint, covering blocks.

    Parameters
    ----------
    model
        The model whose states the partition must cover exactly.
    partition
        Either an iterable of blocks (each an iterable of states) or a mapping
        from each state to a block key.

    Returns
    -------
    list of frozenset
        Blocks ordered by the position of their first member in ``model.states()``.

    Raises
    ------
    ValueError
        If the blocks overlap, reference unknown states, or fail to cover every
        state of ``model``.
    """
    order = {state: index for index, state in enumerate(model.states())}
    state_set = set(order)

    if isinstance(partition, Mapping):
        grouped: dict[Hashable, set[Hashable]] = {}
        for state, key in partition.items():
            grouped.setdefault(key, set()).add(state)
        raw_blocks: Iterable[Iterable[Hashable]] = grouped.values()
    else:
        raw_blocks = partition

    seen: set[Hashable] = set()
    blocks: list[frozenset[Hashable]] = []
    for raw in raw_blocks:
        block = frozenset(raw)
        if not block:
            continue
        unknown = block - state_set
        if unknown:
            raise ValueError(f"partition references unknown states {sorted(map(str, unknown))}")
        overlap = block & seen
        if overlap:
            raise ValueError(f"partition blocks overlap on states {sorted(map(str, overlap))}")
        seen |= block
        blocks.append(block)

    missing = state_set - seen
    if missing:
        raise ValueError(f"partition does not cover states {sorted(map(str, missing))}")

    blocks.sort(key=lambda block: min(order[state] for state in block))
    return blocks


def _block_of(blocks: list[frozenset[Hashable]]) -> dict[Hashable, int]:
    return {state: index for index, block in enumerate(blocks) for state in block}


def _representatives(model: StateMachine, blocks: list[frozenset[Hashable]]) -> list[Hashable]:
    order = {state: index for index, state in enumerate(model.states())}
    return [min(block, key=lambda state: order[state]) for block in blocks]


def _default_label(block: frozenset[Hashable]) -> Hashable:
    if len(block) == 1:
        return next(iter(block))
    return "+".join(sorted(str(state) for state in block))


def _resolve_labels(blocks: list[frozenset[Hashable]], labels: LabelsLike | None) -> list[Hashable]:
    if labels is None:
        resolved = [_default_label(block) for block in blocks]
    elif isinstance(labels, Mapping):
        resolved = [labels.get(block, _default_label(block)) for block in blocks]
    else:
        resolved = [labels(block) for block in blocks]
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"block labels are not distinct: {resolved}")
    return resolved


def _markov_signature(
    model: StateMachine, state: Hashable, block_of: dict[Hashable, int], num_blocks: int
) -> np.ndarray:
    signature = np.zeros(num_blocks, dtype=float)
    for transition in model.graph.out_transitions(state):
        target_block = block_of.get(transition.target)
        if target_block is None:
            continue
        signature[target_block] += float(transition.data.get(ATTR_PROB, 0.0))
    return signature


def _mealy_signature(
    model: StateMachine, state: Hashable, block_of: dict[Hashable, int]
) -> dict[tuple[int, Any], float]:
    signature: dict[tuple[int, Any], float] = {}
    for transition in model.graph.out_transitions(state):
        target_block = block_of.get(transition.target)
        if target_block is None:
            continue
        key = (target_block, transition.data.get(ATTR_EMISSION))
        signature[key] = signature.get(key, 0.0) + float(transition.data.get(ATTR_PROB, 0.0))
    return signature


def _emission_dist(model: StateMachine, state: Hashable) -> dict[Any, float]:
    return dict(model.graph.state_attrs(state).get(ATTR_EMISSION_DIST) or {})


def _dicts_close(left: Mapping[Any, float], right: Mapping[Any, float], *, rtol: float, atol: float) -> bool:
    for key in set(left) | set(right):
        if not np.isclose(left.get(key, 0.0), right.get(key, 0.0), rtol=rtol, atol=atol):
            return False
    return True


def _is_lumpable_markov(model: MarkovChain, blocks: list[frozenset[Hashable]], *, rtol: float, atol: float) -> bool:
    block_of = _block_of(blocks)
    num_blocks = len(blocks)
    for block in blocks:
        members = iter(block)
        reference = _markov_signature(model, next(members), block_of, num_blocks)
        for state in members:
            if not np.allclose(_markov_signature(model, state, block_of, num_blocks), reference, rtol=rtol, atol=atol):
                return False
    return True


def _is_lumpable_mealy(model: MealyHMM, blocks: list[frozenset[Hashable]], *, rtol: float, atol: float) -> bool:
    block_of = _block_of(blocks)
    for block in blocks:
        members = iter(block)
        reference = _mealy_signature(model, next(members), block_of)
        for state in members:
            if not _dicts_close(_mealy_signature(model, state, block_of), reference, rtol=rtol, atol=atol):
                return False
    return True


def _is_lumpable_moore(model: MooreHMM, blocks: list[frozenset[Hashable]], *, rtol: float, atol: float) -> bool:
    block_of = _block_of(blocks)
    num_blocks = len(blocks)
    for block in blocks:
        members = iter(block)
        first = next(members)
        reference_emit = _emission_dist(model, first)
        reference_trans = _markov_signature(model, first, block_of, num_blocks)
        for state in members:
            if not _dicts_close(_emission_dist(model, state), reference_emit, rtol=rtol, atol=atol):
                return False
            if not np.allclose(
                _markov_signature(model, state, block_of, num_blocks), reference_trans, rtol=rtol, atol=atol
            ):
                return False
    return True


def _lumped_initial(
    model: StateMachine, block_of: dict[Hashable, int], labels: list[Hashable]
) -> dict[Hashable, float]:
    initial: dict[Hashable, float] = {}
    for state, mass in getattr(model, "initial_distribution", {}).items():
        label = labels[block_of[state]]
        initial[label] = initial.get(label, 0.0) + float(mass)
    return initial


def _lump_markov(model: MarkovChain, blocks: list[frozenset[Hashable]], labels: LabelsLike | None) -> MarkovChain:
    from pensive.generators.markov import MarkovChain

    resolved = _resolve_labels(blocks, labels)
    block_of = _block_of(blocks)
    representatives = _representatives(model, blocks)

    graph = TransitionGraph()
    for label in resolved:
        graph.add_state(label)
    result = MarkovChain(graph=graph)

    for index, representative in enumerate(representatives):
        merged: dict[int, float] = {}
        for transition in model.graph.out_transitions(representative):
            target_block = block_of.get(transition.target)
            if target_block is None:
                continue
            merged[target_block] = merged.get(target_block, 0.0) + float(transition.data.get(ATTR_PROB, 0.0))
        for target_block, prob in merged.items():
            result.add_transition(resolved[index], resolved[target_block], prob)

    result.initial_distribution = _lumped_initial(model, block_of, resolved)
    result.validate()
    return result


def _lump_mealy(model: MealyHMM, blocks: list[frozenset[Hashable]], labels: LabelsLike | None) -> MealyHMM:
    from pensive.generators.mealy import MealyHMM

    resolved = _resolve_labels(blocks, labels)
    block_of = _block_of(blocks)
    representatives = _representatives(model, blocks)

    graph = TransitionGraph()
    for label in resolved:
        graph.add_state(label)
    result = MealyHMM(graph=graph, observation_alphabet=model.observation_alphabet)

    for index, representative in enumerate(representatives):
        merged: dict[tuple[int, Any], float] = {}
        for transition in model.graph.out_transitions(representative):
            target_block = block_of.get(transition.target)
            if target_block is None:
                continue
            key = (target_block, transition.data.get(ATTR_EMISSION))
            merged[key] = merged.get(key, 0.0) + float(transition.data.get(ATTR_PROB, 0.0))
        for (target_block, emission), prob in merged.items():
            result.add_transition(resolved[index], resolved[target_block], emission, prob)

    result.initial_distribution = _lumped_initial(model, block_of, resolved)
    result.validate()
    return result


def _lump_moore(model: MooreHMM, blocks: list[frozenset[Hashable]], labels: LabelsLike | None) -> MooreHMM:
    from pensive.generators.moore import MooreHMM

    resolved = _resolve_labels(blocks, labels)
    block_of = _block_of(blocks)
    representatives = _representatives(model, blocks)

    graph = TransitionGraph()
    for label in resolved:
        graph.add_state(label)
    result = MooreHMM(graph=graph, observation_alphabet=model.observation_alphabet)

    for index, representative in enumerate(representatives):
        emission = _emission_dist(model, representative)
        if emission:
            result.set_emission_distribution(resolved[index], emission)
        merged: dict[int, float] = {}
        for transition in model.graph.out_transitions(representative):
            target_block = block_of.get(transition.target)
            if target_block is None:
                continue
            merged[target_block] = merged.get(target_block, 0.0) + float(transition.data.get(ATTR_PROB, 0.0))
        for target_block, prob in merged.items():
            result.add_transition(resolved[index], resolved[target_block], prob)

    result.initial_distribution = _lumped_initial(model, block_of, resolved)
    result.validate()
    return result


def _dispatch(model: StateMachine) -> tuple[Callable[..., bool], Callable[..., Any]]:
    from pensive.generators.markov import MarkovChain
    from pensive.generators.mealy import MealyHMM
    from pensive.generators.moore import MooreHMM

    if isinstance(model, MarkovChain):
        return _is_lumpable_markov, _lump_markov
    if isinstance(model, MooreHMM):
        return _is_lumpable_moore, _lump_moore
    if isinstance(model, MealyHMM):
        return _is_lumpable_mealy, _lump_mealy
    raise TypeError(f"lumping is not supported for {type(model).__name__}")


def is_lumpable(model: StateMachine, partition: PartitionLike, *, rtol: float = 1e-8, atol: float = 1e-10) -> bool:
    """Return whether ``partition`` is strongly lumpable for ``model``.

    Strong lumpability follows Kemeny & Snell (:cite:`KemenySnell1976`): the
    aggregated mass into each block must not depend on which state of a block the
    chain occupies. For hidden Markov models the condition is imposed per emitted
    symbol (and, for Moore presentations, additionally on the state emission law).

    Parameters
    ----------
    model
        A :class:`~pensive.generators.markov.MarkovChain`,
        :class:`~pensive.generators.mealy.MealyHMM` (including
        :class:`~pensive.generators.epsilon_machine.EpsilonMachine`), or
        :class:`~pensive.generators.moore.MooreHMM`.
    partition
        Blocks (iterable of iterables) or a state-to-block mapping; must cover
        every state exactly.
    rtol, atol
        Tolerances forwarded to :func:`numpy.isclose`.

    Raises
    ------
    TypeError
        If ``model`` is not a supported generator type.
    ValueError
        If ``partition`` is not a valid partition of ``model``'s states.
    """
    checker, _builder = _dispatch(model)
    blocks = normalize_partition(model, partition)
    return checker(model, blocks, rtol=rtol, atol=atol)


@overload
def lump(
    model: MarkovChain,
    partition: PartitionLike,
    *,
    check: bool = ...,
    labels: LabelsLike | None = ...,
    rtol: float = ...,
    atol: float = ...,
) -> MarkovChain: ...


@overload
def lump(
    model: MooreHMM,
    partition: PartitionLike,
    *,
    check: bool = ...,
    labels: LabelsLike | None = ...,
    rtol: float = ...,
    atol: float = ...,
) -> MooreHMM: ...


@overload
def lump(
    model: MealyHMM,
    partition: PartitionLike,
    *,
    check: bool = ...,
    labels: LabelsLike | None = ...,
    rtol: float = ...,
    atol: float = ...,
) -> MealyHMM: ...


@overload
def lump(
    model: StateMachine,
    partition: PartitionLike,
    *,
    check: bool = ...,
    labels: LabelsLike | None = ...,
    rtol: float = ...,
    atol: float = ...,
) -> StateMachine: ...


def lump(
    model: StateMachine,
    partition: PartitionLike,
    *,
    check: bool = True,
    labels: LabelsLike | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> StateMachine:
    """Aggregate the states of ``model`` according to ``partition``.

    Builds the coarser lumped model (Kemeny & Snell, :cite:`KemenySnell1976`):
    block-to-block transition masses are read from a block representative, and
    initial masses are summed within blocks. A
    :class:`~pensive.generators.markov.MarkovChain` lumps to a ``MarkovChain``, a
    :class:`~pensive.generators.moore.MooreHMM` to a ``MooreHMM``, and any
    :class:`~pensive.generators.mealy.MealyHMM` (including an
    :class:`~pensive.generators.epsilon_machine.EpsilonMachine`) to a plain
    ``MealyHMM`` -- lumping may break unifilarity, so the stricter subtype is not
    preserved.

    Parameters
    ----------
    model
        The generator to lump.
    partition
        Blocks (iterable of iterables) or a state-to-block mapping; must cover
        every state exactly.
    check
        When ``True`` (default), raise :class:`~pensive.exceptions.LumpabilityError`
        if ``partition`` is not strongly lumpable. When ``False``, build the model
        anyway from each block's representative row (the result is exact only when
        the partition is in fact lumpable).
    labels
        Optional mapping from a block (as a ``frozenset``) to its lumped-state
        label, or a callable taking a block and returning a label. By default a
        singleton block keeps its original state label and a merged block becomes
        the ``"+"``-joined string of its members' labels.
    rtol, atol
        Tolerances forwarded to :func:`numpy.isclose` for the lumpability check.

    Raises
    ------
    LumpabilityError
        If ``check`` and ``partition`` is not strongly lumpable.
    TypeError
        If ``model`` is not a supported generator type.
    ValueError
        If ``partition`` is invalid or the resolved block labels collide.
    """
    checker, builder = _dispatch(model)
    blocks = normalize_partition(model, partition)
    if check and not checker(model, blocks, rtol=rtol, atol=atol):
        raise LumpabilityError("partition is not strongly lumpable for this model")
    return builder(model, blocks, labels)
