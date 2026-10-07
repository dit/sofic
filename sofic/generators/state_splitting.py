"""Process-preserving state splitting and amalgamation of hidden Markov models.

Lind & Marcus state splitting (:cite:`LindMarcus1995`, §2.4; implemented for
edge shifts in :mod:`sofic.shifts.state_splitting`) is lifted to Mealy HMMs so
that the generated *process* is unchanged, following the equivalence moves of
:cite:`Upper1997`. Edges are named ``(source, target, key)`` as in
:class:`~sofic.graph.Transition`; :func:`out_edges` and :func:`in_edges` list
them. Splitting state ``s`` into ``m`` copies names them ``(s, 0), ...,
(s, m - 1)``.

Write ``p(e)`` for the joint probability ``P(target, symbol | source)`` of edge
``e`` and ``pi`` for the initial distribution.

*Out-splitting* partitions the out-edges of ``s`` into ``E_0, ..., E_{m-1}``
with masses ``w_i = sum_{e in E_i} p(e)``. Copy ``(s, i)`` keeps the edges of
``E_i`` renormalized to ``p(e) / w_i``; every in-edge ``e`` of ``s`` becomes
``m`` edges into the copies with probabilities ``p(e) w_i``; and
``pi(s, i) = pi(s) w_i``. By induction on ``t`` the forward vector of the split
model is ``alpha_t(s, i) = w_i alpha_t(s)``, so every word has the same
probability. If ``pi`` is stationary, so is the split ``pi``.

*In-splitting* partitions the in-edges of ``s`` into ``F_0, ..., F_{m-1}``.
Edge ``e in F_i`` enters copy ``(s, i)`` only, and every copy keeps all
out-edges of ``s`` with their probabilities. Copies have identical futures, so
the backward vectors agree, ``beta_t(s, i) = beta_t(s)``, and the initial mass
``pi(s)`` may be shared out arbitrarily; it is split in proportion to the
stationary inflow ``sum_{e in F_i} mu(source(e)) p(e)`` (uniformly if ``s``
has no stationary inflow), which keeps a stationary ``pi`` stationary.

:func:`amalgamate` inverts both: in-split copies have identical futures and
merge by Kemeny-Snell strong lumping (:func:`~sofic.generators.lumping.lump`,
:cite:`KemenySnell1976`), while out-split copies receive proportional inflow
and merge into the inflow-weighted mixture of their out-edges.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from sofic.exceptions import LumpabilityError
from sofic.generators.base import HiddenMarkovModel
from sofic.generators.lumping import (
    LabelsLike,
    PartitionLike,
    _block_of,
    _resolve_labels,
    is_lumpable,
    lump,
    normalize_partition,
)
from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_EMISSION, ATTR_PROB, Transition, TransitionGraph

Edge = tuple[Hashable, Hashable, int]


def out_edges(hmm: HiddenMarkovModel, state: Hashable) -> list[Edge]:
    """Return the ``(source, target, key)`` edges leaving ``state`` :cite:`LindMarcus1995`."""
    return [(t.source, t.target, t.key) for t in hmm.to_mealy().transitions() if t.source == state]


def in_edges(hmm: HiddenMarkovModel, state: Hashable) -> list[Edge]:
    """Return the ``(source, target, key)`` edges entering ``state`` :cite:`LindMarcus1995`."""
    return [(t.source, t.target, t.key) for t in hmm.to_mealy().transitions() if t.target == state]


def split_state(
    hmm: HiddenMarkovModel,
    state: Hashable,
    partition: Sequence[Collection[Edge]],
    *,
    kind: Literal["out", "in"] = "out",
) -> MealyHMM:
    """Split ``state`` into copies without changing the generated process.

    The process-preserving lift of the out-/in-splittings of
    :cite:`LindMarcus1995` (Definitions 2.4.3 and 2.4.7) described in the
    module docstring; such splittings are among the HMM equivalence moves of
    :cite:`Upper1997`. Probabilities are combined with exact arithmetic, so
    sympy-valued models stay exact.

    Parameters
    ----------
    hmm
        The generator; non-Mealy presentations are converted with ``to_mealy``.
    state
        The state to split into ``(state, 0), ..., (state, m - 1)``.
    partition
        ``m`` nonempty, disjoint parts covering :func:`out_edges` (``kind="out"``)
        or :func:`in_edges` (``kind="in"``) of ``state``.
    kind
        ``"out"`` partitions out-edges (copies receive proportionally split
        inflow); ``"in"`` partitions in-edges (copies share every out-edge).

    Returns
    -------
    MealyHMM
        A presentation of the same process. In-splitting preserves
        unifilarity; out-splitting generally does not, since one in-edge
        becomes ``m`` parallel edges with the same symbol.

    Raises
    ------
    ValueError
        If ``state`` is unknown, the parts are not a partition of the relevant
        edges, ``kind`` is invalid, or a copy name collides with a state.
    """
    if kind not in ("out", "in"):
        raise ValueError(f"kind must be 'out' or 'in', not {kind!r}")
    gen = hmm.to_mealy()
    if not gen.graph.has_state(state):
        raise ValueError(f"unknown state {state!r}")
    edges = out_edges(gen, state) if kind == "out" else in_edges(gen, state)
    part_of = _part_index(state, edges, partition)
    copies = [(state, index) for index in range(len(partition))]
    if any(gen.graph.has_state(copy) for copy in copies):
        raise ValueError("split state names collide with existing states")

    if kind == "out":
        shares: list[Any] = [0] * len(copies)
        for t in gen.transitions():
            if t.source == state:
                shares[part_of[(t.source, t.target, t.key)]] += t.data[ATTR_PROB]
    else:
        shares = _stationary_inflow(gen, state, part_of, len(copies))

    def source_copies(t: Transition) -> list[tuple[Hashable, Any]]:
        if t.source != state:
            return [(t.source, 1)]
        if kind == "out":
            index = part_of[(t.source, t.target, t.key)]
            return [(copies[index], 1 / shares[index])]
        return [(copy, 1) for copy in copies]

    def target_copies(t: Transition) -> list[tuple[Hashable, Any]]:
        if t.target != state:
            return [(t.target, 1)]
        if kind == "out":
            return list(zip(copies, shares, strict=True))
        return [(copies[part_of[(t.source, t.target, t.key)]], 1)]

    result = MealyHMM(observation_alphabet=gen.observation_alphabet)
    for name in gen.states():
        for copy in copies if name == state else [name]:
            result.graph.add_state(copy)
    for t in gen.transitions():
        targets = target_copies(t)
        for source, source_factor in source_copies(t):
            for target, target_factor in targets:
                prob = t.data[ATTR_PROB] * source_factor * target_factor
                result.add_transition(source, target, t.data.get(ATTR_EMISSION), prob)
    for name, mass in gen.initial_distribution.items():
        if name != state:
            result.initial_distribution[name] = mass
            continue
        for copy, share in zip(copies, shares, strict=True):
            result.initial_distribution[copy] = mass * share
    return result


def amalgamate(
    hmm: HiddenMarkovModel,
    partition: PartitionLike,
    *,
    labels: LabelsLike | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> MealyHMM:
    """Merge the blocks of ``partition`` without changing the generated process.

    Inverts :func:`split_state` (the amalgamations of :cite:`LindMarcus1995`,
    Definition 2.4.9, lifted to processes as in :cite:`Upper1997`). Two exact
    merges are recognized:

    * *same futures* -- the partition is strongly lumpable
      (:func:`~sofic.generators.lumping.is_lumpable`), as for in-split copies,
      and is delegated to :func:`~sofic.generators.lumping.lump`
      :cite:`KemenySnell1976`;
    * *proportional inflow* -- for each block ``B`` there is a weight vector
      ``w`` on ``B`` such that, for every source state and symbol, the edge
      mass into ``B`` and the initial mass on ``B`` are multiples of ``w``, as
      for out-split copies. The forward vector on ``B`` is then always a
      multiple of ``w``, so ``B`` merges exactly into one state whose
      out-edges are the ``w``-mixture of its members' out-edges. A block with
      neither inflow nor initial mass is never visited and is merged with
      uniform ``w``; the process is unchanged, but the out-split weights of
      such a block are unrecoverable, so the original rows need not return.

    The whole partition is tried with each rule; failing that, blocks are
    merged one at a time with whichever rule applies.

    Parameters
    ----------
    hmm
        The generator; non-Mealy presentations are converted with ``to_mealy``.
    partition
        Blocks (iterable of iterables) or a state-to-block mapping covering
        every state, as in :func:`~sofic.generators.lumping.normalize_partition`.
    labels
        Merged-state labels, as in :func:`~sofic.generators.lumping.lump`.
    rtol, atol
        Tolerances forwarded to :func:`numpy.isclose`.

    Returns
    -------
    MealyHMM
        A presentation of the same process with one state per block.

    Raises
    ------
    LumpabilityError
        If some block can be merged by neither rule.
    """
    gen = hmm.to_mealy()
    blocks = normalize_partition(gen, partition)
    resolved = _resolve_labels(blocks, labels)
    if is_lumpable(gen, blocks, rtol=rtol, atol=atol):
        return lump(gen, blocks, labels=labels)
    weights = _inflow_weights(gen, blocks, rtol=rtol, atol=atol)
    if weights is not None:
        return _merge_by_inflow(gen, blocks, resolved, weights)

    result = gen
    for block, label in zip(blocks, resolved, strict=True):
        if len(block) == 1 and next(iter(block)) == label:
            continue
        step = [block, *({state} for state in result.states() if state not in block)]
        step_labels = {block: label}
        if is_lumpable(result, step, rtol=rtol, atol=atol):
            result = lump(result, step, labels=step_labels)
            continue
        step_blocks = normalize_partition(result, step)
        step_weights = _inflow_weights(result, step_blocks, rtol=rtol, atol=atol)
        if step_weights is None:
            raise LumpabilityError(
                f"block {sorted(map(str, block))} has neither identical futures nor proportional inflow"
            )
        result = _merge_by_inflow(result, step_blocks, _resolve_labels(step_blocks, step_labels), step_weights)
    return result


def _part_index(state: Hashable, edges: list[Edge], partition: Sequence[Collection[Edge]]) -> dict[Edge, int]:
    expected = set(edges)
    part_of: dict[Edge, int] = {}
    for index, part in enumerate(partition):
        part = set(part)
        if not part:
            raise ValueError(f"partition of {state!r} has an empty part")
        if not part <= expected or part & set(part_of):
            raise ValueError(f"parts for {state!r} must be disjoint edges of that state")
        part_of.update(dict.fromkeys(part, index))
    if not partition or set(part_of) != expected:
        raise ValueError(f"parts for {state!r} do not cover all of its edges")
    return part_of


def _stationary_inflow(gen: MealyHMM, state: Hashable, part_of: Mapping[Edge, int], m: int) -> list[Any]:
    mu = dict(zip(gen.states(), gen.stationary_distribution().tolist(), strict=True))
    inflow: list[Any] = [0] * m
    for t in gen.transitions():
        if t.target == state:
            inflow[part_of[(t.source, t.target, t.key)]] += mu[t.source] * t.data[ATTR_PROB]
    total = sum(inflow)
    if not np.isclose(float(total), 0.0):
        return [mass / total for mass in inflow]
    return [1 / m] * m


def _inflow_weights(
    gen: MealyHMM, blocks: list[frozenset[Hashable]], *, rtol: float, atol: float
) -> list[dict[Hashable, float]] | None:
    """Return per-block inflow weights ``w`` when every block has proportional inflow."""
    block_of = _block_of(blocks)
    columns: dict[tuple[int, Hashable, Any], dict[Hashable, float]] = defaultdict(lambda: defaultdict(float))
    for t in gen.transitions():
        key = (block_of[t.target], t.source, t.data.get(ATTR_EMISSION))
        columns[key][t.target] += float(t.data[ATTR_PROB])
    vectors: list[list[dict[Hashable, float]]] = [[] for _ in blocks]
    for (index, _source, _symbol), column in columns.items():
        vectors[index].append(column)
    initial = defaultdict(dict)
    for state, mass in gen.initial_distribution.items():
        initial[block_of[state]][state] = float(mass)
    for index, column in initial.items():
        vectors[index].append(column)

    weights: list[dict[Hashable, float]] = []
    for index, block in enumerate(blocks):
        reference: np.ndarray | None = None
        order = sorted(block, key=repr)
        for column in vectors[index]:
            vector = np.array([column.get(state, 0.0) for state in order])
            total = vector.sum()
            if total <= atol:
                continue
            vector = vector / total
            if reference is None:
                reference = vector
            elif not np.allclose(vector, reference, rtol=rtol, atol=atol):
                return None
        if reference is None:
            reference = np.full(len(order), 1.0 / len(order))
        weights.append(dict(zip(order, reference.tolist(), strict=True)))
    return weights


def _merge_by_inflow(
    gen: MealyHMM,
    blocks: list[frozenset[Hashable]],
    labels: list[Hashable],
    weights: list[dict[Hashable, float]],
) -> MealyHMM:
    block_of = _block_of(blocks)
    graph = TransitionGraph()
    for label in labels:
        graph.add_state(label)
    result = MealyHMM(graph=graph, observation_alphabet=gen.observation_alphabet)
    merged: dict[tuple[int, int, Any], float] = defaultdict(float)
    for t in gen.transitions():
        source = block_of[t.source]
        key = (source, block_of[t.target], t.data.get(ATTR_EMISSION))
        merged[key] += weights[source][t.source] * float(t.data[ATTR_PROB])
    for (source, target, symbol), prob in merged.items():
        if prob > 0.0:
            result.add_transition(labels[source], labels[target], symbol, prob)
    for state, mass in gen.initial_distribution.items():
        label = labels[block_of[state]]
        result.initial_distribution[label] = result.initial_distribution.get(label, 0.0) + float(mass)
    result.validate()
    return result


__all__ = ["amalgamate", "in_edges", "out_edges", "split_state"]
