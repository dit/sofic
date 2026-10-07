"""Shift presentations: factor languages, trimming, entropy."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Hashable, Iterator
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np

from sofic.graph import ATTR_MULTIPLICITY, ATTR_SYMBOL
from sofic.shifts.base import SymbolicModel


def trim_transient(model: SymbolicModel) -> SymbolicModel:
    """Remove states not on bi-infinite paths.

    Repeatedly deletes states with no incoming or no outgoing transition; what
    survives is exactly the set of states that lie on some bi-infinite path.
    """
    result = model.copy()
    graph = result.graph.nx
    pending = deque(state for state in graph.nodes if graph.in_degree(state) == 0 or graph.out_degree(state) == 0)
    while pending:
        state = pending.popleft()
        if state not in graph:
            continue
        neighbours = set(graph.predecessors(state)) | set(graph.successors(state))
        graph.remove_node(state)
        for neighbour in neighbours - {state}:
            if neighbour in graph and (graph.in_degree(neighbour) == 0 or graph.out_degree(neighbour) == 0):
                pending.append(neighbour)
    return result


def factor_language(model: SymbolicModel, length: int) -> Iterator[tuple[Any, ...]]:
    """Yield distinct factor words of the given length.

    Walks only the trimmed presentation (:func:`trim_transient`), so every word
    yielded extends to a bi-infinite point of the shift.
    """
    if length <= 0:
        yield ()
        return
    trimmed = trim_transient(model)
    seen: set[tuple[Any, ...]] = set()
    for start in trimmed.states():
        queue: deque[tuple[Hashable, tuple[Any, ...]]] = deque([(start, ())])
        while queue:
            state, prefix = queue.popleft()
            if len(prefix) == length:
                if prefix not in seen:
                    seen.add(prefix)
                    yield prefix
                continue
            for transition in trimmed.graph.out_transitions(state):
                symbol = transition.data.get(ATTR_SYMBOL)
                if symbol is None:
                    continue
                queue.append((transition.target, prefix + (symbol,)))


def adjacency_matrix(model: SymbolicModel, *, multiplicity: bool = True) -> tuple[np.ndarray, tuple[Hashable, ...]]:
    """Return the adjacency matrix and its state order.

    Each transition counts its ``multiplicity`` attribute when ``multiplicity``
    is true (edge-shift semantics), and once otherwise.
    """
    states = tuple(model.states())
    index = {state: i for i, state in enumerate(states)}
    n = len(states)
    matrix = np.zeros((n, n), dtype=float)
    for transition in model.transitions():
        i = index[transition.source]
        j = index[transition.target]
        matrix[i, j] += float(transition.data.get(ATTR_MULTIPLICITY, 1)) if multiplicity else 1.0
    return matrix, states


def topological_entropy_from_matrix(matrix: np.ndarray) -> float:
    """Return ``log2`` of the spectral radius of ``matrix`` (bits per symbol)."""
    if matrix.size == 0:
        return 0.0
    eigenvalues = np.linalg.eigvals(matrix)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    if spectral_radius <= 0.0:
        return 0.0
    return float(np.log2(spectral_radius))


def sofic_topological_entropy(model: SymbolicModel) -> float:
    """Topological entropy (bits) of the shift a labeled presentation generates.

    The ``log2`` spectral radius of a right-resolving presentation
    (:func:`~sofic.shifts.wheeler.right_resolve`), which counts words rather than
    paths :cite:`LindMarcus1995` (Theorem 4.3.3). Edge multiplicities are ignored: two
    parallel edges with one label present the same words as a single edge.
    """
    from sofic.shifts.wheeler import right_resolve

    matrix, _ = adjacency_matrix(right_resolve(model), multiplicity=False)
    return topological_entropy_from_matrix(matrix)


def integer_adjacency_matrix(model: SymbolicModel) -> tuple[np.ndarray, tuple[Hashable, ...]]:
    """Return the exact adjacency matrix (Python ``int`` entries) and its state order.

    Each transition counts its ``multiplicity`` attribute, as for an edge shift.
    The matrix has ``object`` dtype so powers never overflow.
    """
    states = tuple(model.states())
    index = {state: i for i, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)), dtype=object)
    for transition in model.transitions():
        matrix[index[transition.source], index[transition.target]] += edge_multiplicity(transition.data)
    return matrix, states


def edge_multiplicity(data: dict[str, Any]) -> int:
    """Return the integer ``multiplicity`` attribute of a transition (default 1)."""
    multiplicity = data.get(ATTR_MULTIPLICITY, 1)
    if int(multiplicity) != multiplicity:
        raise ValueError(f"edge multiplicity must be an integer, got {multiplicity!r}")
    return int(multiplicity)


def periodic_points(shift: SymbolicModel, n: int) -> int:
    """Return ``p_n``, the number of points ``x`` with ``sigma^n x = x``.

    * :class:`~sofic.shifts.tmc.TopologicalMarkovChain`: ``tr A^n`` for the
      adjacency matrix with multiplicities (edge-shift semantics)
      :cite:`LindMarcus1995` (Proposition 2.2.12).
    * :class:`~sofic.shifts.sft.ShiftOfFiniteType` given by forbidden words of
      length at most ``L``: ``tr A^n`` for the edge graph whose vertices are
      the allowed ``(L - 1)``-blocks and whose edges are the allowed
      ``L``-blocks, an edge shift conjugate to the SFT
      :cite:`LindMarcus1995` (Theorem 2.3.2).
    * Any other labeled presentation (sofic shifts, presentation-only SFTs):
      Manning's formula ``p_n = sum_j (-1)^(j+1) tr A_j^n`` over the signed
      subset matrices ``A_j`` of a right-resolving presentation
      :cite:`LindMarcus1995` (Theorem 6.4.8). Counting cycles of the
      presentation itself would overcount points with several presentations.

    Arithmetic is exact (Python integers).

    Parameters
    ----------
    shift : SymbolicModel
        A TMC, SFT, or sofic-shift presentation.
    n : int
        Period, ``n >= 1``.

    Returns
    -------
    int
        The number of points of period ``n`` (not necessarily least period).
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    return sum(sign * _trace_power(matrix, n) for matrix, sign in _periodic_point_matrices(shift))


def zeta_function(shift: SymbolicModel, t: Any = None) -> Any:
    """Return the Artin-Mazur zeta function ``exp(sum_n p_n t^n / n)`` as a sympy rational function.

    For an edge shift ``X_A`` this is ``1 / det(I - tA)``
    :cite:`LindMarcus1995` (Theorem 6.4.6); for a sofic shift with signed
    subset matrices ``A_j`` it is ``prod_j det(I - tA_j)^((-1)^j)``
    :cite:`LindMarcus1995` (Theorem 6.4.8). The result is reduced to lowest
    terms with numerator and denominator normalized to constant term ``1``.
    See :func:`periodic_points` for how each shift class is presented.
    Requires sympy (``sofic[symbolic]``).

    Parameters
    ----------
    shift : SymbolicModel
        A TMC, SFT, or sofic-shift presentation.
    t : sympy.Symbol, optional
        Variable of the rational function; defaults to ``Symbol("t")``.

    Returns
    -------
    sympy.Expr
        ``numerator / denominator`` in ``t``.
    """
    import sympy as sp

    t = sp.Symbol("t") if t is None else t
    numerator = sp.Poly(1, t)
    denominator = sp.Poly(1, t)
    for matrix, sign in _periodic_point_matrices(shift):
        coefficients = reciprocal_characteristic_polynomial(matrix)
        factor = sp.Poly(list(reversed(coefficients)), t)
        if sign > 0:
            denominator *= factor
        else:
            numerator *= factor
    common = sp.gcd(numerator, denominator)
    numerator = sp.quo(numerator, common)
    denominator = sp.quo(denominator, common)
    if denominator.eval(0) < 0:
        numerator, denominator = -numerator, -denominator
    return numerator.as_expr() / denominator.as_expr()


def reciprocal_characteristic_polynomial(matrix: np.ndarray) -> list[int]:
    """Return the coefficients ``[1, c_1, ..., c_r]`` of ``det(I - tA)`` in increasing powers of ``t``.

    Exact Faddeev-LeVerrier recursion on an integer matrix; ``det(I - tA)`` is
    ``t^r chi_A(1/t)`` :cite:`LindMarcus1995` (proof of Theorem 6.4.6).
    """
    a = np.asarray(matrix, dtype=object)
    size = a.shape[0]
    identity = np.eye(size, dtype=int).astype(object)
    coefficients = [1]
    m = np.zeros((size, size), dtype=object)
    for k in range(1, size + 1):
        m = a.dot(m) + coefficients[-1] * identity
        quotient, remainder = divmod(-int(np.trace(a.dot(m))), k)
        if remainder:
            raise ValueError("matrix must have integer entries")
        coefficients.append(quotient)
    return coefficients


def _trace_power(matrix: np.ndarray, n: int) -> int:
    if matrix.shape[0] == 0:
        return 0
    return int(np.trace(np.linalg.matrix_power(matrix, n)))


def _periodic_point_matrices(shift: SymbolicModel) -> list[tuple[np.ndarray, int]]:
    """Integer matrices ``M`` with signs ``s`` such that ``p_n = sum s * tr M^n``."""
    from sofic.shifts.sft import ShiftOfFiniteType
    from sofic.shifts.sofic_dyck import SoficDyckShift
    from sofic.shifts.tmc import TopologicalMarkovChain

    if isinstance(shift, SoficDyckShift):
        raise TypeError("Dyck shifts are not sofic; periodic points are not supported")
    if isinstance(shift, TopologicalMarkovChain):
        return [(integer_adjacency_matrix(shift)[0], 1)]
    if isinstance(shift, ShiftOfFiniteType) and shift._has_forbidden_word_spec:
        return [(_forbidden_block_matrix(shift), 1)]
    return signed_subset_matrices(shift)


def _forbidden_block_matrix(shift: Any) -> np.ndarray:
    """Adjacency of the edge graph with allowed ``(L-1)``-blocks as vertices and ``L``-blocks as edges."""
    forbidden = shift.forbidden_words()
    if () in forbidden:
        return np.zeros((0, 0), dtype=object)
    length = max((len(word) for word in forbidden), default=1)

    def allowed(word: tuple[Any, ...]) -> bool:
        return not any(len(bad) <= len(word) and word[len(word) - len(bad) :] == bad for bad in forbidden)

    blocks: list[tuple[Any, ...]] = [()]
    for _ in range(length):
        blocks = [
            block + (symbol,) for block in blocks for symbol in shift.symbol_alphabet if allowed(block + (symbol,))
        ]
    vertices = {block[:-1] for block in blocks} | {block[1:] for block in blocks}
    index = {vertex: i for i, vertex in enumerate(vertices)}
    matrix = np.zeros((len(vertices), len(vertices)), dtype=object)
    for block in blocks:
        matrix[index[block[:-1]], index[block[1:]]] += 1
    return matrix


def signed_subset_matrices(model: SymbolicModel) -> list[tuple[np.ndarray, int]]:
    """Return the signed subset matrices ``(A_j, (-1)^(j+1))`` of a right-resolving presentation.

    Following :cite:`LindMarcus1995` (§6.4), the ``j``-th auxiliary graph has
    the ``j``-element state subsets as vertices; a symbol ``a`` defined on
    every state of ``I = {I_1 < ... < I_j}`` with distinct images gives an
    edge to ``J = a(I)`` signed by the parity of the permutation
    ``(a(I_1), ..., a(I_j))`` of ``J``. ``model`` is first right-resolved
    (:func:`~sofic.shifts.wheeler.right_resolve`) and trimmed. Each matrix is
    restricted to the strongly connected parts of its graph that carry a
    cycle, which leaves every ``tr A_j^n`` and ``det(I - tA_j)`` unchanged;
    empty matrices are dropped.
    """
    from sofic.shifts.sofic import SoficShift
    from sofic.shifts.wheeler import right_resolve

    presentation = (
        model
        if isinstance(model, SoficShift)
        else SoficShift(graph=model.graph.copy(), symbol_alphabet=model.symbol_alphabet)
    )
    resolved = trim_transient(right_resolve(presentation))
    states = tuple(resolved.states())
    index = {state: i for i, state in enumerate(states)}
    step: list[dict[Any, int]] = [{} for _ in states]
    for transition in resolved.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        if symbol is not None:
            step[index[transition.source]][symbol] = index[transition.target]
    symbols = {symbol for moves in step for symbol in moves}

    matrices: list[tuple[np.ndarray, int]] = []
    for size in range(1, len(states) + 1):
        entries: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = defaultdict(int)
        for subset in combinations(range(len(states)), size):
            for symbol in symbols:
                images = [step[i].get(symbol) for i in subset]
                if None in images or len(set(images)) < size:
                    continue
                entries[subset, tuple(sorted(images))] += _permutation_sign(images)
        matrix = _cyclic_part(entries)
        if matrix.shape[0]:
            matrices.append((matrix, (-1) ** (size + 1)))
    return matrices


def _permutation_sign(sequence: list[Any]) -> int:
    inversions = sum(1 for i, j in combinations(range(len(sequence)), 2) if sequence[i] > sequence[j])
    return -1 if inversions % 2 else 1


def _cyclic_part(entries: dict[tuple[Hashable, Hashable], int]) -> np.ndarray:
    graph = nx.DiGraph((source, target) for (source, target), value in entries.items() if value)
    nodes = [
        node
        for component in nx.strongly_connected_components(graph)
        if len(component) > 1 or graph.has_edge(next(iter(component)), next(iter(component)))
        for node in component
    ]
    index = {node: i for i, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=object)
    for (source, target), value in entries.items():
        if source in index and target in index:
            matrix[index[source], index[target]] += value
    return matrix
