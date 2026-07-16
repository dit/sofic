"""Exact minimal-DFA identification via a SAT encoding (DFASAT-style).

Given labeled positive/negative sample words, this finds the **minimal** number
of states of any DFA consistent with the sample -- the exact optimum that the
RPNI/EDSM heuristics only approximate. Following Heule & Verwer
:cite:`HeuleVerwer2010`, the augmented prefix-tree acceptor (APTA) is translated
into a graph-colouring SAT instance -- one Boolean per (APTA node, colour), plus
accepting-colour and transition variables -- and an external SAT solver is asked
whether a ``k``-colouring exists. The search runs ``k`` upward from a lower bound
to the EDSM upper bound (:func:`sofic.automata.edsm.learn_dfa_edsm`); the first
satisfiable ``k`` is the provably minimal DFA.

This requires the optional `python-sat <https://pysathq.github.io/>`_ dependency
(``pip install sofic[sat]``).
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

from sofic.automata.dfa import DFA
from sofic.automata.edsm import _ACCEPT, _REJECT, _build_apta, _collect_alphabet, learn_dfa_edsm

__all__ = ["learn_dfa_sat"]


def _require_pysat() -> Any:
    try:
        from pysat.formula import CNF, IDPool
        from pysat.solvers import Solver
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "learn_dfa_sat requires the optional 'python-sat' dependency; install with pip install sofic[sat]"
        ) from exc
    return CNF, IDPool, Solver


def _encode(
    label: Sequence[int],
    trans: Sequence[dict[Any, int]],
    alphabet: Sequence[Any],
    k: int,
    cnf_cls: Any,
    pool: Any,
) -> Any:
    """Build the graph-colouring CNF for a ``k``-state consistent DFA."""
    cnf = cnf_cls()
    n = len(label)
    sym_index = {symbol: s for s, symbol in enumerate(alphabet)}

    def x(v: int, i: int) -> int:
        return pool.id(("x", v, i))

    def a(i: int) -> int:
        return pool.id(("a", i))

    def y(s: int, i: int, j: int) -> int:
        return pool.id(("y", s, i, j))

    # (1) each APTA node gets at least one colour.
    for v in range(n):
        cnf.append([x(v, i) for i in range(k)])
    # (2) ... and at most one.
    for v in range(n):
        for i in range(k):
            for j in range(i + 1, k):
                cnf.append([-x(v, i), -x(v, j)])
    # (3) accepting / rejecting consistency of the colour.
    for v in range(n):
        if label[v] == _ACCEPT:
            for i in range(k):
                cnf.append([-x(v, i), a(i)])
        elif label[v] == _REJECT:
            for i in range(k):
                cnf.append([-x(v, i), -a(i)])
    # symmetry break: the root (APTA node 0) is colour 0.
    cnf.append([x(0, 0)])
    # (4) APTA edges force the coloured transition relation.
    for v in range(n):
        for symbol, w in trans[v].items():
            s = sym_index[symbol]
            for i in range(k):
                for j in range(k):
                    cnf.append([-x(v, i), -x(w, j), y(s, i, j)])
    # (5) the transition relation is a partial function (at most one target).
    for s in range(len(alphabet)):
        for i in range(k):
            for j1 in range(k):
                for j2 in range(j1 + 1, k):
                    cnf.append([-y(s, i, j1), -y(s, i, j2)])

    return cnf, x, a, y, sym_index


def _decode(
    model: set[int],
    label: Sequence[int],
    trans: Sequence[dict[Any, int]],
    alphabet: Sequence[Any],
    k: int,
    x: Any,
    a: Any,
    y: Any,
    sym_index: dict[Any, int],
) -> DFA:
    n = len(label)
    colour_of = {}
    for v in range(n):
        for i in range(k):
            if x(v, i) in model:
                colour_of[v] = i
                break

    used = sorted(set(colour_of.values()) | {0})
    names: dict[int, Hashable] = {colour: f"q{colour}" for colour in used}

    dfa = DFA(input_alphabet=frozenset(alphabet))
    for name in names.values():
        dfa.graph.add_state(name)

    seen: set[tuple[int, Any]] = set()
    for v in range(n):
        i = colour_of[v]
        for symbol, w in trans[v].items():
            j = colour_of[w]
            key = (i, symbol)
            if key in seen:
                continue
            seen.add(key)
            dfa.add_transition(names[i], names[j], symbol)

    dfa.initial_states = frozenset({names[colour_of[0]]})
    dfa.accepting_states = frozenset(names[i] for i in used if a(i) in model)
    dfa.validate()
    return dfa


def learn_dfa_sat(
    positive: Sequence[Sequence[Any]],
    negative: Sequence[Sequence[Any]] | None = None,
    *,
    lower_bound: int = 1,
    upper_bound: int | None = None,
    solver_name: str = "g3",
) -> DFA:
    """Learn the exact minimal DFA consistent with labeled samples via SAT.

    Parameters
    ----------
    positive
        Words that must be accepted.
    negative
        Words that must be rejected.
    lower_bound
        Smallest state count to try (default ``1``).
    upper_bound
        Largest state count to try. Defaults to the number of states of the
        EDSM hypothesis (:func:`sofic.automata.edsm.learn_dfa_edsm`), which is a
        valid upper bound on the minimum.
    solver_name
        Any `python-sat <https://pysathq.github.io/>`_ solver name (default
        ``"g3"`` for Glucose 3).

    Returns
    -------
    DFA
        The provably minimal DFA consistent with the sample
        :cite:`HeuleVerwer2010`.
    """
    cnf_cls, id_pool, solver_cls = _require_pysat()

    pos = [tuple(word) for word in positive]
    neg = [tuple(word) for word in (negative or ())]
    if not pos and not neg:
        raise ValueError("at least one positive or negative sample is required")

    alphabet = _collect_alphabet(pos, neg)
    label, trans = _build_apta(pos, neg)

    if upper_bound is None:
        upper_bound = len(list(learn_dfa_edsm(pos, neg).states()))
    lower_bound = max(1, int(lower_bound))
    if upper_bound < lower_bound:
        upper_bound = lower_bound

    for k in range(lower_bound, upper_bound + 1):
        pool = id_pool()
        cnf, x, a, y, sym_index = _encode(label, trans, alphabet, k, cnf_cls, pool)
        with solver_cls(name=solver_name, bootstrap_with=cnf.clauses) as solver:
            if solver.solve():
                model = {lit for lit in solver.get_model() if lit > 0}
                return _decode(model, label, trans, alphabet, k, x, a, y, sym_index)

    raise RuntimeError(f"no consistent DFA with at most {upper_bound} states (unexpected; check the sample)")
