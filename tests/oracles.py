"""Brute-force reference implementations for differential tests.

Every oracle here is deliberately naive: it enumerates words, hidden paths,
runs, or configurations explicitly and touches only the raw transition graph of
a model (never sofic's own simulation, normalization, or inference code). They
are exponential and intended only for tiny models and short words.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from itertools import product
from typing import Any

import networkx as nx
import numpy as np

from sofic.automata.nwa import NestedWord
from sofic.graph import (
    ATTR_EMISSION,
    ATTR_HIER_STATE,
    ATTR_KIND,
    ATTR_OUTPUT,
    ATTR_PROB,
    ATTR_STACK_SYMBOL,
    ATTR_SYMBOL,
    EPSILON,
    KIND_CALL,
    KIND_INTERNAL,
    KIND_RETURN,
)

Word = tuple[Any, ...]


# --------------------------------------------------------------------------- words


def words(alphabet: Iterable[Any], max_length: int, *, min_length: int = 0) -> list[Word]:
    """All words over ``alphabet`` with ``min_length <= len <= max_length``, shortlex order."""
    symbols = tuple(alphabet)
    return [word for n in range(min_length, max_length + 1) for word in product(symbols, repeat=n)]


# --------------------------------------------------------------------------- NFAs


def _epsilon_closure(model: Any, states: Iterable[Hashable]) -> frozenset[Hashable]:
    closure = set(states)
    stack = list(closure)
    while stack:
        for transition in model.graph.out_transitions(stack.pop()):
            if transition.data.get(ATTR_SYMBOL) is EPSILON and transition.target not in closure:
                closure.add(transition.target)
                stack.append(transition.target)
    return frozenset(closure)


def _symbol_step(model: Any, states: Iterable[Hashable], symbol: Any) -> frozenset[Hashable]:
    return frozenset(
        transition.target
        for state in states
        for transition in model.graph.out_transitions(state)
        if transition.data.get(ATTR_SYMBOL) is not EPSILON and transition.data.get(ATTR_SYMBOL) == symbol
    )


def nfa_reachable(nfa: Any, word: Sequence[Any], *, start: Iterable[Hashable] | None = None) -> frozenset[Hashable]:
    """States reachable from ``start`` (default: initial states) by reading ``word``, epsilon-closed."""
    current = _epsilon_closure(nfa, nfa.initial_states if start is None else start)
    for symbol in word:
        current = _epsilon_closure(nfa, _symbol_step(nfa, current, symbol))
    return current


def nfa_accepts(nfa: Any, word: Sequence[Any]) -> bool:
    """Membership by explicit subset simulation with epsilon closure."""
    return bool(nfa_reachable(nfa, word) & frozenset(nfa.accepting_states))


def language(accepts: Callable[[Word], bool], alphabet: Iterable[Any], max_length: int) -> frozenset[Word]:
    """Words of length ``<= max_length`` accepted by the predicate ``accepts``."""
    return frozenset(word for word in words(alphabet, max_length) if accepts(word))


def nerode_class_count(accepts: Callable[[Word], bool], alphabet: Iterable[Any], k: int) -> int:
    """Count Myhill-Nerode classes visible with prefixes and suffixes of length ``<= k``.

    Each prefix ``u`` gets the signature ``(accepts(u + v) for v in words(<= k))``;
    the result is the number of distinct signatures. For a language whose complete
    minimal DFA has ``m`` states this equals ``m`` once ``k >= m - 1``.
    """
    symbols = tuple(alphabet)
    suffixes = words(symbols, k)
    return len({tuple(accepts(prefix + suffix) for suffix in suffixes) for prefix in words(symbols, k)})


# --------------------------------------------------------------------------- HMMs


def _hmm_tables(hmm: Any, states: Sequence[Hashable] | None) -> tuple[list[Hashable], dict, dict]:
    mealy = hmm.to_mealy() if hasattr(hmm, "to_mealy") else hmm
    order = list(mealy.states()) if states is None else list(states)
    if not mealy.initial_distribution:
        raise ValueError("oracle needs an explicit initial distribution")
    pi = {state: float(mealy.initial_distribution.get(state, 0.0)) for state in order}
    step: dict[tuple[Hashable, Any], dict[Hashable, float]] = {}
    for transition in mealy.transitions():
        key = (transition.source, transition.data.get(ATTR_EMISSION))
        row = step.setdefault(key, {})
        row[transition.target] = row.get(transition.target, 0.0) + float(transition.data.get(ATTR_PROB, 0.0))
    return order, pi, step


def _joint(pi: Mapping, step: Mapping, path: Sequence[Hashable], observations: Sequence[Any]) -> float:
    prob = pi.get(path[0], 0.0)
    for t, symbol in enumerate(observations):
        if prob == 0.0:
            return 0.0
        prob *= step.get((path[t], symbol), {}).get(path[t + 1], 0.0)
    return prob


def _paths(order: Sequence[Hashable], length: int) -> Iterable[tuple[Hashable, ...]]:
    return product(order, repeat=length)


def path_probability(hmm: Any, path: Sequence[Hashable], observations: Sequence[Any]) -> float:
    """Joint probability ``P(X_0..X_n = path, Y_0..Y_{n-1} = observations)``."""
    if len(path) != len(observations) + 1:
        raise ValueError("path must have one more state than there are observations")
    _order, pi, step = _hmm_tables(hmm, None)
    return _joint(pi, step, path, observations)


def word_probability(hmm: Any, observations: Sequence[Any]) -> float:
    """``P(Y_0..Y_{n-1} = observations)`` by summing over all hidden paths ``X_0..X_n``."""
    order, pi, step = _hmm_tables(hmm, None)
    return sum(_joint(pi, step, path, observations) for path in _paths(order, len(observations) + 1))


def word_distribution(hmm: Any, length: int, alphabet: Iterable[Any] | None = None) -> dict[Word, float]:
    """Map every length-``length`` word to its probability (zero-probability words included)."""
    symbols = tuple(hmm.observation_alphabet if alphabet is None else alphabet)
    return {word: word_probability(hmm, word) for word in words(symbols, length, min_length=length)}


def forward(hmm: Any, observations: Sequence[Any], *, states: Sequence[Hashable] | None = None) -> np.ndarray:
    """``alpha[t, s] = P(Y_0..Y_{t-1}, X_t = s)`` for ``t = 0..n`` by path enumeration."""
    order, pi, step = _hmm_tables(hmm, states)
    alpha = np.zeros((len(observations) + 1, len(order)))
    for t in range(len(observations) + 1):
        for path in _paths(order, t + 1):
            alpha[t, order.index(path[-1])] += _joint(pi, step, path, observations[:t])
    return alpha


def backward(hmm: Any, observations: Sequence[Any], *, states: Sequence[Hashable] | None = None) -> np.ndarray:
    """``beta[t, s] = P(Y_t..Y_{n-1} | X_t = s)`` for ``t = 0..n`` by path enumeration."""
    order, _pi, step = _hmm_tables(hmm, states)
    n = len(observations)
    beta = np.zeros((n + 1, len(order)))
    for t in range(n + 1):
        for i, state in enumerate(order):
            uniform = {state: 1.0}
            for tail in _paths(order, n - t):
                beta[t, i] += _joint(uniform, step, (state, *tail), observations[t:])
    return beta


def gamma(hmm: Any, observations: Sequence[Any], *, states: Sequence[Hashable] | None = None) -> np.ndarray:
    """``gamma[t, s] = P(X_t = s | Y_0..Y_{n-1})``; all zeros if the word has probability zero."""
    order, pi, step = _hmm_tables(hmm, states)
    n = len(observations)
    result = np.zeros((n + 1, len(order)))
    for path in _paths(order, n + 1):
        prob = _joint(pi, step, path, observations)
        for t, state in enumerate(path):
            result[t, order.index(state)] += prob
    total = result[0].sum()
    return result / total if total > 0.0 else np.zeros_like(result)


def xi(hmm: Any, observations: Sequence[Any], *, states: Sequence[Hashable] | None = None) -> np.ndarray:
    """``xi[t, i, j] = P(X_t = i, X_{t+1} = j | Y_0..Y_{n-1})`` for ``t = 0..n-1``."""
    order, pi, step = _hmm_tables(hmm, states)
    n = len(observations)
    result = np.zeros((n, len(order), len(order)))
    total = 0.0
    for path in _paths(order, n + 1):
        prob = _joint(pi, step, path, observations)
        total += prob
        for t in range(n):
            result[t, order.index(path[t]), order.index(path[t + 1])] += prob
    return result / total if total > 0.0 else np.zeros_like(result)


def viterbi_paths(
    hmm: Any,
    observations: Sequence[Any],
    *,
    rel_tol: float = 1e-9,
) -> tuple[float, list[tuple[Hashable, ...]]]:
    """Brute-force MAP decoding: ``(max joint probability, all maximizing X_0..X_n)``.

    Every path whose joint probability is within ``rel_tol`` of the maximum is
    returned, so ties can be checked without fixing a tie-breaking rule. The
    list is empty when every path has probability zero.
    """
    order, pi, step = _hmm_tables(hmm, None)
    scored = [(_joint(pi, step, path, observations), path) for path in _paths(order, len(observations) + 1)]
    best = max((prob for prob, _ in scored), default=0.0)
    if best <= 0.0:
        return 0.0, []
    return best, [path for prob, path in scored if math.isclose(prob, best, rel_tol=rel_tol)]


def block_entropy(distribution: Mapping[Any, float], *, base: float = 2.0) -> float:
    """Shannon entropy ``H(L) = -sum p log p`` of a word distribution (bits by default)."""
    return -sum(p * math.log(p, base) for p in distribution.values() if p > 0.0)


# --------------------------------------------------------------------------- VPAs and NWAs


def vpa_accepts(vpa: Any, word: Sequence[Any]) -> bool:
    """Membership by explicit ``(state, stack)`` configurations on the raw VPA.

    The stack holds pushed symbols; the empty stack stands for the bottom.
    Calls push their stack symbol (calls without one never fire), so pending
    calls simply stay on the stack. A return guarded by the bottom symbol fires
    only on the empty stack and leaves it empty; a return guarded by ``g`` pops a
    top ``g``; a wildcard return (no guard) pops any top symbol and, when the VPA
    has a bottom symbol, also fires on the empty stack. Without a bottom symbol a
    pending return can never fire.
    """
    if vpa.initial_state is None:
        return False
    bottom = vpa.bottom_stack_symbol
    configs: set[tuple[Hashable, tuple[Any, ...]]] = {(vpa.initial_state, ())}
    for symbol in word:
        nxt: set[tuple[Hashable, tuple[Any, ...]]] = set()
        for state, stack in configs:
            for transition in vpa.graph.out_transitions(state):
                data = transition.data
                if data.get(ATTR_SYMBOL) is None or data.get(ATTR_SYMBOL) != symbol:
                    continue
                kind, guard, target = data.get(ATTR_KIND), data.get(ATTR_STACK_SYMBOL), transition.target
                if kind == KIND_INTERNAL:
                    nxt.add((target, stack))
                elif kind == KIND_CALL:
                    if guard is not None:
                        nxt.add((target, (*stack, guard)))
                elif kind == KIND_RETURN:
                    if guard is None:
                        if stack:
                            nxt.add((target, stack[:-1]))
                        elif bottom is not None:
                            nxt.add((target, stack))
                    elif bottom is not None and guard == bottom:
                        if not stack:
                            nxt.add((target, stack))
                    elif stack and stack[-1] == guard:
                        nxt.add((target, stack[:-1]))
        configs = nxt
    return any(state in vpa.accepting_states for state, _stack in configs)


def nwa_accepts(nwa: Any, word: NestedWord | Sequence[Any]) -> bool:
    """Membership by enumerating runs that respect the nesting relation.

    A run picks one transition per position. It is valid when each matched
    return carries the hierarchical state stored by its matching call, and each
    pending return carries the bottom hierarchical state (so pending returns
    are impossible without one). Pending calls are unconstrained. A plain
    sequence is first nested by the NWA's visible role alphabets.
    """
    if not isinstance(word, NestedWord):
        word = NestedWord.from_visible_word(
            word,
            call_alphabet=nwa.call_alphabet,
            return_alphabet=nwa.return_alphabet,
            internal_alphabet=nwa.internal_alphabet,
        )
    if nwa.initial_state is None:
        return False
    bottom = nwa.bottom_hier_state
    n = len(word.symbols)

    def search(position: int, state: Hashable, stored: dict[int, Any]) -> bool:
        if position == n:
            return state in nwa.accepting_states
        symbol, kind, partner = word.symbols[position], word.kinds[position], word.matching[position]
        for transition in nwa.graph.out_transitions(state):
            data = transition.data
            if data.get(ATTR_SYMBOL) != symbol or data.get(ATTR_KIND) != kind:
                continue
            hier = data.get(ATTR_HIER_STATE)
            if kind == KIND_RETURN:
                if partner is None and (bottom is None or hier != bottom):
                    continue
                if partner is not None and stored[partner] != hier:
                    continue
            new_stored = {**stored, position: hier} if kind == KIND_CALL else stored
            if search(position + 1, transition.target, new_stored):
                return True
        return False

    return search(0, nwa.initial_state, {})


# --------------------------------------------------------------------------- Büchi automata


def buchi_accepts_lasso(ba: Any, prefix: Sequence[Any], loop: Sequence[Any]) -> bool:
    """Büchi acceptance of ``prefix loop^omega`` via the automaton x loop-position product.

    Product nodes are ``(state, i)`` with ``i`` the next loop index; letter edges
    read ``loop[i]`` and advance ``i`` cyclically, epsilon edges keep ``i``. The
    word is accepted iff some SCC reachable from ``(post-prefix states, 0)``
    contains an accepting state and at least one letter edge (so a run visits an
    accepting state infinitely often while consuming the whole infinite word).
    """
    if not loop:
        raise ValueError("loop must be non-empty")
    graph = nx.DiGraph()
    letter_edges: set[tuple[tuple[Hashable, int], tuple[Hashable, int]]] = set()
    for transition in ba.transitions():
        symbol = transition.data.get(ATTR_SYMBOL)
        for i, letter in enumerate(loop):
            source = (transition.source, i)
            if symbol is EPSILON:
                graph.add_edge(source, (transition.target, i))
            elif symbol == letter:
                target = (transition.target, (i + 1) % len(loop))
                graph.add_edge(source, target)
                letter_edges.add((source, target))
    starts = {(state, 0) for state in nfa_reachable(ba, prefix)}
    graph.add_nodes_from(starts)
    reachable = set(starts).union(*(nx.descendants(graph, node) for node in starts)) if starts else set()
    for component in nx.strongly_connected_components(graph.subgraph(reachable)):
        if not any(state in ba.accepting_states for state, _ in component):
            continue
        if any(source in component and target in component for source, target in letter_edges):
            return True
    return False


# --------------------------------------------------------------------------- transducers


def transducer_outputs(mealy: Any, word: Sequence[Any]) -> set[Word]:
    """All outputs of complete runs of ``mealy`` on ``word``, by path enumeration.

    Epsilon-input edges are followed too; a run is abandoned once its output
    exceeds a bound only reachable by pumping a productive epsilon-input cycle,
    which raises ``ValueError`` because the relation is then infinite.
    """
    limit = (len(word) + 1) * (len(list(mealy.states())) + 1)
    results: set[Word] = set()
    seen: set[tuple[Hashable, int, Word]] = set()
    stack = [(state, 0, ()) for state in mealy.initial_states]
    while stack:
        config = stack.pop()
        if config in seen:
            continue
        seen.add(config)
        state, position, output = config
        if len(output) > limit:
            raise ValueError("productive epsilon-input cycle: infinitely many outputs")
        if position == len(word):
            results.add(output)
        for transition in mealy.graph.out_transitions(state):
            symbol = transition.data.get(ATTR_SYMBOL)
            out = transition.data.get(ATTR_OUTPUT)
            emitted = () if out is None or out is EPSILON else (out,)
            if symbol is EPSILON:
                stack.append((transition.target, position, output + emitted))
            elif position < len(word) and symbol == word[position]:
                stack.append((transition.target, position + 1, output + emitted))
    return results


def transducer_relation(mealy: Any, inputs: Iterable[Sequence[Any]]) -> dict[Word, set[Word]]:
    """Map each input word to the set of outputs ``mealy`` can produce on it."""
    return {tuple(word): transducer_outputs(mealy, word) for word in inputs}


def compose_relations(
    first: Mapping[Word, set[Word]],
    second: Mapping[Word, set[Word]] | Callable[[Word], set[Word]],
) -> dict[Word, set[Word]]:
    """Relational composition ``{x: {z : y in first[x], z in second(y)}}``."""
    apply = second if callable(second) else (lambda y: second.get(y, set()))
    return {x: {z for y in ys for z in apply(y)} for x, ys in first.items()}


# --------------------------------------------------------------------------- lifted common information


def _cmi_bits(joint: np.ndarray) -> float:
    """``I[a : b | c]`` in bits for a joint indexed ``[a, b, c]``, from the four entropies."""

    def h(p: np.ndarray) -> float:
        p = p[p > 0.0]
        return float(-(p * np.log2(p)).sum())

    return max(0.0, h(joint.sum(axis=1)) + h(joint.sum(axis=0)) - h(joint) - h(joint.sum(axis=(0, 1))))


def _cmi_gradient_bits(joint: np.ndarray) -> np.ndarray:
    def log(x: np.ndarray) -> np.ndarray:
        return np.log2(np.maximum(x, 1e-300))

    return (
        log(joint)
        + log(joint.sum(axis=(0, 1)))[None, None, :]
        - log(joint.sum(axis=1))[:, None, :]
        - log(joint.sum(axis=0))[None, :, :]
    )


def lifted_tensors(problem: Any, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``p(s+, s-, g)``, ``p(g_0, x_0, s-_1, g_1)`` and ``p(s+_0, g_0, x_0, g_1)`` by direct ``einsum``."""
    plus, minus, edges = problem.plus, problem.minus, problem.edges
    screen = np.einsum("pa,pb,pg->abg", plus, minus, problem.p[:, None] * c)
    future = np.einsum("pg,pxq,qs,qv->gxsv", c, edges, minus, c)
    past = np.einsum("pa,pg,pxq,qv->agxv", plus, c, edges, c)
    return screen, future, past


def lifted_constraint_informations(problem: Any, c: np.ndarray) -> np.ndarray:
    screen, future, past = lifted_tensors(problem, c)
    return np.array(
        [
            _cmi_bits(screen),
            _cmi_bits(future.reshape(-1, *future.shape[2:])),
            _cmi_bits(np.moveaxis(past, 1, -1).reshape(past.shape[0], -1, past.shape[1])),
        ]
    )


def lifted_constraint_jacobian(problem: Any, c: np.ndarray) -> np.ndarray:
    plus, minus, edges = problem.plus, problem.minus, problem.edges
    screen, future, past = lifted_tensors(problem, c)
    g_screen = _cmi_gradient_bits(screen)
    g_future = _cmi_gradient_bits(future.reshape(-1, *future.shape[2:])).reshape(future.shape)
    moved = np.moveaxis(past, 1, -1)
    g_past = np.moveaxis(
        _cmi_gradient_bits(moved.reshape(past.shape[0], -1, past.shape[1])).reshape(moved.shape), -1, 1
    )
    jac = np.empty((3, *c.shape))
    jac[0] = problem.p[:, None] * np.einsum("pa,pb,abg->pg", plus, minus, g_screen)
    jac[1] = np.einsum("gxsv,pxq,qs,qv->pg", g_future, edges, minus, c) + np.einsum(
        "gxsv,pg,pxq,qs->qv", g_future, c, edges, minus
    )
    jac[2] = np.einsum("agxv,pa,pxq,qv->pg", g_past, plus, edges, c) + np.einsum(
        "agxv,pa,pg,pxq->qv", g_past, plus, c, edges
    )
    return jac


# --------------------------------------------------------------------------- simplex boxes


def _outward(value: Any, down: bool) -> float:
    """``value`` (a :class:`~fractions.Fraction`) rounded to a float in the given direction."""
    from fractions import Fraction

    f = float(value)
    if down and Fraction(f) > value:
        return math.nextafter(f, -math.inf)
    if not down and Fraction(f) < value:
        return math.nextafter(f, math.inf)
    return f


def tighten_exact(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """A box shrunk to its intersection with the probability simplex in exact rational arithmetic."""
    from fractions import Fraction

    for _ in range(3):
        exact_lo = [Fraction(float(v)) for v in lo]
        sum_lo = sum(exact_lo, Fraction(0))
        hi = np.minimum(hi, [_outward(1 - (sum_lo - v), down=False) for v in exact_lo])
        exact_hi = [Fraction(float(v)) for v in hi]
        sum_hi = sum(exact_hi, Fraction(0))
        lo = np.maximum(lo, [_outward(1 - (sum_hi - v), down=True) for v in exact_hi])
    return (lo, hi) if np.all(lo <= hi) else None


# --------------------------------------------------------------------------- rank-1 coverage (mu_1)


def _tree_potentials(tree: frozenset, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    na, nb = L.shape
    adj: list[list[int]] = [[] for _ in range(na + nb)]
    for a, b in tree:
        adj[a].append(na + b)
        adj[na + b].append(a)
    pot = np.full(na + nb, np.nan)
    pot[0] = 0.0
    stack = [0]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if np.isnan(pot[y]):
                a, b = (x, y - na) if x < na else (y, x - na)
                pot[y] = L[a, b] - pot[x]
                stack.append(y)
    return pot[:na], pot[na:]


def mu1_vertex_reference(M: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """``mu_1`` by vertex enumeration, splitting each tree with a fresh union-find per removed edge."""
    from sofic.generators.coverage_bound import (
        _LOG_FLOOR,
        _VERTEX_PERTURBATION,
        _components,
        _initial_tree,
        _logsumexp,
    )

    M = np.asarray(M, dtype=float)
    na, nb = M.shape
    L = np.maximum(np.log(np.maximum(M, 1e-300)), _LOG_FLOOR)
    L = L + _VERTEX_PERTURBATION * np.random.default_rng(0).random((na, nb))
    start = _initial_tree(L)
    seen = {start}
    queue = [start]
    best, best_pq = -np.inf, None
    while queue:
        tree = queue.pop()
        p, q = _tree_potentials(tree, L)
        value = _logsumexp(p) + _logsumexp(q)
        if value > best:
            best, best_pq = value, (p, q)
        slack = L - p[:, None] - q[None, :]
        for edge in tree:
            comp = _components(na, nb, tree - {edge})
            side = comp[edge[0]]
            rows_out = np.array([comp[a] != side for a in range(na)])
            cols_in = np.array([comp[na + b] == side for b in range(nb)])
            if not rows_out.any() or not cols_in.any():
                continue
            sub = slack[np.ix_(rows_out, cols_in)]
            i, j = np.unravel_index(int(np.argmin(sub)), sub.shape)
            entering = (int(np.flatnonzero(rows_out)[i]), int(np.flatnonzero(cols_in)[j]))
            nxt = (tree - {edge}) | {entering}
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    p, q = best_pq
    value = min(float(M.sum()), float(np.exp(best)))
    return value, np.exp(p - _logsumexp(p)), np.exp(q - _logsumexp(q))


def column_box_lp_reference(U: np.ndarray, A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """``max 1.x`` over ``0 <= x <= U[b]``, ``A[p].x <= 0``, ``C[p].x >= 0`` by LAPACK vertex enumeration."""
    from itertools import combinations

    nb, m = U.shape
    npairs = A.shape[0]
    G = np.zeros((nb, npairs, 2 * m + 2, m))
    h = np.zeros((nb, npairs, 2 * m + 2))
    G[:, :, :m] = -np.eye(m)
    G[:, :, m : 2 * m] = np.eye(m)
    h[:, :, m : 2 * m] = U[:, None, :]
    G[:, :, 2 * m] = A[None]
    G[:, :, 2 * m + 1] = -C[None]
    combos = np.array(list(combinations(range(2 * m + 2), m)))
    Gs, hs = G[:, :, combos], h[:, :, combos]
    regular = np.abs(np.linalg.det(Gs)) > 1e-14
    x = np.linalg.solve(np.where(regular[..., None, None], Gs, np.eye(m)), hs[..., None])[..., 0]
    slack = np.einsum("bpkm,bpcm->bpck", G, x) - h[:, :, None, :]
    feasible = regular & (slack.max(axis=-1) <= 1e-12 * (1.0 + float(U.max())))
    return np.where(feasible, x.sum(axis=-1), -np.inf).max(axis=-1)
