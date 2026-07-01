"""Constructors for canonical ε-machines used in computational mechanics.

References
----------
- Even, ABC, SNS, NP2: Jurgens & Marzen, arXiv:1008.4182 (sync paper).
- Golden mean, butterfly, restricted golden mean, Nemo: Mahoney, Ellison &
  Crutchfield, arXiv:0905.4787 (IACP); butterfly/Nemo transition tables in
  Mahoney et al., arXiv:0906.5099.
- Reversible / Fig.~9 example: Ellison, Mahoney, James & Crutchfield,
  arXiv:1107.2168.
- Golden-mean Markov chain (forbid ``00``): Ellison et al., arXiv:0905.3587.
- Golden-mean shift (forbid ``11``, Parry max-entropy): standard symbolic
  dynamics; see e.g. Ellison et al., arXiv:1107.2168 Fig.~2.
- Tent map (Misiurewicz point): James, Burke & Crutchfield (2013), supplement
  to *Chaos Forgets and Remembers*; Figs.~6--8.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np

from pensive.generators.epsilon_machine import EpsilonMachine
from pensive.generators.mealy import MealyHMM
from pensive.graph import ATTR_EMISSION, ATTR_FUTURE_SYMBOL, ATTR_PROB, TransitionGraph
from pensive.shifts.tmc import TopologicalMarkovChain
from pensive.states import sequential_labels


def _stationary_distribution(
    states: Sequence[Hashable],
    symbol_matrices: Mapping[Any, np.ndarray],
) -> dict[Hashable, float]:
    transition = sum(symbol_matrices.values())
    eigenvalues, vectors = np.linalg.eig(transition.T)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    pi = np.real(vectors[:, index])
    if pi.sum() < 0.0:
        pi = -pi
    pi = np.maximum(pi, 0.0)
    total = float(pi.sum())
    if total <= 0.0:
        raise ValueError("failed to compute stationary distribution")
    pi /= total
    return {states[i]: float(pi[i]) for i in range(len(states))}


def from_symbol_matrices(
    states: Sequence[Hashable],
    symbols: Sequence[Any],
    matrices: Mapping[Any, np.ndarray],
    *,
    initial_distribution: Mapping[Hashable, float] | None = None,
) -> EpsilonMachine:
    """Build an ε-machine from edge-labeled transition matrices ``T^(x)``.

    ``matrices[x][i, j]`` is ``Pr(S' = states[j], X = x | S = states[i])``.
    """
    state_list = tuple(states)
    symbol_list = tuple(symbols)
    index = {state: i for i, state in enumerate(state_list)}
    arrays = {symbol: np.asarray(matrix, dtype=float) for symbol, matrix in matrices.items()}
    for symbol, matrix in arrays.items():
        if matrix.shape != (len(state_list), len(state_list)):
            raise ValueError(f"matrix for symbol {symbol!r} has shape {matrix.shape}")

    pi = (
        dict(initial_distribution) if initial_distribution is not None else _stationary_distribution(state_list, arrays)
    )
    eps = EpsilonMachine(
        initial_distribution=pi,
        observation_alphabet=frozenset(symbol_list),
    )
    for state in state_list:
        eps.graph.add_state(state)

    for symbol, matrix in arrays.items():
        for source in state_list:
            i = index[source]
            for target in state_list:
                j = index[target]
                prob = float(matrix[i, j])
                if prob <= 0.0:
                    continue
                eps.graph.add_transition(
                    source,
                    target,
                    **{ATTR_PROB: prob, ATTR_EMISSION: symbol},
                )
    eps.validate()
    return eps


def _add_edges(
    eps: EpsilonMachine,
    edges: Sequence[tuple[Hashable, Hashable, Any, float]],
) -> None:
    for source, target, symbol, prob in edges:
        eps.graph.add_transition(
            source,
            target,
            **{ATTR_PROB: float(prob), ATTR_EMISSION: symbol},
        )


def bernoulli(p: float = 0.5, *, symbols: tuple[Any, Any] = ("0", "1")) -> EpsilonMachine:
    """Memoryless (Bernoulli) source with ``P(symbols[0]) = 1 - p``."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    zero, one = symbols
    state = sequential_labels(1)[0]
    eps = EpsilonMachine(
        initial_distribution={state: 1.0},
        observation_alphabet=frozenset(symbols),
    )
    eps.graph.add_state(state)
    _add_edges(
        eps,
        [
            (state, state, zero, 1.0 - p),
            (state, state, one, p),
        ],
    )
    eps.validate()
    return eps


def fair_coin() -> EpsilonMachine:
    """Fair binary memoryless source (``p = 1/2``)."""
    return bernoulli(0.5)


def even_process(p: float = 0.5) -> EpsilonMachine:
    """Even Process: even-length blocks of 1s bounded by 0s.

    Jurgens & Marzen, arXiv:1008.4182, Fig.~1; Mahoney et al.,
    arXiv:0905.4787, Sec.~III.1.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return from_symbol_matrices(
        sequential_labels(2),
        (0, 1),
        {
            0: np.array([[p, 0.0], [0.0, 0.0]]),
            1: np.array([[0.0, 1.0 - p], [1.0, 0.0]]),
        },
    )


def noisy_random_phase_slip() -> EpsilonMachine:
    """Noisy Random Phase-Slip Process (James et al., 2011, Fig.~11c).

    Five-state ε-machine with stochastic phase slip at state ``A`` and
    emission noise at state ``D``.  Prototype for block-convergence figures in
    *Anatomy of a Bit* :cite:`James2011`.
    """
    from pensive.examples.processes import _edge_machine

    states = sequential_labels(5)
    a, b, c, d, e = states
    return _edge_machine(
        [
            (a, a, 0, 0.5),
            (a, b, 1, 0.5),
            (b, c, 0, 1.0),
            (c, d, 1, 1.0),
            (d, e, 0, 0.5),
            (d, e, 1, 0.5),
            (e, a, 0, 1.0),
        ],
        machine_type=EpsilonMachine,
        normalize=False,
    )


def golden_mean(p: float = 0.5) -> EpsilonMachine:
    """Golden Mean Process (two-state presentation, forbid consecutive ``11``).

    State ``A`` self-loops on ``0`` (probability ``p``) and moves to ``B``
    on ``1``; ``B`` always returns to ``A`` on ``0``.  This is the
    standard golden-mean *shift* topology (Mahoney et al., arXiv:0905.4787,
    Fig.~2 style).  For the order-1 Markov presentation that forbids ``00``,
    see :func:`golden_mean_markov`; for the bidirectional machine in Ellison et
    al., arXiv:0905.3587, Fig.~4, see :func:`golden_mean_forward` and
    :func:`golden_mean_reverse`; for the Parry max-entropy measure on the
    same shift, see :func:`golden_mean_shift_parry`.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return from_symbol_matrices(
        sequential_labels(2),
        (0, 1),
        {
            0: np.array([[p, 0.0], [1.0, 0.0]]),
            1: np.array([[0.0, 1.0 - p], [0.0, 0.0]]),
        },
    )


def golden_mean_forward(p: float = 0.5) -> EpsilonMachine:
    """Forward ε-machine M⁺ for the golden mean (forbid ``00``).

    States ``A`` and ``B``; self-loop probability ``p = Pr(1|A)``.
    Ellison, Mahoney & Crutchfield, arXiv:0905.3587, Fig.~4(a).
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return from_symbol_matrices(
        ("A", "B"),
        (0, 1),
        {
            0: np.array([[0.0, 1.0 - p], [0.0, 0.0]]),
            1: np.array([[p, 0.0], [1.0, 0.0]]),
        },
    )


def golden_mean_reverse(p: float = 0.5) -> EpsilonMachine:
    """Reverse ε-machine M⁻ for the golden mean (forbid ``00``).

    Isomorphic to :func:`golden_mean_forward` with states ``A`` and ``B``.
    When paired for a bidirectional presentation, reverse states are relabeled
    to continue the alphabet (``C``, ``D``, ...). Ellison, Mahoney & Crutchfield,
    arXiv:0905.3587, Fig.~4(b).
    """
    return golden_mean_forward(p)


def golden_mean_bidirectional(p: float = 0.5):
    """Bidirectional ε-machine M± for the golden mean (forbid ``00``).

    Joint states ``(A, C)``, ``(A, D)``, ``(B, C)`` as in Ellison et al.,
    arXiv:0905.3587, Fig.~4(c).
    """
    from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine

    return BidirectionalEpsilonMachine.from_pair(
        golden_mean_forward(p),
        golden_mean_reverse(p),
    )


def golden_mean_markov(p: float = 0.5) -> EpsilonMachine:
    """Order-1 Markov presentation of the golden mean (forbid ``00``).

    States are the most recent symbol ``0`` or ``1``. Ellison et al.,
    arXiv:0905.3587, Fig.~1; stationary ``pi(1) = 2/3`` at ``p = 1/2``.
    For the paper's ``A``/``B`` labeling and bidirectional machine, prefer
    :func:`golden_mean_forward`.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return from_symbol_matrices(
        sequential_labels(2),
        (0, 1),
        {
            0: np.array([[0.0, 0.0], [p, 0.0]]),
            1: np.array([[0.0, 1.0], [0.0, 1.0 - p]]),
        },
    )


def golden_mean_shift_parry() -> EpsilonMachine:
    """Max-entropy (Parry) measure on the golden-mean shift (forbid 11).

    This is the symbolic-dynamics convention (adjacency ``[[1,1],[1,0]]``),
    distinct from :func:`golden_mean` / :func:`golden_mean_markov`, which
    forbid consecutive 0s in the CM literature cited above.
    """
    tmc = TopologicalMarkovChain.from_adjacency(
        np.array([[1, 1], [1, 0]], dtype=float),
        symbol_alphabet=frozenset({0, 1}),
    )
    parry = tmc.parry_measure()
    return EpsilonMachine.from_hmm(parry)


def alternating_biased_coins(p: float = 0.5, q: float = 0.4) -> EpsilonMachine:
    """Alternating Biased Coins (ABC) Process.

    Jurgens & Marzen, arXiv:1008.4182, Fig.~2.
    """
    if not 0.0 < p < 1.0 or not 0.0 < q < 1.0:
        raise ValueError("p and q must be in (0, 1)")
    return from_symbol_matrices(
        sequential_labels(2),
        (0, 1),
        {
            0: np.array([[0.0, 1.0 - p], [1.0 - q, 0.0]]),
            1: np.array([[0.0, p], [q, 0.0]]),
        },
    )


def restricted_golden_mean(k: int = 1) -> EpsilonMachine:
    """Restricted Golden Mean family (``k``-cryptic); distinct from :func:`golden_mean`.

    Mahoney et al., arXiv:0906.5099, Sec.~III.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    states = sequential_labels(k + 1)
    n = k + 1
    t0 = np.zeros((n, n), dtype=float)
    t1 = np.zeros((n, n), dtype=float)
    t0[0, 1] = 0.5
    t1[0, 0] = 0.5
    for i in range(1, k):
        t1[i, i + 1] = 1.0
    t1[k, 0] = 1.0
    return from_symbol_matrices(states, (0, 1), {0: t0, 1: t1})


def nemo_process(p: float = 0.5, q: float = 0.5) -> EpsilonMachine:
    """Nemo Process (three-state, ``infty``-cryptic).

    Mahoney et al., arXiv:0906.5099, Fig.~7 / Sec.~IV.
    """
    if not 0.0 < p < 1.0 or not 0.0 < q < 1.0:
        raise ValueError("p and q must be in (0, 1)")
    states = ("A", "B", "C")
    return from_symbol_matrices(
        states,
        (0, 1),
        {
            0: np.array(
                [
                    [0.0, 1.0 - p, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0 - q, 0.0, 0.0],
                ]
            ),
            1: np.array(
                [
                    [p, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [q, 0.0, 0.0],
                ]
            ),
        },
    )


def phase_slip_backtrack(p: float = 0.5, q: float = 0.5) -> EpsilonMachine:
    """Phase-Slip Backtrack (PSB) Process (``R=3``, ``k_chi=2``).

    James, Mahoney, Ellison & Crutchfield, arXiv:1010.5545, Fig.~2.
    """
    if not 0.0 < p < 1.0 or not 0.0 < q < 1.0:
        raise ValueError("p and q must be in (0, 1)")
    states = ("A", "B", "C", "D")
    return from_symbol_matrices(
        states,
        (0, 1),
        {
            0: np.array(
                [
                    [0.0, 0.0, 1.0 - p, 0.0],
                    [0.0, 0.0, 0.0, 1.0 - q],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            1: np.array(
                [
                    [0.0, p, 0.0, 0.0],
                    [0.0, 0.0, q, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            ),
        },
    )


def butterfly_process() -> EpsilonMachine:
    """Butterfly Process (five-state, ``2``-cryptic) over symbols ``0``--``7``.

    Mahoney et al., arXiv:0906.5099, Fig.~1. Each causal state emits every
    symbol with probability ``1/8``; synchronizing symbols ``2``--``7`` always
    reach the same causal state regardless of the source.
    """
    states = ("A", "B", "C", "D", "E")
    prob = 1.0 / 8.0
    targets = {
        0: {"A": "B", "B": "B", "C": "D", "D": "B", "E": "D"},
        1: {"A": "C", "B": "C", "C": "C", "D": "E", "E": "E"},
        2: "A",
        3: "A",
        4: "B",
        5: "D",
        6: "C",
        7: "E",
    }
    eps = EpsilonMachine(
        initial_distribution=dict.fromkeys(states, 0.2),
        observation_alphabet=frozenset(range(8)),
    )
    for state in states:
        eps.graph.add_state(state)
    for source in states:
        for symbol in range(8):
            target_spec = targets[symbol]
            target = target_spec if isinstance(target_spec, str) else target_spec[source]
            _add_edges(eps, [(source, target, symbol, prob)])
    eps.validate()
    return eps


def ellison_fig9_forward() -> EpsilonMachine:
    """Forward ε-machine from Ellison et al., arXiv:1107.2168, Fig.~9."""
    eps = EpsilonMachine(
        initial_distribution={"A": 0.5, "B": 0.5},
        observation_alphabet=frozenset({0, 1, 2}),
    )
    for state in ("A", "B"):
        eps.graph.add_state(state)
    _add_edges(
        eps,
        [
            ("A", "A", 0, 0.5),
            ("A", "B", 1, 0.5),
            ("B", "B", 1, 0.5),
            ("B", "A", 2, 0.5),
        ],
    )
    eps.validate()
    return eps


def tent_map_misiurewicz_a() -> float:
    """Misiurewicz parameter ``a`` for the tent map (James et al., 2013, Eq. 12)."""
    alpha = (math.sqrt(19 / 27) + 1) ** (1 / 3)
    return alpha + 2 / (3 * alpha)


def tent_map_misiurewicz_fig7_symbol_matrices(
    a: float | None = None,
) -> tuple[tuple[str, ...], tuple[int, ...], dict[int, np.ndarray]]:
    """Return Fig.~7 ε-machine symbol matrices for the tent map at parameter ``a``.

    James, Burke & Crutchfield, *Chaos Forgets and Remembers* (2013), supplement
    Fig.~7.  State ``A`` emits only ``1``; ``D`` has a nontrivial ``0`` branch to
    ``C`` and a ``1`` self-loop.
    """
    if a is None:
        a = tent_map_misiurewicz_a()
    states = ("A", "B", "C", "D")
    denom = 2 * a**2 + 4 * a + 2
    t0 = np.zeros((4, 4), dtype=float)
    t1 = np.zeros((4, 4), dtype=float)
    # Topology from supplement Fig. 7.
    t1[0, 1] = 1.0  # A → B on 1
    t0[1, 2] = (a + 2) / (2 * a + 2)  # B → C on 0
    t1[1, 0] = a / (2 * a + 2)  # B → A on 1
    t0[2, 0] = 1.0 / (a + 2)  # C → A on 0
    t1[2, 3] = (a + 1) / (a + 2)  # C → D on 1
    t0[3, 2] = (a**2 + 2 * a) / denom  # D → C on 0
    t1[3, 3] = (a**2 + 2 * a + 2) / denom  # D → D on 1
    return states, (0, 1), {0: t0, 1: t1}


def tent_map_misiurewicz_forward(a: float | None = None) -> EpsilonMachine:
    """Forward ε-machine for tent-map symbolic dynamics at the Misiurewicz point."""
    states, symbols, matrices = tent_map_misiurewicz_fig7_symbol_matrices(a)
    return from_symbol_matrices(states, symbols, matrices)


def tent_map_misiurewicz_hmm(a: float | None = None) -> MealyHMM:
    """Non-unifilar HMM from supplement Fig.~6 (right) before ε-machine minimization."""
    from pensive.generators.mealy import MealyHMM

    if a is None:
        a = tent_map_misiurewicz_a()
    # State B emits 0 or 1; state D emits only 1.  Row masses are joint P(target, symbol | state).
    p_b0 = (a + 2) / (2 * a + 2)
    p_b1 = a / (2 * a + 2)
    hmm = MealyHMM(observation_alphabet=frozenset({0, 1}))
    for state in ("A", "B", "C", "D"):
        hmm.graph.add_state(state)
    edges = [
        ("A", "B", 0, 0.5),
        ("A", "C", 0, 0.5),
        ("B", "A", 0, p_b0),
        ("B", "D", 1, p_b1),
        ("C", "A", 1, 0.5),
        ("C", "B", 1, 0.5),
        ("D", "C", 1, 1.0),
    ]
    for source, target, symbol, prob in edges:
        hmm.graph.add_transition(
            source,
            target,
            **{ATTR_PROB: float(prob), ATTR_EMISSION: symbol},
        )
    idx = hmm.reindex()
    pi = hmm.stationary_distribution()
    hmm.initial_distribution = {idx.state(i): float(pi[i]) for i in range(len(idx.states))}
    return hmm


def tent_map_misiurewicz_reverse(a: float | None = None) -> EpsilonMachine:
    """Reverse ε-machine for tent-map symbolic dynamics at the Misiurewicz point.

    Projected from the supplement Fig.~8 bidirectional presentation with
    ``future_symbol`` annotations on ``E``, ``F``, and ``G``.
    """
    return tent_map_misiurewicz_bidirectional_fig8(a).reverse_machine


def _annotate_tent_map_misiurewicz_reverse_future_symbols(reverse: EpsilonMachine) -> None:
    """Attach synchronization symbols to reverse causal states (supplement Fig.~8)."""
    # Present-symbol constraints on reverse components E,F,G; H is outside support.
    futures = {"E": 0, "F": 1, "G": 1, "H": -1}
    for state, symbol in futures.items():
        if not reverse.graph.has_state(state):
            continue
        attrs = dict(reverse.graph.state_attrs(state))
        attrs[ATTR_FUTURE_SYMBOL] = symbol
        reverse.graph.add_state(state, **attrs)


def _tent_map_misiurewicz_fig8_reverse_relabel() -> dict[str, str]:
    """Map supplement reverse causal labels to collision-free joint labels."""
    return {"A": "E", "B": "F", "C": "G", "D": "H"}


def _tent_map_misiurewicz_fig8_joint_state(
    forward: str,
    reverse: str,
    *,
    relabel: Mapping[str, str],
) -> tuple[str, str]:
    return forward, relabel[reverse]


def _tent_map_misiurewicz_fig8_edges(
    a: float,
) -> list[tuple[tuple[str, str], tuple[str, str], int, float]]:
    """Directed edges for supplement Fig.~8 with reverse states relabeled E--H.

    Joint labels use ``S⁺:S⁻`` from James et al. (2013), supplement Fig.~8.
    Edge probabilities are the figure's ``1/2`` and ``a/(a+1)`` templates.
    """
    relabel = _tent_map_misiurewicz_fig8_reverse_relabel()
    name = {
        "BA": ("B", "A"),
        "CC": ("C", "C"),
        "AB": ("A", "B"),
        "BC": ("B", "C"),
        "CB": ("C", "B"),
        "DA": ("D", "A"),
        "DB": ("D", "B"),
        "DC": ("D", "C"),
    }
    half = 0.5
    inv_a1 = 1.0 / (a + 1.0)
    frac_a1 = a / (a + 1.0)
    edge_specs = [
        ("BA", "CC", 0, 1.0),
        ("CC", "AB", 0, half),
        ("CC", "DB", 1, half),
        ("AB", "BA", 1, inv_a1),
        ("AB", "BC", 1, frac_a1),
        ("BC", "AB", 1, half),
        ("BC", "CB", 0, half),
        ("DA", "CC", 0, 1.0),
        ("DB", "DA", 1, inv_a1),
        ("DB", "DC", 1, frac_a1),
        ("DC", "DB", 1, half),
        ("DC", "CB", 0, half),
        ("CB", "DC", 1, frac_a1),
        ("CB", "DA", 1, inv_a1),
    ]
    edges: list[tuple[tuple[str, str], tuple[str, str], int, float]] = []
    for row, target, symbol, prob in edge_specs:
        source = _tent_map_misiurewicz_fig8_joint_state(name[row][0], name[row][1], relabel=relabel)
        dest = _tent_map_misiurewicz_fig8_joint_state(name[target][0], name[target][1], relabel=relabel)
        edges.append((source, dest, symbol, prob))
    return edges


def _stationary_distribution_from_joint_graph(
    graph: TransitionGraph,
) -> dict[tuple[str, str], float]:
    states = list(graph.states())
    if not states:
        return {}
    index = {state: position for position, state in enumerate(states)}
    transition = np.zeros((len(states), len(states)), dtype=float)
    for transition_edge in graph.transitions():
        source = index[transition_edge.source]
        target = index[transition_edge.target]
        transition[source, target] += float(transition_edge.data.get(ATTR_PROB, 0.0))
    stationary = np.ones(len(states), dtype=float) / len(states)
    for _ in range(20_000):
        stationary = stationary @ transition
    return {states[position]: float(stationary[position]) for position in range(len(states))}


def _project_bidirectional_side(
    graph: TransitionGraph,
    joint_pi: Mapping[tuple[str, str], float],
    *,
    project_forward: bool,
    future_symbols: Mapping[str, Any] | None = None,
) -> EpsilonMachine:
    """Marginalize a hand-built bidirectional graph to an ε-machine presentation."""
    coordinate = 0 if project_forward else 1
    marginal: dict[str, float] = {}
    for pair, mass in joint_pi.items():
        side_state = pair[coordinate]
        marginal[side_state] = marginal.get(side_state, 0.0) + float(mass)

    side_graph = TransitionGraph()
    for state, mass in marginal.items():
        if mass <= 0.0:
            continue
        attrs: dict[str, Any] = {}
        if future_symbols is not None and state in future_symbols:
            attrs[ATTR_FUTURE_SYMBOL] = future_symbols[state]
        side_graph.add_state(state, **attrs)

    aggregated: dict[tuple[str, str, Any], float] = {}
    for pair, mass in joint_pi.items():
        source = pair[coordinate]
        source_mass = marginal.get(source, 0.0)
        if mass <= 0.0 or source_mass <= 0.0:
            continue
        for transition in graph.out_transitions(pair):
            symbol = transition.data.get(ATTR_EMISSION)
            prob = float(transition.data.get(ATTR_PROB, 0.0))
            if symbol is None or prob <= 0.0:
                continue
            target_state = transition.target[coordinate]
            key = (source, target_state, symbol)
            aggregated[key] = aggregated.get(key, 0.0) + mass * prob / source_mass

    for (source, target, symbol), prob in aggregated.items():
        if prob <= 0.0:
            continue
        side_graph.add_transition(
            source,
            target,
            **{ATTR_PROB: prob, ATTR_EMISSION: symbol},
        )

    eps = EpsilonMachine(
        graph=side_graph,
        initial_distribution=marginal,
        observation_alphabet=frozenset({0, 1}),
    )
    eps.validate_stochastic()
    return eps


def tent_map_misiurewicz_bidirectional_fig8(a: float | None = None):
    """Hand-built supplement Fig.~8 bidirectional ε-machine."""
    from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine

    if a is None:
        a = tent_map_misiurewicz_a()
    forward = tent_map_misiurewicz_forward(a)
    graph = TransitionGraph()
    for source, target, symbol, prob in _tent_map_misiurewicz_fig8_edges(float(a)):
        if not graph.has_state(source):
            graph.add_state(source)
        if not graph.has_state(target):
            graph.add_state(target)
        graph.add_transition(
            source,
            target,
            **{ATTR_PROB: prob, ATTR_EMISSION: symbol},
        )
    joint_pi = _stationary_distribution_from_joint_graph(graph)
    reverse_raw = _project_bidirectional_side(
        graph,
        joint_pi,
        project_forward=False,
        future_symbols={"E": 0, "F": 1, "G": 1, "H": -1},
    )
    try:
        reverse = EpsilonMachine.from_hmm(reverse_raw)
    except Exception:
        from pensive.generators.epsilon_machine import _row_normalized_presentation

        reverse = _row_normalized_presentation(reverse_raw)
    _annotate_tent_map_misiurewicz_reverse_future_symbols(reverse)
    bidir = BidirectionalEpsilonMachine(
        graph=graph,
        initial_distribution=joint_pi,
        observation_alphabet=frozenset({0, 1}),
        forward_machine=forward,
        reverse_machine=reverse,
    )
    bidir._joint_pi = dict(joint_pi)
    bidir.validate()
    return bidir


def tent_map_misiurewicz_bidirectional(a: float | None = None):
    """Bidirectional ε-machine for the tent map at the Misiurewicz parameter."""
    return tent_map_misiurewicz_bidirectional_fig8(a)


def tent_map_misiurewicz_information_expected(a: float | None = None) -> dict[str, float]:
    """Closed-form anatomy rates from James et al. (2013), supplement."""
    if a is None:
        a = tent_map_misiurewicz_a()
    h_mu = math.log2(a)
    # Supplement closed form for I[X₀ : S⁻₁ | S⁺₀] (bound rate b_μ).
    b_mu = h_mu - 0.25 * (3.0 - 2.0 / (a + 1) - 4.0 / (a + 2) + 9.0 / (2 * a + 3))
    r_mu = h_mu - b_mu
    return {
        "bound_mu": b_mu,
        "ephemeral_mu": r_mu,
        "entropy_rate": h_mu,
    }


def ellison_fig9_reverse() -> EpsilonMachine:
    """Reverse ε-machine from Ellison et al., arXiv:1107.2168, Fig.~9.

    MSP-derived presentation used for Fig.~15 bidirectional pairing via Eq.~(15).
    """
    from pensive.generators.reversal import time_reverse_stochastic

    forward = ellison_fig9_forward()
    reverse = EpsilonMachine.from_hmm(time_reverse_stochastic(forward))
    reverse.validate()
    return reverse


def ellison_fig15_bidirectional():
    """Bidirectional ε-machine M± from Ellison et al., arXiv:1107.2168, Fig.~15.

    Built from the separate forward and reverse presentations in Fig.~9 via
    Eq.~(15) in the same paper.
    """
    from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine

    return BidirectionalEpsilonMachine.from_pair(
        ellison_fig9_forward(),
        ellison_fig9_reverse(),
    )
