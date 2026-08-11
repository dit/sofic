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
  to *Chaos Forgets and Remembers*; Figs.~6--8.  The four-letter
  ``preimage`` variants read the same dynamics through a partition refined by
  both preimages of the critical point; that presentation is derived from the
  interval Markov chain rather than taken from a figure.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np

from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_EMISSION, ATTR_FUTURE_SYMBOL, ATTR_PROB, TransitionGraph
from sofic.shifts.tmc import TopologicalMarkovChain
from sofic.states import sequential_labels


def _stationary_distribution(
    states: Sequence[Hashable],
    symbol_matrices: Mapping[Any, np.ndarray],
) -> dict[Hashable, Any]:
    from sofic.generators.prob import as_prob, has_symbolic
    from sofic.generators.stationary import stationary_distribution_from_transition

    transition = sum(symbol_matrices.values())
    pi = stationary_distribution_from_transition(transition)
    if pi.dtype == object or has_symbolic(pi.ravel()):
        return {states[i]: as_prob(pi[i]) for i in range(len(states))}
    return {states[i]: float(pi[i]) for i in range(len(states))}


def from_symbol_matrices(
    states: Sequence[Hashable],
    symbols: Sequence[Any],
    matrices: Mapping[Any, np.ndarray],
    *,
    initial_distribution: Mapping[Hashable, Any] | None = None,
) -> EpsilonMachine:
    """Build an ε-machine from edge-labeled transition matrices ``T^(x)``.

    ``matrices[x][i, j]`` is ``Pr(S' = states[j], X = x | S = states[i])``.
    Entries may be floats or exact sympy expressions.
    """
    from sofic.generators.prob import as_prob, has_symbolic, is_positive_mass

    state_list = tuple(states)
    symbol_list = tuple(symbols)
    index = {state: i for i, state in enumerate(state_list)}
    flat_entries = [entry for matrix in matrices.values() for entry in np.asarray(matrix, dtype=object).ravel()]
    symbolic = has_symbolic(flat_entries)
    arrays = {}
    for symbol, matrix in matrices.items():
        arr = np.asarray(matrix, dtype=object if symbolic else float)
        if arr.shape != (len(state_list), len(state_list)):
            raise ValueError(f"matrix for symbol {symbol!r} has shape {arr.shape}")
        arrays[symbol] = arr

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
                prob = as_prob(matrix[i, j])
                if not is_positive_mass(prob):
                    continue
                eps.graph.add_transition(
                    source,
                    target,
                    **{ATTR_PROB: prob, ATTR_EMISSION: symbol},
                )
    eps.validate()
    return eps


def bernoulli(p: float = 0.5, *, symbols: tuple[Any, Any] = ("0", "1")) -> EpsilonMachine:
    """Memoryless (Bernoulli) source with ``P(symbols[0]) = 1 - p``."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    from sofic.examples.processes import _edge_machine

    zero, one = symbols
    state = sequential_labels(1)[0]
    return _edge_machine(
        [
            (state, state, zero, 1.0 - p),
            (state, state, one, p),
        ],
        initial_distribution={state: 1.0},
        normalize=False,
    )


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
    from sofic.examples.processes import _edge_machine

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
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine

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
    from sofic.examples.processes import _edge_machine

    edges = []
    for source in states:
        for symbol in range(8):
            target_spec = targets[symbol]
            target = target_spec if isinstance(target_spec, str) else target_spec[source]
            edges.append((source, target, symbol, prob))
    return _edge_machine(
        edges,
        initial_distribution=dict.fromkeys(states, 0.2),
        normalize=False,
    )


def ellison_fig9_forward() -> EpsilonMachine:
    """Forward ε-machine from Ellison et al., arXiv:1107.2168, Fig.~9."""
    from sofic.examples.processes import _edge_machine

    return _edge_machine(
        [
            ("A", "A", 0, 0.5),
            ("A", "B", 1, 0.5),
            ("B", "B", 1, 0.5),
            ("B", "A", 2, 0.5),
        ],
        initial_distribution={"A": 0.5, "B": 0.5},
        normalize=False,
    )


def tent_map_misiurewicz_a(symbolic: bool = False):
    """Misiurewicz parameter ``a`` for the tent map (James et al., 2013, Eq. 12).

    With ``symbolic=True``, return the exact sympy expression
    ``α + 2/(3α)`` where ``α = (1 + sqrt(19/27))**(1/3)``.
    """
    if symbolic:
        import sympy as sp

        alpha = (1 + sp.sqrt(sp.Rational(19, 27))) ** sp.Rational(1, 3)
        return sp.simplify(alpha + 2 / (3 * alpha))
    alpha = (math.sqrt(19 / 27) + 1) ** (1 / 3)
    return alpha + 2 / (3 * alpha)


def tent_map_misiurewicz_fig7_symbol_matrices(
    a: Any | None = None,
) -> tuple[tuple[str, ...], tuple[int, ...], dict[int, np.ndarray]]:
    """Return Fig.~7 ε-machine symbol matrices for the tent map at parameter ``a``.

    James, Burke & Crutchfield, *Chaos Forgets and Remembers* (2013), supplement
    Fig.~7.  State ``A`` emits only ``1``; ``D`` has a nontrivial ``0`` branch to
    ``C`` and a ``1`` self-loop.  When ``a`` is a sympy expression the matrices
    use object dtype with exact entries.
    """
    from sofic.generators.prob import is_symbolic, zeros

    if a is None:
        a = tent_map_misiurewicz_a()
    states = ("A", "B", "C", "D")
    symbolic = is_symbolic(a)
    denom = 2 * a**2 + 4 * a + 2
    t0 = zeros((4, 4), symbolic=symbolic)
    t1 = zeros((4, 4), symbolic=symbolic)
    # Topology from supplement Fig. 7.
    t1[0, 1] = 1 if symbolic else 1.0  # A → B on 1
    t0[1, 2] = (a + 2) / (2 * a + 2)  # B → C on 0
    t1[1, 0] = a / (2 * a + 2)  # B → A on 1
    t0[2, 0] = 1 / (a + 2) if symbolic else 1.0 / (a + 2)  # C → A on 0
    t1[2, 3] = (a + 1) / (a + 2)  # C → D on 1
    t0[3, 2] = (a**2 + 2 * a) / denom  # D → C on 0
    t1[3, 3] = (a**2 + 2 * a + 2) / denom  # D → D on 1
    return states, (0, 1), {0: t0, 1: t1}


def tent_map_misiurewicz_forward(a: Any | None = None) -> EpsilonMachine:
    """Forward ε-machine for tent-map symbolic dynamics at the Misiurewicz point."""
    states, symbols, matrices = tent_map_misiurewicz_fig7_symbol_matrices(a)
    return from_symbol_matrices(states, symbols, matrices)


def tent_map_misiurewicz_preimage_symbol_matrices(
    a: Any | None = None,
) -> tuple[tuple[str, ...], tuple[int, ...], dict[int, np.ndarray]]:
    """Symbol matrices for the tent map read through the partition ``{L, c, R}``.

    The kneading partition of :func:`tent_map_misiurewicz_fig7_symbol_matrices`
    cuts ``[0, 1]`` only at the critical point ``c = 1/2``.  Refining it by the
    two order-1 preimages of ``c`` -- ``L = 1/(2a)`` and ``R = 1 - 1/(2a)`` --
    gives the four-letter generating partition

    ==========  ==================
    Symbol      Cell
    ==========  ==================
    ``0``       ``[0, L)``
    ``1``       ``[L, c)``
    ``2``       ``[c, R)``
    ``3``       ``[R, 1]``
    ==========  ==================

    under which the map has the five-state ε-machine ``A``--``E``.  Reducing by
    the parameter's minimal polynomial ``a**3 = 2a + 2`` turns every transition
    probability into a quadratic in ``a`` with rational coefficients, so unlike
    the kneading presentation none of them carries an ``a``-dependent
    denominator.

    Derived from the exact interval Markov chain on the forward-orbit closure of
    ``{c, L, R}``.  The tent map, the Misiurewicz parameter and the information
    anatomy this presentation is used for are from James, Burke & Crutchfield,
    *Chaos Forgets and Remembers* (2013) :cite:`James2013`; that paper's figures
    cover only the kneading partition, so this refined presentation has no
    published figure to cite.
    """
    from sofic.generators.prob import is_symbolic, zeros

    if a is None:
        a = tent_map_misiurewicz_a()
    states = ("A", "B", "C", "D", "E")
    symbolic = is_symbolic(a)
    one = 1 if symbolic else 1.0
    matrices = {symbol: zeros((5, 5), symbolic=symbolic) for symbol in (0, 1, 2, 3)}
    # A=0, B=1, C=2, D=3, E=4.  Rows sum to one identically in ``a``.
    matrices[2][0, 0] = (a**2 - 2) / 2  # A → A on 2
    matrices[3][0, 1] = (4 - a**2) / 2  # A → B on 3
    matrices[0][1, 3] = (a**2 - 2 * a + 2) / 6  # B → D on 0
    matrices[1][1, 0] = (4 + 2 * a - a**2) / 6  # B → A on 1
    matrices[2][2, 4] = (2 + a - a**2) / 2  # C → E on 2
    matrices[3][2, 1] = (a**2 - a) / 2  # C → B on 3
    matrices[1][3, 2] = one  # D → C on 1
    matrices[2][4, 2] = one  # E → C on 2
    return states, (0, 1, 2, 3), matrices


def tent_map_misiurewicz_preimage_forward(a: Any | None = None) -> EpsilonMachine:
    """Forward ε-machine of the tent map under the four-letter ``{L, c, R}`` partition.

    Companion to :func:`tent_map_misiurewicz_forward`, which reads the same
    dynamics through the two-letter kneading partition.  Refining by both
    preimages of the critical point trades states for letters: five causal
    states over a four-letter alphabet instead of four over two.  The process
    stays strictly sofic (infinite Markov and cryptic order) but its ephemeral
    information vanishes -- see
    :func:`tent_map_misiurewicz_preimage_information_expected`.
    """
    states, symbols, matrices = tent_map_misiurewicz_preimage_symbol_matrices(a)
    return from_symbol_matrices(states, symbols, matrices)


def tent_map_misiurewicz_preimage_information_expected(a: Any | None = None) -> dict[str, Any]:
    """Closed-form anatomy of the four-letter ``{L, c, R}`` tent-map partition.

    The ephemeral rate is *exactly* zero, so ``b_mu = h_mu = log2(a)``.  The
    reason is structural rather than numerical: the machine of
    :func:`tent_map_misiurewicz_preimage_forward` is unifilar, no two of its
    edges share both a source and a target, and every branch leads to a state
    with a distinguishable future (``A`` versus ``B``, ``D`` versus ``A``, ``E``
    versus ``B``).  Knowing the past fixes the current causal state, the future
    then fixes the successor, and the two together name the emitted symbol --
    leaving nothing for ``r_mu = H[X_0 | past, future]`` to measure.

    Contrast :func:`tent_map_misiurewicz_information_expected`, where the
    coarser kneading partition of the *same* dynamics splits the same entropy
    rate into a nonzero ephemeral part (James, Burke & Crutchfield, 2013
    :cite:`James2013`).
    """
    from sofic.generators.prob import is_symbolic

    if a is None:
        a = tent_map_misiurewicz_a()
    if is_symbolic(a):
        import sympy as sp

        h_mu = sp.log(a, 2)
        return {"bound_mu": h_mu, "ephemeral_mu": sp.Integer(0), "entropy_rate": h_mu}
    h_mu = math.log2(a)
    return {"bound_mu": h_mu, "ephemeral_mu": 0.0, "entropy_rate": h_mu}


def tent_map_misiurewicz_hmm(a: Any | None = None) -> MealyHMM:
    """Non-unifilar HMM from supplement Fig.~6 (right).

    James, Burke & Crutchfield, *Chaos Forgets and Remembers* (2013), supplement
    Fig.~6 (right): generating partition overlaid on the Markov-partition chain.
    Non-unifilar at ``A`` (two ``0`` outs) and ``D`` (three ``1`` outs).
    :meth:`~sofic.generators.epsilon_machine.EpsilonMachine.from_hmm` recovers
    the Fig.~7 ε-machine.
    """
    from sofic.generators.prob import as_prob, is_symbolic

    if a is None:
        a = tent_map_misiurewicz_a()
    symbolic = is_symbolic(a)
    if symbolic:
        import sympy as sp

        half = sp.Rational(1, 2)
        one = sp.Integer(1)
        inv_a1 = 1 / (a + 1)
        half_a_a1 = a / (2 * (a + 1))
    else:
        half = 0.5
        one = 1.0
        inv_a1 = 1.0 / (a + 1.0)
        half_a_a1 = a / (2.0 * (a + 1.0))

    hmm = MealyHMM(observation_alphabet=frozenset({0, 1}))
    for state in ("A", "B", "C", "D"):
        hmm.graph.add_state(state)
    edges = [
        ("A", "B", 0, half),
        ("A", "C", 0, half),
        ("B", "D", 0, one),
        ("C", "D", 1, one),
        ("D", "A", 1, inv_a1),
        ("D", "B", 1, half_a_a1),
        ("D", "C", 1, half_a_a1),
    ]
    for source, target, symbol, prob in edges:
        hmm.add_transition(source, target, symbol, as_prob(prob))

    if symbolic:
        # The Misiurewicz parameter is the real root of ``a**3 - 2*a - 2`` (James,
        # Burke & Crutchfield, 2013, Eq. 12; the Cardano form ``alpha + 2/(3 alpha)``
        # returned by ``tent_map_misiurewicz_a(symbolic=True)``).  Carrying this
        # minimal polynomial lets ``EpsilonMachine.from_hmm`` recognize the two
        # mixed states that coincide only under the constraint and recover the
        # 4-state Fig.~7 machine.
        from sofic.generators.prob import SymbolConstraints

        hmm.symbol_constraints = SymbolConstraints([a**3 - 2 * a - 2])

    pi = hmm.stationary_distribution()
    idx = hmm.reindex()
    if pi.dtype == object or symbolic:
        hmm.initial_distribution = {idx.state(i): as_prob(pi[i]) for i in range(len(idx.states))}
    else:
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
    a: Any,
) -> list[tuple[tuple[str, str], tuple[str, str], int, Any]]:
    """Directed edges for supplement Fig.~8 with reverse states relabeled E--H.

    Joint labels use ``S⁺:S⁻`` from James et al. (2013), supplement Fig.~8.
    Edge probabilities are the figure's ``1/2`` and ``a/(a+1)`` templates.
    """
    from sofic.generators.prob import is_symbolic

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
    if is_symbolic(a):
        import sympy as sp

        half = sp.Rational(1, 2)
        inv_a1 = 1 / (a + 1)
        frac_a1 = a / (a + 1)
        one = sp.Integer(1)
    else:
        half = 0.5
        inv_a1 = 1.0 / (a + 1.0)
        frac_a1 = a / (a + 1.0)
        one = 1.0
    edge_specs = [
        ("BA", "CC", 0, one),
        ("CC", "AB", 0, half),
        ("CC", "DB", 1, half),
        ("AB", "BA", 1, inv_a1),
        ("AB", "BC", 1, frac_a1),
        ("BC", "AB", 1, half),
        ("BC", "CB", 0, half),
        ("DA", "CC", 0, one),
        ("DB", "DA", 1, inv_a1),
        ("DB", "DC", 1, frac_a1),
        ("DC", "DB", 1, half),
        ("DC", "CB", 0, half),
        ("CB", "DC", 1, frac_a1),
        ("CB", "DA", 1, inv_a1),
    ]
    edges: list[tuple[tuple[str, str], tuple[str, str], int, Any]] = []
    for row, target, symbol, prob in edge_specs:
        source = _tent_map_misiurewicz_fig8_joint_state(name[row][0], name[row][1], relabel=relabel)
        dest = _tent_map_misiurewicz_fig8_joint_state(name[target][0], name[target][1], relabel=relabel)
        edges.append((source, dest, symbol, prob))
    return edges


def _stationary_distribution_from_joint_graph(
    graph: TransitionGraph,
) -> dict[tuple[str, str], Any]:
    from sofic.generators.prob import as_prob, has_symbolic, zeros
    from sofic.generators.stationary import stationary_distribution_from_transition

    states = list(graph.states())
    if not states:
        return {}
    index = {state: position for position, state in enumerate(states)}
    edge_probs = [t.data.get(ATTR_PROB, 0.0) for t in graph.transitions()]
    symbolic = has_symbolic(edge_probs)
    transition = zeros((len(states), len(states)), symbolic=symbolic)
    for transition_edge in graph.transitions():
        source = index[transition_edge.source]
        target = index[transition_edge.target]
        transition[source, target] = as_prob(transition[source, target]) + as_prob(
            transition_edge.data.get(ATTR_PROB, 0.0)
        )
    if symbolic:
        pi = stationary_distribution_from_transition(transition)
        return {states[position]: as_prob(pi[position]) for position in range(len(states))}
    stationary = np.ones(len(states), dtype=float) / len(states)
    for _ in range(20_000):
        stationary = stationary @ np.asarray(transition, dtype=float)
    return {states[position]: float(stationary[position]) for position in range(len(states))}


def _project_bidirectional_side(
    graph: TransitionGraph,
    joint_pi: Mapping[tuple[str, str], Any],
    *,
    project_forward: bool,
    future_symbols: Mapping[str, Any] | None = None,
) -> EpsilonMachine:
    """Marginalize a hand-built bidirectional graph to an ε-machine presentation."""
    from sofic.generators.prob import (
        as_prob,
        is_positive_mass,
        simplify_prob,
    )

    coordinate = 0 if project_forward else 1
    marginal: dict[str, Any] = {}
    for pair, mass in joint_pi.items():
        side_state = pair[coordinate]
        if side_state in marginal:
            marginal[side_state] = simplify_prob(as_prob(marginal[side_state]) + as_prob(mass))
        else:
            marginal[side_state] = as_prob(mass)

    side_graph = TransitionGraph()
    for state, mass in marginal.items():
        if not is_positive_mass(mass):
            continue
        attrs: dict[str, Any] = {}
        if future_symbols is not None and state in future_symbols:
            attrs[ATTR_FUTURE_SYMBOL] = future_symbols[state]
        side_graph.add_state(state, **attrs)

    aggregated: dict[tuple[str, str, Any], Any] = {}
    for pair, mass in joint_pi.items():
        source = pair[coordinate]
        source_mass = marginal.get(source, 0)
        if not is_positive_mass(mass) or not is_positive_mass(source_mass):
            continue
        for transition in graph.out_transitions(pair):
            symbol = transition.data.get(ATTR_EMISSION)
            prob = as_prob(transition.data.get(ATTR_PROB, 0.0))
            if symbol is None or not is_positive_mass(prob):
                continue
            target_state = transition.target[coordinate]
            key = (source, target_state, symbol)
            contrib = simplify_prob(as_prob(mass) * as_prob(prob) / as_prob(source_mass))
            if key in aggregated:
                aggregated[key] = simplify_prob(as_prob(aggregated[key]) + contrib)
            else:
                aggregated[key] = contrib

    for (source, target, symbol), prob in aggregated.items():
        if not is_positive_mass(prob):
            continue
        side_graph.add_transition(
            source,
            target,
            **{ATTR_PROB: as_prob(prob), ATTR_EMISSION: symbol},
        )

    eps = EpsilonMachine(
        graph=side_graph,
        initial_distribution=marginal,
        observation_alphabet=frozenset({0, 1}),
    )
    eps.validate_stochastic()
    return eps


def tent_map_misiurewicz_bidirectional_fig8(a: Any | None = None):
    """Hand-built supplement Fig.~8 bidirectional ε-machine."""
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
    from sofic.generators.prob import as_prob, is_symbolic

    if a is None:
        a = tent_map_misiurewicz_a()
    forward = tent_map_misiurewicz_forward(a)
    graph = TransitionGraph()
    for source, target, symbol, prob in _tent_map_misiurewicz_fig8_edges(a):
        if not graph.has_state(source):
            graph.add_state(source)
        if not graph.has_state(target):
            graph.add_state(target)
        graph.add_transition(
            source,
            target,
            **{ATTR_PROB: as_prob(prob), ATTR_EMISSION: symbol},
        )
    joint_pi = _stationary_distribution_from_joint_graph(graph)
    reverse_raw = _project_bidirectional_side(
        graph,
        joint_pi,
        project_forward=False,
        future_symbols={"E": 0, "F": 1, "G": 1, "H": -1},
    )
    if is_symbolic(a):
        from sofic.generators.epsilon_machine import _row_normalized_presentation

        reverse = _row_normalized_presentation(reverse_raw)
    else:
        try:
            reverse = EpsilonMachine.from_hmm(reverse_raw)
        except Exception:
            from sofic.generators.epsilon_machine import _row_normalized_presentation

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


def tent_map_misiurewicz_information_expected(a: Any | None = None) -> dict[str, Any]:
    """Closed-form anatomy rates from James et al. (2013), supplement.

    Returns floats when ``a`` is numeric, or sympy expressions when ``a`` is
    symbolic.  The ephemeral rate is
    ``r_μ = (1/4)*(3 - 2/(a+1) - 4/(a+2) + 9/(2a+3))``.
    """
    from sofic.generators.prob import is_symbolic

    if a is None:
        a = tent_map_misiurewicz_a()
    if is_symbolic(a):
        import sympy as sp

        h_mu = sp.log(a, 2)
        r_mu = sp.Rational(1, 4) * (3 - 2 / (a + 1) - 4 / (a + 2) + 9 / (2 * a + 3))
        b_mu = sp.simplify(h_mu - r_mu)
        return {
            "bound_mu": b_mu,
            "ephemeral_mu": sp.simplify(r_mu),
            "entropy_rate": h_mu,
        }
    h_mu = math.log2(a)
    r_mu = 0.25 * (3.0 - 2.0 / (a + 1) - 4.0 / (a + 2) + 9.0 / (2 * a + 3))
    b_mu = h_mu - r_mu
    return {
        "bound_mu": b_mu,
        "ephemeral_mu": r_mu,
        "entropy_rate": h_mu,
    }


def ellison_fig9_reverse() -> EpsilonMachine:
    """Reverse ε-machine from Ellison et al., arXiv:1107.2168, Fig.~9.

    MSP-derived presentation used for Fig.~15 bidirectional pairing via Eq.~(15).
    """
    from sofic.generators.reversal import time_reverse_stochastic

    forward = ellison_fig9_forward()
    reverse = EpsilonMachine.from_hmm(time_reverse_stochastic(forward))
    reverse.validate()
    return reverse


def ellison_fig15_bidirectional():
    """Bidirectional ε-machine M± from Ellison et al., arXiv:1107.2168, Fig.~15.

    Built from the separate forward and reverse presentations in Fig.~9 via
    Eq.~(15) in the same paper.
    """
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine

    return BidirectionalEpsilonMachine.from_pair(
        ellison_fig9_forward(),
        ellison_fig9_reverse(),
    )
