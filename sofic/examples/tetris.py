"""Tetromino randomizer ε-machines.

These constructors model the *stationary* piece process. Game-start transients
— TGM's initial ``ZZZZ`` / ``ZZSS`` history and the first-piece ban on S, Z,
and O — are not part of the stationary law and are omitted.

TGM3's 35-pool drought randomizer is omitted: its state (pool occupancy, a
7-piece drought order, and a 4-piece history) has millions of configurations.

References
----------
- NES spawn algorithm: :cite:`TetrisWikiNES`
- Guideline 7-bag (Random Generator): :cite:`TetrisWikiRandomGenerator`
- TGM history randomizer: :cite:`TetrisWikiTGM`
- Game Boy bitwise-OR randomizer: :cite:`HardDropGameBoy`
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from itertools import combinations, product

import numpy as np

from sofic.examples.processes import _edge_machine
from sofic.generators.epsilon_machine import EpsilonMachine

TETROMINOES: tuple[str, ...] = ("I", "J", "L", "O", "S", "T", "Z")

# Game Boy ROM numbering, L = 0b000 through Z = 0b110.
_GAMEBOY_INDEX: dict[str, int] = {"L": 0, "J": 1, "I": 2, "O": 3, "S": 4, "T": 5, "Z": 6}


def _reroll_emission_probs(
    alphabet: Sequence[str],
    rolls: int,
    rejected: set[str],
) -> dict[str, float]:
    """Emission law for a uniform draw with ``rolls`` attempts.

    Each attempt is uniform on ``alphabet``. A draw in ``rejected`` is thrown
    out unless it is the last attempt, which is always accepted.
    """
    n = len(alphabet)
    uniform = 1.0 / n
    if rolls <= 1 or not rejected or len(rejected) == n:
        return dict.fromkeys(alphabet, uniform)
    rho = len(rejected) / n
    p_rejected = uniform * (rho ** (rolls - 1))
    p_free = uniform * (1.0 - rho**rolls) / (1.0 - rho)
    return {piece: p_rejected if piece in rejected else p_free for piece in alphabet}


def _stationary_from_edges(
    states: Sequence[Hashable],
    edges: Sequence[tuple[Hashable, Hashable, object, float]],
    *,
    tol: float = 1e-14,
    max_iter: int = 100_000,
) -> dict[Hashable, float]:
    """Left-stationary distribution by sparse power iteration."""
    n = len(states)
    if n == 0:
        return {}
    index = {state: i for i, state in enumerate(states)}
    outgoing: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for source, target, _symbol, prob in edges:
        outgoing[index[source]].append((index[target], float(prob)))

    pi = np.full(n, 1.0 / n)
    nxt = np.empty(n)
    for _ in range(max_iter):
        nxt.fill(0.0)
        for i, mass in enumerate(pi):
            if mass == 0.0:
                continue
            for j, prob in outgoing[i]:
                nxt[j] += mass * prob
        total = float(nxt.sum())
        if total <= 0.0:
            break
        nxt /= total
        if float(np.max(np.abs(nxt - pi))) < tol:
            pi = nxt
            break
        pi = nxt.copy()
    return {state: float(pi[i]) for i, state in enumerate(states)}


def _history_machine(
    window: int,
    rolls: int,
    *,
    name: str,
    alphabet: Sequence[str] = TETROMINOES,
) -> EpsilonMachine:
    states = list(product(alphabet, repeat=window))
    edges: list[tuple[Hashable, Hashable, str, float]] = []
    for history in states:
        rejected = set(history)
        suffix = history[1:]
        for piece, prob in _reroll_emission_probs(alphabet, rolls, rejected).items():
            if prob > 0.0:
                edges.append((history, (*suffix, piece), piece, prob))
    initial = _stationary_from_edges(states, edges)
    return _edge_machine(
        edges,
        name=name,
        initial_distribution=initial,
        normalize=False,
    )


def tetris_iid() -> EpsilonMachine:
    """Memoryless uniform draw over the seven tetrominoes.

    The original 1984 Electronika 60 game, and many early ports, sample each
    piece independently. Entropy rate is ``log2(7)``.
    """
    state = "A"
    mass = 1.0 / len(TETROMINOES)
    return _edge_machine(
        [(state, state, piece, mass) for piece in TETROMINOES],
        name="Tetris IID",
        initial_distribution={state: 1.0},
        normalize=False,
    )


def tetris_nes() -> EpsilonMachine:
    """Idealized NES Tetris randomizer.

    The first roll is uniform on eight values (the seven pieces plus a dummy).
    A dummy or a repeat of the previous piece triggers a second roll, uniform
    on the seven pieces :cite:`TetrisWikiNES`. With previous piece ``prev``,

    - ``P(prev | prev) = 1/28``
    - ``P(p | prev) = 9/56`` for ``p ≠ prev``

    This is the intended "vaguely avoids duplicates" model. The 6502
    implementation also folds in a spawn-count modulo, which biases the
    second roll; that 49-state machine is not reproduced here.
    """
    # First roll fails with probability 2/8 (dummy or previous), then the
    # second roll is uniform on 7, so a repeat has probability (1/4)*(1/7).
    p_repeat = 1.0 / 28.0
    p_other = 9.0 / 56.0
    edges = []
    for prev in TETROMINOES:
        for piece in TETROMINOES:
            prob = p_repeat if piece == prev else p_other
            edges.append((prev, piece, piece, prob))
    return _edge_machine(edges, name="Tetris NES", normalize=False)


def tetris_bag() -> EpsilonMachine:
    """Guideline 7-bag (Random Generator) :cite:`TetrisWikiRandomGenerator`.

    Causal state is the remaining subset of the current bag (``2^7 - 1 = 127``
    nonempty subsets), stored as a sorted tuple. The last piece of a bag
    refills to a fresh permutation of all seven. Entropy rate is
    ``log2(7!)/7``.
    """
    alphabet = TETROMINOES
    full = alphabet
    edges: list[tuple[Hashable, Hashable, str, float]] = []
    for k in range(1, len(alphabet) + 1):
        for remaining in combinations(alphabet, k):
            if k == 1:
                edges.append((remaining, full, remaining[0], 1.0))
                continue
            mass = 1.0 / k
            for piece in remaining:
                nxt = tuple(symbol for symbol in remaining if symbol != piece)
                edges.append((remaining, nxt, piece, mass))
    return _edge_machine(edges, name="Tetris 7-bag", normalize=False)


def tetris_history(window: int = 4, rolls: int = 4) -> EpsilonMachine:
    """TGM-style history randomizer :cite:`TetrisWikiTGM`.

    The state is the ordered window of the last ``window`` pieces. Each of
    ``rolls`` attempts draws uniformly from the seven tetrominoes; a draw that
    appears in the window is rejected unless it is the last attempt.

    ``window=4, rolls=4`` is TGM1; ``window=4, rolls=6`` is TGM2 / TAP. The
    chain is a Markov process of order ``window`` on ``7**window`` states.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if rolls < 1:
        raise ValueError("rolls must be >= 1")
    return _history_machine(window, rolls, name=f"Tetris history({window}, {rolls})")


def tetris_tgm() -> EpsilonMachine:
    """Tetris The Grand Master (TGM1): 4-piece history, 4 rolls."""
    return _history_machine(4, 4, name="Tetris TGM")


def tetris_tgm2() -> EpsilonMachine:
    """Tetris The Absolute The Grand Master 2: 4-piece history, 6 rolls."""
    return _history_machine(4, 6, name="Tetris TGM2")


def tetris_gameboy() -> EpsilonMachine:
    """Game Boy Tetris (1989) bitwise-OR randomizer :cite:`HardDropGameBoy`.

    State is the last two pieces ``(locking, preview)``. A candidate is
    accepted when ``(locking | preview | candidate) != locking`` in the ROM's
    piece numbering (L=0, …, Z=6). Up to three rolls; the third is always
    taken. Intended to suppress three-in-a-row of the same piece; the bitwise
    test makes L the rarest piece and O/S/T the most common.
    """
    alphabet = TETROMINOES
    rolls = 3
    edges: list[tuple[Hashable, Hashable, str, float]] = []
    for locking, preview in product(alphabet, repeat=2):
        rejected = {
            piece
            for piece in alphabet
            if (_GAMEBOY_INDEX[locking] | _GAMEBOY_INDEX[preview] | _GAMEBOY_INDEX[piece]) == _GAMEBOY_INDEX[locking]
        }
        for piece, prob in _reroll_emission_probs(alphabet, rolls, rejected).items():
            if prob > 0.0:
                edges.append(((locking, preview), (preview, piece), piece, prob))
    return _edge_machine(edges, name="Tetris Game Boy", normalize=False)


__all__ = [
    "TETROMINOES",
    "tetris_bag",
    "tetris_gameboy",
    "tetris_history",
    "tetris_iid",
    "tetris_nes",
    "tetris_tgm",
    "tetris_tgm2",
]
