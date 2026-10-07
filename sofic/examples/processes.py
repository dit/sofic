"""cmpy-compatible process constructors.

This module ports the public constructors from ``cmpy.machines.processes`` to
sofic-native objects.  Quantum and elementary-cellular-automaton helpers are
intentionally out of scope.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np

from sofic.automata.transducers import MealyMachine
from sofic.examples._construction import _edge_machine
from sofic.examples.epsilon_machines import (
    bernoulli,
    even_process,
    golden_mean,
)
from sofic.generators.base import QuasiStochasticModel
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_EMISSION, ATTR_OUTPUT, ATTR_PROB, ATTR_QUASIPROB, ATTR_SYMBOL


def _require_machine_type(machine_type: Any, *allowed: type) -> None:
    """Raise ``NotImplementedError`` unless ``machine_type`` is ``None`` or allowed.

    ``None`` always selects the constructor's default output type, so it is
    accepted implicitly alongside the explicitly-supported classes.
    """
    if machine_type not in (*allowed, None):
        raise NotImplementedError


def _as_alphabet(symbols: int | Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(symbols, int):
        return tuple(str(i) for i in range(symbols))
    return tuple(symbols)


def _from_string(
    spec: str,
    *,
    machine_type: type[MealyHMM] = EpsilonMachine,
    name: str | None = None,
    normalize: bool = True,
) -> MealyHMM:
    edges = []
    for chunk in spec.replace("\n", " ").split(";"):
        fields = chunk.split()
        if not fields:
            continue
        if len(fields) not in (3, 4):
            raise ValueError(f"invalid transition specification {chunk!r}")
        source, target, symbol = fields[:3]
        prob = float(fields[3]) if len(fields) == 4 else 1.0
        edges.append((source, target, symbol, prob))
    return _edge_machine(edges, machine_type=machine_type, name=name, normalize=normalize)


def _word_period(word: Sequence[Any]) -> Sequence[Any]:
    for period in range(1, len(word) + 1):
        base = word[:period]
        if all(word[i] == base[i % period] for i in range(len(word))):
            return base
    return word


def _words(alphabet: Sequence[Any], length: int) -> Iterable[tuple[Any, ...]]:
    return product(alphabet, repeat=length)


def _dirichlet(n: int, rng: Any = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    if hasattr(rng, "dirichlet"):
        return np.asarray(rng.dirichlet(np.ones(n)), dtype=float)
    if hasattr(rng, "rand"):
        values = np.asarray(rng.rand(n), dtype=float)
    else:
        values = np.asarray(np.random.default_rng().random(n), dtype=float)
    return values / values.sum()


def _compatible_machine_type(machine_type: Any, default: type[MealyHMM] = EpsilonMachine) -> type[MealyHMM]:
    if machine_type is None:
        return default
    if machine_type in (EpsilonMachine, MealyHMM):
        return machine_type
    if isinstance(machine_type, str):
        lowered = machine_type.lower()
        if lowered in {"epsilon", "em", "recurrentepsilonmachine"}:
            return EpsilonMachine
        if lowered in {"mealy", "mealyhmm"}:
            return MealyHMM
    return default


def afc(n: int) -> EpsilonMachine:
    if n < 1:
        raise ValueError("n >= 1 required")
    spec = "B B 0; "
    for i in range(1, n):
        spec += f"A{i} A{i + 1} 1; A{i + 1} B 0;"
    spec += "B A1 1; A1 B 0;"
    return _from_string(spec, name=f"Almost Fair Coin Process, order {n}")


def afc2(n: int) -> EpsilonMachine:
    if n < 1:
        raise ValueError("n >= 1 required")
    spec = "A1 B1 1; B1 A1 0; "
    for i in range(1, n):
        spec += f"A{i} A{i + 1} 0; A{i + 1} B1 1; B{i} B{i + 1} 1; B{i + 1} A1 0; "
    return _from_string(spec, name=f"Almost Fair Coin 2 Process, order {n}")


def band_merging(machine_type: Any = MealyHMM) -> MealyHMM:
    cls = _compatible_machine_type(machine_type, MealyHMM)
    if cls is EpsilonMachine:
        edges = [
            ("A", "B", "1", 1),
            ("B", "A", "0", 0.5),
            ("B", "A", "1", 0.5),
            ("T1", "A", "0", 0.25),
            ("T1", "T2", "1", 0.75),
            ("T2", "A", "0", 1 / 3),
            ("T2", "T1", "1", 2 / 3),
            ("E", "E", "1", math.sqrt(2) / 2),
            ("E", "A", "0", 1 - math.sqrt(2) / 2),
        ]
        return _edge_machine(edges, machine_type=EpsilonMachine, name="Band Merging Process", normalize=False)
    return _edge_machine(
        [("A", "B", "1", 1), ("B", "A", "0", 0.5), ("B", "A", "1", 0.5)],
        machine_type=MealyHMM,
        name="Band Merging Process",
        normalize=False,
    )


def beads_on_necklace(
    machine_type: Any = MealyHMM,
    beads: Sequence[Any] | Sequence[Sequence[Any]] = ("a", "b"),
    necklace: Sequence[Any] = ("0", "1", "3"),
    probs: float | Sequence[float] = 0.5,
) -> MealyHMM:
    _require_machine_type(machine_type, MealyHMM)
    if not necklace or not beads:
        raise ValueError("beads and necklace are required")
    bead_rows: list[Sequence[Any]] = (
        list(beads) if isinstance(beads[0], (list, tuple)) else [beads for _ in range(len(necklace))]
    )  # type: ignore[index,arg-type,list-item]
    if len(necklace) != len(bead_rows):
        raise ValueError("necklace and beads must have the same length")
    prob_rows = [float(probs)] * len(necklace) if isinstance(probs, (int, float)) else [float(p) for p in probs]
    if len(prob_rows) != len(necklace):
        raise ValueError("necklace and probs must have the same length")

    edges = []
    curr_state = 0
    for j, bead in enumerate(bead_rows):
        attach_state = curr_state
        bead = tuple(bead)
        if len(bead) == 0:
            prob_rows[j] = 1.0
        elif len(bead) == 1:
            edges.append((curr_state, curr_state, bead[0], 1.0 - prob_rows[j]))
        else:
            for i, symbol in enumerate(bead):
                if i == 0:
                    next_state = curr_state + 1
                    edges.append((curr_state, next_state, symbol, 1.0 - prob_rows[j]))
                    curr_state += 1
                elif i == len(bead) - 1:
                    edges.append((curr_state, attach_state, symbol, 1.0))
                else:
                    next_state = curr_state + 1
                    edges.append((curr_state, next_state, symbol, 1.0))
                    curr_state += 1
        new_attach_state = 0 if j == len(bead_rows) - 1 else curr_state + 1
        edges.append((attach_state, new_attach_state, necklace[j], prob_rows[j]))
        curr_state = new_attach_state
    return _edge_machine(edges, machine_type=MealyHMM, name="Beads On Necklace", normalize=False)


def before_after(machine_type: Any = MealyHMM, style: str = "simple") -> MealyHMM:
    _require_machine_type(machine_type, MealyHMM)
    if style == "simple":
        edges = [("A", "A", "0", 0.5), ("A", "B", "1", 0.5), ("B", "B", "0", 0.5), ("B", "A", "2", 0.5)]
    elif style == "cayley":
        a, b = 0.6, 0.4
        edges = [
            ("a", "a", "0", a),
            ("a", "ca", "2", b / 2),
            ("a", "ba", "1", b / 2),
            ("ca", "ca", "0", a),
            ("ca", "bca", "1", b),
            ("bca", "bca", "0", a),
            ("bca", "ca", "2", b),
            ("ba", "ba", "0", a),
            ("ba", "cba", "2", b),
            ("cba", "cba", "0", a),
            ("cba", "ba", "1", b),
        ]
    else:
        raise ValueError("style must be 'simple' or 'cayley'")
    return _edge_machine(edges, machine_type=MealyHMM, name="before_after", normalize=False)


def random_biased_coin(machine_type: Any = EpsilonMachine, rng: np.random.Generator | None = None) -> EpsilonMachine:
    generator = rng if rng is not None else np.random.default_rng()
    bias = float(generator.random())
    return _coin(bias, _compatible_machine_type(machine_type))


def _coin(p: float, machine_type: type[MealyHMM] = EpsilonMachine) -> MealyHMM:
    """Single-state coin with ``P("1") = p``."""
    return _edge_machine(
        [("A", "A", "0", 1 - p), ("A", "A", "1", p)],
        machine_type=machine_type,
        name=f"Coin, p = {p}",
        normalize=False,
    )


def iid(k: int | Sequence[Any], machine_type: Any = EpsilonMachine) -> EpsilonMachine:
    alphabet = _as_alphabet(k)
    return _edge_machine(
        [("A", "A", symbol, 1.0) for symbol in alphabet],
        machine_type=_compatible_machine_type(machine_type),
        name=f"iid({len(alphabet)})",
        normalize=True,
    )


def binary_markov_chain(
    p: float = 0.4,
    q: float = 0.3,
    a: float | None = None,
    b: float | None = None,
    machine_type: Any = EpsilonMachine,
    branch: Any = None,
) -> MealyHMM:
    if machine_type in (EpsilonMachine, None):
        return bmc_em(p, q)
    if isinstance(machine_type, str):
        lowered = machine_type.lower()
        if lowered == "generative":
            return bmc_gen(p, q, branch=branch)
        if lowered == "parametrized":
            if a is None or b is None:
                raise ValueError("a and b are required for parametrized binary_markov_chain")
            return bmc_param(p, q, a, b)
        if lowered == "lohr":
            return bmc_lohr(p)
    raise NotImplementedError


def bmc_em(p: float, q: float) -> EpsilonMachine:
    if math.isclose(p, 1 - q):
        return _coin(p)
    return _edge_machine(
        [("A", "A", "0", 1 - p), ("A", "B", "1", p), ("B", "A", "0", q), ("B", "B", "1", 1 - q)],
        machine_type=EpsilonMachine,
        name="BinaryMarkovChain eM",
        normalize=False,
    )


def bmc_gen(p: float, q: float, branch: Any = None) -> MealyHMM:
    del branch
    if p == 1 and q == 1:
        spec = "A B 1 1.; B A 0 1."
    elif p == 0 and q == 0:
        spec = "A A 0 1.; B B 1 1."
    elif p + q > 1:
        spec = f"A B 0 1; B A 1 {p + q - 1}; B B 0 {1 - p}; B B 1 {1 - q}"
    elif p < q:
        spec = (
            f"A A 1 {(1 - p - q) / (1 - p)}; A B 1 {q / (1 - p)}; "
            f"B A 1 {p * (1 - p - q) / (1 - p)}; B B 0 {1 - p}; B B 1 {p * q / (1 - p)}"
        )
    else:
        spec = (
            f"A A 0 {(1 - p - q) / (1 - q)}; A B 0 {p / (1 - q)}; "
            f"B A 0 {q * (1 - p - q) / (1 - q)}; B B 0 {p * q / (1 - q)}; B B 1 {1 - q}"
        )
    return _from_string(spec, machine_type=MealyHMM, name="BinaryMarkovChain gen", normalize=False)


def _bmc_param_get_a_range(p: float, q: float) -> list[float]:
    return [0, min(q, 1 - p)]


def _bmc_param_get_b_range(p: float, q: float) -> list[float]:
    return [max(q, 1 - p), 1]


def _bmc_param_check_a_range(p: float, q: float, a: float) -> bool:
    low, high = _bmc_param_get_a_range(p, q)
    return low <= a <= high


def _bmc_param_check_b_range(p: float, q: float, b: float) -> bool:
    low, high = _bmc_param_get_b_range(p, q)
    return low <= b <= high


def bmc_param(p: float, q: float, a: float, b: float) -> MealyHMM:
    if not _bmc_param_check_a_range(p, q, a) or not _bmc_param_check_b_range(p, q, b):
        raise ValueError("a or b is outside the allowed range")
    probs = [
        a * (b + p - 1) / (b - a),
        (1 - a) * (b - q) / (b - a),
        a * (1 - p - a) / (b - a),
        (1 - a) * (q - a) / (b - a),
        b * (b + p - 1) / (b - a),
        (1 - b) * (b - q) / (b - a),
        b * (1 - p - a) / (b - a),
        (1 - b) * (q - a) / (b - a),
    ]
    edges = [
        ("A", "A", "0", probs[0]),
        ("A", "A", "1", probs[1]),
        ("A", "B", "0", probs[2]),
        ("A", "B", "1", probs[3]),
        ("B", "A", "0", probs[4]),
        ("B", "A", "1", probs[5]),
        ("B", "B", "0", probs[6]),
        ("B", "B", "1", probs[7]),
    ]
    return _edge_machine(edges, machine_type=MealyHMM, name="BinaryMarkovChain parametrized", normalize=False)


def bmc_lohr(p: float) -> MealyHMM:
    if p > 0.5:
        raise ValueError("currently requires p <= 0.5")
    return _edge_machine(
        [
            ("0", "0", "0", 1 - 2 * p),
            ("0", "2", "0", 2 * p),
            ("1", "1", "1", 1 - 2 * p),
            ("1", "2", "1", 2 * p),
            ("2", "0", "0", 0.5 - p),
            ("2", "1", "1", 0.5 - p),
            ("2", "2", "0", p),
            ("2", "2", "1", p),
        ],
        machine_type=MealyHMM,
        name="BinaryMarkovChain Lohr",
        normalize=False,
    )


def cantor(machine_type: Any = MealyHMM) -> MealyHMM:
    _require_machine_type(machine_type, MealyHMM)
    edges = [
        ("A", "A", "0", 0.55),
        ("B", "A", "0", 0.30),
        ("B", "B", "0", 0.15),
        ("A", "A", "1", 0.15),
        ("A", "B", "1", 0.30),
        ("B", "B", "1", 0.55),
    ]
    return _edge_machine(edges, machine_type=MealyHMM, name="Cantor Process", normalize=False)


def coupled_gmps(
    epsilon: float = 0.01, p: float = 0.5, alt: bool = True, machine_type: Any = EpsilonMachine
) -> EpsilonMachine:
    """Two golden-mean components joined by ``epsilon``-weak transitions.

    The ``C``/``D`` component is :func:`~sofic.examples.golden_mean` (forbid
    ``11``); the ``A``/``B`` component is its ``0 <-> 1`` mirror (forbid ``00``),
    so the two components are distinguishable.
    """
    if epsilon < 0 or p < 0:
        raise ValueError("epsilon and p cannot be less than zero")
    if alt:
        q1 = epsilon * (1 - p)
        q2 = epsilon * p
        edges = [
            ("A", "A", "1", 0.5),
            ("A", "B", "0", 0.5),
            ("B", "A", "1", 1 - q1),
            ("B", "C", "0", q1),
            ("C", "C", "0", 0.5),
            ("C", "D", "1", 0.5),
            ("D", "C", "0", 1 - q2),
            ("D", "A", "1", q2),
        ]
    else:
        q1 = epsilon * (1 - p)
        q2 = epsilon * p
        edges = [
            ("A", "A", "1", (1 - q1) / 2),
            ("A", "B", "0", (1 - q1) / 2),
            ("B", "A", "1", 1),
            ("C", "C", "0", (1 - q2) / 2),
            ("C", "D", "1", (1 - q2) / 2),
            ("D", "C", "0", 1),
        ]
        if q1:
            edges.append(("A", "C", "2", q1))
        if q2:
            edges.append(("C", "A", "2", q2))
    initial = None
    if epsilon == 0:
        initial = {"A": p * 2 / 3, "B": p * 1 / 3, "C": (1 - p) * 2 / 3, "D": (1 - p) * 1 / 3}
    return _edge_machine(
        edges,
        machine_type=_compatible_machine_type(machine_type),
        name="Coupled Golden Mean Processes",
        initial_distribution=initial,
        normalize=False,
    )


def uncoupled_gmps(p: float = 0.5, machine_type: Any = EpsilonMachine) -> EpsilonMachine:
    """Mixture of the golden mean (forbid ``11``, weight ``1 - p``) and its ``0 <-> 1`` mirror (weight ``p``)."""
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    return _edge_machine(
        [
            ("A", "A", "1", 0.5),
            ("A", "B", "0", 0.5),
            ("B", "A", "1", 1),
            ("C", "C", "0", 0.5),
            ("C", "D", "1", 0.5),
            ("D", "C", "0", 1),
        ],
        machine_type=_compatible_machine_type(machine_type),
        name="Uncoupled Golden Mean Processes",
        initial_distribution={"A": p * 2 / 3, "B": p * 1 / 3, "C": (1 - p) * 2 / 3, "D": (1 - p) * 1 / 3},
        normalize=False,
    )


def cyclic_branching(num_states: int, num_branchings: int, num_symbols: int = 2) -> MealyHMM:
    if num_branchings >= num_states:
        raise ValueError("number of branchings must be less than number of states")
    edges = []
    for x in range(num_states):
        y = 0 if x == num_states - 1 else x + 1
        symbol = "1" if x == num_states - 1 else "0"
        if x >= num_states - num_branchings:
            for s in range(num_symbols):
                edges.append((x, y, str(s), 1 / num_symbols))
        else:
            edges.append((x, y, symbol, 1))
    return _edge_machine(
        edges,
        machine_type=MealyHMM,
        name=f"Noisy Period-{num_states} with {num_branchings} branchings",
        normalize=False,
    )


def ehrenfest(p: float = 0.5, N: int = 5, machine_type: Any = EpsilonMachine) -> EpsilonMachine:
    _require_machine_type(machine_type, EpsilonMachine)
    edges = []
    for state in range(N + 1):
        edges.append((state, state, str(state), 1 - p))
    for state in range(N):
        edges.append((state, state + 1, str(state + 1), p * (N - state) / N))
    for state in range(1, N + 1):
        edges.append((state, state - 1, str(state - 1), p * state / N))
    return _edge_machine(edges, machine_type=EpsilonMachine, name="ehrenfest", normalize=False)


def random_even(machine_type: Any = EpsilonMachine, rng: np.random.Generator | None = None) -> EpsilonMachine:
    return uniform_mealyhmm(
        even_process(),
        name="Random Even Process",
        create_using=_compatible_machine_type(machine_type),
        prng=rng,
    )


def even_redundant(machine_type: Any = EpsilonMachine, bias: float = 0.5) -> EpsilonMachine:
    """Deliberately non-minimal four-state presentation of the Even Process.

    States ``A``/``C`` and ``B``/``D`` are pairwise equivalent, so this is *not*
    an ε-machine despite the return type: it exists to exercise minimization
    (state merging should recover the two-state Even Process).
    """
    _require_machine_type(machine_type, EpsilonMachine)
    return _from_string(
        f"A A 0 {bias}; A B 1 {1 - bias}; B C 1 1.; C C 0 {bias}; C D 1 {1 - bias}; D A 1 1.",
        name="Even Process (4-state)",
    )


def three_even(machine_type: Any = EpsilonMachine) -> EpsilonMachine:
    _require_machine_type(machine_type, EpsilonMachine)
    return _from_string("A A 0 1; A B 1 1; A C 2 1; B A 1 1; C A 2 1;", name="ThreEven Process")


def flower(
    N: int = 4,
    M: int = 3,
    forward_dist: Sequence[float] | None = None,
    reverse_dists: np.ndarray | None = None,
    machine_type: Any = EpsilonMachine,
) -> EpsilonMachine:
    _require_machine_type(machine_type, EpsilonMachine)
    if N < 2 or M < 2:
        raise ValueError("N and M must be at least 2")
    if forward_dist is None:
        forward_dist = [1.0 / (N - 1)] * (N - 1)
    if reverse_dists is None:
        reverse_dists = np.vstack([_dirichlet(M - 1) for _ in range(N - 1)])
    edges = []
    for i in range(N - 1):
        petal = str(i + 1)
        edges.append(("0", petal, petal, float(forward_dist[i])))
        for j in range(M - 1):
            edges.append((petal, "0", str(N + j), float(reverse_dists[i, j])))
    return _edge_machine(edges, machine_type=EpsilonMachine, name="Flower Process", normalize=False)


def four_state_almost_iid(delta: float = 0.2) -> EpsilonMachine:
    delta = max(0.0, min(float(delta), 0.24))
    p, q, r, s = 0.50 - 2 * delta, 0.50 - delta, 0.50 + delta, 0.50 + 2 * delta
    spec = f"A A 0 {p}; A B 1 {1 - p}; B C 0 {q}; B C 1 {1 - q}; C A 0 {r}; C D 1 {1 - r}; D A 0 {s}; D D 1 {1 - s};"
    return _from_string(spec, name="FourStateAlmostIID Process")


def girvan_fig6b(machine_type: Any = EpsilonMachine, alpha: float = 0.5, pi: float = 0.4) -> EpsilonMachine:
    return _edge_machine(
        [("A", "A", "1", alpha), ("A", "P", "0", 1 - alpha), ("P", "A", "1", 1 - pi), ("P", "P", "0", pi)],
        machine_type=_compatible_machine_type(machine_type),
        name="girvan_fig6b",
        normalize=False,
    )


def girvan_fig6c(
    machine_type: Any = EpsilonMachine, alpha: float = 0.5, pi: float = 0.4, rho: float = 0.3
) -> EpsilonMachine:
    return _edge_machine(
        [
            ("A", "A", "1", alpha),
            ("A", "P", "0", 1 - alpha),
            ("P", "A", "1", 1 - pi),
            ("P", "R", "0", pi),
            ("R", "R", "0", rho),
            ("R", "A", "1", 1 - rho),
        ],
        machine_type=_compatible_machine_type(machine_type),
        name="girvan_fig6c",
        normalize=False,
    )


def girvan_fig6d(
    machine_type: Any = EpsilonMachine,
    alpha: float = 0.5,
    pi: float = 0.4,
    rho: float = 0.3,
    iota: float = 0.2,
) -> EpsilonMachine:
    return _edge_machine(
        [
            ("A", "I", "1", alpha),
            ("A", "P", "0", 1 - alpha),
            ("P", "A", "1", 1 - pi),
            ("P", "R", "0", pi),
            ("R", "R", "0", rho),
            ("R", "A", "1", 1 - rho),
            ("I", "I", "1", iota),
            ("I", "P", "0", 1 - iota),
        ],
        machine_type=_compatible_machine_type(machine_type),
        name="girvan_fig6d",
        normalize=False,
    )


def stretched_gm(k: int) -> EpsilonMachine:
    """Golden mean with each ``1`` stretched to a block of exactly ``k`` ones; ``k = 1`` forbids ``11``."""
    if k <= 0:
        raise ValueError("minimum k is 1")
    spec = "0 0 0 0.5; 0 1 1 0.5;"
    for kk in range(2, k + 1):
        spec += f"{kk - 1} {kk} 1 1.0;"
    spec += f"{k} 0 0 1.0"
    return _from_string(spec, name=f"Stretched Golden Mean Process, k={k}")


def rn_gm(R: int, N: int, p: float = 0.5) -> EpsilonMachine:
    """R-N golden mean: from the ``0`` self-loop, a ``1`` starts ``N`` ones then ``R`` zeros (forbids ``11`` at ``N = 1``)."""
    if R <= 0 or N <= 0 or R < N:
        raise ValueError("requires 1 <= N <= R")
    spec = f"0 0 0 {p}; 0 1 1 {1 - p};"
    for kk in range(2, N + 1):
        spec += f"{kk - 1} {kk} 1 1.0;"
    for kk in range(N + 1, N + R):
        spec += f"{kk - 1} {kk} 0 1.0;"
    spec += f"{N + R - 1} 0 0 1.0"
    return _from_string(spec, name=f"R-N Golden Mean Process, R={R} N={N}")


def rk_gm(R: int, k: int, p: float = 0.5) -> EpsilonMachine:
    """R-k golden mean: from the ``0`` self-loop, a ``1`` starts ``R`` ones then ``k`` zeros (Markov order ``R + k - 1``)."""
    if R <= 0 or k <= 0:
        raise ValueError("R and k must be positive")
    spec = f"0 0 0 {p}; 0 1 1 {1 - p};"
    for kk in range(2, R + 1):
        spec += f"{kk - 1} {kk} 1 1.0;"
    for kk in range(R + 1, R + k):
        spec += f"{kk - 1} {kk} 0 1.0;"
    spec += f"{R + k - 1} 0 0 1.0"
    return _from_string(spec, name=f"R-k Golden Mean Process, R={R} k={k}")


def random_golden_mean(machine_type: Any = EpsilonMachine, rng: np.random.Generator | None = None) -> EpsilonMachine:
    """:func:`~sofic.examples.golden_mean` topology (forbid ``11``) with random transition probabilities."""
    return uniform_mealyhmm(
        golden_mean(),
        name="Random Golden Mean Process",
        create_using=_compatible_machine_type(machine_type),
        prng=rng,
    )


def golden_mean_ghmm() -> QuasiStochasticModel:
    """Golden mean process (forbid ``11``, ``p = 1/2``) as a quasi-stochastic model with negative weights."""
    q = QuasiStochasticModel(initial_quasidistribution={"A": 4 / 3, "B": -1 / 3})
    q.observation_alphabet = frozenset({"0", "1"})
    q.name = "Golden Mean Process"
    for state in ("A", "B"):
        q.graph.add_state(state)
    for source, target, symbol, prob in [
        ("A", "A", "1", 1.0),
        ("A", "B", "1", -0.5),
        ("B", "A", "1", 2.0),
        ("B", "B", "1", -1.0),
        ("A", "A", "0", 0.5),
    ]:
        q.graph.add_transition(source, target, **{ATTR_EMISSION: symbol, ATTR_QUASIPROB: prob})
    q.validate()
    return q


def nonunifilar_golden_mean(bias: float = 0.5, free: float = 2 / 3) -> MealyHMM:
    """Non-unifilar presentation of :func:`~sofic.examples.golden_mean` ``(bias)`` (forbid ``11``).

    ``free`` is the one free transition weight of the two-state family.
    """
    pGM = bias
    pA = 1 / (1 + pGM)
    pB = pGM / (1 + pGM)
    p0 = pA * pGM
    p010 = pA * pGM**2
    p_min = pGM
    p_max = min(1, pA / pB / pGM)
    if not p_min <= free <= p_max:
        raise ValueError(f"free must be in [{p_min:.2f}, {p_max:.2f}]")
    tab0 = free
    tab1 = p010 / p0**2 - tab0 - p010 / p0 / tab0
    tba1 = p010 / p0 / tab0
    taa1 = 1 - tab1 - tab0
    tbb1 = 1 - tba1
    edges = []
    for edge in [
        ("A", "B", "1", tab0),
        ("A", "B", "0", tab1),
        ("B", "A", "0", tba1),
        ("A", "A", "0", taa1),
        ("B", "B", "0", tbb1),
    ]:
        if not math.isclose(edge[3], 0.0):
            edges.append(edge)
    return _edge_machine(edges, machine_type=MealyHMM, name=f"Nonunifilar Golden Mean, p = {bias:.2f}", normalize=False)


def irreversible_two_state(p: float = 0.5, q: float = 0.5, machine_type: Any = EpsilonMachine) -> EpsilonMachine:
    """Two-state irreversible process over ``"0"``, ``"1"``, ``"2"``.

    ``A`` emits ``0`` and stays (probability ``p``) or emits ``1`` into ``B``;
    ``B`` emits ``1`` and stays (probability ``q``) or emits ``2`` back to ``A``.
    At the default ``p = q = 1/2`` this is the forward ε-machine of Ellison,
    Mahoney, James & Crutchfield, arXiv:1107.2168, Fig.~9 :cite:`Ellison2011`;
    see :func:`~sofic.examples.ellison_fig9_reverse` for its reverse machine.
    """
    return _edge_machine(
        [("A", "A", "0", p), ("A", "B", "1", 1 - p), ("B", "B", "1", q), ("B", "A", "2", 1 - q)],
        machine_type=_compatible_machine_type(machine_type),
        name="IrrevTwoState",
        normalize=False,
    )


def ising(machine_type: Any = EpsilonMachine, J: float = 1.0, B: float = 0.3, T: float = 1.0) -> EpsilonMachine:
    _require_machine_type(machine_type, EpsilonMachine)
    beta = 1.0 / T
    rad = (np.sinh(beta * B) ** 2 + np.exp(-4 * beta * J)) ** 0.5
    two_p = 1.0 - (2 * np.exp(-4 * beta * J)) / (rad * (np.cosh(beta * B) + rad))
    one_p = np.sinh(beta * B) / rad
    p = (two_p / 2 + one_p + 0.5) / (one_p + 1)
    q = (two_p / 2 - one_p + 0.5) / (1 - one_p)
    if not np.all(np.isfinite([p, q])):
        raise FloatingPointError("non-finite Ising transition probability")
    return _edge_machine(
        [("A", "A", "0", p), ("A", "B", "1", 1 - p), ("B", "B", "1", q), ("B", "A", "0", 1 - q)],
        machine_type=EpsilonMachine,
        name="Ising Process",
        normalize=False,
    )


def lollipop(
    N: int, M: int, p: float = 0.5, q: float = 0.5, r: float = 0.1, machine_type: Any = EpsilonMachine
) -> EpsilonMachine:
    _require_machine_type(machine_type, EpsilonMachine)
    hns = [str(ind) for ind in range(N)]
    sns = [str(ind) for ind in range(N, N + 2 * (M - 1) + 1)]
    edges = []
    for ind in range(N - 1):
        prob = 1 - p if ind == 0 else 1 - q if ind == 1 else 1
        edges.append((hns[ind], hns[ind + 1], "0", prob))
    edges.append((hns[N - 1], hns[0], "0", 1 - p if N == 2 else 1))
    for ind in range(M - 1):
        edges.append((hns[0] if ind == 0 else sns[ind - 1], sns[ind], "1", p if ind == 0 else 1))
    for ind in range(M - 1, 2 * (M - 1)):
        edges.append((hns[1] if ind == M - 1 else sns[ind - 1], sns[ind], "1", q if ind == M - 1 else 1))
    last = 2 * (M - 1)
    edges.append((sns[M - 2], sns[last], "1", 1))
    edges.append((sns[2 * (M - 1) - 1], sns[last], "1", 1 - r))
    edges.append((sns[2 * (M - 1) - 1], sns[last], "0", r))
    edges.append((sns[last], hns[0], "2", 1))
    return _edge_machine(edges, machine_type=EpsilonMachine, name="lollipop", normalize=False)


def logic_machine(
    logic: str, bias: float | Sequence[float] = 0.5, noise: float | Sequence[float] = 0.5, minimize: bool = True
) -> MealyHMM:
    del minimize
    if logic == "rrx":
        return rrx(machine_type=MealyHMM)
    if logic == "rn1c":
        return rn1c(noise=float(np.atleast_1d(noise)[0]), bias=float(np.atleast_1d(bias)[0]))
    raise NotImplementedError("logic_machine currently supports the common 'RRX' and 'Rn1C' logics")


def markov_skeleton(R: int, k: int | Sequence[Any], join: bool | None = None) -> MealyHMM:
    alphabet = _as_alphabet(k)
    if join is None:
        join = len(alphabet) <= 10
    if join:
        alphabet = tuple(map(str, alphabet))
        if any(len(symbol) > 1 for symbol in alphabet):
            raise ValueError("cannot join symbols with more than one character")
    if R == 0:
        return _edge_machine(
            [("A", "A", symbol, 1.0) for symbol in alphabet], machine_type=MealyHMM, name="Markov skeleton"
        )
    states: list[Hashable] = ["".join(word) if join else word for word in _words(alphabet, R)]
    edges = []
    for state in states:
        for symbol in alphabet:
            target = state[1:] + symbol if join else tuple(state[1:]) + (symbol,)
            edges.append((state, target, symbol, 1.0))
    return _edge_machine(edges, machine_type=MealyHMM, name=f"Order-{R} Markov skeleton")


def misiurewicz(machine_type: Any = MealyHMM) -> MealyHMM:
    _require_machine_type(machine_type, MealyHMM)
    return _edge_machine(
        [
            ("A", "B", "0", 0.364),
            ("B", "C", "0", 0.276),
            ("D", "B", "0", 0.521),
            ("A", "A", "1", 0.636),
            ("B", "A", "1", 0.724),
            ("C", "D", "1", 1),
            ("D", "C", "1", 0.479),
        ],
        machine_type=MealyHMM,
        name="Misiurewicz Process (Forward)",
        normalize=False,
    )


def misiurewicz_simplified(machine_type: Any = MealyHMM) -> MealyHMM:
    _require_machine_type(machine_type, MealyHMM)
    return _edge_machine(
        [
            ("A", "B", "0", 0.4),
            ("B", "C", "0", 0.25),
            ("D", "B", "0", 0.5),
            ("A", "A", "1", 0.6),
            ("B", "A", "1", 0.75),
            ("C", "D", "1", 1),
            ("D", "C", "1", 0.5),
        ],
        machine_type=MealyHMM,
        name="Simplified Misiurewicz Process (Forward)",
        normalize=False,
    )


def misiurewicz_uniform(machine_type: Any = MealyHMM) -> MealyHMM:
    _require_machine_type(machine_type, MealyHMM)
    return _edge_machine(
        [
            ("A", "B", "0", 0.5),
            ("B", "C", "0", 0.5),
            ("D", "B", "0", 0.5),
            ("A", "A", "1", 0.5),
            ("B", "A", "1", 0.5),
            ("C", "D", "1", 1),
            ("D", "C", "1", 0.5),
        ],
        machine_type=MealyHMM,
        name="Uniform Misiurewicz Process (Forward)",
        normalize=False,
    )


def multiple3(machine_type: Any = MealyHMM, bias: float = 0.5) -> MealyHMM:
    return multiple_n(3, machine_type=machine_type, bias=bias)


def multiple4(machine_type: Any = MealyHMM, bias: float = 0.5) -> MealyHMM:
    return multiple_n(4, machine_type=machine_type, bias=bias)


def multiple_n(n: int, machine_type: Any = MealyHMM, bias: float = 0.5) -> MealyHMM:
    _require_machine_type(machine_type, MealyHMM, EpsilonMachine)
    edges = [(0, 0, "0", bias), (0, 1, "1", 1 - bias)]
    for x in range(1, int(n)):
        edges.append((x, 0 if x == n - 1 else x + 1, "1", 1))
    cls = EpsilonMachine if machine_type is EpsilonMachine else MealyHMM
    return _edge_machine(edges, machine_type=cls, name=f"'Multiples of {n}' Process", normalize=False)


def nemo_redundant(machine_type: Any = MealyHMM, p: float = 0.5, q: float = 0.5) -> MealyHMM:
    _require_machine_type(machine_type, MealyHMM)
    return _from_string(
        f"""
        A A 1 {p}; A B 0 {1 - p}; B C 0 1; C D 0 {1 - q}; C D 1 {q};
        D D 1 {p}; D E 0 {1 - p}; E F 0 1; F A 0 {1 - q}; F A 1 {q};
        """,
        machine_type=MealyHMM,
        name="Nemo Process (6-state)",
    )


def noisy_period2(noise: float = 0.5) -> EpsilonMachine:
    return _edge_machine(
        [("A", "B", "0", 1), ("B", "A", "0", noise), ("B", "A", "1", 1 - noise)],
        machine_type=EpsilonMachine,
        name="Noisy Period-2",
        normalize=False,
    )


def odd(machine_type: Any = EpsilonMachine, bias1: float = 0.5, bias2: float = 0.5) -> EpsilonMachine:
    _require_machine_type(machine_type, EpsilonMachine)
    edges = [
        ("A", "A", "0", bias1),
        ("A", "B", "1", 1 - bias1),
        ("B", "A", "0", bias2),
        ("B", "C", "1", 1 - bias2),
        ("C", "B", "1", 1),
    ]
    return _edge_machine(edges, machine_type=EpsilonMachine, name="Odd Process", normalize=False)


def odd_ghmm(variant: int = 1) -> QuasiStochasticModel:
    matrices = (
        {"0": [[0.5, 0, 0], [1.0, 0, 0], [1.0, 0, 0]], "1": [[0.5, -0.5, 0.5], [0, 0, 0], [0.5, 0, -0.5]]}
        if variant == 1
        else {"0": [[0.5, 0.5, -0.5], [1, 1, -1], [1, 1, -1]], "1": [[1, 0.5, -1], [0, 0, 0], [0.5, 0.5, -1]]}
    )
    states = ("A", "B", "C")
    q = QuasiStochasticModel(initial_quasidistribution={"A": 1.0, "B": 0.0, "C": 0.0})
    q.observation_alphabet = frozenset(matrices)
    q.name = "Odd Process"
    for state in states:
        q.graph.add_state(state)
    for symbol, matrix in matrices.items():
        arr = np.asarray(matrix, dtype=float)
        for i, source in enumerate(states):
            for j, target in enumerate(states):
                prob = float(arr[i, j])
                if prob:
                    q.graph.add_transition(source, target, **{ATTR_EMISSION: symbol, ATTR_QUASIPROB: prob})
    q.validate()
    return q


def even_odd(machine_type: Any = EpsilonMachine) -> EpsilonMachine:
    _require_machine_type(machine_type, EpsilonMachine)
    return _from_string("A B 0 0.5; B A 0 1; A C 1 0.5; C B 0 0.5; C D 1 0.5; D C 1 1;", name="EvenOdd Process")


def three_even_odd(machine_type: Any = EpsilonMachine) -> EpsilonMachine:
    _require_machine_type(machine_type, EpsilonMachine)
    return _from_string(
        "A A 0 1; A B 1 1; B A 1 1; A C 2 1; C A 0 1; C B 1 1; C D 2 1; D C 2 1;", name="ThreEvenOdd Process"
    )


def period(P: int) -> MealyHMM:
    return periodic("0" * (P - 1) + "1")


def periodic(word: Sequence[Any], reduce: bool = True) -> MealyHMM:
    base = _word_period(word) if reduce else word
    edges = []
    for i, symbol in enumerate(base):
        edges.append((i, (i + 1) % len(base), symbol, 1.0))
    return _edge_machine(
        edges, machine_type=MealyHMM, name=f"Period-{len(base)} Process ({''.join(map(str, base))})", normalize=False
    )


def period1(machine_type: Any = MealyHMM) -> MealyHMM:
    return periodic("1") if machine_type is MealyHMM else _from_string("A A 1 1", name="Period-1 Process")


def period2(machine_type: Any = MealyHMM) -> MealyHMM:
    return periodic("01") if machine_type is MealyHMM else _from_string("A B 0 1; B A 1 1", name="Period-2 Process")


def period4(machine_type: Any = MealyHMM) -> MealyHMM:
    return (
        periodic("1110")
        if machine_type is MealyHMM
        else _from_string("A B 1 1; B C 1 1; C D 1 1; D A 0 1", name="Period-4 Process")
    )


def period8(machine_type: Any = MealyHMM) -> MealyHMM:
    if machine_type is not MealyHMM:
        raise NotImplementedError
    return periodic("10101110")


def period12(machine_type: Any = MealyHMM) -> MealyHMM:
    if machine_type is not MealyHMM:
        raise NotImplementedError
    return periodic("101011101110")


def period16(machine_type: Any = MealyHMM) -> MealyHMM:
    if machine_type is not MealyHMM:
        raise NotImplementedError
    return periodic("1010111011101110")


def perturbed_coin(p: float = 0.2, q: float | None = None, machine_type: Any = EpsilonMachine) -> MealyHMM:
    if p == 0.5:
        return bernoulli()
    if q is None:
        q = p
    if machine_type in (EpsilonMachine, None):
        return _edge_machine(
            [("A", "A", "0", 1 - p), ("A", "B", "1", p), ("B", "A", "0", q), ("B", "B", "1", 1 - q)],
            machine_type=EpsilonMachine,
            name="Perturbed Coin",
            normalize=False,
        )
    lowered = machine_type.lower() if isinstance(machine_type, str) else ""
    if lowered in {"lohr", "generative"}:
        if q != p:
            raise ValueError("Lohr/generative variants only support q == p")
        return bmc_lohr(p) if lowered == "lohr" else bmc_gen(p, p)
    raise NotImplementedError(f"cannot build machine type {machine_type!r}")


def phase_slip_backtrack_cmpy() -> EpsilonMachine:
    """cmpy's Phase-Slip Backtrack; a different process from :func:`~sofic.examples.epsilon_machines.phase_slip_backtrack`."""
    return _from_string("A B 1; A D 0; B B 1; B C 0; C D 0; D A 1", name="Phase-Slip Backtrack")


def rip(p: float = 0.5, q: float = 0.5, reverse: bool = False) -> EpsilonMachine:
    if reverse:
        spec = f"D E 1 {1 - p * q}; D F 0 {p * q}; E D 1 {(1 - p) / (1 - p * q)}; E G 0 {p * (1 - q) / (1 - p * q)}; F G 0 1; G D 1 1"
    else:
        spec = f"A B 0 {p}; A C 1 {1 - p}; B C 0 {q}; B C 1 {1 - q}; C A 1 1"
    return _from_string(spec, name="Random Insertion Process")


def rn1c(noise: float = 0.5, bias: float = 0.5) -> EpsilonMachine:
    return _edge_machine(
        [
            ("A", "B", "0", bias),
            ("B", "A", "0", 1),
            ("A", "C", "1", 1 - bias),
            ("C", "A", "0", noise),
            ("C", "A", "1", 1 - noise),
        ],
        machine_type=EpsilonMachine,
        name=f"Random Noisy-1 Copy, p(0|flip is 1) = {noise:.02f}",
        normalize=False,
    )


def rn1n(machine_type: Any = EpsilonMachine, bias: float = 0.5, noise: float = 0.1) -> EpsilonMachine:
    _require_machine_type(machine_type, EpsilonMachine)
    return _edge_machine(
        [
            ("A", "B", "0", bias),
            ("A", "C", "1", 1 - bias),
            ("B", "A", "1", 1),
            ("C", "A", "1", noise),
            ("C", "A", "0", 1 - noise),
        ],
        machine_type=EpsilonMachine,
        name="Rn1N Process",
        normalize=False,
    )


def rrx(machine_type: Any = EpsilonMachine) -> MealyHMM:
    if machine_type in (EpsilonMachine, None):
        edges = [
            ("S", "0", "0", 0.5),
            ("S", "1", "1", 0.5),
            ("0", "01|10", "1", 0.5),
            ("0", "00|11", "0", 0.5),
            ("1", "01|10", "0", 0.5),
            ("1", "00|11", "1", 0.5),
            ("00|11", "S", "0", 1),
            ("01|10", "S", "1", 1),
        ]
        return _edge_machine(edges, machine_type=EpsilonMachine, name="rrx", normalize=False)
    if machine_type is MealyHMM:
        return _edge_machine(
            [
                ("A", "B", "0", 0.5),
                ("A", "C", "1", 0.5),
                ("B", "D", "0", 0.5),
                ("B", "E", "1", 0.5),
                ("C", "F", "0", 0.5),
                ("C", "G", "1", 0.5),
                ("D", "A", "0", 1),
                ("E", "A", "1", 1),
                ("F", "A", "1", 1),
                ("G", "A", "0", 1),
            ],
            machine_type=MealyHMM,
            name="Non-Minimal RRX",
            normalize=False,
        )
    raise NotImplementedError


def rrxro(machine_type: Any = EpsilonMachine) -> EpsilonMachine:
    _require_machine_type(machine_type, EpsilonMachine)
    return _from_string(
        """
        S 0 0 .5; S 1 1 .5; 0 00 0 .5; 0 01 1 .5; 1 10 0 .5; 1 11 1 .5;
        01 orTrue 1 1; 10 orTrue 1 1; 00 000 0 1; 000 orFalse 0 .5; 000 orTrue 1 .5;
        11 110 0 1; 110 orFalse 0 .5; 110 orTrue 1 .5; orFalse S 0 1; orTrue S 1 1;
        """,
        name="rrxro",
    )


def sns(machine_type: Any = MealyHMM) -> MealyHMM:
    _require_machine_type(machine_type, MealyHMM)
    return _edge_machine(
        [("A", "A", "1", 0.5), ("A", "B", "1", 0.5), ("B", "A", "0", 0.5), ("B", "B", "1", 0.5)],
        machine_type=MealyHMM,
        name="Simple Nondeterministic Source",
        normalize=False,
    )


def three_hundred(
    machine_type: Any = EpsilonMachine, biases: tuple[float, float, float] = (0.5, 0.5, 0.5)
) -> EpsilonMachine:
    _require_machine_type(machine_type, EpsilonMachine)
    p, q, r = biases
    spec = f"A B 0 {p}; A C 1 {1 - p}; B D 0 1; C E 0 1; D F 0 {q}; D F 1 {1 - q}; E A 0 1; F A 0 {r}; F A 1 {1 - r};"
    return _from_string(spec, name="ThreeHundred Process")


def uniform_mealyhmm(
    topology: MealyHMM | Mapping[Any, np.ndarray] | Sequence[np.ndarray],
    name: str | None = None,
    nodes: Sequence[Hashable] | None = None,
    symbols: Sequence[Any] | None = None,
    create_using: type[MealyHMM] | None = None,
    prng: Any = None,
) -> MealyHMM:
    cls = create_using or MealyHMM
    if isinstance(topology, MealyHMM):
        grouped: dict[Hashable, list[tuple[Hashable, Any]]] = {}
        for transition in topology.transitions():
            grouped.setdefault(transition.source, []).append((transition.target, transition.data[ATTR_EMISSION]))
        edges = []
        for source, targets in grouped.items():
            probs = _dirichlet(len(targets), prng)
            for prob, (target, symbol) in zip(probs, targets, strict=True):
                edges.append((source, target, symbol, float(prob)))
        return _edge_machine(edges, machine_type=cls, name=name or "Random Mealy HMM", normalize=False)

    if isinstance(topology, Mapping):
        matrices = {symbol: np.asarray(matrix, dtype=float) for symbol, matrix in topology.items()}
    else:
        matrices = {
            symbol: np.asarray(matrix, dtype=float)
            for symbol, matrix in zip(symbols or _as_alphabet(len(topology)), topology, strict=False)
        }
    first = next(iter(matrices.values()))
    if nodes is None:
        nodes = tuple(range(first.shape[0]))
    edges = []
    for i, source in enumerate(nodes):
        support = []
        for symbol, matrix in matrices.items():
            for j, target in enumerate(nodes):
                if matrix[i, j] > 0:
                    support.append((target, symbol))
        probs = _dirichlet(len(support), prng)
        for prob, (target, symbol) in zip(probs, support, strict=True):
            edges.append((source, target, symbol, float(prob)))
    return _edge_machine(edges, machine_type=cls, name=name or "Random Mealy HMM", normalize=False)


def uniform_mealymc(
    order: int,
    symbols: int | Sequence[Any],
    name: str | None = None,
    create_using: type[MealyHMM] | None = None,
    prng: Any = None,
) -> MealyHMM:
    alphabet = _as_alphabet(symbols)
    cls = create_using or MealyHMM
    edges = []
    for state in _words(alphabet, order):
        probs = _dirichlet(len(alphabet), prng)
        for prob, symbol in zip(probs, alphabet, strict=True):
            target = (*state[1:], symbol) if order else ()
            edges.append((state, target, symbol, float(prob)))
    return _edge_machine(
        edges, machine_type=cls, name=(name or "Random Markov Chain") + f" (k={order})", normalize=False
    )


def _transducer(
    edges: Iterable[tuple[Hashable, Hashable, Any, Any, float]],
    *,
    initial: Hashable | None = None,
    name: str | None = None,
) -> MealyMachine:
    edge_list = list(edges)
    states = list(dict.fromkeys([source for source, *_ in edge_list] + [target for _source, target, *_ in edge_list]))
    inputs = frozenset(input_symbol for _source, _target, input_symbol, _output_symbol, _prob in edge_list)
    outputs = frozenset(output_symbol for _source, _target, _input_symbol, output_symbol, _prob in edge_list)
    start = initial if initial is not None else (states[0] if states else None)
    machine = MealyMachine(
        input_alphabet=inputs,
        output_alphabet=outputs,
        initial_states=frozenset({start} if start is not None else set()),
    )
    if name is not None:
        machine.name = name
    for state in states:
        machine.graph.add_state(state)
    for source, target, input_symbol, output_symbol, prob in edge_list:
        machine.graph.add_transition(
            source, target, **{ATTR_SYMBOL: input_symbol, ATTR_OUTPUT: output_symbol, ATTR_PROB: float(prob)}
        )
    machine.validate()
    return machine


def gm_to_even(bias: float = 0.5, create_using: Any = None) -> MealyMachine:
    """Map the golden mean (forbid ``11``) onto the Even Process: ``0 -> 0`` and ``10 -> 11``."""
    del bias, create_using
    return _transducer([("A", "A", "0", "0", 1), ("A", "B", "1", "1", 1), ("B", "A", "0", "1", 1)], name="GM to Even")


def rct(bias: float = 0.5, create_using: Any = None) -> MealyMachine:
    del create_using
    return _transducer(
        [("A", "B", "0", "0", bias), ("A", "C", "0", "1", bias), ("B", "A", "0", "0", 1), ("C", "A", "1", "1", 1)]
    )


def bit_flip(create_using: Any = None) -> MealyMachine:
    del create_using
    return _transducer([("A", "A", "0", "1", 1), ("A", "A", "1", "0", 1)])


def flip_every_other(parity: str = "Even", create_using: Any = None) -> MealyMachine:
    del create_using
    initial = "A" if parity == "Even" else "B"
    return _transducer(
        [("A", "B", "0", "1", 1), ("A", "B", "1", "0", 1), ("B", "A", "0", "0", 1), ("B", "A", "1", "1", 1)],
        initial=initial,
    )


def delay(length: int = 2, symbols: int | Sequence[Any] = 2, create_using: Any = None) -> MealyMachine:
    del create_using
    alphabet = tuple(map(str, _as_alphabet(symbols)))
    edges = []
    for state in product(alphabet, repeat=length):
        source = "".join(state)
        for input_symbol in alphabet:
            full = source + input_symbol
            edges.append((source, full[1:], input_symbol, full[0], 1))
    return _transducer(edges)


def two_perm(symbols: int | Sequence[Any] = 2, create_using: Any = None) -> MealyMachine:
    del create_using
    alphabet = tuple(map(str, _as_alphabet(symbols)))
    edges = []
    for even_symbol in alphabet:
        edges.append(("??", even_symbol + "?", even_symbol, "?", 1))
        for odd_symbol in alphabet:
            edges.append((even_symbol + "?", even_symbol + odd_symbol, odd_symbol, odd_symbol, 1))
            for next_even in alphabet:
                edges.append((even_symbol + odd_symbol, next_even + "?", next_even, even_symbol, 1))
    return _transducer(edges, initial="??")


def binary_channel(p: float = 0.0, q: float = 0.0, create_using: Any = None) -> MealyMachine:
    del create_using
    return _transducer(
        [("A", "A", "0", "0", 1 - p), ("A", "A", "0", "1", p), ("A", "A", "1", "1", 1 - q), ("A", "A", "1", "0", q)]
    )


def sliding_nor(create_using: Any = None) -> MealyMachine:
    del create_using
    return _transducer(
        [("A", "A", "0", "1", 1), ("A", "B", "1", "0", 1), ("B", "A", "0", "0", 1), ("B", "B", "1", "0", 1)]
    )


def parity(create_using: Any = None) -> MealyMachine:
    del create_using
    return _transducer(
        [("A", "A", "0", "0", 1), ("A", "B", "1", "1", 1), ("B", "A", "0", "0", 1), ("B", "A", "1", "0", 1)]
    )


def gme(create_using: Any = None) -> MealyMachine:
    del create_using
    return _transducer(
        [
            ("A", "A", "0", "1", 0.5),
            ("A", "B", "0", "0", 0.5),
            ("B", "A", "0", "1", 1),
            ("A", "A", "1", "0", 0.5),
            ("A", "B", "1", "1", 0.5),
            ("B", "A", "1", "1", 1),
        ]
    )


def binary_channel_et(p: float = 0.1, q: float = 0.2) -> Any:
    """Memoryless binary channel as its minimal (single-state) ε-transducer."""
    from sofic.generators.epsilon_transducer import EpsilonTransducer

    return EpsilonTransducer.from_channel(binary_channel(p, q))


def gm_to_even_et() -> Any:
    """Golden-mean-to-even map as an ε-transducer (Barnett & Crutchfield 2015)."""
    from sofic.generators.epsilon_transducer import EpsilonTransducer

    return EpsilonTransducer.from_channel(gm_to_even())


processes = [
    "band_merging",
    "beads_on_necklace",
    "before_after",
    "binary_markov_chain",
    "cantor",
    "coupled_gmps",
    "irreversible_two_state",
    "nonunifilar_golden_mean",
    "ehrenfest",
    "even_odd",
    "even_redundant",
    "flower",
    "four_state_almost_iid",
    "girvan_fig6b",
    "girvan_fig6c",
    "girvan_fig6d",
    "ising",
    "lollipop",
    "misiurewicz",
    "misiurewicz_simplified",
    "misiurewicz_uniform",
    "multiple3",
    "multiple4",
    "nemo_redundant",
    "noisy_period2",
    "odd",
    "period1",
    "period2",
    "period4",
    "period8",
    "period12",
    "period16",
    "perturbed_coin",
    "phase_slip_backtrack_cmpy",
    "random_biased_coin",
    "random_golden_mean",
    "random_even",
    "rip",
    "rn1c",
    "rn1n",
    "rrx",
    "rrxro",
    "sns",
    "three_hundred",
    "three_even",
    "three_even_odd",
]
nonergodic_generators = ["uncoupled_gmps"]
transducers = [
    "gm_to_even",
    "rct",
    "bit_flip",
    "flip_every_other",
    "delay",
    "two_perm",
    "binary_channel",
    "sliding_nor",
    "parity",
    "gme",
]
epsilon_transducers = [
    "binary_channel_et",
    "gm_to_even_et",
]

process_list = [globals()[name] for name in processes]
process_list.extend(globals()[name] for name in nonergodic_generators)
transducer_list = [globals()[name] for name in transducers]
epsilon_transducer_list = [globals()[name] for name in epsilon_transducers]

__all__ = processes + nonergodic_generators + transducers + epsilon_transducers
__all__ += [
    "afc",
    "afc2",
    "bmc_em",
    "bmc_gen",
    "bmc_param",
    "bmc_lohr",
    "cyclic_branching",
    "golden_mean_ghmm",
    "iid",
    "logic_machine",
    "markov_skeleton",
    "multiple_n",
    "odd_ghmm",
    "period",
    "periodic",
    "stretched_gm",
    "rk_gm",
    "rn_gm",
    "uniform_mealyhmm",
    "uniform_mealymc",
    "processes",
    "process_list",
    "transducers",
    "transducer_list",
    "epsilon_transducers",
    "epsilon_transducer_list",
]
