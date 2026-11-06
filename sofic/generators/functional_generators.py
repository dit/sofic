r"""Minimal functional generators: generator states that are functions of :math:`(S^+, S^-)`.

If the state is :math:`W = f(S^+, S^-)`, consecutive states are functions of the bidirectional edge,
so the only candidate HMM is the coarse-graining of the bidirectional ε-machine by :math:`f`, and
:math:`f` is a generator exactly when that coarse-graining satisfies the lifting conditions of
:mod:`~sofic.generators.lifted_common_information`. Each block :math:`f^{-1}(w)` must render
:math:`S^+` and :math:`S^-` independent, so it is a combinatorial rectangle of the joint support
carrying a product distribution, and :math:`\H{W}` is a sum of per-block costs. The least such cost
is the functional common information :math:`\op{F}{S^+ : S^-}` :cite:`james2017multivariate`, an
exact cover found by integer programming.

:func:`functional_cover_search` solves that exact cover, tests the lifting conditions, and if they
fail adds a cut excluding that cover and re-solves. Successive optima are nondecreasing, so each
rejected cover's entropy is a certified lower bound, and the first cover that generates is a minimal
functional generator. The identity cover (the bidirectional machine) always generates, so the search
terminates.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from sofic.exceptions import StochasticValidationError
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.coupled_generators import _GeneratorProblem
from sofic.generators.epsilon_machine import EpsilonMachine

_MAX_RECTANGLES = 200_000
_PRODUCT_RTOL = 1e-8
_PRODUCT_ATOL = 1e-12
_LIFT_ATOL = 1e-10


def _entropy_cost(mass: float) -> float:
    return float(-mass * np.log2(mass)) if mass > 0.0 else 0.0


def product_rectangles(pairs: list[tuple], p: np.ndarray) -> list[frozenset[int]]:
    """Blocks of joint states ``A x B`` contained in the support whose distribution is a product.

    Both conditions pass to sub-rectangles, so rectangles are grown one row or column at a time from
    single cells. Blocks are sets of indices into ``pairs``.
    """
    index = {pair: i for i, pair in enumerate(pairs)}
    rows = sorted({a for a, _ in pairs}, key=repr)
    cols = sorted({b for _, b in pairs}, key=repr)

    def block_of(rect: tuple[frozenset, frozenset]) -> frozenset[int] | None:
        A, B = sorted(rect[0], key=repr), sorted(rect[1], key=repr)
        cells = [index[cell] for cell in product(A, B) if cell in index]
        if len(cells) < len(A) * len(B):
            return None
        if len(cells) > 1:
            q = p[cells].reshape(len(A), len(B))
            q = q / q.sum()
            if not np.allclose(q, np.outer(q.sum(1), q.sum(0)), rtol=_PRODUCT_RTOL, atol=_PRODUCT_ATOL):
                return None
        return frozenset(cells)

    seen: set[tuple[frozenset, frozenset]] = set()
    blocks: set[frozenset[int]] = set()
    frontier = [(frozenset([a]), frozenset([b])) for a, b in pairs]
    while frontier:
        rect = frontier.pop()
        if rect in seen:
            continue
        seen.add(rect)
        block = block_of(rect)
        if block is None:
            continue
        blocks.add(block)
        if len(blocks) > _MAX_RECTANGLES:
            raise StochasticValidationError("functional generator search: too many product rectangles")
        A, B = rect
        frontier += [(A | {a}, B) for a in rows if a not in A]
        frontier += [(A, B | {b}) for b in cols if b not in B]
    return sorted(blocks, key=sorted)


@dataclass(frozen=True)
class FunctionalCoverSearch:
    """Outcome of :func:`functional_cover_search`.

    ``labels[i]`` is the state of joint state ``i``; ``value`` its entropy; ``lower`` a certified lower
    bound on the least entropy of a functional generator (``lower == value`` when ``exact``);
    ``functional_common_information`` the first cover's entropy :math:`\\op{F}{S^+ : S^-}`;
    ``rejected`` how many covers failed the lifting conditions.
    """

    labels: np.ndarray
    value: float
    lower: float
    exact: bool
    functional_common_information: float
    rejected: int


def functional_cover_search(problem: _GeneratorProblem, *, max_covers: int = 2000) -> FunctionalCoverSearch:
    """Least-entropy functional generator by exact cover with no-good cuts; see the module docstring."""
    lifted = problem.lifted
    pairs, p = list(lifted.pairs), lifted.p
    n = len(pairs)
    blocks = product_rectangles(pairs, p)
    costs = np.array([_entropy_cost(float(p[list(b)].sum())) for b in blocks])
    cover = np.zeros((n, len(blocks)))
    for j, block in enumerate(blocks):
        cover[list(block), j] = 1.0

    best_labels = np.arange(n)
    best_value = float(sum(_entropy_cost(float(m)) for m in p))
    cuts: list[np.ndarray] = []
    lower = 0.0
    first = None
    for _ in range(max_covers):
        constraints = [LinearConstraint(cover, 1, 1)]
        if cuts:
            cut_rows = np.array(cuts)
            constraints.append(LinearConstraint(cut_rows, -np.inf, cut_rows.sum(axis=1) - 1))
        res = milp(costs, constraints=constraints, integrality=np.ones(len(blocks)), bounds=Bounds(0, 1))
        if not res.success:
            return FunctionalCoverSearch(best_labels, best_value, best_value, True, first or best_value, len(cuts))
        chosen = np.flatnonzero(res.x > 0.5)
        cost = float(costs[chosen].sum())
        first = cost if first is None else first
        lower = max(lower, cost)
        if cost >= best_value - 1e-12:
            return FunctionalCoverSearch(best_labels, best_value, best_value, True, first, len(cuts))
        labels = np.empty(n, dtype=int)
        for w, j in enumerate(chosen):
            labels[list(blocks[j])] = w
        if lifted.markov_residual(np.eye(len(chosen))[labels]) <= _LIFT_ATOL:
            return FunctionalCoverSearch(labels, cost, cost, True, first, len(cuts))
        cut = np.zeros(len(blocks))
        cut[chosen] = 1.0
        cuts.append(cut)
    return FunctionalCoverSearch(best_labels, best_value, lower, False, first or best_value, len(cuts))


def minimal_functional_generator(
    machine: BidirectionalEpsilonMachine | EpsilonMachine,
    *,
    reverse_max_states: int = 10_000,
    use_ladder: bool = True,
    max_covers: int = 2000,
    reproduction_atol: float = 1e-5,
):
    r"""Least-state-entropy generator whose state is a function of :math:`(S^+, S^-)`.

    Returns a :class:`~sofic.generators.generative_search.GenerativeModelResult` with
    ``objective="functional"``. ``lower`` comes from the exact-cover search
    (:func:`functional_cover_search`) and meets ``upper`` unless ``max_covers`` covers fail the
    lifting conditions first, in which case ``upper`` is the bidirectional machine. Since every
    functional generator is a generator, :math:`C_g \le` ``upper``.
    """
    from sofic.generators.coupled_generators import CoupledSearchResult, _model_from_kernel
    from sofic.generators.generative_search import GenerativeModelResult, _finalize_result, resolve_bidirectional
    from sofic.generators.minimal_generative_model import _reproduction_error, _reproduction_length

    bidir, build_meta = resolve_bidirectional(machine, reverse_max_states=reverse_max_states, use_ladder=use_ladder)
    problem = _GeneratorProblem(bidir)
    search = functional_cover_search(problem, max_covers=max_covers)
    log = [
        {"stage": "bidirectional build", **build_meta},
        {"stage": "functional common information", "value": search.functional_common_information},
        {
            "stage": "functional cover search",
            "value": search.value,
            "lower": search.lower,
            "exact": search.exact,
            "rejected covers": search.rejected,
        },
    ]
    c = np.eye(search.labels.max() + 1)[search.labels]
    pi, Q, R, K = problem.channel_seed(c)
    seed = CoupledSearchResult(pi, Q, R, K, search.value, problem.residual(pi, Q, R, K), "functional cover")
    model = _model_from_kernel(problem, seed, 1e-9)
    reference = bidir.forward_machine
    error = _reproduction_error(model, reference, max_length=_reproduction_length(reference))
    if error > reproduction_atol:
        raise StochasticValidationError(
            f"minimal_functional_generator: cover passes the lifting conditions but misses the source by {error:.3g}"
        )
    result = GenerativeModelResult(
        model=model,
        upper=float(model.state_entropy()),
        lower=float(search.lower),
        upper_source="functional cover search",
        lower_source="exact cover with no-good cuts" if search.rejected else "functional common information",
        exact_common_information_estimate=None,
        separating_variables_generate=search.rejected == 0,
        stages=log,
        objective="functional",
    )
    return _finalize_result(result)


__all__ = [
    "FunctionalCoverSearch",
    "functional_cover_search",
    "minimal_functional_generator",
    "product_rectangles",
]
