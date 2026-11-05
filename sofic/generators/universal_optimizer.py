r"""Direct optimization of generative complexity from the coupled generator equations.

Minimize :math:`\H{\pi}` subject to the bidirectional edge law and forward / reverse
Markov consistency (:func:`~sofic.generators.coupled_generators._GeneratorProblem.residuals`),
with word-level verification. This is the universal problem from the method note,
without the lower-bound or :math:`\op{G}{S^+ : S^-}` decision tree of
:func:`~sofic.generators.generative_search.generative_model`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from sofic.exceptions import StochasticValidationError
from sofic.generators.coupled_generators import (
    CoupledGenerativeModel,
    CoupledSearchResult,
    _GeneratorProblem,
    _model_from_kernel,
    coupled_search,
    generator_state_cap,
    memory_lift,
    memory_lift_search,
    polish_separating_channel,
    separating_optima,
)

if TYPE_CHECKING:
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine


@dataclass
class UniversalGenerativeResult:
    """Best word-verified generator from the coupled equations over several state counts."""

    model: CoupledGenerativeModel
    value: float
    """State entropy ``H[pi]`` of the returned generator (upper bound on ``C_g``)."""
    k: int
    residual: float
    start: str
    n_feasible: int
    n_verified: int
    by_k: list[dict[str, Any]] = field(default_factory=list)
    search: CoupledSearchResult | None = None
    """Winning coupled search point (for pipeline ``offer``)."""


@dataclass
class MethodComparison:
    """Side-by-side upper bounds from the universal optimizer and the full pipeline."""

    universal: UniversalGenerativeResult
    pipeline_upper: float
    pipeline_certified: bool
    pipeline_lower: float


def _memory_lift_seeds(
    problem: _GeneratorProblem,
    *,
    m_values: Sequence[int],
    starts: int,
    rng: np.random.Generator,
) -> list[tuple[str, tuple[np.ndarray, ...]]]:
    out: list[tuple[str, tuple[np.ndarray, ...]]] = []
    bidir = problem.bidir
    for m in m_values:
        if m * m > len(problem.p_pair) + 4:
            continue
        for rank, (_entropy, residual, nu) in enumerate(
            memory_lift_search(bidir, m, starts=max(3, starts // 2), rng=rng)[:3]
        ):
            if residual > 1e-4:
                continue
            seed = memory_lift(problem, nu)
            out.append((f"memory_lift_{m}_{rank}", seed))
    return out


def _g_optima_seeds(
    problem: _GeneratorProblem,
    *,
    exact_starts: int,
    rng: np.random.Generator,
) -> list[tuple[str, tuple[np.ndarray, ...]]]:
    bidir = problem.bidir
    _, optima = separating_optima(bidir, starts=exact_starts, rng=rng)
    out: list[tuple[str, tuple[np.ndarray, ...]]] = []
    for i, c in enumerate(optima[:12]):
        c = polish_separating_channel(problem, c)
        out.append((f"g_optimum_{i}", problem.channel_seed(c)))
    return out


def universal_generative_optimizer(
    bidir: BidirectionalEpsilonMachine,
    *,
    max_states: int | None = None,
    starts: int = 20,
    memory_lift_m: Sequence[int] = (2, 3, 4),
    include_g_seeds: bool = True,
    exact_starts: int = 15,
    maxiter: int = 3000,
    residual_atol: float = 1e-6,
    reproduction_atol: float = 1e-5,
    alternating: bool = False,
    dual_strategy: bool = True,
    cutoff: float = 1e-9,
    rng: np.random.Generator | None = None,
) -> UniversalGenerativeResult:
    """Search over hidden state count and seeds; return the best verified coupled generator."""
    from sofic.generators.minimal_generative_model import _reproduction_error, _reproduction_length

    rng = rng or np.random.default_rng()
    problem = _GeneratorProblem(bidir)
    n_pair = len(problem.p_pair)
    k_max = generator_state_cap(problem, hard_cap=max_states if max_states is not None else min(n_pair, 8))
    extra_all = _memory_lift_seeds(problem, m_values=memory_lift_m, starts=starts, rng=rng)
    if include_g_seeds and n_pair <= 36:
        extra_all += _g_optima_seeds(problem, exact_starts=exact_starts, rng=rng)

    reference = bidir.forward_machine
    length = _reproduction_length(reference)
    best_verified: tuple[CoupledSearchResult, CoupledGenerativeModel] | None = None
    by_k: list[dict[str, Any]] = []
    n_feasible = 0
    n_verified = 0

    for k in range(2, k_max + 1):
        candidates = coupled_search(
            bidir,
            k,
            starts=starts,
            extra=extra_all,
            maxiter=maxiter,
            residual_atol=residual_atol,
            alternating=alternating,
            dual_strategy=dual_strategy,
            rng=rng,
        )
        n_feasible += len(candidates)
        verified_here: list[tuple[CoupledSearchResult, CoupledGenerativeModel]] = []
        for result in candidates:
            model = _model_from_kernel(problem, result, cutoff)
            if _reproduction_error(model, reference, max_length=length) <= reproduction_atol:
                verified_here.append((result, model))
        n_verified += len(verified_here)
        best_k = verified_here[0] if verified_here else None
        by_k.append(
            {
                "k": k,
                "n_feasible": len(candidates),
                "n_verified": len(verified_here),
                "best_entropy": None if best_k is None else best_k[0].entropy,
                "best_start": None if best_k is None else best_k[0].start,
            }
        )
        if verified_here and (best_verified is None or verified_here[0][0].entropy < best_verified[0].entropy):
            best_verified = verified_here[0]

    if best_verified is None:
        raise StochasticValidationError(
            "universal_generative_optimizer: no k-state coupled generator reproduced the process"
        )
    result, model = best_verified
    return UniversalGenerativeResult(
        model=model,
        value=result.entropy,
        k=len(result.pi),
        residual=result.residual,
        start=result.start,
        n_feasible=n_feasible,
        n_verified=n_verified,
        by_k=by_k,
        search=result,
    )


def compare_universal_to_pipeline(
    bidir: BidirectionalEpsilonMachine,
    *,
    rng: np.random.Generator | None = None,
    **optimizer_kwargs: Any,
) -> MethodComparison:
    """Run :func:`universal_generative_optimizer` and :func:`generative_model_from_bidirectional` on the same machine."""
    from sofic.generators.generative_search import generative_model_from_bidirectional

    rng = rng or np.random.default_rng()
    universal = universal_generative_optimizer(bidir, rng=rng, **optimizer_kwargs)
    pipeline = generative_model_from_bidirectional(
        bidir,
        rng=rng,
        universal_coupled=True,
        early_theorem_g_exit=True,
    )
    return MethodComparison(
        universal=universal,
        pipeline_upper=pipeline.upper,
        pipeline_certified=pipeline.certified,
        pipeline_lower=pipeline.lower,
    )


__all__ = [
    "MethodComparison",
    "UniversalGenerativeResult",
    "compare_universal_to_pipeline",
    "universal_generative_optimizer",
]
