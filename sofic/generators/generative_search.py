r"""Search for minimal-state-entropy generators, bracketing the generative complexity.

:func:`generative_model` runs a pipeline over nested model classes and returns the best generator it can
verify together with a rigorous bracket :math:`[\text{lower}, \text{upper}]` on the generative complexity
:math:`C_g` of Löhr & Ay :cite:`LohrAy2009`:

1. *Lower bounds.* Excess entropy; support-type and flow bounds when the joint has few maximal rectangles;
   optional heuristic coverage bound on full supports (:mod:`~sofic.generators.coverage_bound`).
2. *Persistent crypticity.* When both :math:`p.\chi^\pm` vanish on the bidirectional edge,
   :math:`C_g = \op{G}{S^+ : S^-}`.
3. *Exact optima + kernel LP.* Enumerate :math:`\op{G}{S^+ : S^-}` optima; optional
   :func:`~sofic.generators.coupled_generators.bilevel_search` when none generate; then an
   **entropy-ordered sweep** over all multistart separating decompositions (not only
   :math:`\op{G}` optima) up to the current best upper bound.
4. *Upper bounds:* ε-machines, memory lifts, functional/lifted channels (small joints), coupled search.

Every candidate is word-verified. ``certified`` reports when ``lower`` meets ``upper``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from sofic.exceptions import StochasticValidationError
from sofic.generators.coupled_generators import (
    CoupledGenerativeModel,
    CoupledSearchResult,
    _GeneratorProblem,
    _model_from_kernel,
    bilevel_search,
    coupled_search,
    first_generating_separator,
    generator_kernel,
    memory_lift,
    memory_lift_search,
    polish_separating_channel,
    separating_decomposition_channels,
    separating_optima,
)
from sofic.generators.generator_bounds import flow_bound_result, maximal_rectangles, support_type_bound

if TYPE_CHECKING:
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine

_MAX_RECTANGLES = 7
_FLOW_THRESHOLDS = ((0.6,), (0.55, 0.65))
_CERT_TOL = 1e-6


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())


@dataclass
class GenerativeModelResult:
    """Best verified generator and a bracket on the generative complexity."""

    model: CoupledGenerativeModel
    upper: float
    lower: float
    upper_source: str
    lower_source: str
    exact_common_information_estimate: float | None
    separating_variables_generate: bool
    stages: list[dict[str, Any]] = field(default_factory=list)
    persistent_crypticity_theorem_applies: bool = False
    n_g_optima: int = 0
    n_g_optima_generating: int = 0
    cg_exceeds_g: bool | None = None
    lower_certified: bool = True

    @property
    def certified(self) -> bool:
        """Whether the upper and lower bounds meet (to 1e-6 bits)."""
        return self.upper - self.lower <= _CERT_TOL

    @property
    def gap(self) -> float:
        return self.upper - self.lower

    @property
    def g_optima_generate(self) -> bool:
        """Whether some :math:`\\op{G}{S^+ : S^-}` optimum passed the generator kernel LP."""
        return self.n_g_optima_generating > 0


def _persistent_criteria(problem: _GeneratorProblem) -> tuple[float, float]:
    lifted = problem.lifted
    J = np.einsum("pb,pxq,qa->bxa", lifted.minus, lifted.edges, lifted.plus)
    h_joint = _entropy(J.ravel())
    return h_joint - _entropy(J.sum(axis=2).ravel()), h_joint - _entropy(J.sum(axis=0).ravel())


def _channel_matrix(problem: _GeneratorProblem, channel: dict) -> np.ndarray:
    states = sorted({w for row in channel.values() for w in row}, key=repr)
    return np.array([[channel[pair].get(w, 0.0) for w in states] for pair in problem.lifted.pairs])


def _coverage_lower_bound(
    joint: dict[tuple, float],
    rng: np.random.Generator,
    *,
    mu_starts: int,
    certified: bool,
) -> tuple[float, dict[str, float], bool]:
    from sofic.generators.coverage_bound import (
        coverage_entropy_lower,
        joint_matrix,
        mu1_search,
        mu_caps,
        mu_k_bilinear,
    )

    m, _, _ = joint_matrix(joint)
    if certified and np.all(m > 0):
        caps, caps_certified = mu_caps(m, k=2, seed=int(rng.integers(0, 2**31)), mu_starts=mu_starts)
        meta = {
            "mu_1": caps[0],
            "mu_2": caps[1] if len(caps) > 1 else caps[0],
            "certified": caps_certified,
        }
        return coverage_entropy_lower(caps), meta, caps_certified
    seed = int(rng.integers(0, 2**31))
    mu1 = mu1_search(m, seed=seed, maxiter=min(250, mu_starts * 8))
    mu2 = mu_k_bilinear(m, 2, starts=min(40, mu_starts))
    caps = (mu1, mu2)
    return coverage_entropy_lower(caps), {"mu_1": mu1, "mu_2": mu2, "certified": False}, False


def _try_best_candidate(
    problem: _GeneratorProblem,
    reference: Any,
    length: int,
    candidates: list[tuple[float, str, tuple[np.ndarray, ...]]],
    *,
    lower: float,
    lower_source: str,
    lower_certified: bool,
    g_estimate: float | None,
    theorem: bool,
    n_g_optima: int,
    n_generating: int,
    cg_exceeds_g: bool | None,
    log: list[dict[str, Any]],
    reproduction_atol: float,
) -> GenerativeModelResult | None:
    from sofic.generators.minimal_generative_model import _reproduction_error

    g_generates = n_generating > 0
    sep_gen = g_generates if n_g_optima > 0 else theorem

    for _value, name, seed in sorted(candidates, key=lambda t: t[0]):
        pi, Q, R, K = seed
        result = CoupledSearchResult(pi, Q, R, K, _entropy(pi), problem.residual(pi, Q, R, K), name)
        model = _model_from_kernel(problem, result, 1e-9)
        if _reproduction_error(model, reference, max_length=length) > reproduction_atol:
            continue
        upper = float(model.state_entropy())
        return GenerativeModelResult(
            model=model,
            upper=upper,
            lower=float(min(lower, upper)),
            upper_source=name,
            lower_source=lower_source,
            exact_common_information_estimate=g_estimate,
            separating_variables_generate=sep_gen,
            stages=log,
            persistent_crypticity_theorem_applies=theorem,
            n_g_optima=n_g_optima,
            n_g_optima_generating=n_generating,
            cg_exceeds_g=cg_exceeds_g,
            lower_certified=lower_certified,
        )
    return None


def generative_model(
    bidir: BidirectionalEpsilonMachine,
    *,
    max_states: int | None = None,
    starts: int = 20,
    memory_starts: int = 10,
    residual_atol: float = 1e-7,
    reproduction_atol: float = 1e-5,
    stages: Sequence[str] = ("machines", "exact", "memory", "functional", "lifted", "coupled"),
    large: int = 9,
    exact_starts: int = 30,
    dit_fallback: bool = True,
    coverage: Literal["off", "heuristic", "certified"] = "heuristic",
    coverage_mu_starts: int = 25,
    entropy_sweep_starts: int = 80,
    bilevel: bool = True,
    early_certified: bool = True,
    rng: np.random.Generator | None = None,
) -> GenerativeModelResult:
    """Run the generator-search pipeline; see the module docstring."""
    from sofic.generators.lifted_common_information import _unconstrained_channel
    from sofic.generators.minimal_generative_model import (
        _normalized_joint_distribution,
        functional_generative_model,
        minimal_generative_model,
    )

    rng = rng or np.random.default_rng()
    problem = _GeneratorProblem(bidir)
    lifted = problem.lifted
    log: list[dict[str, Any]] = []
    reference = bidir.forward_machine
    length = __import__(
        "sofic.generators.minimal_generative_model", fromlist=["_reproduction_length"]
    )._reproduction_length(reference)

    joint = {pair: float(m) for pair, m in zip(lifted.pairs, lifted.p, strict=True)}
    n_rect = len(maximal_rectangles(set(joint)))

    # ---- lower bounds -------------------------------------------------------------------------------
    excess = _entropy(problem.p_plus) + _entropy(problem.p_minus) - _entropy(problem.p_pair)
    lower, lower_source = excess, "excess entropy"
    lower_certified = True
    log.append({"stage": "excess entropy", "value": excess})

    if n_rect <= _MAX_RECTANGLES:
        support, _, _ = support_type_bound(joint)
        log.append({"stage": "support bound", "value": support})
        if support > lower:
            lower, lower_source = support, "support-type bound"
            lower_certified = True
        for thresholds in _FLOW_THRESHOLDS:
            fb = flow_bound_result(bidir, thresholds=thresholds)
            log.append({"stage": f"flow bound {thresholds}", "value": fb.value, "certified": fb.certified})
            if fb.value > lower + 1e-9:
                lower, lower_source = fb.value, f"flow bound {thresholds}"
                lower_certified = fb.certified

    if coverage in ("heuristic", "certified") and n_rect > _MAX_RECTANGLES:
        try:
            use_certified = coverage == "certified"
            cov, meta, caps_ok = _coverage_lower_bound(
                joint, rng, mu_starts=coverage_mu_starts, certified=use_certified
            )
            stage = "coverage bound (certified)" if use_certified else "coverage bound (heuristic)"
            log.append({"stage": stage, "value": cov, **meta})
            if cov > lower + 1e-9:
                lower, lower_source = cov, stage
                lower_certified = caps_ok if use_certified else False
        except Exception as exc:  # pragma: no cover - defensive
            log.append({"stage": "coverage bound", "error": str(exc)})

    f_crit, r_crit = _persistent_criteria(problem)
    theorem = f_crit < 1e-10 and r_crit < 1e-10
    log.append({"stage": "persistent criteria", "value": (f_crit, r_crit), "theorem applies": theorem})

    # ---- upper search -------------------------------------------------------------------------------
    candidates: list[tuple[float, str, tuple[np.ndarray, ...]]] = []
    repair: list[tuple[str, tuple[np.ndarray, ...]]] = []
    small = len(lifted.pairs) <= large
    g_estimate: float | None = None
    n_g_optima = 0
    n_generating = 0
    cg_exceeds_g: bool | None = None
    optima: list[np.ndarray] = []

    def offer(name: str, seed: tuple[np.ndarray, ...]) -> None:
        residual = problem.residual(*seed)
        value = _entropy(seed[0])
        log.append({"stage": name, "value": value, "residual": residual})
        if residual <= residual_atol:
            candidates.append((value, name, seed))

    def maybe_return_early() -> GenerativeModelResult | None:
        if not early_certified or not candidates:
            return None
        if min(c[0] for c in candidates) - lower > _CERT_TOL:
            return None
        return _try_best_candidate(
            problem,
            reference,
            length,
            candidates,
            lower=lower,
            lower_source=lower_source,
            lower_certified=lower_certified,
            g_estimate=g_estimate,
            theorem=theorem,
            n_g_optima=n_g_optima,
            n_generating=n_generating,
            cg_exceeds_g=cg_exceeds_g,
            log=log,
            reproduction_atol=reproduction_atol,
        )

    if "machines" in stages:
        offer("epsilon machine", problem.epsilon_seed())
        offer("reverse epsilon machine", problem.reverse_seed())
        offer("bidirectional machine", problem.pair_seed())
        early = maybe_return_early()
        if early is not None and early.certified:
            return early

    if "exact" in stages:
        best, optima = separating_optima(problem, starts=exact_starts, rng=rng)
        if np.isfinite(best):
            g_estimate = best
        if dit_fallback:
            try:
                joint_n = _normalized_joint_distribution(bidir, cutoff=1e-10)
                c = _unconstrained_channel(
                    lifted,
                    joint_n,
                    "exact",
                    niter=10,
                    maxiter=1000,
                    polish=1e-8,
                    backend="numpy",
                    cutoff=1e-10,
                    rng=rng,
                )
                if c is not None:
                    c = polish_separating_channel(problem, c)
                    value = _entropy(lifted.p @ c)
                    if g_estimate is None or value < g_estimate - 1e-9:
                        g_estimate, optima = value, [c]
                    elif value <= g_estimate + 1e-7:
                        optima.append(c)
            except (StochasticValidationError, ImportError):
                pass
        n_g_optima = len(optima)
        for i, c in enumerate(optima):
            violation, K = generator_kernel(problem, c)
            pi, Q, R, _ = problem.channel_seed(c)
            log.append(
                {
                    "stage": f"exact common information, optimum {i}",
                    "value": _entropy(pi),
                    "kernel violation": violation,
                }
            )
            if violation <= 1e-9:
                offer("exact-common-information optimum", (pi, Q, R, K))
                n_generating += 1
            else:
                repair.append((f"exact-common-information optimum {i} (repair)", (pi, Q, R, K)))
        if n_g_optima > 0:
            cg_exceeds_g = n_generating == 0
        log.append(
            {
                "stage": "exact common information",
                "value": g_estimate,
                "optima": n_g_optima,
                "generating optima": n_generating,
                "cg exceeds g": cg_exceeds_g,
            }
        )
        if bilevel and n_generating == 0 and optima:
            for ent, viol, c in bilevel_search(bidir, optima[:8])[:4]:
                log.append({"stage": "bilevel search", "entropy": ent, "kernel violation": viol})
                if viol <= 1e-9:
                    pi, Q, R, K = problem.channel_seed(c)
                    offer("bilevel feasible channel", (pi, Q, R, K))
                elif viol < 0.05:
                    pi, Q, R, K = problem.channel_seed(c)
                    repair.append(("bilevel near-feasible", (pi, Q, R, K)))

        sweep_channels = separating_decomposition_channels(problem, starts=entropy_sweep_starts, rng=rng)
        best_upper = min((c[0] for c in candidates), default=np.inf)
        hit = None
        if n_generating == 0 and sweep_channels:
            max_h = best_upper if np.isfinite(best_upper) else None
            if lower_certified:
                max_h = min(max_h, lower) if max_h is not None else lower
            hit = first_generating_separator(
                problem,
                sweep_channels,
                max_entropy=max_h,
            )
        log.append(
            {
                "stage": "entropy sweep (separating decompositions)",
                "channels tried": len(sweep_channels),
                "max entropy": best_upper if np.isfinite(best_upper) else None,
                "found generating separator": hit is not None,
            }
        )
        if hit is not None:
            entropy, c, K = hit
            pi, Q, R, _ = problem.channel_seed(c)
            offer("entropy-sweep separating generator", (pi, Q, R, K))
            n_generating += 1
            cg_exceeds_g = False

    if "memory" in stages:
        for _value, residual, nu in memory_lift_search(bidir, 2, starts=memory_starts, rng=rng)[:3]:
            if residual <= residual_atol:
                offer("memory-1 lift", memory_lift(problem, nu))

    for name, builder in (("functional", functional_generative_model), ("lifted", minimal_generative_model)):
        if name not in stages or not small:
            if name in stages:
                log.append({"stage": f"{name} channel", "skipped": f"more than {large} causal-state pairs"})
            continue
        try:
            kwargs = {"rng": rng} if name == "lifted" else {}
            model = builder(bidir, **kwargs)
            offer(f"{name} channel", problem.channel_seed(_channel_matrix(problem, model.pair_state_channel)))
        except (StochasticValidationError, ImportError) as exc:
            log.append({"stage": f"{name} channel", "error": str(exc)})

    if "coupled" in stages:
        k_max = max_states or min(len(lifted.pairs), 8)
        extra = [(name, seed) for _v, name, seed in sorted(candidates, key=lambda t: t[0])[:6]] + repair
        for k in range(2, k_max + 1):
            for res in coupled_search(bidir, k, starts=starts, extra=extra, residual_atol=residual_atol, rng=rng)[:2]:
                offer(f"coupled k={k} ({res.start})", (res.pi, res.Q, res.R, res.K))

    result = _try_best_candidate(
        problem,
        reference,
        length,
        candidates,
        lower=lower,
        lower_source=lower_source,
        lower_certified=lower_certified,
        g_estimate=g_estimate,
        theorem=theorem,
        n_g_optima=n_g_optima,
        n_generating=n_generating,
        cg_exceeds_g=cg_exceeds_g,
        log=log,
        reproduction_atol=reproduction_atol,
    )
    if result is None:
        raise StochasticValidationError("generative_model: no candidate reproduced the source process")
    return result


__all__ = ["GenerativeModelResult", "generative_model"]
