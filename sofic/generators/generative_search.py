r"""Search for minimal-state-entropy generators, bracketing the generative complexity.

:func:`generative_model` builds (when needed) a bidirectional presentation and runs the full
pipeline. :func:`generative_model_from_bidirectional` runs the same search when the bidirectional
machine is already known. Both return the best generator they can verify together with a rigorous
bracket :math:`[\text{lower}, \text{upper}]` on the generative complexity
:math:`C_g` of Löhr & Ay :cite:`LohrAy2009`:

1. *Lower bounds.* Excess entropy; support-type and flow bounds when the joint has few maximal rectangles;
   the certified coverage bound (:mod:`~sofic.generators.coverage_bound`), the informative one on
   full supports, where the support-type bound vanishes.
2. *Persistent crypticity.* When both :math:`p.\chi^\pm` vanish on the bidirectional edge,
   :math:`C_g = \op{G}{S^+ : S^-}`.
3. *Exact optima + kernel LP.* Enumerate :math:`\op{G}{S^+ : S^-}` optima; optional
   :func:`~sofic.generators.coupled_generators.bilevel_search` when none generate; then an
   **entropy-ordered sweep** over all multistart separating decompositions (not only
   :math:`\op{G}` optima) up to the current best upper bound.
4. *Upper bounds:* ε-machines, memory lifts, the exact minimal functional generator
   (:func:`~sofic.generators.functional_generators.functional_cover_search`), lifted channels
   (small joints), coupled search.

Every candidate is word-verified. ``certified`` reports when ``lower`` meets ``upper``.
:func:`minimal_wyner_model` runs the same stages ranked by :math:`\I{W : S^+, S^-}`; the functional
variant lives in :mod:`~sofic.generators.functional_generators`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from sofic.exceptions import StochasticValidationError
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.coupled_generators import (
    CoupledGenerativeModel,
    CoupledSearchResult,
    _GeneratorProblem,
    _model_from_kernel,
    bilevel_search,
    coupled_search,
    first_generating_separator,
    generator_kernel,
    generator_state_cap,
    memory_lift,
    memory_lift_search,
    polish_separating_channel,
    separating_decomposition_channels,
    separating_optima,
    systematic_coupled_seeds,
)
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.generator_bounds import (
    flow_bound_result,
    maximal_rectangles,
    support_exact_cover_bound,
    support_type_bound,
)

_MAX_RECTANGLES = 7
_FLOW_THRESHOLDS = ((0.6,), (0.55, 0.65))
_CERT_TOL = 1e-6
# ``I[S+ : S- | W]`` (bits) above which a channel is not treated as an exact decomposition.
_SEPARATION_ATOL = 1e-9


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())


@dataclass
class GenerativeModelResult:
    """Best verified generator and a bracket on the minimized objective.

    ``objective`` names what ``upper`` and ``lower`` bound: ``"entropy"`` (state entropy, the generative
    complexity :math:`C_g`), ``"wyner"`` (:math:`\\I{W : S^+, S^-}` over generators) or ``"functional"``
    (state entropy over generators whose state is a function of :math:`(S^+, S^-)`).
    """

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
    objective: str = "entropy"

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


def _reliability_summary(result: GenerativeModelResult) -> dict[str, Any]:
    return {
        "stage": "reliability summary",
        "objective": result.objective,
        "certified": result.certified,
        "gap": result.gap,
        "upper": result.upper,
        "lower": result.lower,
        "upper_source": result.upper_source,
        "lower_source": result.lower_source,
        "lower_certified": result.lower_certified,
        "n_g_optima": result.n_g_optima,
        "n_g_optima_generating": result.n_g_optima_generating,
        "cg_exceeds_g": result.cg_exceeds_g,
        "separating_variables_generate": result.separating_variables_generate,
        "persistent_crypticity_theorem_applies": result.persistent_crypticity_theorem_applies,
    }


def _finalize_result(result: GenerativeModelResult) -> GenerativeModelResult:
    if not result.stages or result.stages[-1].get("stage") != "reliability summary":
        result.stages.append(_reliability_summary(result))
    return result


def _persistent_criteria(problem: _GeneratorProblem) -> tuple[float, float]:
    lifted = problem.lifted
    J = np.einsum("pb,pxq,qa->bxa", lifted.minus, lifted.edges, lifted.plus)
    h_joint = _entropy(J.ravel())
    return h_joint - _entropy(J.sum(axis=2).ravel()), h_joint - _entropy(J.sum(axis=0).ravel())


def _wyner_information(problem: _GeneratorProblem, seed: tuple[np.ndarray, ...]) -> float:
    r"""``I[W : S+, S-]`` of a generator seed: ``H[S+, S-] - sum_w pi_w (H[Q_w] + H[R_w])``.

    Every generator state separates the causal states, so ``H[S+, S- | W] = H[S+ | W] + H[S- | W]``.
    """
    pi, Q, R = seed[0], seed[1], seed[2]
    h_given_w = sum(float(pi[w]) * (_entropy(Q[w]) + _entropy(R[w])) for w in range(len(pi)))
    return max(0.0, _entropy(problem.p_pair) - h_given_w)


def _theorem_applies(problem: _GeneratorProblem, *, atol: float = 1e-10) -> bool:
    f_crit, r_crit = _persistent_criteria(problem)
    return f_crit < atol and r_crit < atol


def resolve_bidirectional(
    machine: BidirectionalEpsilonMachine | EpsilonMachine,
    *,
    reverse_max_states: int = 10_000,
    use_ladder: bool = True,
) -> tuple[BidirectionalEpsilonMachine, dict[str, Any]]:
    """Build or accept a bidirectional ε-machine for generator search."""
    from sofic.generators.bidirectional_build import try_build_bidirectional

    if isinstance(machine, BidirectionalEpsilonMachine):
        return machine, {"bidir_class": "finite", "source": "bidirectional input"}
    forward = machine if isinstance(machine, EpsilonMachine) else EpsilonMachine.from_hmm(machine)
    built = try_build_bidirectional(forward, reverse_max_states=reverse_max_states, use_ladder=use_ladder)
    meta: dict[str, Any] = {
        "bidir_class": built.preflight.bidir_class,
        "reverse_max_states": built.preflight.reverse_max_states,
        "reverse_recurrence_finite": built.preflight.reverse_recurrence_finite,
        "pe_drift": built.pe_drift,
        "edge_stabilized": built.edge_stabilized,
    }
    if built.preflight.error:
        meta["build_error"] = built.preflight.error
    if built.bidir is None:
        raise StochasticValidationError(f"generative_model: bidirectional build failed ({built.preflight.bidir_class})")
    return built.bidir, meta


def _channel_matrix(problem: _GeneratorProblem, channel: dict) -> np.ndarray:
    states = sorted({w for row in channel.values() for w in row}, key=repr)
    return np.array([[channel[pair].get(w, 0.0) for w in states] for pair in problem.lifted.pairs])


def _coverage_lower_bound(
    joint: dict[tuple, float],
    rng: np.random.Generator,
    *,
    mu_starts: int,
    certified: bool,
    k_max: int = 8,
    certified_only: bool = False,
) -> tuple[float, dict[str, float], bool]:
    from sofic.generators.coverage_bound import (
        coverage_bound,
        coverage_entropy_lower,
        joint_matrix,
        mu1_search,
        mu_k_bilinear,
    )

    m, _, _ = joint_matrix(joint)
    seed = int(rng.integers(0, 2**31))
    if certified:
        cov = coverage_bound(m, k_max=k_max, seed=seed)
        meta: dict[str, float | int | bool | tuple[float, ...]] = {
            "mu_1": cov.mu1,
            "k_used": cov.k_used,
            "caps": cov.caps,
            "certified": cov.certified,
        }
        if certified_only and not cov.certified:
            return float("-inf"), {**meta, "skipped": True}, False
        return cov.value, meta, cov.certified
    if certified_only:
        return float("-inf"), {"certified": False, "skipped": True}, False
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
    objective: str = "entropy",
) -> GenerativeModelResult | None:
    from sofic.generators.minimal_generative_model import _reproduction_error

    g_generates = n_generating > 0
    sep_gen = g_generates if n_g_optima > 0 else theorem

    for value, name, seed in sorted(candidates, key=lambda t: t[0]):
        pi, Q, R, K = seed
        result = CoupledSearchResult(pi, Q, R, K, _entropy(pi), problem.residual(pi, Q, R, K), name)
        model = _model_from_kernel(problem, result, 1e-9)
        if _reproduction_error(model, reference, max_length=length) > reproduction_atol:
            continue
        upper = float(value) if objective == "wyner" else float(model.state_entropy())
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
            objective=objective,
        )
    return None


def _bracket_search(
    bidir: BidirectionalEpsilonMachine,
    *,
    objective: Literal["entropy", "wyner"],
    max_states: int | None = None,
    starts: int = 20,
    memory_starts: int = 10,
    residual_atol: float = 1e-7,
    reproduction_atol: float = 1e-5,
    stages: Sequence[str] = ("machines", "exact", "memory", "functional", "lifted", "coupled"),
    large: int = 9,
    exact_starts: int = 30,
    dit_fallback: bool = True,
    coverage: Literal["off", "heuristic", "certified"] = "certified",
    coverage_mu_starts: int = 25,
    coverage_k_max: int = 8,
    coverage_certified_only: bool = False,
    entropy_sweep_starts: int = 80,
    bilevel: bool = True,
    early_certified: bool = True,
    early_theorem_g_exit: bool = False,
    universal_coupled: bool = False,
    coupled_alternating: bool = False,
    coupled_dual_strategy: bool = True,
    rng: np.random.Generator | None = None,
) -> GenerativeModelResult:
    """Generator search ranked by state entropy or by ``I[W : S+, S-]``; see the module docstring."""
    from sofic.generators.functional_generators import functional_cover_search
    from sofic.generators.lifted_common_information import _unconstrained_channel
    from sofic.generators.minimal_generative_model import (
        _normalized_joint_distribution,
        minimal_generative_model,
        wyner_generative_model,
    )

    wyner = objective == "wyner"
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
    if wyner:
        log.append({"stage": "entropy lower bounds", "skipped": "they bound H[W], not I[W : S+, S-]"})

    if not wyner and n_rect <= _MAX_RECTANGLES:
        exact_cover = support_exact_cover_bound(joint)
        if exact_cover is not None:
            log.append(
                {
                    "stage": "exact rectangle partition",
                    "value": exact_cover.value,
                    "certified": exact_cover.certified,
                }
            )
            if exact_cover.value > lower + 1e-9:
                lower, lower_source = exact_cover.value, "exact rectangle partition"
                lower_certified = True
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

    if not wyner and coverage in ("heuristic", "certified"):
        try:
            use_certified = coverage == "certified"
            cov, meta, caps_ok = _coverage_lower_bound(
                joint,
                rng,
                mu_starts=coverage_mu_starts,
                certified=use_certified,
                k_max=coverage_k_max,
                certified_only=coverage_certified_only,
            )
            if cov > float("-inf"):
                stage = "coverage bound (certified)" if caps_ok else "coverage bound (heuristic)"
                if use_certified and not caps_ok and coverage_certified_only:
                    stage = "coverage bound (skipped: not certified)"
                log.append({"stage": stage, "value": cov, **meta})
                if cov > lower + 1e-9:
                    lower, lower_source = cov, stage
                    lower_certified = caps_ok if use_certified else False
            elif coverage_certified_only:
                log.append({"stage": "coverage bound", "skipped": True, "certified": False})
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
    systematic_channels: list[np.ndarray] = []

    def offer(name: str, seed: tuple[np.ndarray, ...]) -> None:
        residual = problem.residual(*seed)
        value = _wyner_information(problem, seed) if wyner else _entropy(seed[0])
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
            objective=objective,
        )

    if "machines" in stages:
        offer("epsilon machine", problem.epsilon_seed())
        offer("reverse epsilon machine", problem.reverse_seed())
        offer("bidirectional machine", problem.pair_seed())
        early = maybe_return_early()
        if early is not None and early.certified:
            return _finalize_result(early)

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
                    separation = float(lifted.constraint_informations(c)[0])
                    value = _entropy(lifted.p @ c)
                    log.append({"stage": "dit exact common information", "value": value, "separation": separation})
                    if separation <= _SEPARATION_ATOL:
                        if g_estimate is None or value < g_estimate - 1e-9:
                            g_estimate, optima = value, [c]
                        elif value <= g_estimate + 1e-7:
                            optima.append(c)
            except (StochasticValidationError, ImportError):
                pass
        if wyner and dit_fallback:
            try:
                c = _unconstrained_channel(
                    lifted,
                    _normalized_joint_distribution(bidir, cutoff=1e-10),
                    "wyner",
                    niter=10,
                    maxiter=1000,
                    polish=1e-8,
                    backend="numpy",
                    cutoff=1e-10,
                    rng=rng,
                )
            except (StochasticValidationError, ImportError):
                c = None
            if c is not None:
                violation, K = generator_kernel(problem, c)
                log.append(
                    {
                        "stage": "Wyner common information",
                        "value": lifted.objective(c, "wyner")[0],
                        "kernel violation": violation,
                    }
                )
                pi, Q, R, _ = problem.channel_seed(c)
                if violation <= 1e-9:
                    offer("Wyner-common-information optimum", (pi, Q, R, K))
                else:
                    repair.append(("Wyner-common-information optimum (repair)", (pi, Q, R, K)))
        n_g_optima = len(optima)
        systematic_channels = list(optima)
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
                systematic_channels.append(c)
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
            max_h = best_upper if np.isfinite(best_upper) and not wyner else None
            if lower_certified and not wyner:
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
            systematic_channels.append(c)
            pi, Q, R, _ = problem.channel_seed(c)
            offer("entropy-sweep separating generator", (pi, Q, R, K))
            n_generating += 1
            cg_exceeds_g = False

        if early_theorem_g_exit and not wyner and theorem and n_generating > 0:
            early_g = _try_best_candidate(
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
            if early_g is not None:
                return _finalize_result(early_g)

        repair.extend(systematic_coupled_seeds(problem, systematic_channels))
        log.append(
            {
                "stage": "systematic coupled seeds",
                "channels": len(systematic_channels),
                "seeds": len(repair),
            }
        )

    if "memory" in stages:
        for _value, residual, nu in memory_lift_search(bidir, 2, starts=memory_starts, rng=rng)[:3]:
            if residual <= residual_atol:
                offer("memory-1 lift", memory_lift(problem, nu))

    if "functional" in stages:
        try:
            cover = functional_cover_search(problem)
        except StochasticValidationError as exc:
            log.append({"stage": "functional cover search", "error": str(exc)})
        else:
            log.append(
                {
                    "stage": "functional cover search",
                    "value": cover.value,
                    "exact": cover.exact,
                    "rejected covers": cover.rejected,
                }
            )
            offer("minimal functional generator", problem.channel_seed(np.eye(cover.labels.max() + 1)[cover.labels]))

    if "lifted" in stages:
        if small:
            builder = wyner_generative_model if wyner else minimal_generative_model
            try:
                model = builder(bidir, rng=rng)
                offer("lifted channel", problem.channel_seed(_channel_matrix(problem, model.pair_state_channel)))
            except (StochasticValidationError, ImportError) as exc:
                log.append({"stage": "lifted channel", "error": str(exc)})
        else:
            log.append({"stage": "lifted channel", "skipped": f"more than {large} causal-state pairs"})

    if "coupled" in stages:
        hard_cap = max_states if max_states is not None else 8
        k_cap = generator_state_cap(problem, hard_cap=hard_cap)
        log.append({"stage": "coupled state cap", "k_max": k_cap})
        if universal_coupled:
            from sofic.generators.universal_optimizer import universal_generative_optimizer

            try:
                uni = universal_generative_optimizer(
                    bidir,
                    max_states=k_cap,
                    starts=starts,
                    maxiter=3000,
                    residual_atol=residual_atol,
                    reproduction_atol=reproduction_atol,
                    alternating=coupled_alternating,
                    dual_strategy=coupled_dual_strategy,
                    rng=rng,
                )
            except StochasticValidationError as exc:
                log.append({"stage": "universal coupled search", "error": str(exc)})
            else:
                log.append(
                    {
                        "stage": "universal coupled search",
                        "value": uni.value,
                        "k": uni.k,
                        "start": uni.start,
                        "n_verified": uni.n_verified,
                        "by_k": uni.by_k,
                    }
                )
                if uni.search is not None:
                    s = uni.search
                    offer(f"universal coupled k={uni.k} ({s.start})", (s.pi, s.Q, s.R, s.K))
        else:
            extra = [(name, seed) for _v, name, seed in sorted(candidates, key=lambda t: t[0])[:6]] + repair
            for k in range(2, k_cap + 1):
                for res in coupled_search(
                    bidir,
                    k,
                    starts=starts,
                    extra=extra,
                    residual_atol=residual_atol,
                    alternating=coupled_alternating,
                    dual_strategy=coupled_dual_strategy,
                    rng=rng,
                )[: None if wyner else 2]:
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
        objective=objective,
    )
    if result is None:
        raise StochasticValidationError(f"{objective} generator search: no candidate reproduced the source process")
    return _finalize_result(result)


def generative_model_from_bidirectional(
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
    coverage: Literal["off", "heuristic", "certified"] = "certified",
    coverage_mu_starts: int = 25,
    coverage_k_max: int = 8,
    coverage_certified_only: bool = False,
    entropy_sweep_starts: int = 80,
    bilevel: bool = True,
    early_certified: bool = True,
    early_theorem_g_exit: bool = False,
    universal_coupled: bool = False,
    coupled_alternating: bool = False,
    coupled_dual_strategy: bool = True,
    rng: np.random.Generator | None = None,
) -> GenerativeModelResult:
    """Run the generator-search pipeline on an existing bidirectional machine; see the module docstring."""
    return _bracket_search(
        bidir,
        objective="entropy",
        max_states=max_states,
        starts=starts,
        memory_starts=memory_starts,
        residual_atol=residual_atol,
        reproduction_atol=reproduction_atol,
        stages=stages,
        large=large,
        exact_starts=exact_starts,
        dit_fallback=dit_fallback,
        coverage=coverage,
        coverage_mu_starts=coverage_mu_starts,
        coverage_k_max=coverage_k_max,
        coverage_certified_only=coverage_certified_only,
        entropy_sweep_starts=entropy_sweep_starts,
        bilevel=bilevel,
        early_certified=early_certified,
        early_theorem_g_exit=early_theorem_g_exit,
        universal_coupled=universal_coupled,
        coupled_alternating=coupled_alternating,
        coupled_dual_strategy=coupled_dual_strategy,
        rng=rng,
    )


def minimal_wyner_model(
    machine: BidirectionalEpsilonMachine | EpsilonMachine,
    *,
    reverse_max_states: int = 10_000,
    use_ladder: bool = True,
    max_states: int | None = None,
    starts: int = 20,
    memory_starts: int = 10,
    residual_atol: float = 1e-7,
    reproduction_atol: float = 1e-5,
    large: int = 9,
    exact_starts: int = 30,
    dit_fallback: bool = True,
    entropy_sweep_starts: int = 80,
    bilevel: bool = True,
    coupled_alternating: bool = False,
    coupled_dual_strategy: bool = True,
    rng: np.random.Generator | None = None,
) -> GenerativeModelResult:
    r"""Generator minimizing :math:`\I{W : S^+, S^-}`, the Wyner analogue of :func:`generative_model`.

    Every generator state separates the causal states :cite:`Ellison2009`, so
    :math:`\I{W : S^+, S^-}` is at least the Wyner common information :math:`\op{C}{S^+ : S^-}`
    :cite:`Wyner1975`, which is at least the excess entropy :math:`\I{S^+ : S^-}`; when both
    persistent crypticities vanish the minimum over generators equals :math:`\op{C}{S^+ : S^-}`.

    The candidates are those of :func:`generative_model` (ε-machines, exact-common-information optima,
    the separator sweep, memory lifts, functional and lifted channels, coupled search) plus the lifted
    Wyner model and the unconstrained Wyner optimum when it passes the generator kernel. They are ranked
    by :math:`\I{W : S^+, S^-} = \H{S^+, S^-} - \sum_w \pi_w (\H{S^+ \mid w} + \H{S^- \mid w})` and
    word-verified. The state-entropy lower bounds do not apply, so ``lower`` is the excess entropy and
    the bracket is certified only when a candidate attains it.
    """
    bidir, build_meta = resolve_bidirectional(machine, reverse_max_states=reverse_max_states, use_ladder=use_ladder)
    result = _bracket_search(
        bidir,
        objective="wyner",
        max_states=max_states,
        starts=starts,
        memory_starts=memory_starts,
        residual_atol=residual_atol,
        reproduction_atol=reproduction_atol,
        large=large,
        exact_starts=exact_starts,
        dit_fallback=dit_fallback,
        coverage="off",
        entropy_sweep_starts=entropy_sweep_starts,
        bilevel=bilevel,
        coupled_alternating=coupled_alternating,
        coupled_dual_strategy=coupled_dual_strategy,
        rng=rng,
    )
    result.stages.insert(0, {"stage": "bidirectional build", **build_meta})
    return result


def generative_model(
    machine: BidirectionalEpsilonMachine | EpsilonMachine,
    *,
    reverse_max_states: int = 10_000,
    use_ladder: bool = True,
    max_states: int | None = None,
    starts: int = 20,
    memory_starts: int = 10,
    residual_atol: float = 1e-7,
    reproduction_atol: float = 1e-5,
    large: int = 9,
    exact_starts: int = 30,
    dit_fallback: bool = True,
    coverage: Literal["off", "heuristic", "certified"] = "certified",
    coverage_mu_starts: int = 25,
    coverage_k_max: int = 8,
    coverage_certified_only: bool = False,
    entropy_sweep_starts: int = 80,
    bilevel: bool = True,
    early_certified: bool = True,
    coupled_alternating: bool = False,
    coupled_dual_strategy: bool = True,
    rng: np.random.Generator | None = None,
) -> GenerativeModelResult:
    """Full generative-complexity search from a forward ε-machine or bidirectional presentation.

    1. Build a bidirectional ε-machine (preflight / optional retry ladder).
    2. Certified lower bounds and persistent-crypticity test on the edge.
    3. When :math:`p.\\chi^+ \\approx p.\\chi^- \\approx 0` and some :math:`\\op{G}{S^+ : S^-}` optimum
       passes the generator kernel, return that lifted / exact-CI generator (theorem path).
    4. Otherwise memory lifts, lifted channels on small joints, then
       :func:`~sofic.generators.universal_optimizer.universal_generative_optimizer`
       (full coupled equations).

    The final ``stages`` entry is a **reliability summary** (``certified``, ``gap``, sources, …).
    """
    bidir, build_meta = resolve_bidirectional(machine, reverse_max_states=reverse_max_states, use_ladder=use_ladder)
    result = generative_model_from_bidirectional(
        bidir,
        max_states=max_states,
        starts=starts,
        memory_starts=memory_starts,
        residual_atol=residual_atol,
        reproduction_atol=reproduction_atol,
        stages=("machines", "exact", "memory", "functional", "lifted", "coupled"),
        large=large,
        exact_starts=exact_starts,
        dit_fallback=dit_fallback,
        coverage=coverage,
        coverage_mu_starts=coverage_mu_starts,
        coverage_k_max=coverage_k_max,
        coverage_certified_only=coverage_certified_only,
        entropy_sweep_starts=entropy_sweep_starts,
        bilevel=bilevel,
        early_certified=early_certified,
        early_theorem_g_exit=True,
        universal_coupled=True,
        coupled_alternating=coupled_alternating,
        coupled_dual_strategy=coupled_dual_strategy,
        rng=rng,
    )
    result.stages.insert(0, {"stage": "bidirectional build", **build_meta})
    return result


def ultimate_generative_model(
    machine: BidirectionalEpsilonMachine | EpsilonMachine,
    **kwargs: Any,
) -> GenerativeModelResult:
    """Deprecated alias for :func:`generative_model`."""
    warnings.warn(
        "ultimate_generative_model is renamed to generative_model; update call sites",
        DeprecationWarning,
        stacklevel=2,
    )
    return generative_model(machine, **kwargs)


__all__ = [
    "GenerativeModelResult",
    "generative_model",
    "generative_model_from_bidirectional",
    "minimal_wyner_model",
    "resolve_bidirectional",
    "ultimate_generative_model",
]
