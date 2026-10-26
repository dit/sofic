r"""Search for minimal-state-entropy generators, bracketing the generative complexity.

:func:`generative_model` runs a pipeline over nested model classes and returns the best generator it can
verify together with a rigorous bracket :math:`[\text{lower}, \text{upper}]` on the generative complexity
:math:`C_g` of Löhr & Ay :cite:`LohrAy2009`:

1. *Lower bounds.* The excess entropy :math:`\mathbf{E} = \op{I}{S^+ : S^-} \le \op{G}{S^+ : S^-} \le C_g`
   and, when the causal-state joint has few maximal rectangles, the support-type bound of
   :mod:`~sofic.generators.generator_bounds`.
2. *Theorem check.* If :math:`\H{S^+_{t+1} \mid S^-_t, X_t} = 0` and :math:`\H{S^-_t \mid S^+_{t+1}, X_t} = 0`
   (both persistent crypticities vanish), every variable separating :math:`S^+` from :math:`S^-` is a generator
   state, so :math:`C_g = \op{G}{S^+ : S^-}`; this is reported but needs no special handling, since the
   unconstrained optimum then passes the next test.
3. *Exact test of the unconstrained optima.* Exact-common-information channels (Kumar, Li & El Gamal
   :cite:`Kumar2014`, found with dit from several starts, since there can be several optima of which only some
   generate) are polished to exact separation and checked with the linear program of
   :func:`~sofic.generators.coupled_generators.generator_kernel`; if a kernel exists, :math:`C_g` equals the
   exact-common-information value found.
4. *Upper bounds over nested classes:* the ε-machines and bidirectional machine; deterministic channels
   (:func:`~sofic.generators.minimal_generative_model.functional_generative_model`); lifted channels
   (:func:`~sofic.generators.minimal_generative_model.minimal_generative_model`); memory-:math:`m` lifts
   (:func:`~sofic.generators.coupled_generators.memory_lift_search`, :math:`m = 1`); and general
   :math:`k`-state generators (:func:`~sofic.generators.coupled_generators.coupled_search`), seeded with every
   earlier solution.

Every candidate is checked against the source's word probabilities before it counts. The searches are local,
so ``upper`` is an upper bound; ``certified`` reports whether it meets ``lower``.
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
    generator_kernel,
    memory_lift,
    memory_lift_search,
    polish_separating_channel,
)
from sofic.generators.generator_bounds import maximal_rectangles, support_type_bound

if TYPE_CHECKING:
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine

_MAX_RECTANGLES = 7


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

    @property
    def certified(self) -> bool:
        """Whether the upper and lower bounds meet (to 1e-6 bits)."""
        return self.upper - self.lower <= 1e-6

    @property
    def gap(self) -> float:
        return self.upper - self.lower


def _persistent_criteria(problem: _GeneratorProblem) -> tuple[float, float]:
    """``H[S+_1 | S-_0, X_0]`` and ``H[S-_0 | S+_1, X_0]`` in bits (both zero iff both persistent crypticities
    vanish)."""
    lifted = problem.lifted
    J = np.einsum("pb,pxq,qa->bxa", lifted.minus, lifted.edges, lifted.plus)
    h_joint = _entropy(J.ravel())
    return h_joint - _entropy(J.sum(axis=2).ravel()), h_joint - _entropy(J.sum(axis=0).ravel())


def _channel_matrix(problem: _GeneratorProblem, channel: dict) -> np.ndarray:
    states = sorted({w for row in channel.values() for w in row}, key=repr)
    return np.array([[channel[pair].get(w, 0.0) for w in states] for pair in problem.lifted.pairs])


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
    exact_runs: int = 8,
    rng: np.random.Generator | None = None,
) -> GenerativeModelResult:
    """Run the generator-search pipeline of the module docstring.

    ``max_states`` caps the coupled search (default: the number of causal-state pairs, at most 8);
    ``starts`` and ``memory_starts`` are random starts per state count for the coupled and memory-lift
    searches; ``stages`` selects which upper-bound stages run. The functional and lifted stages are skipped
    when there are more than ``large`` causal-state pairs (they dominate the run time there and, on the examples
    tried, add nothing the memory-lift and coupled stages miss). Near-feasible candidates (e.g. an unconstrained
    optimum whose kernel program has a small violation) seed the coupled search for repair.
    """
    from sofic.generators.lifted_common_information import _unconstrained_channel
    from sofic.generators.minimal_generative_model import (
        _normalized_joint_distribution,
        _reproduction_error,
        _reproduction_length,
        functional_generative_model,
        minimal_generative_model,
    )

    rng = rng or np.random.default_rng()
    problem = _GeneratorProblem(bidir)
    lifted = problem.lifted
    log: list[dict[str, Any]] = []
    reference = bidir.forward_machine
    length = _reproduction_length(reference)

    # ---- lower bounds -------------------------------------------------------------------------------
    excess = _entropy(problem.p_plus) + _entropy(problem.p_minus) - _entropy(problem.p_pair)
    lower, lower_source = excess, "excess entropy"
    joint = {pair: float(m) for pair, m in zip(lifted.pairs, lifted.p, strict=True)}
    if len(maximal_rectangles(set(joint))) <= _MAX_RECTANGLES:
        support, _, _ = support_type_bound(joint)
        log.append({"stage": "support bound", "value": support})
        if support > lower:
            lower, lower_source = support, "support-type bound"
    log.append({"stage": "excess entropy", "value": excess})

    f_crit, r_crit = _persistent_criteria(problem)
    theorem = f_crit < 1e-10 and r_crit < 1e-10
    log.append({"stage": "persistent criteria", "value": (f_crit, r_crit), "theorem applies": theorem})

    # ---- upper bounds ---------------------------------------------------------------------------------
    candidates: list[tuple[float, str, tuple[np.ndarray, ...]]] = []
    repair: list[tuple[str, tuple[np.ndarray, ...]]] = []
    small = len(lifted.pairs) <= large

    def offer(name: str, seed: tuple[np.ndarray, ...]) -> None:
        residual = problem.residual(*seed)
        value = _entropy(seed[0])
        log.append({"stage": name, "value": value, "residual": residual})
        if residual <= residual_atol:
            candidates.append((value, name, seed))

    if "machines" in stages:
        offer("epsilon machine", problem.epsilon_seed())
        offer("reverse epsilon machine", problem.reverse_seed())
        offer("bidirectional machine", problem.pair_seed())

    g_estimate = None
    if "exact" in stages:
        # The unconstrained problem can have several optima, only some of them generator states; dit lands on
        # different ones from different starts, and satisfies the constraints only to a tolerance, so each run is
        # polished exactly before the kernel program decides.
        joint_n = _normalized_joint_distribution(bidir, cutoff=1e-10)
        for run in range(exact_runs):
            try:
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
            except (StochasticValidationError, ImportError):
                c = None
            if c is None:
                continue
            c = polish_separating_channel(problem, c)
            value = _entropy(lifted.p @ c)
            g_estimate = value if g_estimate is None else min(g_estimate, value)
            violation, K = generator_kernel(problem, c)
            pi, Q, R, _ = problem.channel_seed(c)
            log.append({"stage": f"exact common information, run {run}", "value": value, "kernel violation": violation})
            if violation <= 1e-9:
                offer("exact-common-information optimum", (pi, Q, R, K))
                break
            repair.append((f"exact-common-information optimum {run} (repair)", (pi, Q, R, K)))

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

    # ---- verify and choose -------------------------------------------------------------------------------
    for value, name, seed in sorted(candidates, key=lambda t: t[0]):
        result = CoupledSearchResult(*seed, value, problem.residual(*seed), name)
        model = _model_from_kernel(problem, result, 1e-9)
        if _reproduction_error(model, reference, max_length=length) <= reproduction_atol:
            upper = model.state_entropy()
            return GenerativeModelResult(
                model=model,
                upper=float(upper),
                lower=float(min(lower, upper)),
                upper_source=name,
                lower_source=lower_source,
                exact_common_information_estimate=g_estimate,
                separating_variables_generate=theorem,
                stages=log,
            )
    raise StochasticValidationError("generative_model: no candidate reproduced the source process")


__all__ = ["GenerativeModelResult", "generative_model"]
