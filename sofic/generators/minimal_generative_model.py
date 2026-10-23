"""Minimal generative models from bidirectional epsilon-machines."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from sofic.exceptions import StochasticValidationError
from sofic.generators.mealy import MealyHMM
from sofic.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph

if TYPE_CHECKING:
    from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine


_EXACT_ATOL = 1e-10


class _CommonInformationGenerativeModel(MealyHMM):
    pair_state_channel: dict[tuple[Hashable, Hashable], dict[Hashable, float]]
    joint_pair_distribution: dict[tuple[Hashable, Hashable], float]
    source_bidirectional: BidirectionalEpsilonMachine | None
    markov_residual: float
    _entropy_rate: float | None

    def __init__(
        self,
        *,
        pair_state_channel: dict[tuple[Hashable, Hashable], dict[Hashable, float]] | None = None,
        joint_pair_distribution: dict[tuple[Hashable, Hashable], float] | None = None,
        source_bidirectional: BidirectionalEpsilonMachine | None = None,
        entropy_rate: float | None = None,
        markov_residual: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pair_state_channel = pair_state_channel or {}
        self.joint_pair_distribution = joint_pair_distribution or {}
        self.source_bidirectional = source_bidirectional
        self.markov_residual = float(markov_residual)
        self._entropy_rate = entropy_rate

    def entropy_rate(self) -> float:
        """Return the source process entropy rate when the source is known."""
        if self._entropy_rate is not None:
            return float(self._entropy_rate)
        return super().entropy_rate()

    def generative_complexity(self) -> float:
        """State entropy ``C_g`` of the generator."""
        return self.state_entropy()


class MinimalGenerativeModel(_CommonInformationGenerativeModel):
    """Non-unifilar minimal-state-entropy generator.

    The states are an auxiliary variable ``G`` of the forward and reverse causal
    states that renders them conditionally independent and lifts to a generator
    of the process (see :func:`minimal_generative_model`).
    ``exact_common_information`` is ``H[G]``, an upper bound on the generative
    complexity, and ``markov_residual`` the largest constraint conditional mutual
    information (bits) at the returned channel.
    """

    exact_common_information: float

    def __init__(self, *, exact_common_information: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.exact_common_information = float(exact_common_information)


class WynerGenerativeModel(_CommonInformationGenerativeModel):
    """Non-unifilar generator from a Wyner-common-information auxiliary.

    The optimized value is ``I[(S+, S-) : G]`` over auxiliaries that lift to a
    generator (see :func:`wyner_generative_model`), stored as
    ``wyner_common_information``. The model's state entropy is ``H[G]`` and can
    be strictly larger.
    """

    wyner_common_information: float

    def __init__(self, *, wyner_common_information: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wyner_common_information = float(wyner_common_information)


class FunctionalGenerativeModel(_CommonInformationGenerativeModel):
    """Deterministic generator from a functional-common-information auxiliary.

    The auxiliary state ``G`` is a *deterministic function* of the joint causal
    state ``(S+, S-)`` -- the smallest such function rendering ``S+`` and ``S-``
    conditionally independent and lifting to a generator (see
    :func:`functional_generative_model`). ``H[G]`` is stored as
    ``functional_common_information``; because ``G`` is deterministic, the
    model's state entropy equals that value exactly.
    """

    functional_common_information: float

    def __init__(self, *, functional_common_information: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.functional_common_information = float(functional_common_information)


class GacsKornerGenerativeModel(_CommonInformationGenerativeModel):
    """Gács-Körner common-information model of the meet ``S+ ⩘ S-``.

    The states are the meet of the forward and reverse causal states: the
    largest random variable that is simultaneously a deterministic function of
    both :cite:`GacsKorner1973`. Unlike the Exact and Wyner models this is
    combinatorial, not variational — each joint state pair maps to exactly one
    state (its connected component in the joint support graph). Its state
    entropy therefore equals the Gács-Körner common information
    ``K[S+ : S-]`` exactly, and captures only the conserved "core" (phase /
    ergodic-component structure), which is often trivial for mixing processes.

    This is *not* a generative model of the source process: the meet need not
    render ``S+`` and ``S-`` conditionally independent, so the HMM obtained by
    coarse-graining the bidirectional machine onto it generally emits a
    different process (``is_equal_process(source)`` is ``False`` in general,
    e.g. the golden mean collapses to an i.i.d. coin). Accordingly
    :meth:`entropy_rate` is this model's own rate, not the source's.
    """

    def entropy_rate(self) -> float:
        """Entropy rate of this model's own output process (not the source's)."""
        return MealyHMM.entropy_rate(self)

    gk_common_information: float

    def __init__(self, *, gk_common_information: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gk_common_information = float(gk_common_information)


def minimal_generative_model(
    bidir: BidirectionalEpsilonMachine,
    *,
    bound: int | None = None,
    niter: int | None = 10,
    maxiter: int = 1000,
    polish: float | bool = 1e-8,
    backend: str = "numpy",
    starts: int = 8,
    markov_atol: float = 1e-8,
    cutoff: float = 1e-10,
    reproduction_atol: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> MinimalGenerativeModel:
    """Construct a minimal generative model from a bidirectional epsilon-machine.

    The returned model is an edge-emitting HMM over a generative state ``G``,
    built by applying ``p(g | s+, s-)`` at both ends of every bidirectional edge.
    ``H[G]`` is minimized subject to ``I[S+ : S- | G] = 0`` (the exact common
    information :cite:`Kumar2014`) and the two conditions under which that HMM
    generates the source, ``I[G_0 X_0 : S-_1 | G_1] = 0`` and
    ``I[S+_0 : X_0 G_1 | G_0] = 0`` (see
    :mod:`sofic.generators.lifted_common_information`). ``exact_common_information``
    is the model's own state entropy, an upper bound on the generative complexity
    :cite:`LohrAy2009`; the unconstrained value is
    :meth:`~sofic.generators.bidirectional_epsilon_machine.BidirectionalEpsilonMachine.exact_common_information`.
    Lifted generators draw consecutive states independently given the
    bidirectional edge; a generator that couples them can have smaller state
    entropy, so this is not necessarily the minimum over all generators.

    The starts are ``S+``, ``S-``, ``(S+, S-)``, the unconstrained optimum (found
    with dit; ``niter``, ``maxiter``, ``polish`` and ``backend`` as in
    :class:`dit.multivariate.common_informations.ExactCommonInformation`),
    perturbations of it (blended with random channels), time-sharing mixtures of
    it with each of the three ε-machine channels, and ``starts`` random channels
    over ``bound`` states (default: the number of joint states). Constraint informations up to ``markov_atol`` bits are
    accepted and the best candidates refined toward exact closure; a residual
    ``r`` allows the value and word probabilities to be off by roughly
    ``sqrt(r)``. The model is checked against the source up to
    ``reproduction_atol`` in word probability.
    """
    from sofic.generators.lifted_common_information import lifted_model

    return cast(
        MinimalGenerativeModel,
        lifted_model(
            bidir,
            objective="exact",
            bound=bound,
            starts=starts,
            niter=niter,
            maxiter=maxiter,
            polish=polish,
            backend=backend,
            markov_atol=markov_atol,
            reproduction_atol=reproduction_atol,
            cutoff=cutoff,
            rng=rng,
        ),
    )


def wyner_generative_model(
    bidir: BidirectionalEpsilonMachine,
    *,
    bound: int | None = None,
    niter: int | None = 10,
    maxiter: int = 1000,
    polish: float | bool = 1e-8,
    backend: str = "numpy",
    starts: int = 8,
    markov_atol: float = 1e-8,
    cutoff: float = 1e-10,
    reproduction_atol: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> WynerGenerativeModel:
    """Construct a Wyner generative model from a bidirectional epsilon-machine.

    The auxiliary state ``G`` minimizes ``I[(S+, S-) : G]`` :cite:`Wyner1975`
    subject to rendering ``S+`` and ``S-`` conditionally independent and to the
    two lifting conditions of :func:`minimal_generative_model`, whose parameters
    it shares, and like it searches lifted generators only. The returned model's
    state entropy ``H[G]`` is not generally equal to that optimized mutual
    information. The unconstrained value is
    :meth:`~sofic.generators.bidirectional_epsilon_machine.BidirectionalEpsilonMachine.wyner_common_information`.
    """
    from sofic.generators.lifted_common_information import lifted_model

    return cast(
        WynerGenerativeModel,
        lifted_model(
            bidir,
            objective="wyner",
            bound=bound,
            starts=starts,
            niter=niter,
            maxiter=maxiter,
            polish=polish,
            backend=backend,
            markov_atol=markov_atol,
            reproduction_atol=reproduction_atol,
            cutoff=cutoff,
            rng=rng,
        ),
    )


def wyner_common_information(
    bidir: BidirectionalEpsilonMachine,
    *,
    bound: int | None = None,
    niter: int | None = 10,
    maxiter: int = 1000,
    polish: float | bool = 1e-8,
    backend: str = "numpy",
    cutoff: float = 1e-10,
    rng: np.random.Generator | None = None,
) -> float:
    """Wyner common information ``C[S+ : S-]`` :cite:`Wyner1975` of the causal states.

    Because the past and future are conditionally independent given either causal
    state :cite:`Ellison2009`, this equals the Wyner common information of the
    past and future. Its optimizing auxiliary need not be the state of a
    generator; :func:`wyner_generative_model` adds that requirement. Parameters
    mirror :class:`dit.multivariate.common_informations.WynerCommonInformation`.
    """
    return _unconstrained_common_information(
        bidir,
        _optimize_wyner_common_information,
        "Wyner common information",
        bound=bound,
        niter=niter,
        maxiter=maxiter,
        polish=polish,
        backend=backend,
        cutoff=cutoff,
        rng=rng,
    )


def exact_common_information(
    bidir: BidirectionalEpsilonMachine,
    *,
    bound: int | None = None,
    niter: int | None = 10,
    maxiter: int = 1000,
    polish: float | bool = 1e-8,
    backend: str = "numpy",
    cutoff: float = 1e-10,
    rng: np.random.Generator | None = None,
) -> float:
    """Exact common information ``G[S+ : S-]`` :cite:`Kumar2014` of the causal states.

    The analogue of :func:`wyner_common_information` for ``H[G]``; it equals the
    exact common information of the past and future, and its optimizing
    auxiliary need not be the state of a generator (compare
    :func:`minimal_generative_model`).
    """
    return _unconstrained_common_information(
        bidir,
        _optimize_exact_common_information,
        "exact common information",
        bound=bound,
        niter=niter,
        maxiter=maxiter,
        polish=polish,
        backend=backend,
        cutoff=cutoff,
        rng=rng,
    )


def _unconstrained_common_information(
    bidir: BidirectionalEpsilonMachine,
    optimizer: Any,
    optimizer_name: str,
    *,
    bound: int | None,
    niter: int | None,
    maxiter: int,
    polish: float | bool,
    backend: str,
    cutoff: float,
    rng: np.random.Generator | None,
) -> float:
    if cutoff < 0.0:
        raise ValueError("cutoff must be nonnegative")
    joint = _normalized_joint_distribution(bidir, cutoff=cutoff)
    _channel, value = _auxiliary_state_channel(
        joint,
        optimizer=optimizer,
        optimizer_name=optimizer_name,
        bound=bound,
        niter=niter,
        maxiter=maxiter,
        polish=polish,
        backend=backend,
        cutoff=cutoff,
        rng=rng,
    )
    return value


@dataclass(frozen=True)
class GenerativeGap:
    """Gap between a common information of ``(S+, S-)`` and its value over lifted generator states.

    Attributes
    ----------
    measure
        ``"exact"`` or ``"wyner"``.
    lower
        The unconstrained common information: ``G[S+ : S-]`` or ``C[S+ : S-]``.
    upper
        The same quantity over lifted generator states: ``H[G]`` of
        :func:`minimal_generative_model`, or ``I[(S+, S-) : G]`` of
        :func:`wyner_generative_model`.
    model
        The generator achieving ``upper``.
    """

    measure: str
    lower: float
    upper: float
    model: MinimalGenerativeModel | WynerGenerativeModel

    @property
    def gap(self) -> float:
        """``upper - lower``, in bits."""
        return self.upper - self.lower


def generative_gap(
    bidir: BidirectionalEpsilonMachine,
    *,
    measure: str = "exact",
    bound: int | None = None,
    niter: int | None = 10,
    maxiter: int = 1000,
    polish: float | bool = 1e-8,
    backend: str = "numpy",
    starts: int = 8,
    markov_atol: float = 1e-8,
    cutoff: float = 1e-10,
    reproduction_atol: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> GenerativeGap:
    """How much requiring a lifted generator raises the exact (or Wyner) common information.

    Every generator state renders ``S+`` and ``S-`` conditionally independent, so
    for ``measure="exact"``

    .. math:: \\op{G}{S^+ : S^-} \\le C_g \\le \\H{G},

    with ``G`` the state of :func:`minimal_generative_model` and ``C_g`` the
    generative complexity :cite:`LohrAy2009`: ``gap`` bounds ``C_g - G[S+ : S-]``
    from above. Because only lifted generators are searched, ``gap`` can be
    positive even when a non-lifted generator attains the unconstrained value (see
    :mod:`sofic.generators.lifted_common_information`). ``measure="wyner"``
    compares ``C[S+ : S-]`` with :func:`wyner_generative_model`. Both sides come
    from local optimizers, so the gap is an estimate; it vanishes for first-order
    Markov processes. Since both sides upper-bound the true unconstrained value,
    ``lower`` is the smaller of the two. Parameters are those of
    :func:`minimal_generative_model`.
    """
    if measure not in ("exact", "wyner"):
        raise ValueError(f"measure must be 'exact' or 'wyner', not {measure!r}")
    rng = rng or np.random.default_rng()
    optimizer_kwargs = {
        "bound": bound,
        "niter": niter,
        "maxiter": maxiter,
        "polish": polish,
        "backend": backend,
        "cutoff": cutoff,
        "rng": rng,
    }
    if measure == "exact":
        lower = exact_common_information(bidir, **optimizer_kwargs)
        model = minimal_generative_model(
            bidir, starts=starts, markov_atol=markov_atol, reproduction_atol=reproduction_atol, **optimizer_kwargs
        )
        upper = model.exact_common_information
    else:
        lower = wyner_common_information(bidir, **optimizer_kwargs)
        model = wyner_generative_model(
            bidir, starts=starts, markov_atol=markov_atol, reproduction_atol=reproduction_atol, **optimizer_kwargs
        )
        upper = model.wyner_common_information
    return GenerativeGap(measure=measure, lower=min(lower, upper), upper=upper, model=model)


def functional_generative_model(
    bidir: BidirectionalEpsilonMachine,
    *,
    cutoff: float = 1e-10,
    strategy: str = "auto",
) -> FunctionalGenerativeModel:
    """Construct a functional generative model from a bidirectional epsilon-machine.

    The auxiliary state ``G`` is the smallest-entropy *deterministic function* of
    the joint causal state ``(S+, S-)`` that renders ``S+`` and ``S-``
    conditionally independent and satisfies the two lifting conditions of
    :func:`minimal_generative_model`, so the model always generates the source
    (``G = (S+, S-)``, the bidirectional machine, is always feasible). Its entropy
    is exposed as ``functional_common_information``; because ``G`` is
    deterministic, the model's state entropy equals that value exactly.

    dit's functional-common-information partition (``strategy`` is passed to it)
    is used when it already satisfies the lifting conditions, in which case it is
    optimal. Otherwise the partitions of the joint support are searched
    exhaustively when there are at most nine joint states, and greedily (merging
    blocks of ``(S+, S-)`` while the conditions hold) beyond that.
    """
    if cutoff < 0.0:
        raise ValueError("cutoff must be nonnegative")

    joint = _normalized_joint_distribution(bidir, cutoff=cutoff)
    pair_channel, functional_common_information = _functional_state_channel(
        bidir,
        joint,
        cutoff=cutoff,
        strategy=strategy,
    )
    return cast(
        FunctionalGenerativeModel,
        _model_from_channel(
            bidir,
            joint,
            pair_channel,
            model_cls=FunctionalGenerativeModel,
            model_name="functional generative model",
            measure_kwargs={"functional_common_information": functional_common_information},
            cutoff=cutoff,
        ),
    )


def gacs_korner_generative_model(
    bidir: BidirectionalEpsilonMachine,
    *,
    cutoff: float = 1e-10,
) -> GacsKornerGenerativeModel:
    """Construct the Gács-Körner common-information model of a bidirectional epsilon-machine.

    The state ``G`` is the meet ``S+ ⩘ S-`` of the forward and reverse causal
    states — the largest random variable that is simultaneously a deterministic
    function of both, obtained combinatorially as the connected components of
    the joint support graph :cite:`GacsKorner1973`. Because the meet is
    deterministic there is nothing to optimize, so this factory takes no
    optimizer arguments; the returned model's state entropy ``H[G]`` equals the
    Gács-Körner common information ``K[S+ : S-]``.

    Unlike the other constructors here, the result is not verified against the
    source and does not reproduce it in general: it is the common-information
    variable dressed as an HMM, not a generator of the process (see
    :class:`GacsKornerGenerativeModel`).
    """
    if cutoff < 0.0:
        raise ValueError("cutoff must be nonnegative")

    joint = _normalized_joint_distribution(bidir, cutoff=cutoff)
    pair_channel = _gacs_korner_meet_channel(joint)
    component_mass: dict[Hashable, float] = defaultdict(float)
    for pair, pair_mass in joint.items():
        (state,) = pair_channel[pair]
        component_mass[state] += pair_mass
    gk_common_information = _entropy(component_mass.values())
    return cast(
        GacsKornerGenerativeModel,
        _model_from_channel(
            bidir,
            joint,
            pair_channel,
            model_cls=GacsKornerGenerativeModel,
            model_name="Gács-Körner generative model",
            measure_kwargs={"gk_common_information": gk_common_information},
            cutoff=cutoff,
        ),
    )


def _gacs_korner_meet_channel(
    joint: dict[tuple[Hashable, Hashable], float],
) -> dict[tuple[Hashable, Hashable], dict[Hashable, float]]:
    """Deterministic channel assigning each state pair to its meet component.

    Builds the bipartite graph linking a forward state ``alpha`` to a reverse
    state ``gamma`` whenever ``p(alpha, gamma) > 0``; the connected components
    are the atoms of the meet ``S+ ⩘ S-``.
    """
    import networkx as nx

    graph = nx.Graph()
    for alpha, gamma in joint:
        graph.add_edge(("+", alpha), ("-", gamma))

    component_of: dict[Hashable, int] = {}
    for rank, component in enumerate(nx.connected_components(graph)):
        for node in component:
            component_of[node] = rank

    return {pair: {f"G{component_of[('+', pair[0])]}": 1.0} for pair in joint}


def _functional_state_channel(
    bidir: BidirectionalEpsilonMachine,
    joint: dict[tuple[Hashable, Hashable], float],
    *,
    cutoff: float,
    strategy: str,
) -> tuple[dict[tuple[Hashable, Hashable], dict[Hashable, float]], float]:
    """Deterministic channel ``W = f(S+, S-)`` of least entropy that separates the causal states and lifts.

    dit's functional-common-information partition is returned when it satisfies
    the lifting conditions (it is then optimal); otherwise the lifted partition
    search of :mod:`sofic.generators.lifted_common_information` is used.
    """
    from sofic.generators.lifted_common_information import lifted_partition_channel, partition_residual

    channel, value = _dit_functional_channel(joint, strategy=strategy)
    if partition_residual(bidir, joint, channel) <= _EXACT_ATOL:
        return channel, value
    return lifted_partition_channel(bidir, joint)


def _dit_functional_channel(
    joint: dict[tuple[Hashable, Hashable], float],
    *,
    strategy: str,
) -> tuple[dict[tuple[Hashable, Hashable], dict[Hashable, float]], float]:
    """Deterministic channel from dit's functional-common-information partition (separation only)."""
    plus_states = {pair[0] for pair in joint}
    minus_states = {pair[1] for pair in joint}
    if len(plus_states) <= 1 or len(minus_states) <= 1:
        return {pair: {"G0": 1.0} for pair in joint}, 0.0
    matching_channel = _matching_support_channel(joint)
    if matching_channel is not None:
        return matching_channel, _entropy(joint.values())

    from dit.multivariate.common_informations.functional_common_information import functional_markov_chain

    dist, plus_index, minus_index = _indexed_joint_distribution(joint)

    stats: dict[str, Any] = {}
    value = _as_float(functional_markov_chain(dist, [[0], [1]], _strategy=strategy, _stats=stats))
    partition = stats.get("partition")
    if partition is None:
        raise StochasticValidationError("functional common information search returned no partition")

    outcome_rank: dict[tuple[int, int], int] = {}
    for rank, block in enumerate(partition):
        for outcome in block:
            outcome_rank[tuple(outcome)] = rank

    pair_rank: dict[tuple[Hashable, Hashable], int] = {}
    for pair in joint:
        key = (plus_index[pair[0]], minus_index[pair[1]])
        if key not in outcome_rank:
            raise StochasticValidationError(f"functional partition is missing joint state {pair!r}")
        pair_rank[pair] = outcome_rank[key]

    used_ranks = sorted(set(pair_rank.values()))
    relabel = {rank: f"G{i}" for i, rank in enumerate(used_ranks)}
    channel = {pair: {relabel[rank]: 1.0} for pair, rank in pair_rank.items()}
    return channel, value


def _require_dit():
    from sofic.generators.measures import require_dit

    return require_dit("minimal generative models")


def _as_numpy(array: Any) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=float)


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)


def _normalized_joint_distribution(
    bidir: BidirectionalEpsilonMachine,
    *,
    cutoff: float,
) -> dict[tuple[Hashable, Hashable], float]:
    joint = {pair: float(mass) for pair, mass in bidir.joint_distribution().items() if float(mass) > cutoff}
    total = sum(joint.values())
    if total <= 0.0:
        raise StochasticValidationError("bidirectional machine has empty stationary joint distribution")
    return {pair: mass / total for pair, mass in joint.items()}


def _indexed_joint_distribution(
    joint: dict[tuple[Hashable, Hashable], float],
) -> tuple[Any, dict[Hashable, int], dict[Hashable, int]]:
    dit = _require_dit()
    plus_states = tuple(sorted({pair[0] for pair in joint}, key=repr))
    minus_states = tuple(sorted({pair[1] for pair in joint}, key=repr))
    plus_index = {state: i for i, state in enumerate(plus_states)}
    minus_index = {state: i for i, state in enumerate(minus_states)}

    outcomes = [(plus_index[alpha], minus_index[gamma]) for alpha, gamma in joint]
    probs = [joint[pair] for pair in joint]
    return dit.Distribution(outcomes, probs), plus_index, minus_index


def _optimized_auxiliary_joint(
    dist: Any,
    *,
    optimizer: Any,
    optimizer_name: str,
    bound: int | None,
    niter: int | None,
    maxiter: int,
    polish: float | bool,
    backend: str,
    rng: np.random.Generator | None,
) -> tuple[Any, np.ndarray]:
    opt = optimizer(
        dist,
        bound=bound,
        niter=niter,
        maxiter=maxiter,
        polish=polish,
        backend=backend,
        rng=rng,
    )

    aux_joint = _as_numpy(opt.construct_joint(opt._optima))
    if aux_joint.ndim < 3:
        raise StochasticValidationError(f"{optimizer_name} optimizer returned an invalid joint shape")
    if aux_joint.ndim > 3:
        aux_joint = aux_joint.sum(axis=tuple(range(2, aux_joint.ndim - 1)))
    # dit numbers each variable's symbols in order of first appearance; put them back in symbol order.
    unqs = getattr(opt, "_unqs", None)
    if unqs is not None:
        plus_order = [unqs[0].mapping[(i,)] for i in range(len(unqs[0].mapping))]
        minus_order = [unqs[1].mapping[(j,)] for j in range(len(unqs[1].mapping))]
        aux_joint = aux_joint[np.ix_(plus_order, minus_order)]
    return opt, np.maximum(aux_joint, 0.0)


def _raw_channel_from_auxiliary_joint(
    aux_joint: np.ndarray,
    joint: dict[tuple[Hashable, Hashable], float],
    plus_index: dict[Hashable, int],
    minus_index: dict[Hashable, int],
    *,
    optimizer_name: str,
    cutoff: float,
    polish: float | bool,
    niter: int | None,
) -> tuple[dict[tuple[Hashable, Hashable], np.ndarray] | None, str | None]:
    raw_channel: dict[tuple[Hashable, Hashable], np.ndarray] = {}
    for pair, target_mass in joint.items():
        alpha, gamma = pair
        row = aux_joint[plus_index[alpha], minus_index[gamma], :]
        total = float(row.sum())
        if total <= cutoff:
            return None, (
                f"missing generative-state channel row for joint state {pair!r}: "
                f"target joint mass={target_mass:.17g}, returned row mass={total:.17g}, "
                f"cutoff={cutoff:.17g}, polish={polish!r}, niter={niter!r}, optimizer={optimizer_name!r}"
            )
        raw_channel[pair] = row / total
    return raw_channel, None


def _auxiliary_state_channel(
    joint: dict[tuple[Hashable, Hashable], float],
    *,
    optimizer: Any,
    optimizer_name: str,
    bound: int | None,
    niter: int | None,
    maxiter: int,
    polish: float | bool,
    backend: str,
    cutoff: float,
    rng: np.random.Generator | None,
) -> tuple[dict[tuple[Hashable, Hashable], dict[Hashable, float]], float]:
    plus_states = {pair[0] for pair in joint}
    minus_states = {pair[1] for pair in joint}
    if len(plus_states) <= 1 or len(minus_states) <= 1:
        return {pair: {"G0": 1.0} for pair in joint}, 0.0
    matching_channel = _matching_support_channel(joint)
    if matching_channel is not None:
        return matching_channel, _entropy(joint.values())

    dist, plus_index, minus_index = _indexed_joint_distribution(joint)
    opt, aux_joint = _optimized_auxiliary_joint(
        dist,
        optimizer=optimizer,
        optimizer_name=optimizer_name,
        bound=bound,
        niter=niter,
        maxiter=maxiter,
        polish=polish,
        backend=backend,
        rng=rng,
    )
    raw_channel, validation_error = _raw_channel_from_auxiliary_joint(
        aux_joint,
        joint,
        plus_index,
        minus_index,
        optimizer_name=optimizer_name,
        cutoff=cutoff,
        polish=polish,
        niter=niter,
    )
    if validation_error is not None and polish:
        opt, aux_joint = _optimized_auxiliary_joint(
            dist,
            optimizer=optimizer,
            optimizer_name=optimizer_name,
            bound=bound,
            niter=niter,
            maxiter=maxiter,
            polish=False,
            backend=backend,
            rng=rng,
        )
        raw_channel, retry_error = _raw_channel_from_auxiliary_joint(
            aux_joint,
            joint,
            plus_index,
            minus_index,
            optimizer_name=optimizer_name,
            cutoff=cutoff,
            polish=False,
            niter=niter,
        )
        if retry_error is not None:
            raise StochasticValidationError(f"{validation_error}; retry without polishing also failed: {retry_error}")
    elif validation_error is not None:
        raise StochasticValidationError(validation_error)

    assert raw_channel is not None
    state_masses = _state_masses(joint, raw_channel)
    active_indices = [i for i, mass in enumerate(state_masses) if mass > cutoff]
    if not active_indices:
        active_indices = [int(np.argmax(state_masses))]
    state_labels = {index: f"G{rank}" for rank, index in enumerate(active_indices)}

    channel: dict[tuple[Hashable, Hashable], dict[Hashable, float]] = {}
    for pair, row in raw_channel.items():
        entries = {state_labels[i]: float(row[i]) for i in active_indices if row[i] > cutoff}
        total = sum(entries.values())
        if total <= cutoff:
            best = max(active_indices, key=lambda i: row[i])
            entries = {state_labels[best]: 1.0}
        else:
            entries = {state: prob / total for state, prob in entries.items()}
        channel[pair] = entries

    optimized_value = _as_float(opt.objective(opt._optima))
    return channel, optimized_value


def _matching_support_channel(
    joint: dict[tuple[Hashable, Hashable], float],
) -> dict[tuple[Hashable, Hashable], dict[Hashable, float]] | None:
    plus_to_minus: dict[Hashable, Hashable] = {}
    minus_to_plus: dict[Hashable, Hashable] = {}
    for alpha, gamma in joint:
        if alpha in plus_to_minus and plus_to_minus[alpha] != gamma:
            return None
        if gamma in minus_to_plus and minus_to_plus[gamma] != alpha:
            return None
        plus_to_minus[alpha] = gamma
        minus_to_plus[gamma] = alpha

    return {pair: {f"G{i}": 1.0} for i, pair in enumerate(sorted(joint, key=repr))}


def _entropy(masses: Any) -> float:
    from sofic.generators.stochastic import shannon_entropy

    return shannon_entropy(masses, atol=0.0)


def _optimize_exact_common_information(
    dist: Any,
    *,
    bound: int | None,
    niter: int | None,
    maxiter: int,
    polish: float | bool,
    backend: str,
    rng: np.random.Generator | None,
) -> Any:
    from dit.multivariate._backend import _make_backend_subclass
    from dit.multivariate.common_informations.exact_common_information import ExactCommonInformation

    cls = _make_backend_subclass(ExactCommonInformation, backend)
    opt = cls(dist, [[0], [1]], bound=bound)
    if isinstance(polish, int | float) and not isinstance(polish, bool) and polish > 0.0:
        options = opt._additional_options.setdefault("options", {})
        options["ftol"] = min(float(polish), float(options.get("ftol", polish)))
    opt.optimize(niter=niter, maxiter=maxiter, polish=polish, rng=rng)
    return opt


def _optimize_wyner_common_information(
    dist: Any,
    *,
    bound: int | None,
    niter: int | None,
    maxiter: int,
    polish: float | bool,
    backend: str,
    rng: np.random.Generator | None,
) -> Any:
    from dit.multivariate._backend import _make_backend_subclass
    from dit.multivariate.common_informations.wyner_common_information import WynerCommonInformation

    cls = _make_backend_subclass(WynerCommonInformation, backend)
    opt = cls(dist, [[0], [1]], bound=bound)
    if isinstance(polish, int | float) and not isinstance(polish, bool) and polish > 0.0:
        options = opt._additional_options.setdefault("options", {})
        options["ftol"] = min(float(polish), float(options.get("ftol", polish)))
    opt.optimize(niter=niter, maxiter=maxiter, polish=polish, rng=rng)
    return opt


def _state_masses(
    joint: dict[tuple[Hashable, Hashable], float],
    pair_channel: dict[tuple[Hashable, Hashable], np.ndarray],
) -> np.ndarray:
    size = max(len(row) for row in pair_channel.values())
    masses = np.zeros(size, dtype=float)
    for pair, pair_mass in joint.items():
        masses += pair_mass * pair_channel[pair]
    return masses


def _reproduction_length(reference: Any) -> int:
    """Word length at which to compare a generative model against the source process."""
    n_states = len(list(reference.states()))
    return min(8, max(4, 2 * n_states))


def _reproduction_error(model: Any, reference: Any, *, max_length: int) -> float:
    """Max abs word-probability deviation of ``model`` from ``reference`` up to ``max_length``."""
    error = 0.0
    for length in range(max_length + 1):
        model_words = model.word_probabilities(length, sparse=False)
        reference_words = reference.word_probabilities(length, sparse=False)
        for word in set(model_words) | set(reference_words):
            error = max(error, abs(model_words.get(word, 0.0) - reference_words.get(word, 0.0)))
    return error


def _model_from_channel(
    bidir: BidirectionalEpsilonMachine,
    joint: dict[tuple[Hashable, Hashable], float],
    pair_channel: dict[tuple[Hashable, Hashable], dict[Hashable, float]],
    *,
    model_cls: type[_CommonInformationGenerativeModel],
    model_name: str,
    measure_kwargs: dict[str, float],
    cutoff: float,
) -> _CommonInformationGenerativeModel:
    state_mass: dict[Hashable, float] = defaultdict(float)
    for pair, pair_mass in joint.items():
        for state, prob in pair_channel[pair].items():
            state_mass[state] += pair_mass * prob

    active_states = tuple(
        state for state, mass in sorted(state_mass.items(), key=lambda item: repr(item[0])) if mass > cutoff
    )
    total_state_mass = sum(state_mass[state] for state in active_states)
    if total_state_mass <= 0.0:
        raise StochasticValidationError(f"{model_name} has empty state support")
    initial = {state: state_mass[state] / total_state_mass for state in active_states}

    flows: dict[tuple[Hashable, Any, Hashable], float] = defaultdict(float)
    for pair, pair_mass in joint.items():
        if pair_mass <= cutoff:
            continue
        for transition in bidir.graph.out_transitions(pair):
            symbol = transition.data.get(ATTR_EMISSION)
            prob = float(transition.data.get(ATTR_PROB, 0.0))
            target_pair = transition.target
            if symbol is None or prob <= cutoff or target_pair not in pair_channel:
                continue
            for state, state_prob in pair_channel[pair].items():
                if state not in initial or state_prob <= cutoff:
                    continue
                for target_state, target_prob in pair_channel[target_pair].items():
                    if target_state not in initial or target_prob <= cutoff:
                        continue
                    flows[(state, symbol, target_state)] += pair_mass * state_prob * prob * target_prob

    graph = TransitionGraph()
    for state in active_states:
        graph.add_state(state)

    by_source: dict[Hashable, dict[tuple[Hashable, Any], float]] = defaultdict(lambda: defaultdict(float))
    for (source, symbol, target), flow in flows.items():
        if flow > 0.0:
            by_source[source][(target, symbol)] += flow

    for source in active_states:
        outgoing = by_source.get(source, {})
        total = sum(outgoing.values())
        if total <= cutoff:
            raise StochasticValidationError(f"{model_name} state {source!r} has no outgoing mass")
        added = False
        for (target, symbol), flow in outgoing.items():
            prob = flow / total
            if prob <= cutoff:
                continue
            graph.add_transition(source, target, **{ATTR_PROB: float(prob), ATTR_EMISSION: symbol})
            added = True
        if not added:
            target, symbol = max(outgoing, key=outgoing.__getitem__)
            graph.add_transition(source, target, **{ATTR_PROB: 1.0, ATTR_EMISSION: symbol})

    model = model_cls(
        graph=graph,
        initial_distribution=initial,
        observation_alphabet=bidir.observation_alphabet,
        pair_state_channel=pair_channel,
        joint_pair_distribution=joint,
        source_bidirectional=bidir,
        entropy_rate=bidir.entropy_rate(),
        **measure_kwargs,
    )
    model.validate()
    return model


__all__ = [
    "FunctionalGenerativeModel",
    "GacsKornerGenerativeModel",
    "GenerativeGap",
    "MinimalGenerativeModel",
    "WynerGenerativeModel",
    "exact_common_information",
    "functional_generative_model",
    "gacs_korner_generative_model",
    "generative_gap",
    "minimal_generative_model",
    "wyner_common_information",
    "wyner_generative_model",
]
