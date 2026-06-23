"""Minimal generative models from bidirectional epsilon-machines."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pensive.exceptions import StochasticValidationError
from pensive.generators.mealy import MealyHMM
from pensive.graph import ATTR_EMISSION, ATTR_PROB, TransitionGraph

if TYPE_CHECKING:
    from pensive.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine


class _CommonInformationGenerativeModel(MealyHMM):
    pair_state_channel: dict[tuple[Hashable, Hashable], dict[Hashable, float]]
    joint_pair_distribution: dict[tuple[Hashable, Hashable], float]
    source_bidirectional: BidirectionalEpsilonMachine | None
    _entropy_rate: float | None

    def __init__(
        self,
        *,
        pair_state_channel: dict[tuple[Hashable, Hashable], dict[Hashable, float]] | None = None,
        joint_pair_distribution: dict[tuple[Hashable, Hashable], float] | None = None,
        source_bidirectional: BidirectionalEpsilonMachine | None = None,
        entropy_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pair_state_channel = pair_state_channel or {}
        self.joint_pair_distribution = joint_pair_distribution or {}
        self.source_bidirectional = source_bidirectional
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

    The states are the optimized exact-common-information auxiliary variable
    between the forward and reverse causal states of a bidirectional
    epsilon-machine.
    """

    exact_common_information: float

    def __init__(self, *, exact_common_information: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.exact_common_information = float(exact_common_information)


class WynerGenerativeModel(_CommonInformationGenerativeModel):
    """Non-unifilar generator from a Wyner-common-information auxiliary.

    The optimized value is ``I[(S+, S-) : G]`` and is stored as
    ``wyner_common_information``. The model's state entropy is ``H[G]`` and can
    be strictly larger.
    """

    wyner_common_information: float

    def __init__(self, *, wyner_common_information: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wyner_common_information = float(wyner_common_information)


def minimal_generative_model(
    bidir: BidirectionalEpsilonMachine,
    *,
    bound: int | None = None,
    niter: int | None = None,
    maxiter: int = 1000,
    polish: float | bool = 1e-6,
    backend: str = "numpy",
    cutoff: float = 1e-10,
    rng: np.random.Generator | None = None,
) -> MinimalGenerativeModel:
    """Construct a minimal generative model from a bidirectional epsilon-machine.

    Parameters mirror :class:`dit.multivariate.common_informations.ExactCommonInformation`.
    The returned model is an edge-emitting HMM over the optimized generative
    state ``G``.
    """
    if cutoff < 0.0:
        raise ValueError("cutoff must be nonnegative")

    joint = _normalized_joint_distribution(bidir, cutoff=cutoff)
    pair_channel, exact_common_information = _auxiliary_state_channel(
        joint,
        optimizer=_optimize_exact_common_information,
        optimizer_name="exact common information",
        bound=bound,
        niter=niter,
        maxiter=maxiter,
        polish=polish,
        backend=backend,
        cutoff=cutoff,
        rng=rng,
    )
    return cast(
        MinimalGenerativeModel,
        _model_from_channel(
            bidir,
            joint,
            pair_channel,
            model_cls=MinimalGenerativeModel,
            model_name="minimal generative model",
            measure_kwargs={"exact_common_information": exact_common_information},
            cutoff=cutoff,
        ),
    )


def wyner_generative_model(
    bidir: BidirectionalEpsilonMachine,
    *,
    bound: int | None = None,
    niter: int | None = None,
    maxiter: int = 1000,
    polish: float | bool = 1e-6,
    backend: str = "numpy",
    cutoff: float = 1e-10,
    rng: np.random.Generator | None = None,
) -> WynerGenerativeModel:
    """Construct a Wyner generative model from a bidirectional epsilon-machine.

    The auxiliary state ``G`` is optimized for Wyner common information, i.e.
    it minimizes ``I[(S+, S-) : G]`` subject to rendering ``S+`` and ``S-``
    conditionally independent. The returned model's state entropy ``H[G]`` is
    not generally equal to that optimized mutual information.
    """
    if cutoff < 0.0:
        raise ValueError("cutoff must be nonnegative")

    joint = _normalized_joint_distribution(bidir, cutoff=cutoff)
    pair_channel, wyner_common_information = _auxiliary_state_channel(
        joint,
        optimizer=_optimize_wyner_common_information,
        optimizer_name="Wyner common information",
        bound=bound,
        niter=niter,
        maxiter=maxiter,
        polish=polish,
        backend=backend,
        cutoff=cutoff,
        rng=rng,
    )
    return cast(
        WynerGenerativeModel,
        _model_from_channel(
            bidir,
            joint,
            pair_channel,
            model_cls=WynerGenerativeModel,
            model_name="Wyner generative model",
            measure_kwargs={"wyner_common_information": wyner_common_information},
            cutoff=cutoff,
        ),
    )


def _require_dit():
    try:
        import dit
    except ImportError as exc:
        raise ImportError("dit is required for minimal generative models; install pensive[measures]") from exc
    return dit


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
    aux_joint = np.maximum(aux_joint, 0.0)

    raw_channel: dict[tuple[Hashable, Hashable], np.ndarray] = {}
    for pair in joint:
        alpha, gamma = pair
        row = aux_joint[plus_index[alpha], minus_index[gamma], :]
        total = float(row.sum())
        if total <= cutoff:
            raise StochasticValidationError(f"missing generative-state channel row for joint state {pair!r}")
        raw_channel[pair] = row / total

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
    probs = np.asarray(list(masses), dtype=float)
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


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

    active_states = tuple(state for state, mass in sorted(state_mass.items(), key=lambda item: repr(item[0])) if mass > cutoff)
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
        if flow > cutoff:
            by_source[source][(target, symbol)] += flow

    for source in active_states:
        outgoing = by_source.get(source, {})
        total = sum(outgoing.values())
        if total <= cutoff:
            raise StochasticValidationError(f"{model_name} state {source!r} has no outgoing mass")
        for (target, symbol), flow in outgoing.items():
            prob = flow / total
            if prob <= cutoff:
                continue
            graph.add_transition(source, target, **{ATTR_PROB: float(prob), ATTR_EMISSION: symbol})

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
    "MinimalGenerativeModel",
    "WynerGenerativeModel",
    "minimal_generative_model",
    "wyner_generative_model",
]
