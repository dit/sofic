"""Preflight and retry-ladder construction of bidirectional ε-machines for surveys.

When the reverse mixed-state presentation does not close at the default cap,
``build_bidirectional_with_ladder`` raises ``max_states`` until the edge tensor
``Pe`` of :class:`~sofic.generators.coupled_generators._GeneratorProblem` stabilizes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from sofic.exceptions import MixedStateExplosionError, StochasticValidationError
from sofic.generators.bidirectional_construction import (
    build_bidirectional_epsilon_machine,
    infer_reverse_epsilon_machine,
)
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.epsilon_machine import EpsilonMachine

BidirClass = Literal["finite", "truncated", "explosion", "failed_build"]

_DEFAULT_CAPS = (10_000, 50_000, 200_000)


@dataclass(frozen=True)
class BidirPreflightResult:
    """Classification before or after attempting a bidirectional build."""

    bidir_class: BidirClass
    reverse_max_states: int | None = None
    reverse_recurrence_finite: bool | None = None
    reverse_twins_finite: bool | None = None
    error: str | None = None


@dataclass(frozen=True)
class BidirBuildResult:
    """Bidirectional machine plus build metadata for logging and surveys."""

    bidir: BidirectionalEpsilonMachine | None
    preflight: BidirPreflightResult
    pe_drift: float | None = None
    edge_stabilized: bool | None = None
    budget_exhausted: bool = False


def _edge_tensor(bidir: BidirectionalEpsilonMachine) -> np.ndarray:
    from sofic.generators.coupled_generators import _GeneratorProblem

    problem = _GeneratorProblem(bidir)
    return np.asarray(problem.Pe, float)


def _pe_max_drift(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return np.inf
    return float(np.max(np.abs(a - b)))


def preflight_bidir(forward: EpsilonMachine, *, reverse_max_states: int = 10_000) -> BidirPreflightResult:
    """Classify reverse presentation and whether a bidirectional machine builds at ``reverse_max_states``."""
    twins = forward.reverse_is_finite()
    recurrence = forward.reverse_epsilon_machine_is_finite()
    try:
        reverse = infer_reverse_epsilon_machine(forward, max_states=reverse_max_states)
        build_bidirectional_epsilon_machine(forward, reverse)
    except MixedStateExplosionError as exc:
        return BidirPreflightResult(
            "explosion",
            reverse_max_states=reverse_max_states,
            reverse_recurrence_finite=recurrence,
            reverse_twins_finite=twins,
            error=str(exc)[:300],
        )
    except (StochasticValidationError, ValueError) as exc:
        return BidirPreflightResult(
            "failed_build",
            reverse_max_states=reverse_max_states,
            reverse_recurrence_finite=recurrence,
            reverse_twins_finite=twins,
            error=str(exc)[:300],
        )
    if recurrence:
        return BidirPreflightResult(
            "finite",
            reverse_max_states=reverse_max_states,
            reverse_recurrence_finite=True,
            reverse_twins_finite=twins,
        )
    return BidirPreflightResult(
        "truncated",
        reverse_max_states=reverse_max_states,
        reverse_recurrence_finite=False,
        reverse_twins_finite=twins,
    )


def build_bidirectional_with_ladder(
    forward: EpsilonMachine,
    *,
    caps: tuple[int, ...] = _DEFAULT_CAPS,
    edge_tol: float = 1e-8,
    time_budget: float | None = None,
) -> BidirBuildResult:
    """Build a bidirectional machine, increasing reverse ``max_states`` until ``Pe`` stabilizes.

    With ``time_budget`` (seconds) the ladder stops before a cap whose forecast cost would overrun
    the budget, and ``budget_exhausted`` is set on the result. A single mixed-state construction
    cannot be interrupted, so the forecast scales the previous cap's time by the square of the cap
    ratio (construction time grows faster than linearly in the cap); the first cap always runs.
    """
    start = time.monotonic()
    last_pe: np.ndarray | None = None
    last_bidir: BidirectionalEpsilonMachine | None = None
    last_preflight: BidirPreflightResult | None = None
    previous: tuple[int, float] | None = None
    exhausted = False

    for cap in caps:
        if time_budget is not None and previous is not None:
            prev_cap, prev_seconds = previous
            forecast = prev_seconds * (cap / prev_cap) ** 2
            if time.monotonic() - start + forecast > time_budget:
                exhausted = True
                break
        attempt = time.monotonic()
        pre = preflight_bidir(forward, reverse_max_states=cap)
        last_preflight = pre
        if pre.bidir_class in ("explosion", "failed_build"):
            previous = (cap, time.monotonic() - attempt)
            continue
        reverse = infer_reverse_epsilon_machine(forward, max_states=cap)
        bidir = build_bidirectional_epsilon_machine(forward, reverse)
        pe = _edge_tensor(bidir)
        previous = (cap, time.monotonic() - attempt)
        if last_pe is not None:
            drift = _pe_max_drift(last_pe, pe)
            if drift <= edge_tol:
                return BidirBuildResult(
                    bidir,
                    pre,
                    pe_drift=drift,
                    edge_stabilized=True,
                )
        last_pe, last_bidir = pe, bidir

    if last_bidir is not None and last_preflight is not None:
        drift = None if last_pe is None else 0.0
        return BidirBuildResult(
            last_bidir,
            last_preflight,
            pe_drift=drift,
            edge_stabilized=False,
            budget_exhausted=exhausted,
        )

    if last_preflight is not None:
        return BidirBuildResult(None, last_preflight, budget_exhausted=exhausted)
    pre = preflight_bidir(forward, reverse_max_states=caps[-1])
    return BidirBuildResult(None, pre)


def try_build_bidirectional(
    forward: EpsilonMachine,
    *,
    reverse_max_states: int = 10_000,
    use_ladder: bool = False,
    caps: tuple[int, ...] = _DEFAULT_CAPS,
    edge_tol: float = 1e-8,
    time_budget: float | None = None,
) -> BidirBuildResult:
    """Single-cap build, or ladder when ``use_ladder`` and the default cap explodes.

    ``time_budget`` (seconds, including the first preflight) is passed on to
    :func:`build_bidirectional_with_ladder`.
    """
    start = time.monotonic()
    pre = preflight_bidir(forward, reverse_max_states=reverse_max_states)
    if pre.bidir_class == "explosion" and use_ladder:
        remaining = None if time_budget is None else max(0.0, time_budget - (time.monotonic() - start))
        return build_bidirectional_with_ladder(forward, caps=caps, edge_tol=edge_tol, time_budget=remaining)
    if pre.bidir_class in ("explosion", "failed_build"):
        return BidirBuildResult(None, pre)
    reverse = infer_reverse_epsilon_machine(forward, max_states=reverse_max_states)
    bidir = build_bidirectional_epsilon_machine(forward, reverse)
    return BidirBuildResult(bidir, pre, edge_stabilized=True, pe_drift=0.0)


__all__ = [
    "BidirBuildResult",
    "BidirPreflightResult",
    "build_bidirectional_with_ladder",
    "preflight_bidir",
    "try_build_bidirectional",
]
