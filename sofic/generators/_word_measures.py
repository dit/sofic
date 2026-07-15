"""Dit-backed word-distribution measures for block curves.

Leaf module shared by :mod:`sofic.generators.block_entropy` and
:mod:`sofic.generators.block_convergence`.  It depends only on ``dit`` (via
:func:`sofic.generators.measures.require_dit`) and must not import either
block module, so that both can depend on it without a circular import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sofic.generators.epsilon_machine import EpsilonMachine

_TOL = 1e-15


def _require_dit() -> Any:
    from sofic.generators.measures import require_dit

    return require_dit("block convergence measures")


def _positional_rvs(length: int) -> list[list[int]]:
    return [[index] for index in range(length)]


def _word_distribution(distribution: dict[tuple[Any, ...], float]) -> Any:
    dit = _require_dit()
    if not distribution:
        return dit.Distribution([()], [1.0])
    total = sum(float(prob) for prob in distribution.values())
    if total <= _TOL:
        return dit.Distribution([()], [1.0])
    outcomes = list(distribution.keys())
    probs = [float(distribution[outcome]) / total for outcome in outcomes]
    dist = dit.Distribution(outcomes, probs)
    dist.set_rv_names([f"X_{index}" for index in range(len(outcomes[0]))])
    return dist


def _block_word_distribution(machine: EpsilonMachine, length: int) -> Any:
    if length == 0:
        dit = _require_dit()
        return dit.Distribution([()], [1.0])
    return _word_distribution(machine.word_probabilities(length))


def _block_total_correlation(dist: Any, length: int, block_entropy: float, h1: float) -> float:
    if length <= 1:
        return 0.0
    _require_dit()
    from dit.multivariate import total_correlation

    return float(total_correlation(dist, rvs=_positional_rvs(length)))


def _block_residual_entropy(dist: Any, length: int) -> float:
    if length == 0:
        return 0.0
    _require_dit()
    from dit.multivariate import residual_entropy

    return float(residual_entropy(dist, rvs=_positional_rvs(length)))


def _block_coinformation(dist: Any, length: int, block_entropy: float) -> float:
    if length <= 1:
        return 0.0
    _require_dit()
    from dit.multivariate import coinformation

    return float(coinformation(dist, rvs=_positional_rvs(length)))


def _block_caekl(dist: Any, length: int) -> float:
    if length <= 1:
        return 0.0
    _require_dit()
    from dit.multivariate import caekl_mutual_information

    return float(caekl_mutual_information(dist, rvs=_positional_rvs(length)))


def residual_entropy_from_distribution(distribution: dict[tuple[Any, ...], float]) -> float:
    """Dit-backed residual entropy for a word distribution."""
    if not distribution:
        return 0.0
    length = len(next(iter(distribution)))
    if length == 0:
        return 0.0
    dist = _word_distribution(distribution)
    return _block_residual_entropy(dist, length)
