"""Significance tests on two-row contingency tables of next-symbol counts.

Shared by process CSSR (:mod:`sofic.generators.epsilon_inference`), stack CSSR,
and transCSSR (:mod:`sofic.generators.epsilon_transducer_inference`). Each table
holds the counts of the symbols following two sets of histories, one row per set.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from scipy import stats

#: Contingency-table tests: G-test, Pearson chi-squared, or Monte Carlo exact G-test.
TableTest = Literal["g", "chi2", "exact"]

#: Monte Carlo tables drawn per ``"exact"`` test.
EXACT_DRAWS = 999

#: Smallest expected count at which the ``"exact"`` test trusts the chi-squared limit.
EXACT_MIN_EXPECTED = 5.0


def contingency_table(left: Mapping[Any, int], right: Mapping[Any, int], alphabet: Sequence[Any]) -> np.ndarray | None:
    """Two-row table of ``left`` and ``right`` counts over the symbols either one observed.

    Returns ``None`` when there is nothing to test: no observed symbol, identical
    rows, or a single symbol with an empty row or equal proportions.
    """
    active = [symbol for symbol in alphabet if left.get(symbol, 0) + right.get(symbol, 0) > 0]
    if not active:
        return None
    table = np.array(
        [[left.get(symbol, 0) for symbol in active], [right.get(symbol, 0) for symbol in active]],
        dtype=float,
    )
    if np.allclose(table[0], table[1]):
        return None
    if table.shape[1] < 2:
        left_total = table[0].sum()
        right_total = table[1].sum()
        if left_total == 0.0 or right_total == 0.0:
            return None
        if np.isclose(table[0, 0] / left_total, table[1, 0] / right_total):
            return None
    return table


def g_statistic(table: np.ndarray) -> float | None:
    """G-test statistic of a contingency table, as ``scipy.stats.chi2_contingency`` computes it.

    Applies Yates' continuity correction when the table has one degree of freedom
    (``chi2_contingency`` does so for every ``lambda_``, including
    ``"log-likelihood"``), and returns ``None`` if an expected count is zero.
    Inlined because the test runs once per pair of histories, and the general
    scipy routine dominated inference time.
    """
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    if np.any(expected == 0):
        return None
    observed = table
    if (table.shape[0] - 1) * (table.shape[1] - 1) == 1:
        diff = expected - observed
        observed = observed + np.sign(diff) * np.minimum(0.5, np.abs(diff))
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(observed > 0, observed * np.log(observed / expected), 0.0)
    return 2.0 * float(terms.sum())


@lru_cache(maxsize=256)
def chi2_critical(alpha: float, dof: int) -> float:
    """Upper ``alpha`` quantile of the chi-squared distribution with ``dof`` degrees of freedom."""
    return float(stats.chi2.ppf(1.0 - alpha, dof))


def g_significant(table: np.ndarray, alpha: float) -> bool:
    """Asymptotic G-test (see :func:`g_statistic`) at level ``alpha``."""
    statistic = g_statistic(table)
    if statistic is None or not np.isfinite(statistic):
        return False
    return statistic > chi2_critical(alpha, max(1, table.shape[1] - 1))


def chi2_significant(table: np.ndarray, alpha: float) -> bool:
    """Pearson chi-squared test (with Yates' correction at one degree of freedom) at level ``alpha``."""
    try:
        _statistic, p_value, _dof, expected = stats.chi2_contingency(table)
    except ValueError:
        return False
    if np.any(expected == 0):
        return False
    return float(p_value) < alpha


def exact_g_pvalue(table: np.ndarray) -> float:
    """Monte Carlo p-value of the G statistic among tables with the same margins.

    Tables are drawn uniformly given both margins (``scipy.stats.random_table``),
    the exact null of equal morphs. The generator is seeded from the table itself,
    so reconstruction stays deterministic.
    """
    counts = table.astype(np.int64)
    rows, cols = counts.sum(axis=1), counts.sum(axis=0)
    expected = np.outer(rows, cols) / counts.sum()

    def g(observed: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(observed > 0, observed * np.log(observed / expected), 0.0)
        return 2.0 * terms.sum(axis=(-2, -1))

    rng = np.random.default_rng(zlib.crc32(counts.tobytes()))
    draws = stats.random_table(rows, cols, seed=rng).rvs(size=EXACT_DRAWS)
    observed = g(counts.astype(float))
    extreme = np.sum(g(draws.astype(float)) >= observed - 1e-9 * max(1.0, observed))
    return float((1 + extreme) / (1 + EXACT_DRAWS))


def exact_significant(table: np.ndarray, alpha: float) -> bool:
    """The ``"exact"`` decision for a contingency table of next-symbol counts.

    Uses the Monte Carlo exact G-test when an expected count is below
    :data:`EXACT_MIN_EXPECTED` and the asymptotic G-test otherwise.
    """
    if np.any(table.sum(axis=1) == 0):
        return False
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    if expected.min() < EXACT_MIN_EXPECTED:
        return exact_g_pvalue(table) < alpha
    return g_significant(table, alpha)


def table_significant(table: np.ndarray, alpha: float, test: TableTest) -> bool:
    """Whether the rows of ``table`` differ significantly at level ``alpha`` under ``test``."""
    if test == "exact":
        return exact_significant(table, alpha)
    if test == "g":
        return g_significant(table, alpha)
    return chi2_significant(table, alpha)


def table_score(table: np.ndarray, test: TableTest) -> float:
    """Test statistic of ``table`` (lower is more similar): G for ``"g"``/``"exact"``, else Pearson."""
    if test == "chi2":
        try:
            statistic, _p, _dof, _expected = stats.chi2_contingency(table)
        except ValueError:
            return 0.0
        return float(statistic)
    statistic = g_statistic(table)
    return statistic if statistic is not None and np.isfinite(statistic) else 0.0
