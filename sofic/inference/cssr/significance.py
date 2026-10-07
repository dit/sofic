"""Significance tests that decide whether CSSR histories share a morph.

Two-row contingency tables of next-symbol counts are compared by a G-test,
Pearson chi-squared test, or Monte Carlo exact G-test. Shared by process CSSR
(:mod:`sofic.inference.cssr.process`), stack CSSR (:mod:`sofic.inference.cssr.stack`),
and transCSSR (:mod:`sofic.inference.cssr.transducer`).
"""

from __future__ import annotations

import zlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from scipy import stats

from sofic.inference.cssr.counts import History, StateAggregate, SuffixCounts

#: Contingency-table tests: G-test, Pearson chi-squared, or Monte Carlo exact G-test.
TableTest = Literal["g", "chi2", "exact"]

#: Morph-equality tests: G-test, chi-squared, total-variation threshold, or Monte Carlo exact G-test.
MorphTest = Literal["g", "chi2", "tv", "exact"]


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


def _observed_counts_for_morph(
    counts: SuffixCounts,
    histories: set[History],
) -> Counter[Any]:
    observed = Counter()
    for history in histories:
        observed.update(counts.next_counts.get(history, Counter()))
    return observed


def _contingency_rows(
    counts: SuffixCounts,
    left_histories: set[History],
    right_histories: set[History],
) -> np.ndarray | None:
    return contingency_table(
        _observed_counts_for_morph(counts, left_histories),
        _observed_counts_for_morph(counts, right_histories),
        counts.alphabet,
    )


def _bonferroni_alpha(
    counts: SuffixCounts,
    alpha: float,
    *,
    max_length: int,
    min_count: int,
    suffix_length: Callable[[History], int] = len,
    resolution_tests: bool = False,
) -> float:
    """``alpha`` divided by the number of suffixes eligible for a split test.

    With ``resolution_tests`` the observed length-``max_length + 1`` suffixes are
    counted too: process CSSR tests each of them when resolving the successor of a
    length-``max_length`` suffix.
    """
    eligible = 0
    for history, following in counts.next_counts.items():
        length = suffix_length(history)
        total = sum(following.values())
        if (0 < length <= max_length and total >= max(1, min_count)) or (
            resolution_tests and length == max_length + 1 and total > 0
        ):
            eligible += 1
    return alpha / max(1, eligible)


def morphs_differ(
    counts: SuffixCounts,
    left_histories: set[History],
    right_histories: set[History],
    *,
    alpha: float = 0.05,
    test: MorphTest = "g",
    delta: float = 0.0,
) -> bool:
    """Return whether two history sets have significantly different morphs.

    ``"g"`` is the G-test (log-likelihood ratio) with Yates' continuity correction
    when the table has one degree of freedom, i.e. two observed symbols; this is
    the statistic ``scipy.stats.chi2_contingency(table, lambda_="log-likelihood")``
    reports. transCSSR uses the same test. ``"g"`` and ``"chi2"`` use the
    chi-squared limit, which is unreliable when expected counts are small.
    ``"exact"`` instead compares the G statistic with tables drawn uniformly given
    the observed margins whenever an expected count is below 5, and uses the
    G-test otherwise.
    """
    if test == "tv":
        left = counts.state_morph(left_histories)
        right = counts.state_morph(right_histories)
        distance = 0.5 * sum(abs(left[s] - right[s]) for s in counts.alphabet)
        return distance > delta

    table = _contingency_rows(counts, left_histories, right_histories)
    if table is None:
        return False
    return table_significant(table, alpha, test)


def morph_test_score(
    counts: SuffixCounts,
    left_histories: set[History],
    right_histories: set[History],
    *,
    test: MorphTest = "g",
) -> float:
    """Score for matching morphs (lower is more similar)."""
    if test == "tv":
        left = counts.state_morph(left_histories)
        right = counts.state_morph(right_histories)
        return 0.5 * sum(abs(left[s] - right[s]) for s in counts.alphabet)
    table = _contingency_rows(counts, left_histories, right_histories)
    if table is None:
        return 0.0
    return table_score(table, test)


def aggregates_differ(
    left: StateAggregate,
    right: StateAggregate,
    *,
    input_alphabet: tuple[Any, ...],
    output_alphabet: tuple[Any, ...],
    alpha: float,
    test: TableTest = "g",
) -> bool:
    """Return whether two aggregated morphs differ on ``P(output | ., input)`` for some input.

    Each input symbol's output counts are compared with the same test as process
    CSSR (:func:`~sofic.inference.cssr.morphs_differ`); in particular
    ``"g"`` is the G-test with Yates' continuity correction at one degree of freedom.
    """
    for input_symbol in input_alphabet:
        table = contingency_table(
            left.get(input_symbol, Counter()),
            right.get(input_symbol, Counter()),
            output_alphabet,
        )
        if table is None:
            continue
        if table_significant(table, alpha, test):
            return True
    return False


def _aggregate_score(
    left: StateAggregate,
    right: StateAggregate,
    *,
    input_alphabet: tuple[Any, ...],
    output_alphabet: tuple[Any, ...],
) -> float:
    total = 0.0
    for input_symbol in input_alphabet:
        table = contingency_table(
            left.get(input_symbol, Counter()),
            right.get(input_symbol, Counter()),
            output_alphabet,
        )
        if table is not None:
            total += table_score(table, "g")
    return total
