"""Rigorous LP/SDP lower bounds from approximate duals."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sofic.generators.certificates import PSDBlock, certified_lp_minimum, certified_psd, dual_bound


@st.composite
def _feasible_lps(draw):
    n = draw(st.integers(2, 6))
    m_eq = draw(st.integers(0, 3))
    m_ub = draw(st.integers(0, 3))
    ints = st.integers(-3, 3)
    x0 = np.array([draw(st.integers(0, 8)) / 8 for _ in range(n)])
    c = np.array([draw(ints) for _ in range(n)], dtype=float)
    A_eq = np.array([[draw(ints) for _ in range(n)] for _ in range(m_eq)], dtype=float).reshape(m_eq, n)
    A_ub = np.array([[draw(ints) for _ in range(n)] for _ in range(m_ub)], dtype=float).reshape(m_ub, n)
    slack = np.array([draw(st.integers(0, 4)) / 4 for _ in range(m_ub)])
    return c, x0, A_eq, A_eq @ x0, A_ub, A_ub @ x0 + slack


@settings(max_examples=60, deadline=None)
@given(_feasible_lps())
def test_lp_bound_is_valid_and_tight(lp) -> None:
    c, x0, A_eq, b_eq, A_ub, b_ub = lp
    n = len(c)
    kwargs = {}
    if len(A_eq):
        kwargs.update(A_eq=A_eq, b_eq=b_eq)
    if len(A_ub):
        kwargs.update(A_ub=A_ub, b_ub=b_ub)
    result = certified_lp_minimum(c, np.zeros(n), np.ones(n), **kwargs)
    assert result.status == "optimal" and result.bound is not None and result.float_value is not None
    # x0 is exactly feasible (dyadic data), so its value is an exact upper reference for the minimum
    assert result.bound <= sum(Fraction(float(ci)) * Fraction(float(xi)) for ci, xi in zip(c, x0, strict=True))
    assert float(result.bound) >= result.float_value - 1e-7


def test_wrong_duals_still_give_valid_bounds() -> None:
    c = np.array([1.0, 1.0])
    A_eq = np.array([[1.0, 1.0]])
    for y in (-5.0, 0.0, 0.3, 1.0, 7.0):
        bound = dual_bound(c, [0, 0], [1, 1], A_eq=A_eq, b_eq=[1.0], y_eq=[y])
        assert bound is not None and bound <= 1
    assert dual_bound(c, [0, 0], [1, 1], A_eq=A_eq, b_eq=[1.0], y_eq=[1.0]) == 1


def test_infeasibility_is_certified() -> None:
    result = certified_lp_minimum(
        [1.0, 0.0], [0, 0], [1, 1], A_eq=np.array([[1.0, 1.0]]), b_eq=[3.0], A_ub=np.array([[1.0, 0.0]]), b_ub=[0.5]
    )
    assert result.status == "infeasible"


def test_unbounded_box_with_bad_sign_gives_none() -> None:
    assert dual_bound([1.0], [-np.inf], [1.0]) is None


def test_certified_psd_shifts_indefinite_matrices() -> None:
    Z = certified_psd(np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert Z[0][0] > 1 and Z[0][0] * Z[1][1] - Z[0][1] ** 2 > 0


def test_sdp_bound() -> None:
    cvxpy = pytest.importorskip("cvxpy")
    # min x  s.t.  [[x, 1], [1, x]] >= 0  has value 1
    x = cvxpy.Variable()
    M = cvxpy.bmat([[x, 1], [1, x]])
    con = M >> 0
    cvxpy.Problem(cvxpy.Minimize(x), [con, x <= 10, x >= -10]).solve(solver="CLARABEL")
    block = PSDBlock(F0=np.array([[0.0, 1.0], [1.0, 0.0]]), F={0: np.eye(2)})
    bound = dual_bound([1.0], [-10.0], [10.0], psd=[block], Z=[con.dual_value])
    assert bound is not None and 1 - 1e-6 <= bound <= 1
