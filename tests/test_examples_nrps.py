"""Regression tests for the NRPS example process."""

from __future__ import annotations

import pytest

from sofic.examples import even_process, golden_mean, noisy_random_phase_slip
from sofic.examples.processes import NRPS


def test_nrps_alias_matches_canonical_constructor():
    assert NRPS().is_equal_process(noisy_random_phase_slip())


def test_nrps_topology():
    eps = noisy_random_phase_slip()
    edges = {(t.source, t.data["emission"], t.target): t.data["prob"] for t in eps.transitions()}
    assert edges[("A", 0, "A")] == pytest.approx(0.5)
    assert edges[("A", 1, "B")] == pytest.approx(0.5)
    assert edges[("D", 0, "E")] == pytest.approx(0.5)
    assert edges[("D", 1, "E")] == pytest.approx(0.5)
    assert len(list(eps.states())) == 5


def test_nrps_distinct_from_other_prototypes():
    nrps = noisy_random_phase_slip()
    assert nrps.block_convergence_estimates(8).block_entropy[1] != pytest.approx(
        golden_mean(0.5).block_convergence_estimates(8).block_entropy[1],
        abs=1e-3,
    )
    assert nrps.block_convergence_estimates(8).r_mu != pytest.approx(
        even_process(0.5).block_convergence_estimates(8).r_mu,
        abs=1e-3,
    )
