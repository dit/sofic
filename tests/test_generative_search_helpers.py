"""Fast checks of the helpers around :func:`sofic.generators.generative_search.generative_model`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sofic.examples import even_process, golden_mean
from sofic.examples.epsilon_machines import golden_mean_bidirectional
from sofic.generators import generative_search as gs
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.coupled_generators import _GeneratorProblem

DSBS = {("0", "0"): 0.4, ("0", "1"): 0.1, ("1", "0"): 0.1, ("1", "1"): 0.4}


def test_resolve_bidirectional_accepts_or_builds():
    given = golden_mean_bidirectional()
    bidir, meta = gs.resolve_bidirectional(given)
    assert bidir is given
    assert meta == {"bidir_class": "finite", "source": "bidirectional input"}
    built, meta = gs.resolve_bidirectional(even_process(), use_ladder=False)
    assert isinstance(built, BidirectionalEpsilonMachine)
    assert meta["bidir_class"] == "finite"
    assert meta["reverse_recurrence_finite"]
    assert "build_error" not in meta


def test_persistent_crypticity_theorem_applies_to_the_golden_mean():
    bidir, _meta = gs.resolve_bidirectional(golden_mean())
    assert gs._theorem_applies(_GeneratorProblem(bidir))


def test_coverage_lower_bound_modes():
    rng = np.random.default_rng(0)
    heuristic, meta, certified = gs._coverage_lower_bound(DSBS, rng, mu_starts=3, certified=False)
    assert not certified and not meta["certified"]
    assert {"mu_1", "mu_2"} <= set(meta)
    skipped, meta, certified = gs._coverage_lower_bound(DSBS, rng, mu_starts=3, certified=False, certified_only=True)
    assert skipped == -math.inf and meta["skipped"] and not certified
    exact, meta, certified = gs._coverage_lower_bound(DSBS, rng, mu_starts=3, certified=True, certified_only=True)
    assert certified and meta["certified"]
    assert meta["mu_1"] == pytest.approx(0.625, abs=1e-6)
    assert exact == pytest.approx(heuristic, abs=1e-4)
