"""Tests for belief-map semigroups of hidden Markov models."""

import pytest

from sofic.algebra import krohn_rhodes_complexity, transformation_semigroup
from sofic.algebra.belief import mixed_state_semigroup, projective_semigroup
from sofic.examples.epsilon_machines import even_process, golden_mean
from sofic.examples.processes import nonunifilar_golden_mean, sns
from sofic.exceptions import SemigroupTooLargeError


@pytest.mark.parametrize("factory", [golden_mean, even_process])
def test_unifilar_machines_have_finite_projective_semigroups(factory):
    machine = factory()
    projective = projective_semigroup(machine)
    assert len(projective.states) == len(transformation_semigroup(machine))
    assert projective.states[0] == ()
    assert krohn_rhodes_complexity(projective) == krohn_rhodes_complexity(machine)


def test_projective_semigroup_acts_faithfully_on_itself():
    projective = projective_semigroup(even_process())
    assert len(projective) == len(projective.states)


def test_nonunifilar_presentations_can_still_be_finite():
    assert len(projective_semigroup(nonunifilar_golden_mean()).states) == 6


def test_simple_nonunifilar_source_does_not_close():
    with pytest.raises(SemigroupTooLargeError):
        projective_semigroup(sns(), max_size=500)


def test_mixed_state_semigroup_uses_the_mixed_state_presentation():
    sg = mixed_state_semigroup(even_process())
    assert len(sg.states) == 5
    assert krohn_rhodes_complexity(sg).exact == 1
