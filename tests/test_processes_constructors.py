"""Variants and argument checks of the cmpy-compatible constructors in :mod:`sofic.examples.processes`."""

from __future__ import annotations

import numpy as np
import pytest

from sofic.examples import processes as pr
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.mealy import MealyHMM

VARIANTS = [
    (pr.band_merging, {"machine_type": EpsilonMachine}),
    (pr.beads_on_necklace, {"beads": [(), ("a",), ("a", "b", "c")], "necklace": ("0", "1", "2"), "probs": 0.4}),
    (pr.beads_on_necklace, {"beads": ("a", "b"), "necklace": ("0", "1"), "probs": [0.3, 0.6]}),
    (pr.before_after, {"style": "cayley"}),
    (pr.binary_markov_chain, {"p": 0.4, "q": 0.3, "machine_type": "generative"}),
    (pr.binary_markov_chain, {"p": 0.4, "q": 0.3, "a": 0.1, "b": 0.8, "machine_type": "parametrized"}),
    (pr.binary_markov_chain, {"p": 0.3, "machine_type": "lohr"}),
    (pr.bmc_em, {"p": 0.3, "q": 0.7}),
    (pr.bmc_gen, {"p": 1, "q": 1}),
    (pr.bmc_gen, {"p": 0, "q": 0}),
    (pr.bmc_gen, {"p": 0.7, "q": 0.6}),
    (pr.bmc_gen, {"p": 0.2, "q": 0.4}),
    (pr.bmc_gen, {"p": 0.4, "q": 0.2}),
    (pr.coupled_gmps, {"epsilon": 0.1, "alt": False}),
    (pr.coupled_gmps, {"epsilon": 0.0, "alt": False}),
    (pr.coupled_gmps, {"epsilon": 0.0}),
    (pr.iid, {"k": 2, "machine_type": "em"}),
    (pr.iid, {"k": 2, "machine_type": "mealy"}),
    (pr.iid, {"k": 2, "machine_type": "unknown"}),
    (pr.iid, {"k": ("x", "y", "z"), "machine_type": MealyHMM}),
    (pr.logic_machine, {"logic": "rrx"}),
    (pr.logic_machine, {"logic": "rn1c", "bias": [0.3], "noise": 0.2}),
    (pr.markov_skeleton, {"R": 1, "k": 11}),
    (pr.markov_skeleton, {"R": 0, "k": 2}),
    (pr.perturbed_coin, {"p": 0.5}),
    (pr.perturbed_coin, {"p": 0.2, "machine_type": "lohr"}),
    (pr.perturbed_coin, {"p": 0.2, "machine_type": "generative"}),
    (pr.rip, {"reverse": True}),
    (pr.rrx, {"machine_type": MealyHMM}),
    (pr.stretched_gm, {"k": 3}),
    (pr.rn_gm, {"R": 3, "N": 2}),
    (pr.rk_gm, {"R": 2, "k": 2}),
]


@pytest.mark.parametrize(("factory", "kwargs"), VARIANTS, ids=lambda v: getattr(v, "__name__", None))
def test_constructor_variants_are_valid_machines(factory, kwargs):
    machine = factory(**kwargs)
    machine.validate()
    pi = np.asarray(machine.stationary_distribution(), dtype=float)
    assert pi.sum() == pytest.approx(1.0)


ERRORS = [
    (pr.afc, {"n": 0}, ValueError),
    (pr.afc2, {"n": 0}, ValueError),
    (pr.beads_on_necklace, {"beads": (), "necklace": ("0",)}, ValueError),
    (pr.beads_on_necklace, {"beads": [("a",), ("b",)], "necklace": ("0",)}, ValueError),
    (pr.beads_on_necklace, {"beads": ("a",), "necklace": ("0", "1"), "probs": [0.5]}, ValueError),
    (pr.beads_on_necklace, {"machine_type": EpsilonMachine}, NotImplementedError),
    (pr.before_after, {"style": "fancy"}, ValueError),
    (pr.binary_markov_chain, {"machine_type": "parametrized"}, ValueError),
    (pr.binary_markov_chain, {"machine_type": "quantum"}, NotImplementedError),
    (pr.binary_markov_chain, {"machine_type": MealyHMM}, NotImplementedError),
    (pr.bmc_param, {"p": 0.4, "q": 0.3, "a": 0.9, "b": 0.8}, ValueError),
    (pr.bmc_lohr, {"p": 0.6}, ValueError),
    (pr.coupled_gmps, {"epsilon": -0.1}, ValueError),
    (pr.uncoupled_gmps, {"p": 1.5}, ValueError),
    (pr.cyclic_branching, {"num_states": 2, "num_branchings": 2}, ValueError),
    (pr.flower, {"N": 1}, ValueError),
    (pr.stretched_gm, {"k": 0}, ValueError),
    (pr.rn_gm, {"R": 1, "N": 2}, ValueError),
    (pr.rk_gm, {"R": 0, "k": 1}, ValueError),
    (pr.nonunifilar_golden_mean, {"free": 0.01}, ValueError),
    (pr.logic_machine, {"logic": "xor"}, NotImplementedError),
    (pr.markov_skeleton, {"R": 1, "k": ("ab", "c"), "join": True}, ValueError),
    (pr.perturbed_coin, {"p": 0.2, "q": 0.3, "machine_type": "lohr"}, ValueError),
    (pr.perturbed_coin, {"p": 0.2, "machine_type": "quantum"}, NotImplementedError),
    (pr.period16, {"machine_type": EpsilonMachine}, NotImplementedError),
]


@pytest.mark.parametrize(("factory", "kwargs", "error"), ERRORS, ids=lambda v: getattr(v, "__name__", None))
def test_constructor_argument_checks(factory, kwargs, error):
    with pytest.raises(error):
        factory(**kwargs)


def test_transition_specifications_need_three_or_four_fields():
    with pytest.raises(ValueError, match="invalid transition specification"):
        pr._from_string("A B; B A 1")


def test_ising_rejects_non_finite_probabilities():
    with np.errstate(all="ignore"), pytest.raises(FloatingPointError):
        pr.ising(J=1.0, B=0.3, T=1e-3)


def test_markov_skeleton_with_tuple_states():
    machine = pr.markov_skeleton(2, ("a", "b"), join=False)
    assert ("a", "b") in set(machine.states())
    assert len(list(machine.transitions())) == 8


def test_uniform_mealyhmm_topologies():
    rng = np.random.default_rng(0)
    from_machine = pr.uniform_mealyhmm(pr.even_process(), prng=rng)
    assert {(t.source, t.target) for t in from_machine.transitions()} == {
        (t.source, t.target) for t in pr.even_process().transitions()
    }
    one = np.ones((2, 2))
    from_mapping = pr.uniform_mealyhmm({"x": one, "y": np.eye(2)}, nodes=("p", "q"), prng=rng)
    assert set(from_mapping.states()) == {"p", "q"}
    assert from_mapping.observation_alphabet == frozenset("xy")
    from_list = pr.uniform_mealyhmm([one, one], symbols=("u", "v"), prng=np.random.RandomState(0))
    assert from_list.observation_alphabet == frozenset("uv")
    default_rng = pr.uniform_mealyhmm([one], prng=object())
    for machine in (from_machine, from_mapping, from_list, default_rng):
        machine.validate()
