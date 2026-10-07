"""Every example factory builds, validates, and matches its documented numbers."""

from __future__ import annotations

import inspect
import math
from collections import defaultdict
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import sofic.examples as examples
from sofic.base import StateMachine
from sofic.examples import processes
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.graph import ATTR_EMISSION, ATTR_EMISSION_DIST, ATTR_OUTPUT, ATTR_PROB, ATTR_SYMBOL, EPSILON

ALPHABET_FIELDS = (
    "symbol_alphabet",
    "observation_alphabet",
    "input_alphabet",
    "output_alphabet",
    "call_alphabet",
    "return_alphabet",
    "internal_alphabet",
)
MAX_ORACLE_STATES = 200

PARAMETRIZED_FACTORIES = {
    "afc": lambda: examples.afc(3),
    "afc2": lambda: examples.afc2(3),
    "bmc_em": lambda: examples.bmc_em(0.3, 0.6),
    "bmc_gen": lambda: examples.bmc_gen(0.3, 0.6),
    "bmc_lohr": lambda: examples.bmc_lohr(0.3),
    "bmc_param": lambda: examples.bmc_param(0.3, 0.6, 0.2, 0.7),
    "cyclic_branching": lambda: examples.cyclic_branching(3, 2),
    "iid": lambda: examples.iid(3),
    "lollipop": lambda: examples.lollipop(2, 3),
    "markov_skeleton": lambda: examples.markov_skeleton(2, 2),
    "multiple_n": lambda: examples.multiple_n(3),
    "period": lambda: examples.period(3),
    "periodic": lambda: examples.periodic("0011"),
    "rk_gm": lambda: examples.rk_gm(2, 2),
    "rn_gm": lambda: examples.rn_gm(2, 2),
    "stretched_gm": lambda: examples.stretched_gm(2),
    "uniform_mealymc": lambda: examples.uniform_mealymc(1, 2),
}


def _required(factory: Any) -> list[str]:
    return [
        p.name
        for p in inspect.signature(factory).parameters.values()
        if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def _model_factories() -> dict[str, Any]:
    found: dict[str, Any] = {}
    modules = (examples, processes)
    candidates = [(name, getattr(module, name)) for module in modules for name in sorted(set(module.__all__))]
    for name, factory in candidates:
        if not inspect.isfunction(factory) or inspect.signature(factory).return_annotation in ("float", "dict"):
            continue
        if not _required(factory):
            found[name] = factory
    found.update(PARAMETRIZED_FACTORIES)
    return found


FACTORIES = _model_factories()


def test_every_required_argument_factory_is_curated():
    names = {name for module in (examples, processes) for name in module.__all__}
    needs_args = {
        name
        for name in names
        if inspect.isfunction(getattr(examples, name, None) or getattr(processes, name))
        and _required(getattr(examples, name, None) or getattr(processes, name))
        and inspect.signature(getattr(examples, name, None) or getattr(processes, name)).return_annotation
        not in ("float", "dict")
    }
    missing = needs_args - set(PARAMETRIZED_FACTORIES)
    assert not {name for name in missing if not name.startswith(("tent_map", "logic_machine", "uniform_mealyhmm"))}


def _emitted_symbols(model: StateMachine) -> set[Any]:
    symbols: set[Any] = set()
    for transition in model.transitions():
        for key in (ATTR_SYMBOL, ATTR_EMISSION, ATTR_OUTPUT):
            if key in transition.data and transition.data[key] is not EPSILON:
                symbols.add(transition.data[key])
    for state in model.states():
        symbols.update(model.graph.state_attrs(state).get(ATTR_EMISSION_DIST, {}) or {})
    return symbols


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_factory_builds_validates_and_uses_string_symbols(name):
    model = FACTORIES[name]()
    if not isinstance(model, StateMachine):
        pytest.skip(f"{name} returns {type(model).__name__}")
    model.validate()
    assert list(model.states()), name
    emitted = _emitted_symbols(model)
    assert all(isinstance(symbol, str) for symbol in emitted), sorted(map(repr, emitted))
    for field in ALPHABET_FIELDS:
        alphabet = getattr(model, field, None)
        if alphabet:
            assert all(isinstance(symbol, str) for symbol in alphabet), (field, sorted(map(repr, alphabet)))


# --------------------------------------------------------------------------- raw-graph oracle


def _entropy(probs: Any) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 1e-15)


def unifilar_oracle(machine: Any) -> tuple[float, float]:
    """``(h_mu, C_mu)`` from the raw edge list: stationary ``pi`` by eigenvector, then ``sum pi H(row)`` and ``H(pi)``."""
    states = list(machine.states())
    index = {state: i for i, state in enumerate(states)}
    matrix = np.zeros((len(states), len(states)))
    rows: dict[Any, list[float]] = defaultdict(list)
    for transition in machine.transitions():
        prob = float(transition.data[ATTR_PROB])
        matrix[index[transition.source], index[transition.target]] += prob
        rows[transition.source].append(prob)
    eigenvalues, vectors = np.linalg.eig(matrix.T)
    pi = np.real(vectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
    pi = pi / pi.sum()
    h = sum(pi[index[state]] * _entropy(rows[state]) for state in states)
    return h, _entropy(pi)


EPSILON_MACHINES = sorted(
    name for name in FACTORIES if FACTORIES[name].__annotations__.get("return") in ("EpsilonMachine", EpsilonMachine)
)


@pytest.mark.parametrize("name", EPSILON_MACHINES)
def test_epsilon_machine_examples_match_raw_graph_oracle(name):
    machine = FACTORIES[name]()
    if not isinstance(machine, EpsilonMachine) or len(list(machine.states())) > MAX_ORACLE_STATES:
        pytest.skip("not a small epsilon-machine")
    if not machine.is_irreducible():
        pytest.skip("stationary distribution is not unique")
    h, c = unifilar_oracle(machine)
    assert machine.entropy_rate() == pytest.approx(h, abs=1e-8)
    assert machine.statistical_complexity() == pytest.approx(c, abs=1e-8)


# --------------------------------------------------------------------------- documented numbers

PHI = (1 + math.sqrt(5)) / 2
H_THIRD = _entropy([1 / 3, 2 / 3])

DOCUMENTED = {
    "golden_mean()": (examples.golden_mean, {"entropy_rate": 2 / 3, "statistical_complexity": H_THIRD}),
    "golden_mean(1/phi)": (
        lambda: examples.golden_mean(1 / PHI),
        {"entropy_rate": math.log2(PHI)},
    ),
    "even_process()": (examples.even_process, {"entropy_rate": 2 / 3, "statistical_complexity": H_THIRD}),
    "butterfly_process()": (
        examples.butterfly_process,
        {"entropy_rate": 1.0, "statistical_complexity": math.log2(5), "crypticity": 0.3},
    ),
    "bernoulli()": (examples.bernoulli, {"entropy_rate": 1.0, "statistical_complexity": 0.0}),
}


@pytest.mark.parametrize("label", sorted(DOCUMENTED))
def test_documented_numbers(label):
    factory, expected = DOCUMENTED[label]
    machine = factory()
    for measure, value in expected.items():
        assert getattr(machine, measure)() == pytest.approx(value, abs=1e-9), measure


@given(st.floats(0.01, 0.99))
def test_bernoulli_entropy_rate_is_binary_entropy(p):
    machine = examples.bernoulli(p)
    assert machine.entropy_rate() == pytest.approx(_entropy([p, 1 - p]), abs=1e-9)
    assert machine.statistical_complexity() == pytest.approx(0.0, abs=1e-9)


@given(st.floats(0.05, 0.95))
def test_golden_mean_family_matches_oracle(p):
    machine = examples.golden_mean(p)
    h, c = unifilar_oracle(machine)
    assert machine.entropy_rate() == pytest.approx(h, abs=1e-9)
    assert machine.statistical_complexity() == pytest.approx(c, abs=1e-9)
    assert machine.entropy_rate() == pytest.approx(_entropy([p, 1 - p]) / (2 - p), abs=1e-9)


@given(st.floats(0.05, 0.95), st.floats(0.05, 0.95))
def test_nemo_process_matches_oracle(p, q):
    machine = examples.nemo_process(p, q)
    h, c = unifilar_oracle(machine)
    assert machine.entropy_rate() == pytest.approx(h, abs=1e-8)
    assert machine.statistical_complexity() == pytest.approx(c, abs=1e-8)


@pytest.mark.parametrize("k", range(1, 7))
def test_period_k_is_deterministic_with_log_k_memory(k):
    machine = EpsilonMachine.from_hmm(examples.period(k))
    assert len(list(machine.states())) == k
    assert machine.entropy_rate() == pytest.approx(0.0, abs=1e-9)
    assert machine.statistical_complexity() == pytest.approx(math.log2(k), abs=1e-9)


def test_tent_map_forward_matches_documented_information():
    machine = examples.tent_map_misiurewicz_forward()
    expected = examples.tent_map_misiurewicz_information_expected()
    assert machine.entropy_rate() == pytest.approx(float(expected["entropy_rate"]), abs=1e-8)
