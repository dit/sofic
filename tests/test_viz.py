"""Tests for Graphviz visualization."""

from __future__ import annotations

import pytest

from pensive.automata.dfa import DFA
from pensive.examples.epsilon_machines import golden_mean, golden_mean_bidirectional
from pensive.generators.markov import MarkovChain
from pensive.graph import ATTR_PROB
from pensive.viz._context import viz_context
from pensive.viz._format import format_belief, format_distribution, format_prob_rational
from pensive.viz.graphviz import model_to_graphviz

graphviz = pytest.importorskip("graphviz")


def _dfa() -> DFA:
    dfa = DFA(
        input_alphabet=frozenset({"a", "b"}),
        initial_states=frozenset({"q0"}),
        accepting_states=frozenset({"q1"}),
    )
    for state in ("q0", "q1"):
        dfa.graph.add_state(state)
    dfa.add_transition("q0", "q1", "a")
    dfa.add_transition("q0", "q0", "b")
    dfa.add_transition("q1", "q1", "a")
    dfa.add_transition("q1", "q0", "b")
    return dfa


def test_repr_summary():
    dfa = _dfa()
    assert repr(dfa) == "DFA(2 states, 4 transitions)"


def test_viz_context_labels_dfa():
    context = viz_context(_dfa())
    assert context.initial_states == frozenset({"q0"})
    assert context.accepting_states == frozenset({"q1"})
    transitions = list(_dfa().transitions())
    assert context.edge_label(transitions[0]) in {"a", "b"}


def test_model_to_graphviz_contains_states_and_edges():
    dot = model_to_graphviz(_dfa())
    source = dot.source
    assert "q0" in source
    assert "q1" in source
    assert "a" in source
    assert "b" in source
    assert "__start__" in source


def test_epsilon_machine_renders():
    eps = golden_mean()
    dot = model_to_graphviz(eps)
    source = dot.source
    assert "EpsilonMachine" in source or "digraph" in source
    assert "0" in source
    assert "1" in source
    assert "1/2" in source
    assert "0.5" not in source


def test_format_prob_rational_two_digit_fractions():
    assert format_prob_rational(0.5) == "1/2"
    assert format_prob_rational(2 / 3) == "2/3"
    assert format_prob_rational(1 / 99) == "1/99"
    assert format_prob_rational(0.0) == "0"
    assert format_prob_rational(1.0) == "1"


def test_format_prob_rational_decimal_fallback():
    assert format_prob_rational(0.01) == "0.01"
    assert format_prob_rational(0.123456789) == "0.123"


def test_format_belief_uses_fractions():
    assert format_belief((1 / 3, 2 / 3)) == "μ=(1/3, 2/3)"


def test_format_distribution_uses_fractions():
    assert format_distribution({"0": 0.5, "1": 0.5}) == "0:1/2, 1:1/2"


def test_markov_chain_stationary_annotation_uses_fractions():
    chain = MarkovChain(initial_distribution={"A": 0.5, "B": 0.5})
    for state in ("A", "B"):
        chain.graph.add_state(state)
    chain.graph.add_transition("A", "A", **{ATTR_PROB: 0.5})
    chain.graph.add_transition("A", "B", **{ATTR_PROB: 0.5})
    chain.graph.add_transition("B", "A", **{ATTR_PROB: 1.0})
    source = model_to_graphviz(chain).source
    assert "π=1/2" in source
    assert "π=0.5" not in source


def test_repr_mimebundle_uses_graphviz_svg():
    dfa = _dfa()
    bundle = dfa._repr_mimebundle_()
    assert bundle is not None
    assert bundle["text/plain"] == repr(dfa)
    assert "image/svg+xml" in bundle
    assert bundle["image/svg+xml"].lstrip().startswith("<")
    assert "text/latex" not in bundle


def test_repr_png_method():
    png = _dfa()._repr_png_()
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_find_executable_discovers_mactex_without_path():
    import os

    from pensive.viz._tikz_compile import find_executable

    mactex = "/Library/TeX/texbin/pdflatex"
    if not os.path.isfile(mactex):
        pytest.skip("MacTeX pdflatex not installed")
    saved = os.environ.get("PATH")
    os.environ["PATH"] = "/usr/bin:/bin"
    try:
        assert find_executable("pdflatex") == mactex
    finally:
        if saved is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = saved


def test_repr_mimebundle_without_graphviz(monkeypatch):
    dfa = _dfa()

    def _fail_svg(_self):
        return None

    monkeypatch.setattr(DFA, "_graphviz_svg", _fail_svg)
    bundle = dfa._repr_mimebundle_()
    assert bundle is None


def test_epsilon_machine_recurrent_states_colored():
    from pensive.examples.epsilon_machines import golden_mean_bidirectional, golden_mean_forward

    forward = model_to_graphviz(golden_mean_forward(0.5)).source
    assert "honeydew" in forward
    assert "mistyrose" not in forward

    bidir = model_to_graphviz(golden_mean_bidirectional(0.5), style="paper").source
    assert "honeydew" in bidir


def test_msp_state_fillcolors():
    from pensive.examples.epsilon_machines import golden_mean

    source = model_to_graphviz(golden_mean(0.5).mixed_state_presentation()).source
    assert "mistyrose" in source
    assert "honeydew" in source


def test_draw_method(tmp_path):
    dfa = _dfa()
    out = dfa.draw(filename=str(tmp_path / "dfa"), format="svg", view=False)
    assert out is not None
    assert out.endswith(".svg")


def test_golden_mean_bidirectional_paper_style_dot():
    bidir = golden_mean_bidirectional(0.5)
    source = model_to_graphviz(bidir).source
    assert "__start__" not in source
    assert "π=" not in source
    assert "state_" not in source
    assert "(A, C)" in source
    assert "(A, D)" in source
    assert "(B, C)" in source
    assert "1/2" in source
    assert "0.5" not in source


def test_bidirectional_transition_endpoints_use_registered_nodes():
    bidir = golden_mean_bidirectional(0.5)
    source = model_to_graphviz(bidir).source
    node_ids = {
        line.split("[", 1)[0].strip()
        for line in source.splitlines()
        if "[" in line and "__start__" not in line and "->" not in line
    }
    for line in source.splitlines():
        if "->" not in line:
            continue
        left, right = line.split("->", 1)
        src = left.strip().strip('"')
        dst = right.split("[", 1)[0].strip().strip('"')
        assert src in node_ids
        assert dst in node_ids
