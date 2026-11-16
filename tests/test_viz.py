"""Tests for Graphviz visualization."""

from __future__ import annotations

import pytest

from sofic.automata.dfa import DFA
from sofic.examples.epsilon_machines import golden_mean, golden_mean_bidirectional
from sofic.generators.markov import MarkovChain
from sofic.graph import ATTR_PROB
from sofic.viz._context import EMISSION_PALETTE, viz_context
from sofic.viz._format import (
    format_belief,
    format_distribution,
    format_prob_label,
    format_prob_rational,
    format_state,
)
from sofic.viz._tikz_format import format_state_tikz_node
from sofic.viz.graphviz import model_to_graphviz
from sofic.viz.tikz import model_to_tikz

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


def _nested_frozenset_dfa() -> DFA:
    start = frozenset({frozenset({0}), frozenset({1, 2})})
    accepting = frozenset()
    dfa = DFA(
        input_alphabet=frozenset({"a"}),
        initial_states=frozenset({start}),
        accepting_states=frozenset({accepting}),
    )
    dfa.graph.add_state(start)
    dfa.graph.add_state(accepting)
    dfa.add_transition(start, accepting, "a")
    dfa.add_transition(accepting, accepting, "a")
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


def test_sofic_dyck_graphviz_marks_matched_edges():
    from sofic.examples import sofic_dyck_nondeterminizable_shift

    source = model_to_graphviz(sofic_dyck_nondeterminizable_shift()).source

    assert 'label="a | call | m1"' in source
    assert 'label="b | return | m1"' in source
    assert 'label="b | return"' in source
    assert source.count("m1") == 2
    assert "color=seagreen" in source
    assert "color=firebrick" in source


def test_epsilon_machine_renders():
    eps = golden_mean()
    dot = model_to_graphviz(eps)
    source = dot.source
    assert "EpsilonMachine" in source or "digraph" in source
    assert "0" in source
    assert "1" in source
    assert "1/2" in source
    assert "0.5" not in source


def test_edge_machine_renders_readable_state_labels():
    edge = golden_mean().to_edge_machine()
    source = model_to_graphviz(edge).source
    assert "\x1e" not in source
    assert "(A, 0, A)" in source

    bundle = edge._repr_mimebundle_()
    assert bundle is not None
    assert "image/svg+xml" in bundle
    assert bundle["image/svg+xml"].lstrip().startswith("<")


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


def test_format_state_formats_frozensets_readably():
    nested = frozenset({frozenset({0}), frozenset({1, 2})})
    assert format_state(frozenset({0, 1})) == r"\{0, 1\}"
    assert format_state(nested) == r"\{\{0\}, \{1, 2\}\}"
    assert format_state(frozenset()) == r"\{\}"


def test_graphviz_nested_frozenset_state_labels_are_readable():
    source = model_to_graphviz(_nested_frozenset_dfa()).source
    assert "frozenset(" not in source
    assert r"\{\{0\}, \{1, 2\}\}" in source
    assert r"\{\}" in source


def test_tikz_nested_frozenset_state_labels_are_readable():
    assert format_state_tikz_node(frozenset({frozenset({0}), frozenset({1, 2})})) == (r"\{\{0\}{,} \{1{,} 2\}\}")
    source = model_to_tikz(_nested_frozenset_dfa())
    assert "frozenset(" not in source
    assert r"\{\{0\}{,} \{1{,} 2\}\}" in source


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

    from sofic.viz._tikz_compile import find_executable

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
    from sofic.examples.epsilon_machines import golden_mean, golden_mean_bidirectional

    forward = model_to_graphviz(golden_mean(0.5)).source
    assert "honeydew" in forward
    assert "mistyrose" not in forward

    bidir = model_to_graphviz(golden_mean_bidirectional(0.5), style="paper").source
    assert "honeydew" in bidir


def test_msp_state_fillcolors():
    from sofic.examples.epsilon_machines import golden_mean

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


def test_format_prob_label_symbolic():
    pytest.importorskip("sympy")
    import sympy as sp

    a = sp.symbols("a", positive=True)
    assert format_prob_label(sp.Rational(1, 2)) == "1/2"
    assert "a" in format_prob_label(a / (a + 1))
    assert format_prob_label(0.5) == "1/2"


def test_edges_colored_by_emission_by_default():
    eps = golden_mean()
    source = model_to_graphviz(eps).source
    assert f'color="{EMISSION_PALETTE[0]}"' in source
    assert f'color="{EMISSION_PALETTE[1]}"' in source

    context = viz_context(eps)
    by_symbol: dict[object, set[str | None]] = {}
    for transition in eps.transitions():
        by_symbol.setdefault(transition.data["emission"], set()).add(context.edge_color(transition))
    assert by_symbol["0"] == {EMISSION_PALETTE[0]}
    assert by_symbol["1"] == {EMISSION_PALETTE[1]}


def test_color_by_emission_can_be_disabled():
    source = model_to_graphviz(golden_mean(), color_by_emission=False).source
    assert 'color="#' not in source


def test_dfa_edges_colored_by_input_symbol():
    source = model_to_graphviz(_dfa()).source
    assert f'color="{EMISSION_PALETTE[0]}"' in source
    assert f'color="{EMISSION_PALETTE[1]}"' in source

    plain = model_to_graphviz(_dfa(), color_by_emission=False).source
    assert 'color="#' not in plain


def test_node_names_are_injective():
    from sofic.viz._names import node_names

    states = ["q_1", "q-1", 1, "1", "s_1", ("q", 1)]
    names = node_names(states)
    assert len(set(names.values())) == len(states)
    assert {names["q_1"], names["q-1"]} == {"q_1", "q_1_2"}
    assert node_names(["A", "B"]) == {"A": "A", "B": "B"}


def test_colliding_states_render_as_distinct_nodes():
    from sofic.shifts.sofic import SoficShift

    shift = SoficShift(symbol_alphabet=frozenset({"a"}))
    shift.add_transition("q_1", "q-1", "a")
    shift.add_transition("q-1", "q_1", "a")
    assert model_to_graphviz(shift).source.count("label=") >= 4
    tikz = model_to_tikz(shift)
    assert tikz.count(r"\node [state") == 2
    assert len({line.split("(")[1].split(")")[0] for line in tikz.splitlines() if r"\node [state" in line}) == 2


def _spec_parts(model, **data):
    from sofic.viz._edge import edge_spec

    model.graph.add_state(0)
    key = model.graph.add_transition(0, 0, **data)
    transition = next(t for t in model.transitions() if t.key == key)
    spec = edge_spec(model, transition)
    return spec.style, [(part.kind, part.value) for part in spec.parts]


def test_edge_specs_for_each_model_family():
    from sofic.automata.transducers import MooreMachine
    from sofic.base import StateMachine
    from sofic.generators.base import QuasiStochasticModel
    from sofic.generators.mealy import MealyHMM
    from sofic.generators.moore import MooreHMM
    from sofic.generators.nmachine import NMachine
    from sofic.graph import ATTR_EMISSION, ATTR_QUASIPROB, ATTR_SYMBOL

    class Bare(StateMachine):
        def validate(self) -> None:
            pass

    assert _spec_parts(MooreMachine(), **{ATTR_SYMBOL: "a"}) == ("symbol_only", [("symbol", "a")])
    assert _spec_parts(MooreMachine()) == ("symbol_only", [])
    assert _spec_parts(NMachine(), **{ATTR_EMISSION: "x", ATTR_QUASIPROB: -0.5}) == (
        "edge",
        [("emission", "x"), ("quasiprob", -0.5)],
    )
    assert _spec_parts(MooreHMM(), **{ATTR_PROB: 0.25}) == ("prob_only", [("prob", 0.25)])
    assert _spec_parts(MooreHMM()) == ("prob_only", [])
    assert _spec_parts(MealyHMM(), **{ATTR_SYMBOL: "s", ATTR_PROB: 1.0}) == ("edge", [("symbol", "s"), ("prob", 1.0)])
    assert _spec_parts(MealyHMM()) == ("edge", [])
    assert _spec_parts(QuasiStochasticModel(), **{ATTR_EMISSION: "x", ATTR_QUASIPROB: 2.0}) == (
        "edge",
        [("emission", "x"), ("quasiprob", 2.0)],
    )
    assert _spec_parts(QuasiStochasticModel()) == ("edge", [])
    full = {ATTR_SYMBOL: "a", ATTR_EMISSION: "x", ATTR_PROB: 0.5, ATTR_QUASIPROB: 0.5}
    assert _spec_parts(Bare(), **full) == (
        "fallback",
        [("symbol", "a"), ("emission", "x"), ("prob", 0.5), ("quasiprob", 0.5)],
    )
    assert _spec_parts(Bare()) == ("fallback", [])


def test_prob_part_prefers_probability_then_quasiprobability():
    from sofic.viz._edge import _primary_symbol_part, _prob_part

    assert _prob_part(0.5, 0.1).kind == "prob"
    assert _prob_part(None, 0.1).kind == "quasiprob"
    assert _prob_part(None, None) is None
    assert _primary_symbol_part(None, "a", symbol_fallback=False) is None


def test_idiagram_legend_layout_helpers():
    from sofic.viz import idiagram as idg

    assert idg._legend_ncols(5.0, 0) == 1
    assert idg._legend_ncols(100.0, 3) == 3
    assert idg._legend_ncols(1.0, 3) == 1
    handles = {group: group for group in idg.GROUP_ORDER}
    packed, ncol = idg._pack_legend_handles(handles, 4, empty_patch=None)
    assert ncol == 4
    height = len(packed) // ncol
    assert packed[:height] == ["r_mu"] + [None] * (height - 1)
    assert set(idg.GROUP_ORDER) <= set(packed)
    assert idg._pack_legend_handles({"structure": "s"}, 4, empty_patch=None) == (["s"], 1)
    assert idg._pack_legend_handles({}, 4, empty_patch=None) == ([], 1)
