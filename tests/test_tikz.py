"""Tests for TikZ / Vaucanson visualization."""

from __future__ import annotations

import pytest

from pensive.automata.dfa import DFA
from pensive.examples.epsilon_machines import golden_mean_bidirectional, golden_mean_forward
from pensive.viz.tikz import compile_tikz, draw_tikz, model_to_tikz


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


def test_golden_mean_forward_tikz_vaucanson_style():
    tikz = model_to_tikz(golden_mean_forward(0.5))
    assert "style=vaucanson" in tikz
    assert r"\Edge{" in tikz
    assert r"\half" in tikz or r"\nicefrac{1}{2}" in tikz
    assert "(A)" in tikz
    assert "(B)" in tikz


def test_dfa_tikz_symbol_only():
    tikz = model_to_tikz(_dfa())
    assert r"\Symbol{" in tikz
    assert r"\Edge{" not in tikz


def test_bidirectional_tikz_uses_edge_labels():
    tikz = model_to_tikz(golden_mean_bidirectional(0.5), style="paper")
    assert r"\Edge{" in tikz
    assert r"\TEdge{" not in tikz
    assert "(A{,} C)" in tikz or "(A, C)" in tikz
    assert r"font=\footnotesize" in tikz
    assert "tikzpicture" in tikz


def test_loop_avoids_outgoing_corridor():
    from pensive.viz._tikz_layout import plan_loop_styles

    positions = {"A": (0.0, 2.0), "B": (0.0, -2.0)}
    grouped = {
        ("A", "B"): [object()],
        ("B", "A"): [object()],
        ("B", "B"): [object()],
    }
    styles = plan_loop_styles(positions, grouped)
    assert styles[("B", "B", 0)] != "loop above"


def test_msp_self_loop_avoids_reciprocal_edge():
    from pensive.examples.epsilon_machines import golden_mean
    from pensive.viz.tikz import model_to_tikz

    tikz = model_to_tikz(golden_mean(0.5).mixed_state_presentation(), style="paper")
    a_loop_lines = [line for line in tikz.splitlines() if "s_1_0)" in line and "loop" in line]
    assert len(a_loop_lines) == 1
    assert "loop above" not in a_loop_lines[0]


def test_reciprocal_edges_bend_same_direction():
    from pensive.examples.epsilon_machines import golden_mean_forward
    from pensive.viz._tikz_layout import edge_style

    assert edge_style("A", "B", parallel_index=0, total_parallel=1, has_reverse=True) == "bend left"
    assert edge_style("B", "A", parallel_index=0, total_parallel=1, has_reverse=True) == "bend left"
    assert edge_style("A", "B", parallel_index=0, total_parallel=1, has_reverse=False) == ""

    tikz = model_to_tikz(golden_mean_forward(0.5), style="paper")
    ab = [line for line in tikz.splitlines() if "(A)" in line and "(B)" in line and "edge" in line]
    assert len(ab) == 2
    assert all("bend left" in line for line in ab)


def test_graphviz_layout_spreads_nodes():
    graphviz = pytest.importorskip("graphviz")
    del graphviz
    from pensive.viz._tikz_layout import layout_graphviz

    coords = layout_graphviz(golden_mean_bidirectional(0.5), style="paper")
    positions = []
    for placement in coords.values():
        assert placement.startswith("at (")
        inner = placement.removeprefix("at (").removesuffix("cm)")
        x_str, y_str = inner.split("cm, ")
        positions.append((float(x_str), float(y_str)))
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    assert max(xs) - min(xs) > 1.0
    assert max(ys) - min(ys) > 1.0


def test_mixed_state_presentation_tikz_compiles():
    from pensive.examples.epsilon_machines import golden_mean
    from pensive.viz._tikz_compile import find_executable

    if find_executable("pdflatex") is None:
        pytest.skip("pdflatex not available")
    msp = golden_mean(0.5).mixed_state_presentation()
    tikz = model_to_tikz(msp, style="paper")
    assert r"\mu=" not in tikz
    assert r"\scriptstyle" in tikz
    assert "μ" not in tikz
    assert "initial" not in tikz
    assert "line width=5pt" in tikz
    png = compile_tikz(tikz, format="png")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_bidirectional_tikz_png():
    from pensive.viz._tikz_compile import find_executable
    from pensive.viz.tikz import model_to_tikz_image

    if find_executable("pdflatex") is None:
        pytest.skip("pdflatex not available")
    png = model_to_tikz_image(golden_mean_bidirectional(0.5))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_circle_layout_polar_coordinates():
    tikz = model_to_tikz(golden_mean_forward(0.5), layout="circle", radius="2cm")
    assert "at (90:2cm)" in tikz or "at (90:2cm)" in tikz.replace(" ", "")


@pytest.mark.parametrize("layout", ["circle", "graphviz"])
def test_layout_modes(layout: str):
    graphviz = pytest.importorskip("graphviz")
    del graphviz
    tikz = model_to_tikz(golden_mean_forward(0.5), layout=layout)
    assert r"\begin{tikzpicture}" in tikz
    assert r"\path" in tikz


def test_draw_tikz_writes_file(tmp_path):
    path = tmp_path / "machine.tikz"
    result = draw_tikz(golden_mean_forward(0.5), filename=str(path))
    assert result == str(path)
    content = path.read_text(encoding="utf-8")
    assert r"\Edge{" in content


def test_to_tikz_method():
    tikz = golden_mean_forward(0.5).to_tikz()
    assert "style=vaucanson" in tikz


def test_standalone_document_includes_preamble():
    doc = model_to_tikz(golden_mean_forward(0.5), fragment=False)
    assert r"\documentclass" in doc
    assert "vaucanson.tikz" in doc


def test_epsilon_machine_tikz_fillcolor():
    from pensive.examples.epsilon_machines import golden_mean_forward

    tikz = model_to_tikz(golden_mean_forward(0.5))
    assert "fill=honeydew" in tikz


def test_compile_tikz_fragment_to_png():
    import shutil

    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex not available")
    fragment = model_to_tikz(golden_mean_forward(0.5))
    png = compile_tikz(fragment, format="png")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_draw_tikz_compiles_png(tmp_path):
    import shutil

    if shutil.which("pdflatex") is None:
        pytest.skip("pdflatex not available")
    path = tmp_path / "machine.png"
    result = draw_tikz(golden_mean_forward(0.5), filename=str(path))
    assert result == str(path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
