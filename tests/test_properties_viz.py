"""Property-based tests for Graphviz / TikZ output on adversarially labeled models."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Hashable
from typing import Any

import networkx as nx
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from sofic.base import StateMachine
from sofic.graph import TransitionGraph
from sofic.testing import dfas, markov_chains, mealy_hmms, mealy_transducers, nfas, sofic_shifts, vpas
from sofic.viz._names import node_names

# Each example spawns a Graphviz subprocess (~0.1 s), so the DOT test runs a quarter of the profile.
_FEW = settings(max_examples=max(10, settings.default.max_examples // 4))
STATE_FIELDS = frozenset({"initial_states", "accepting_states", "initial_distribution", "initial_state"})
NASTY = " \"'\\{}$%&#_^~<>,;=[]()|@*-+.:/éλ→\t\n"

labels = st.one_of(
    st.integers(-20, 20),
    st.text(alphabet=NASTY + "ab01", max_size=5),
    st.sampled_from(["node", "edge", "graph", "digraph", "subgraph", "strict", "__start__", "start", "1", "-1"]),
    st.tuples(st.integers(0, 2), st.text(alphabet="a,{}", max_size=2)),
    st.frozensets(st.integers(0, 2), max_size=2),
)
symbol_alphabets = st.sampled_from([("0", "1"), (0, "0"), ("a b", 'q"'), ("\\", "{}"), ("$x$", "%"), ("é", "→")])

models = st.one_of(
    symbol_alphabets.flatmap(lambda a: sofic_shifts(alphabet=a)),
    symbol_alphabets.flatmap(lambda a: dfas(alphabet=a)),
    symbol_alphabets.flatmap(lambda a: nfas(alphabet=a)),
    symbol_alphabets.flatmap(lambda a: mealy_hmms(alphabet=a)),
    markov_chains(),
    mealy_transducers(input_alphabet=("^", "_"), output_alphabet=("~", "#")),
    vpas(),
)


def relabel_states(model: StateMachine, mapping: dict[Hashable, Hashable]) -> StateMachine:
    result = model.copy()
    result.graph = TransitionGraph(nx.relabel_nodes(model.graph.nx, mapping, copy=True))
    for field in STATE_FIELDS & set(vars(result)):
        value = getattr(result, field)
        if isinstance(value, frozenset):
            setattr(result, field, frozenset(mapping[s] for s in value))
        elif isinstance(value, dict):
            setattr(result, field, {mapping[s]: p for s, p in value.items()})
        elif value is not None:
            setattr(result, field, mapping[value])
    return result


@st.composite
def adversarial_models(draw: Any) -> StateMachine:
    model = draw(models)
    states = list(model.states())
    new = draw(st.lists(labels, min_size=len(states), max_size=len(states), unique_by=repr))
    assume(len(set(new)) == len(new))
    return relabel_states(model, dict(zip(states, new, strict=True)))


# --------------------------------------------------------------------------- node ids


@given(st.lists(labels, max_size=8))
def test_node_names_are_unique_identifiers(states):
    names = node_names(states)
    assert set(names) == set(states)
    assert len(set(names.values())) == len(names)
    assert all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in names.values())
    assert "__start__" not in names.values()


# --------------------------------------------------------------------------- Graphviz


def _balanced_dot(source: str) -> None:
    depth, in_string, escaped = 0, False, False
    for char in source:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            assert depth >= 0
    assert not in_string and depth == 0


@_FEW
@given(adversarial_models())
def test_dot_output_parses_with_unique_node_ids(model):
    graphviz = pytest.importorskip("graphviz")
    source = model.to_graphviz().source
    _balanced_dot(source)
    if shutil.which("dot") is None:
        return
    parsed = json.loads(graphviz.Source(source).pipe(format="json"))
    nodes = parsed.get("objects", [])
    names = [node["name"] for node in nodes]
    start = "__start__" in names
    assert len(names) == len(set(names))
    assert len(names) == len(list(model.states())) + start
    starts = sum(1 for edge in parsed.get("edges", []) if nodes[edge["tail"]]["name"] == "__start__")
    assert len(parsed.get("edges", [])) - starts == len(list(model.transitions()))


# --------------------------------------------------------------------------- TikZ


def _strip_controls(text: str) -> str:
    """Drop every ``\\x`` control symbol so escaped braces and dollars don't count."""
    return re.sub(r"\\.", "", text, flags=re.DOTALL)


def _balanced_braces(text: str) -> bool:
    depth = 0
    for char in _strip_controls(text):
        depth += {"{": 1, "}": -1}.get(char, 0)
        if depth < 0:
            return False
    return depth == 0


def _remove_mbox_groups(math: str) -> str:
    out, i = [], 0
    while i < len(math):
        if math.startswith(r"\mbox{", i):
            depth, i = 1, i + len(r"\mbox{")
            while depth:
                if math[i] == "\\":
                    i += 2
                    continue
                depth += {"{": 1, "}": -1}.get(math[i], 0)
                i += 1
            continue
        out.append(math[i])
        i += 1
    return "".join(out)


def _math_segments(text: str) -> list[str]:
    positions = [m.start() for m in re.finditer(r"(?<!\\)\$", text)]
    assert len(positions) % 2 == 0, "unbalanced $"
    return [text[a + 1 : b] for a, b in zip(positions[::2], positions[1::2], strict=True)]


def test_tikz_labels_with_blank_lines_do_not_emit_par():
    from sofic.shifts import SoficShift

    shift = SoficShift(symbol_alphabet=frozenset({"x\n\ny"}))
    shift.add_transition("a\n\nb", "a\n\nb", "x\n\ny")
    tikz = shift.to_tikz()
    assert "\n\n" not in tikz.strip()
    assert _balanced_braces(tikz)


@given(adversarial_models())
def test_tikz_output_is_well_formed(model):
    tikz = model.to_tikz()
    assert _balanced_braces(tikz)
    body = tikz.split(r"\begin{tikzpicture}", 1)[1]
    assert "\n\n" not in body.strip(), "a blank line inside tikzpicture is a \\par"
    for math in _math_segments(tikz):
        bare = _remove_mbox_groups(math)
        for text_only in (r"\textbackslash", r"\textasciitilde", r"\^{}"):
            assert text_only not in bare
    declared = re.findall(r"\\node \[[^\]]*\] \(([A-Za-z0-9_]+)\)", tikz)
    assert len(declared) == len(set(declared)) == len(list(model.states()))
    endpoints = re.findall(r"\(([A-Za-z0-9_]+)\) edge", tikz) + re.findall(r"\(([A-Za-z0-9_]+)\)\s*(?:\n|;)", tikz)
    assert set(endpoints) <= set(declared)
