"""Abstract base for all sofic state-machine models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any, Self

import networkx as nx

from sofic.exceptions import SoficValidationError
from sofic.graph import Transition, TransitionGraph
from sofic.indexing import StateIndex


class StateMachine(ABC):
    """Common interface for graph-backed models in sofic."""

    graph: TransitionGraph

    def __init__(self, graph: TransitionGraph | None = None) -> None:
        self.graph = graph if graph is not None else TransitionGraph()

    @abstractmethod
    def validate(self) -> None:
        """Raise :class:`~sofic.exceptions.SoficValidationError` on failure."""

    def states(self) -> Iterator[Hashable]:
        yield from self.graph.states()

    def transitions(self) -> Iterator[Transition]:
        yield from self.graph.transitions()

    def reindex(self) -> StateIndex:
        return StateIndex(self.states())

    def copy(self) -> Self:
        cloned = object.__new__(self.__class__)
        cloned.graph = self.graph.copy()
        for name, value in self.__dict__.items():
            if name == "graph":
                continue
            cloned.__dict__[name] = deepcopy(value)
        return cloned

    def to_networkx(self) -> nx.MultiDiGraph:
        return self.graph.nx.copy()

    @classmethod
    def from_networkx(cls, g: nx.MultiDiGraph, **kwargs: Any) -> Self:
        return cls(graph=TransitionGraph(g.copy()), **kwargs)

    def to_yaml(self) -> str:
        """Return a YAML representation of this model."""
        from sofic.serialization import model_to_yaml

        return model_to_yaml(self)

    def write_yaml(self, path: str | Path) -> None:
        """Write this model as YAML."""
        Path(path).write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_yaml(cls, text: str, *, validate: bool = True) -> Self:
        """Reconstruct a model from YAML text."""
        from sofic.serialization import model_from_yaml

        model = model_from_yaml(text, validate=validate)
        if not isinstance(model, cls):
            raise TypeError(f"YAML contains {type(model).__qualname__}, not a {cls.__qualname__}")
        return model

    @classmethod
    def read_yaml(cls, path: str | Path, *, validate: bool = True) -> Self:
        """Read a model from a YAML file."""
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"), validate=validate)

    def reverse(self) -> Self:
        """Return a model with the transition graph transposed."""
        result = self.copy()
        result.graph = self.graph.reverse()
        return result

    def is_wheeler(self) -> bool:
        """Return whether this presentation admits a Wheeler order.

        Wheelerness is a property of a *presentation*, not of the language it
        generates: a process whose minimal presentation is not Wheeler may
        still have a larger one that is. Nor does it coincide with finite
        Markov order -- every definite presentation has a Wheeler order-``R``
        de Bruijn form, yet Wheeler presentations of infinite Markov order also
        exist :cite:`Gagie2017` :cite:`Alanko2020`.
        """
        from sofic.automata.wheeler import is_wheeler

        return is_wheeler(self)

    def wheeler_order(self) -> Any:
        """Return a :class:`~sofic.automata.wheeler.WheelerOrder`, or ``None``.

        Sorts states by the co-lexicographic rank of the words reaching them,
        so each state owns an interval of the sorted prefixes
        :cite:`Gagie2017`.
        """
        from sofic.automata.wheeler import wheeler_order

        return wheeler_order(self)

    def colex_width(self) -> int:
        """Co-lexicographic width of this presentation; Wheeler is width one.

        Bounds the cost of indexing, encoding, and determinizing the machine
        :cite:`CotumaccioPrezza2021`. Width one implies Wheelerness only for
        input-consistent presentations -- prefer :meth:`is_wheeler`.
        """
        from sofic.automata.wheeler import colex_width

        return colex_width(self)

    def __repr__(self) -> str:
        states = list(self.states())
        transitions = list(self.transitions())
        return f"{self.__class__.__name__}({len(states)} states, {len(transitions)} transitions)"

    def _repr_png_(self) -> bytes | None:
        """Notebook image: Graphviz PNG, or ``None`` if rendering is unavailable."""
        try:
            from sofic.viz.graphviz import model_to_png

            return model_to_png(self)
        except Exception:
            return None

    def _graphviz_svg(self) -> str | None:
        """Graphviz SVG for notebook vector display."""
        try:
            from sofic.viz.graphviz import model_to_svg

            return model_to_svg(self)
        except Exception:
            return None

    def _repr_mimebundle_(
        self,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> dict[str, Any] | None:
        bundle: dict[str, Any] = {"text/plain": repr(self)}

        svg = self._graphviz_svg()
        if svg is not None:
            bundle["image/svg+xml"] = svg

        if include is not None:
            bundle = {key: value for key, value in bundle.items() if key in include}
        if exclude is not None:
            bundle = {key: value for key, value in bundle.items() if key not in exclude}

        return bundle if len(bundle) > 1 else None

    def to_graphviz(self, **kwargs: Any) -> Any:
        """Build a Graphviz diagram for this model."""
        from sofic.viz.graphviz import model_to_graphviz

        return model_to_graphviz(self, **kwargs)

    def draw(self, filename: str | None = None, **kwargs: Any) -> str | None:
        """Render this model with Graphviz."""
        from sofic.viz.graphviz import draw

        return draw(self, filename=filename, **kwargs)

    def to_tikz(self, **kwargs: Any) -> str:
        """Return a Vaucanson-style TikZ picture for this model."""
        from sofic.viz.tikz import model_to_tikz

        return model_to_tikz(self, **kwargs)

    def draw_tikz(self, filename: str | None = None, **kwargs: Any) -> str | None:
        """Write a TikZ fragment for this model."""
        from sofic.viz.tikz import draw_tikz

        return draw_tikz(self, filename=filename, **kwargs)

    def _require(self, condition: bool, message: str) -> None:
        if not condition:
            raise SoficValidationError(message)
