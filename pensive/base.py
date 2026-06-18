"""Abstract base for all pensive state-machine models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator
from copy import deepcopy
from typing import Any, Self

import networkx as nx

from pensive.exceptions import PensiveValidationError
from pensive.graph import Transition, TransitionGraph
from pensive.indexing import StateIndex


class StateMachine(ABC):
    """Common interface for graph-backed models in pensive."""

    graph: TransitionGraph

    def __init__(self, graph: TransitionGraph | None = None) -> None:
        self.graph = graph if graph is not None else TransitionGraph()

    @abstractmethod
    def validate(self) -> None:
        """Raise :class:`~pensive.exceptions.PensiveValidationError` on failure."""

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

    def reverse(self) -> Self:
        """Return a model with the transition graph transposed."""
        result = self.copy()
        result.graph = self.graph.reverse()
        return result

    def __repr__(self) -> str:
        states = list(self.states())
        transitions = list(self.transitions())
        return f"{self.__class__.__name__}({len(states)} states, {len(transitions)} transitions)"

    def _repr_png_(self) -> bytes | None:
        """Notebook image: Graphviz PNG, or ``None`` if rendering is unavailable."""
        try:
            from pensive.viz.graphviz import model_to_png

            return model_to_png(self)
        except Exception:
            return None

    def _graphviz_svg(self) -> str | None:
        """Graphviz SVG for notebook vector display."""
        try:
            from pensive.viz.graphviz import model_to_svg

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
        from pensive.viz.graphviz import model_to_graphviz

        return model_to_graphviz(self, **kwargs)

    def draw(self, filename: str | None = None, **kwargs: Any) -> str | None:
        """Render this model with Graphviz."""
        from pensive.viz.graphviz import draw

        return draw(self, filename=filename, **kwargs)

    def to_tikz(self, **kwargs: Any) -> str:
        """Return a Vaucanson-style TikZ picture for this model."""
        from pensive.viz.tikz import model_to_tikz

        return model_to_tikz(self, **kwargs)

    def draw_tikz(self, filename: str | None = None, **kwargs: Any) -> str | None:
        """Write a TikZ fragment for this model."""
        from pensive.viz.tikz import draw_tikz

        return draw_tikz(self, filename=filename, **kwargs)

    def _require(self, condition: bool, message: str) -> None:
        if not condition:
            raise PensiveValidationError(message)
