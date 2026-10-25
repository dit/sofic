"""Cascade products with explicit dependency functions.

A cascade over components ``(X_1, S_1), ..., (X_n, S_n)`` acts on tuples
``(x_1, ..., x_n)`` coordinatewise: an input applies ``d_i(x_1, ..., x_{i-1})
∈ S_i`` to ``x_i``, so information only flows down the hierarchy
:cite:`EgriNagyNehaniv2015`. It is the finite, explicitly tabulated part of
the iterated wreath product that a particular decomposition actually uses.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from sofic.exceptions import SemigroupTooLargeError

if TYPE_CHECKING:
    from sofic.algebra.holonomy import HolonomyDecomposition
    from sofic.algebra.transformation import TransformationSemigroup
    from sofic.automata.dfa import DFA

__all__ = ["CascadeProduct", "LevelAction", "build_holonomy_cascade"]

Coordinates = tuple[int | None, ...]


@dataclass(frozen=True)
class LevelAction:
    """A component transformation: a reset to ``value`` or a permutation.

    ``mapping`` lists the moved states of a permutation; every other state is
    fixed. A reset to ``None`` clears a coordinate the new chain does not use.
    """

    kind: Literal["constant", "permutation"]
    value: int | None = None
    mapping: tuple[tuple[int, int], ...] = ()

    @classmethod
    def constant(cls, value: int | None) -> LevelAction:
        return cls("constant", value=value)

    @classmethod
    def permutation(cls, mapping: Mapping[int, int]) -> LevelAction:
        return cls("permutation", mapping=tuple(sorted((a, b) for a, b in mapping.items() if a != b)))

    def apply(self, x: int | None) -> int | None:
        if self.kind == "constant":
            return self.value
        return dict(self.mapping).get(x, x) if x is not None else None

    @property
    def is_identity(self) -> bool:
        return self.kind == "permutation" and not self.mapping


class CascadeProduct:
    """A finite cascade product given by dependency tables.

    Parameters
    ----------
    level_sizes : sequence of int
        Number of (non-``None``) states of each component, top level first.
    dependencies : mapping
        ``dependencies[label][i]`` maps a prefix ``coords[:i]`` to the
        :class:`LevelAction` applied at level ``i``.
    decoding : mapping
        Valid coordinate tuple to the original state it emulates.
    encoding : mapping
        Original state to a chosen coordinate tuple (a lift).
    """

    def __init__(
        self,
        level_sizes: Sequence[int],
        dependencies: Mapping[Hashable, Sequence[Mapping[Coordinates, LevelAction]]],
        decoding: Mapping[Coordinates, Hashable],
        encoding: Mapping[Hashable, Coordinates],
    ) -> None:
        self.level_sizes = tuple(level_sizes)
        self.dependencies = {label: tuple(dict(t) for t in tables) for label, tables in dependencies.items()}
        self.decoding = dict(decoding)
        self.encoding = dict(encoding)

    @property
    def n_levels(self) -> int:
        return len(self.level_sizes)

    @property
    def states(self) -> tuple[Coordinates, ...]:
        return tuple(self.decoding)

    @property
    def labels(self) -> tuple[Hashable, ...]:
        return tuple(self.dependencies)

    def __repr__(self) -> str:
        return f"CascadeProduct(levels={list(self.level_sizes)}, states={len(self.decoding)}, inputs={len(self.dependencies)})"

    def act(self, coords: Sequence[int | None], label: Hashable) -> Coordinates:
        """Apply input ``label``: level ``i`` sees only the old ``coords[:i]``."""
        coords = tuple(coords)
        tables = self.dependencies[label]
        return tuple(tables[i][coords[:i]].apply(coords[i]) for i in range(self.n_levels))

    def run(self, word: Iterable[Hashable], start: Hashable) -> Coordinates:
        """Read ``word`` from ``start`` (an original state or a coordinate tuple)."""
        coords = cast("Coordinates", start) if start in self.decoding else self.encoding[start]
        for label in word:
            coords = self.act(coords, label)
        return coords

    def truncate(self, k: int) -> CascadeProduct:
        """Keep the top ``k`` levels: the coarse-grained "effective theory".

        Level ``i`` reads only ``coords[:i]``, so the first ``k`` levels are a
        cascade in their own right and the projection onto them commutes with
        every input :cite:`DeDeo2011`. In the result, ``decoding`` maps each
        prefix to the frozenset of original states it covers (a *superstate*;
        superstates may overlap) and ``encoding`` maps each state to the
        prefix of its lift.
        """
        if not 0 <= k <= self.n_levels:
            raise ValueError(f"k must be between 0 and {self.n_levels}")
        covers: dict[Coordinates, set[Hashable]] = {}
        for coords, state in self.decoding.items():
            covers.setdefault(coords[:k], set()).add(state)
        return CascadeProduct(
            self.level_sizes[:k],
            {label: tables[:k] for label, tables in self.dependencies.items()},
            {prefix: frozenset(states) for prefix, states in covers.items()},
            {state: coords[:k] for state, coords in self.encoding.items()},
        )

    def as_dfa(self, initial: Hashable | None = None) -> DFA:
        """Return the cascade as a complete DFA on its coordinate tuples."""
        from sofic.automata.dfa import DFA

        dfa = DFA(
            input_alphabet=frozenset(self.labels),
            initial_states=frozenset() if initial is None else frozenset({self.encoding[initial]}),
        )
        for coords in self.decoding:
            dfa.graph.add_state(coords)
        for coords in self.decoding:
            for label in self.labels:
                dfa.add_transition(coords, self.act(coords, label), label)
        return dfa

    def emulates(self, semigroup: TransformationSemigroup) -> bool:
        """Return whether the cascade covers ``semigroup``'s generators.

        Checks that the valid coordinate tuples are closed under every input,
        that decoding is onto the states, and that decoding commutes with the
        action: ``decode(act(c, a)) == decode(c) · a``.
        """
        if set(self.decoding.values()) != set(semigroup.states):
            return False
        gens = semigroup.generators
        if set(gens) != set(self.labels):
            return False
        position = {s: i for i, s in enumerate(semigroup.states)}
        for coords, state in self.decoding.items():
            for label, g in gens.items():
                try:
                    image = self.act(coords, label)
                except KeyError:
                    return False
                if image not in self.decoding:
                    return False
                if position[self.decoding[image]] != g(position[state]):
                    return False
        return True


def build_holonomy_cascade(decomposition: HolonomyDecomposition, *, max_states: int) -> CascadeProduct:
    """Tabulate the holonomy cascade over all maximal tile chains.

    Raises ``RuntimeError`` if a level's action is not determined by the
    coordinates above it, or does not reproduce the lifted action -- either
    would mean the lift is not consistent with chain structure.
    """
    sg = decomposition.semigroup
    gens = sg.generators
    n_levels = decomposition.n_levels
    dependencies: dict[Hashable, list[dict[Coordinates, LevelAction]]] = {
        label: [{} for _ in range(n_levels)] for label in gens
    }
    decoding: dict[Coordinates, Hashable] = {}
    for chain in decomposition.chains():
        if len(decoding) >= max_states:
            raise SemigroupTooLargeError(f"holonomy cascade exceeds max_states={max_states} coordinate tuples")
        coords = decomposition.encode(chain)
        (x,) = chain[-1]
        decoding[coords] = sg.states[x]
        for label, g in gens.items():
            actions = decomposition.level_actions(chain, g)
            expected = decomposition.encode(decomposition.step(chain, g))
            for i, action in enumerate(actions):
                table = dependencies[label][i]
                previous = table.setdefault(coords[:i], action)
                if previous != action:
                    raise RuntimeError(f"level {i} action on {label!r} depends on more than the levels above it")
                if action.apply(coords[i]) != expected[i]:
                    raise RuntimeError(f"level {i} action on {label!r} does not reproduce the lifted chain")
    encoding = {state: decomposition.coordinates(state) for state in sg.states}
    sizes = [len(level.states) for level in decomposition.levels]
    return CascadeProduct(sizes, dependencies, decoding, encoding)
