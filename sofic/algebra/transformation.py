"""Finite transformation semigroups.

A transformation semigroup ``(Q, S)`` is a finite state set ``Q`` together
with a semigroup ``S`` of total maps ``Q -> Q`` closed under composition. Maps
act on the right, ``x · (st) = (x · s) · t``, so a word read left to right is
the product of its letters in the same order :cite:`Eilenberg1976`. A
deterministic semiautomaton (a DFA without initial or accepting states) is the
same data with a chosen generating set: one transformation per input symbol.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import lcm
from typing import TYPE_CHECKING, Any, Literal

from sofic.exceptions import SemigroupTooLargeError

if TYPE_CHECKING:
    import networkx as nx

    from sofic.algebra.complexity import ComplexityBounds
    from sofic.algebra.green import GreenStructure
    from sofic.algebra.holonomy import HolonomyDecomposition

DEFAULT_MAX_SIZE = 100_000

__all__ = [
    "DEFAULT_MAX_SIZE",
    "SINK",
    "Transformation",
    "TransformationSemigroup",
]


class _Sink:
    """Absorbing state added when completing partial transformations."""

    _instance: _Sink | None = None

    def __new__(cls) -> _Sink:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "SINK"

    def __reduce__(self) -> tuple[Any, ...]:
        return (_Sink, ())


SINK = _Sink()


@dataclass(frozen=True, slots=True)
class Transformation:
    """A total map on ``{0, ..., n - 1}`` acting on the right.

    ``x · s = images[x]``, and ``s * t`` is "``s`` then ``t``", so
    ``x · (s * t) = (x · s) · t``.

    Examples
    --------
    >>> s = Transformation((1, 0, 2))
    >>> t = Transformation((2, 2, 2))
    >>> (s * t).images
    (2, 2, 2)
    >>> s.index_period()
    (1, 2)
    """

    images: tuple[int, ...]

    @classmethod
    def identity(cls, degree: int) -> Transformation:
        return cls(tuple(range(degree)))

    @classmethod
    def constant(cls, degree: int, value: int) -> Transformation:
        return cls((value,) * degree)

    @property
    def degree(self) -> int:
        return len(self.images)

    def __call__(self, x: int) -> int:
        return self.images[x]

    def __mul__(self, other: object) -> Transformation:
        if not isinstance(other, Transformation):
            return NotImplemented
        return Transformation(_compose(self.images, other.images))

    def act(self, subset: Iterable[int]) -> frozenset[int]:
        """Return the setwise image ``P · s``."""
        return frozenset(self.images[x] for x in subset)

    def image(self) -> frozenset[int]:
        return frozenset(self.images)

    def rank(self) -> int:
        return len(set(self.images))

    def kernel(self) -> tuple[frozenset[int], ...]:
        """Return the kernel partition: blocks of states with a common image."""
        blocks: dict[int, set[int]] = {}
        for x, y in enumerate(self.images):
            blocks.setdefault(y, set()).add(x)
        return tuple(sorted((frozenset(b) for b in blocks.values()), key=sorted))

    def is_permutation(self) -> bool:
        return self.rank() == self.degree

    def is_idempotent(self) -> bool:
        return all(self.images[y] == y for y in self.images)

    def is_constant(self) -> bool:
        return self.rank() <= 1

    def power(self, exponent: int) -> Transformation:
        """Return ``s**exponent`` for ``exponent >= 0`` (``s**0`` is the identity)."""
        if exponent < 0:
            raise ValueError("exponent must be nonnegative")
        result = tuple(range(self.degree))
        base = self.images
        while exponent:
            if exponent & 1:
                result = _compose(result, base)
            base = _compose(base, base)
            exponent >>= 1
        return Transformation(result)

    def __pow__(self, exponent: int) -> Transformation:
        return self.power(exponent)

    def index_period(self) -> tuple[int, int]:
        """Return ``(m, r)``, the least pair with ``s**(m + r) == s**m``.

        The element generates an aperiodic cyclic semigroup exactly when
        ``r == 1``.
        """
        seen: dict[tuple[int, ...], int] = {}
        current = self.images
        k = 1
        while current not in seen:
            seen[current] = k
            current = _compose(current, self.images)
            k += 1
        index = seen[current]
        return index, k - index

    def restrict(self, subset: Sequence[int]) -> tuple[int, ...]:
        """Return the action on the ordered ``subset`` as positions in it.

        Requires ``subset · s`` to be contained in ``subset``.
        """
        position = {x: i for i, x in enumerate(subset)}
        return tuple(position[self.images[x]] for x in subset)


def _compose(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    """Return ``left`` then ``right`` as raw image tuples."""
    return tuple(right[y] for y in left)


def _permutation_order(images: tuple[int, ...], support: Iterable[int]) -> int:
    """Order of ``images`` restricted to ``support`` (which it must permute)."""
    order = 1
    seen: set[int] = set()
    for start in support:
        if start in seen:
            continue
        length = 0
        x = start
        while x not in seen:
            seen.add(x)
            x = images[x]
            length += 1
        order = lcm(order, length)
    return order


class TransformationSemigroup:
    """The transformation semigroup generated by labeled maps of a finite set.

    Elements are enumerated lazily by breadth-first closure the first time
    they are needed, so every element carries a shortest generating word.
    Constructions that only need the generators -- notably the holonomy
    decomposition -- never trigger the enumeration.

    Parameters
    ----------
    generators : mapping
        Generator label (typically an input symbol) to a :class:`Transformation`
        or a sequence of images on ``range(degree)``.
    states : sequence, optional
        Labels for the points ``0, ..., degree - 1``. Defaults to the integers.
    monoid : bool
        Adjoin the identity (the empty word). This is the convention for the
        transition monoid of an automaton.
    max_size : int
        Raise :class:`~sofic.exceptions.SemigroupTooLargeError` if the closure
        exceeds this many elements.

    Examples
    --------
    >>> flip = TransformationSemigroup({"a": (1, 0)})
    >>> len(flip), flip.is_aperiodic()
    (2, False)
    """

    def __init__(
        self,
        generators: Mapping[Hashable, Transformation | Sequence[int]],
        *,
        states: Sequence[Hashable] | None = None,
        monoid: bool = True,
        max_size: int = DEFAULT_MAX_SIZE,
    ) -> None:
        gens = {
            label: g if isinstance(g, Transformation) else Transformation(tuple(int(x) for x in g))
            for label, g in generators.items()
        }
        if states is not None:
            degree = len(states)
        elif gens:
            degree = next(iter(gens.values())).degree
        else:
            raise ValueError("need at least one generator or an explicit state list")
        if degree == 0:
            raise ValueError("a transformation semigroup needs at least one state")
        for label, g in gens.items():
            if g.degree != degree or any(not 0 <= y < degree for y in g.images):
                raise ValueError(f"generator {label!r} is not a map on {degree} states")
        if not gens and not monoid:
            raise ValueError("a semigroup with no generators must be a monoid")
        self._generators = gens
        self._labels = tuple(gens)
        self._states = tuple(states) if states is not None else tuple(range(degree))
        if len(set(self._states)) != degree:
            raise ValueError("state labels must be distinct")
        self._degree = degree
        self._monoid = monoid
        self._max_size = max_size
        self._elements: list[tuple[int, ...]] | None = None
        self._words: list[tuple[Hashable, ...]] = []
        self._index: dict[tuple[int, ...], int] = {}
        self._right: list[list[int]] = []
        self._left: list[list[int]] | None = None

    # -- construction helpers -------------------------------------------------

    @classmethod
    def from_maps(
        cls,
        maps: Mapping[Hashable, Mapping[Hashable, Hashable]],
        *,
        states: Iterable[Hashable] | None = None,
        complete: Literal["sink", "error"] = "sink",
        monoid: bool = True,
        max_size: int = DEFAULT_MAX_SIZE,
    ) -> TransformationSemigroup:
        """Build from per-label state maps, completing partial maps.

        ``maps[label][state]`` is the successor of ``state`` on ``label``; a
        missing entry is undefined. With ``complete="sink"`` undefined moves go
        to an absorbing :data:`SINK` state, added only when needed; with
        ``complete="error"`` they raise ``ValueError``.
        """
        if states is None:
            seen: dict[Hashable, None] = {}
            for table in maps.values():
                for source, target in table.items():
                    seen.setdefault(source)
                    seen.setdefault(target)
            ordered = _sorted_states(seen)
        else:
            ordered = list(dict.fromkeys(states))
        position = {s: i for i, s in enumerate(ordered)}
        partial = any(s not in table for table in maps.values() for s in ordered)
        if partial:
            if complete == "error":
                raise ValueError("partial transformation; pass complete='sink' to add an absorbing state")
            if SINK not in position:
                position[SINK] = len(ordered)
                ordered.append(SINK)
        generators: dict[Hashable, tuple[int, ...]] = {}
        for label, table in maps.items():
            images: list[int] = []
            for state in ordered:
                target = table.get(state, SINK) if state is not SINK else SINK
                if target not in position:
                    raise ValueError(f"unknown target state {target!r} on {label!r}")
                images.append(position[target])
            generators[label] = tuple(images)
        return cls(generators, states=ordered, monoid=monoid, max_size=max_size)

    # -- basic data -----------------------------------------------------------

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def states(self) -> tuple[Hashable, ...]:
        return self._states

    @property
    def generators(self) -> dict[Hashable, Transformation]:
        return dict(self._generators)

    @property
    def generator_labels(self) -> tuple[Hashable, ...]:
        return self._labels

    @property
    def is_monoid(self) -> bool:
        return self._monoid

    def state_index(self, state: Hashable) -> int:
        return self._states.index(state)

    # -- enumeration ----------------------------------------------------------

    def _closure(self) -> list[tuple[int, ...]]:
        if self._elements is not None:
            return self._elements
        try:
            return self._enumerate()
        except SemigroupTooLargeError:
            self._index, self._words, self._right = {}, [], []
            raise

    def _enumerate(self) -> list[tuple[int, ...]]:
        elements: list[tuple[int, ...]] = []
        gens = [self._generators[label].images for label in self._labels]

        def add(element: tuple[int, ...], word: tuple[Hashable, ...]) -> int:
            index = self._index.get(element)
            if index is not None:
                return index
            if len(elements) >= self._max_size:
                raise SemigroupTooLargeError(
                    f"semigroup exceeds max_size={self._max_size} elements; raise the cap if intended"
                )
            self._index[element] = len(elements)
            elements.append(element)
            self._words.append(word)
            self._right.append([])
            return len(elements) - 1

        if self._monoid:
            add(tuple(range(self._degree)), ())
        for label, g in zip(self._labels, gens, strict=True):
            add(g, (label,))
        queue = deque(range(len(elements)))
        while queue:
            i = queue.popleft()
            row = []
            for label, g in zip(self._labels, gens, strict=True):
                before = len(elements)
                j = add(_compose(elements[i], g), (*self._words[i], label))
                if j >= before:
                    queue.append(j)
                row.append(j)
            self._right[i] = row
        self._elements = elements
        return elements

    def __len__(self) -> int:
        return len(self._closure())

    def __iter__(self) -> Iterator[Transformation]:
        return (Transformation(e) for e in self._closure())

    def __getitem__(self, index: int) -> Transformation:
        return Transformation(self._closure()[index])

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, Transformation):
            return False
        self._closure()
        return item.images in self._index

    def __repr__(self) -> str:
        size = len(self._elements) if self._elements is not None else "?"
        return (
            f"{type(self).__name__}(degree={self._degree}, generators={len(self._labels)}, "
            f"size={size}, monoid={self._monoid})"
        )

    def index(self, element: Transformation) -> int:
        self._closure()
        return self._index[element.images]

    def word(self, element: Transformation | int) -> tuple[Hashable, ...]:
        """Return a shortest generator word evaluating to ``element``."""
        index = element if isinstance(element, int) else self.index(element)
        self._closure()
        return self._words[index]

    def evaluate(self, word: Iterable[Hashable]) -> Transformation:
        """Return the transformation of a generator word (identity for ``()``)."""
        images = tuple(range(self._degree))
        for label in word:
            images = _compose(images, self._generators[label].images)
        return Transformation(images)

    def multiply(self, i: int, j: int) -> int:
        """Return the index of ``self[i] * self[j]``."""
        elements = self._closure()
        return self._index[_compose(elements[i], elements[j])]

    def right_cayley(self) -> list[list[int]]:
        """``right[i][g]`` is the index of ``self[i] * generator g``."""
        self._closure()
        return self._right

    def left_cayley(self) -> list[list[int]]:
        """``left[i][g]`` is the index of ``generator g * self[i]``."""
        elements = self._closure()
        if self._left is None:
            gens = [self._generators[label].images for label in self._labels]
            self._left = [[self._index[_compose(g, e)] for g in gens] for e in elements]
        return self._left

    def cayley_graph(self, side: Literal["right", "left"] = "right") -> nx.MultiDiGraph:
        """Return the Cayley graph with edges labeled by generator."""
        import networkx as nx

        table = self.right_cayley() if side == "right" else self.left_cayley()
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(range(len(table)))
        for i, row in enumerate(table):
            for label, j in zip(self._labels, row, strict=True):
                graph.add_edge(i, j, symbol=label)
        return graph

    # -- element-level properties ---------------------------------------------

    def idempotents(self) -> list[int]:
        return [i for i, e in enumerate(self._closure()) if all(e[y] == y for y in e)]

    def is_aperiodic(self) -> bool:
        """Return whether every cyclic subsemigroup has period one.

        Equivalently every subgroup is trivial, every H-class is a singleton
        :cite:`Pin1986`, and -- for a syntactic monoid -- the language is
        star-free :cite:`Schutzenberger1965`.
        """
        return all(Transformation(e).index_period()[1] == 1 for e in self._closure())

    def is_synchronizing(self) -> bool:
        """Return whether some word sends every state to one state (``S`` contains a constant).

        Equivalently the minimal ideal has rank one. A word reaches a constant
        iff every pair of states can be merged, so this is decided on pairs
        without enumerating ``S`` :cite:`Volkov2008`.
        """
        if self._degree == 1:
            return True
        gens = [g.images for g in self._generators.values()]
        pairs = {(x, y) for x in range(self._degree) for y in range(x + 1, self._degree)}
        mergeable: set[tuple[int, int]] = set()
        changed = True
        while changed:
            changed = False
            for pair in pairs - mergeable:
                x, y = pair
                for g in gens:
                    a, b = sorted((g[x], g[y]))
                    if a == b or (a, b) in mergeable:
                        mergeable.add(pair)
                        changed = True
                        break
        return mergeable == pairs

    def is_group(self) -> bool:
        """Return whether ``S`` is a permutation group (all generators are permutations)."""
        return all(g.is_permutation() for g in self._generators.values())

    # -- delegated constructions ----------------------------------------------

    def green(self) -> GreenStructure:
        """Return Green's relations; see :func:`~sofic.algebra.green.green_structure`."""
        from sofic.algebra.green import green_structure

        return green_structure(self)

    def holonomy(self, **kwargs: Any) -> HolonomyDecomposition:
        """Return the holonomy decomposition; see :func:`~sofic.algebra.holonomy.holonomy_decomposition`."""
        from sofic.algebra.holonomy import holonomy_decomposition

        return holonomy_decomposition(self, **kwargs)

    def krohn_rhodes_complexity(self) -> ComplexityBounds:
        """Return complexity bounds; see :func:`~sofic.algebra.complexity.krohn_rhodes_complexity`."""
        from sofic.algebra.complexity import krohn_rhodes_complexity

        return krohn_rhodes_complexity(self)


def _sorted_states(states: Iterable[Hashable]) -> list[Hashable]:
    items = list(states)
    try:
        return sorted(items)  # type: ignore[type-var]
    except TypeError:
        return sorted(items, key=repr)
