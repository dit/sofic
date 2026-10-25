"""Green's relations of a finite transformation semigroup.

Two elements are R-related when they generate the same right ideal
(``sS¹ = tS¹``), L-related for left ideals, J-related for two-sided ideals,
and H = R ∩ L. In a finite semigroup D = J, every D-class is a rectangular
"egg box" of H-classes, and the H-class of an idempotent is a maximal subgroup
:cite:`Green1951` :cite:`Pin1986`. Here R- and L-classes are the strongly
connected components of the right and left Cayley graphs and J-classes those
of their union, which is exactly the "reachable by multiplying on that side"
characterization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx

from sofic.algebra._groups import permutation_group

if TYPE_CHECKING:
    from sofic.algebra.transformation import TransformationSemigroup

__all__ = ["DClass", "GreenStructure", "green_structure"]


@dataclass(frozen=True)
class DClass:
    """One D-class laid out as an egg box.

    ``cells[r][c]`` is the H-class at the intersection of the ``r``-th R-class
    and the ``c``-th L-class, as a tuple of element indices.
    """

    r_classes: tuple[tuple[int, ...], ...]
    l_classes: tuple[tuple[int, ...], ...]
    cells: tuple[tuple[tuple[int, ...], ...], ...]
    idempotents: frozenset[int]

    @property
    def is_regular(self) -> bool:
        return bool(self.idempotents)

    @property
    def size(self) -> int:
        return sum(len(r) for r in self.r_classes)

    @property
    def h_class_size(self) -> int:
        return len(self.cells[0][0])

    def elements(self) -> tuple[int, ...]:
        return tuple(sorted(i for r in self.r_classes for i in r))


class GreenStructure:
    """Green's R, L, H and D (= J) classes of a :class:`TransformationSemigroup`.

    ``j_order`` is the condensation DAG of the two-sided Cayley graph on
    D-class ids, with an edge from a class to each class strictly below it in
    one multiplication step.
    """

    def __init__(self, semigroup: TransformationSemigroup) -> None:
        self.semigroup = semigroup
        right = semigroup.right_cayley()
        left = semigroup.left_cayley()
        n = len(right)
        self.r_of, self.r_classes = _scc_labels(n, right)
        self.l_of, self.l_classes = _scc_labels(n, left)
        both = [r + l for r, l in zip(right, left, strict=True)]
        self.d_of, d_members = _scc_labels(n, both)

        h_map: dict[tuple[int, int], list[int]] = {}
        for i in range(n):
            h_map.setdefault((self.r_of[i], self.l_of[i]), []).append(i)
        self.h_classes = [tuple(v) for v in h_map.values()]
        self.h_of = [0] * n
        for k, members in enumerate(self.h_classes):
            for i in members:
                self.h_of[i] = k

        idempotents = set(semigroup.idempotents())
        self.d_classes: list[DClass] = []
        for members in d_members:
            r_ids = list(dict.fromkeys(self.r_of[i] for i in members))
            l_ids = list(dict.fromkeys(self.l_of[i] for i in members))
            cells = tuple(tuple(tuple(h_map[(r, c)]) for c in l_ids) for r in r_ids)
            self.d_classes.append(
                DClass(
                    r_classes=tuple(self.r_classes[r] for r in r_ids),
                    l_classes=tuple(self.l_classes[c] for c in l_ids),
                    cells=cells,
                    idempotents=frozenset(i for i in members if i in idempotents),
                )
            )

        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(d_members)))
        for i, row in enumerate(both):
            for j in row:
                if self.d_of[i] != self.d_of[j]:
                    graph.add_edge(self.d_of[i], self.d_of[j])
        self.j_order = graph

    def is_aperiodic(self) -> bool:
        """Return whether every H-class is trivial."""
        return all(len(h) == 1 for h in self.h_classes)

    def maximal_subgroup(self, idempotent: int) -> Any:
        """Return the maximal subgroup ``H_e`` as a sympy group on ``im(e)``.

        Each element of ``H_e`` restricts to a permutation of the image of
        ``e``, and the restriction is faithful, so the group is returned as a
        permutation group on ``len(im(e))`` points (the image in sorted order).
        """
        e = self.semigroup[idempotent]
        if not e.is_idempotent():
            raise ValueError(f"element {idempotent} is not idempotent")
        image = sorted(e.image())
        members = self.h_classes[self.h_of[idempotent]]
        return permutation_group((self.semigroup[i].restrict(image) for i in members), len(image))

    def maximal_subgroups(self) -> list[tuple[int, Any]]:
        """One ``(idempotent, group)`` pair per regular D-class.

        All maximal subgroups in a D-class are isomorphic, so one
        representative idempotent per regular D-class suffices.
        """
        return [(min(d.idempotents), self.maximal_subgroup(min(d.idempotents))) for d in self.d_classes if d.is_regular]

    def minimal_ideal(self) -> DClass:
        """Return the minimal ideal ``K(S)``, the unique J-class below all others.

        Every finite semigroup has one; it is a completely simple semigroup
        and, for a transformation semigroup, consists of the elements of
        minimal rank :cite:`Howie1995`.
        """
        (bottom,) = [d for d in self.j_order.nodes if self.j_order.out_degree(d) == 0]
        return self.d_classes[bottom]

    def minimal_rank(self) -> int:
        """Rank of the elements of the minimal ideal (``1`` iff ``S`` contains a constant map)."""
        return self.semigroup[self.minimal_ideal().elements()[0]].rank()

    def schutzenberger_group(self, element: int) -> Any:
        """Return the Schützenberger group of an element's H-class, regular or not.

        The right Schützenberger group of ``H = H_x`` is induced by
        ``{u ∈ S¹ : x u ∈ H}``; such ``u`` permute ``im(x)``, and ``xu = xu'``
        iff they agree there, so the group is faithful on ``im(x)`` with order
        ``|H|`` :cite:`Howie1995`. It is returned as a sympy group on the
        sorted image. For an idempotent it is the maximal subgroup
        (:meth:`maximal_subgroup`). Compare the *permutator group* of
        ``im(x)``, induced by its whole setwise stabilizer, which can be larger
        :cite:`Linton1998`.
        """
        sg = self.semigroup
        x = sg[element]
        image = sorted(x.image())
        target = self.h_of[element]
        perms = [tuple(range(len(image)))]
        for u in sg:
            product = x * u
            if product in sg and self.h_of[sg.index(product)] == target:
                perms.append(u.restrict(image))
        return permutation_group(perms, len(image))

    def egg_box(self) -> str:
        """Render every D-class as a grid of H-classes labeled by shortest words.

        Idempotents are starred; the empty word is written ``1``.
        """
        sg = self.semigroup
        idempotents = {i for d in self.d_classes for i in d.idempotents}

        def label(i: int) -> str:
            word = sg.word(i)
            text = "".join(map(str, word)) if word else "1"
            return f"*{text}" if i in idempotents else text

        blocks = []
        for k, d in enumerate(self.d_classes):
            rows = [" | ".join(",".join(label(i) for i in cell) for cell in row) for row in d.cells]
            header = f"D{k} (size {d.size}, {'regular' if d.is_regular else 'non-regular'})"
            blocks.append("\n".join([header, *rows]))
        return "\n\n".join(blocks)


def green_structure(semigroup: TransformationSemigroup) -> GreenStructure:
    """Return Green's relations of ``semigroup``; enumerates its elements."""
    return GreenStructure(semigroup)


def _scc_labels(n: int, adjacency: list[list[int]]) -> tuple[list[int], list[tuple[int, ...]]]:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for i, row in enumerate(adjacency):
        graph.add_edges_from((i, j) for j in row)
    labels = [0] * n
    components = sorted((sorted(int(i) for i in c) for c in nx.strongly_connected_components(graph)), key=min)
    classes = [tuple(component) for component in components]
    for k, component in enumerate(components):
        for i in component:
            labels[i] = k
    return labels, classes
