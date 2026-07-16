"""Sofic shift presentations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sofic.shifts.base import SymbolicModel

if TYPE_CHECKING:
    from sofic.generators.mealy import MealyHMM


class SoficShift(SymbolicModel):
    """Sofic subshift given by a labeled directed graph presentation."""

    def topological_entropy(self) -> float:
        from sofic.shifts.tmc_construction import topological_entropy

        return topological_entropy(self)

    def parry_measure(self) -> MealyHMM:
        """Measure of maximal entropy (Parry measure) as a labeled Mealy HMM.

        The unique measure attaining ``h_top`` (Parry 1964): built from the Perron
        data of a right-resolving presentation of this shift, with each edge's
        emitted symbol preserved. See
        :func:`~sofic.shifts.topological_anatomy.parry_measure_sofic`.
        """
        from sofic.shifts.topological_anatomy import parry_measure_sofic

        return parry_measure_sofic(self)

    def topological_anatomy(self) -> dict[str, float]:
        """Topological information anatomy: the MME split ``h_top = b_top + r_top``.

        Evaluates the metric information anatomy (James et al. 2013) of the
        observed symbol process at the measure of maximal entropy, returning
        ``{h_top, b_top, r_top, excess_entropy}``. Here ``r_top`` is the MME
        erasure entropy rate (Verdu & Weissman 2008). See
        :func:`~sofic.shifts.topological_anatomy.topological_anatomy`.
        """
        from sofic.shifts.topological_anatomy import topological_anatomy

        return topological_anatomy(self)

    def markov_order(self) -> int | float:
        """Markov order ``R`` when the presentation is right-resolving (unifilar)."""
        from sofic.generators.synchronization import graph_from_sofic_shift, markov_order_from_graph

        return markov_order_from_graph(graph_from_sofic_shift(self))

    def cryptic_order(self) -> int | float:
        """Cryptic order ``k_chi`` when the presentation is right-resolving."""
        from sofic.generators.synchronization import cryptic_order_from_graph, graph_from_sofic_shift

        return cryptic_order_from_graph(graph_from_sofic_shift(self))

    def reset_threshold(self) -> int | float:
        """Reset threshold: shortest synchronizing (reset) word length.

        The complement of :meth:`markov_order`; ``math.inf`` when not exactly
        synchronizable. For the Cerny bound (complete automata only) see Volkov
        (2008).
        """
        from sofic.generators.synchronization import graph_from_sofic_shift, reset_threshold_from_graph

        return reset_threshold_from_graph(graph_from_sofic_shift(self))

    def synchronizing_word(self) -> list[Any] | None:
        """Return a shortest synchronizing word, or ``None`` if not exactly synchronizable.

        The list of symbols has length :meth:`reset_threshold` (Travers &
        Crutchfield, arXiv:1008.4182).
        """
        from sofic.generators.synchronization import graph_from_sofic_shift, shortest_synchronizing_word_from_graph

        return shortest_synchronizing_word_from_graph(graph_from_sofic_shift(self))

    def is_exactly_synchronizable(self) -> bool:
        """Return whether this presentation is exactly synchronizable.

        True iff a finite synchronizing word exists (the reset threshold is
        finite; Travers & Crutchfield, arXiv:1008.4182). Strictly weaker than
        finite Markov order: see :meth:`is_definite`.
        """
        from sofic.generators.synchronization import graph_from_sofic_shift, is_exactly_synchronizable

        return is_exactly_synchronizable(graph_from_sofic_shift(self))

    def is_asymptotically_synchronizable(self) -> bool:
        """Return whether this presentation is asymptotically synchronizable.

        True for every finite-state ε-machine (Travers & Crutchfield,
        arXiv:1008.4182); theorem-backed rather than computed.
        """
        from sofic.generators.synchronization import (
            graph_from_sofic_shift,
            is_asymptotically_synchronizable_from_graph,
        )

        return is_asymptotically_synchronizable_from_graph(graph_from_sofic_shift(self))

    def is_definite(self) -> bool:
        """Return whether this presentation is a definite automaton (finite Markov order).

        Definiteness (Perles-Rabin-Shamir) means the state is fixed by the last
        ``R`` symbols. Implies :meth:`is_exactly_synchronizable`.
        """
        from sofic.generators.synchronization import graph_from_sofic_shift, is_definite_from_graph

        return is_definite_from_graph(graph_from_sofic_shift(self))
