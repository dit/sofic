"""Sofic shift presentations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pensive.shifts.base import SymbolicModel

if TYPE_CHECKING:
    from pensive.generators.mealy import MealyHMM


class SoficShift(SymbolicModel):
    """Sofic subshift given by a labeled directed graph presentation."""

    def topological_entropy(self) -> float:
        from pensive.shifts.tmc_construction import topological_entropy

        return topological_entropy(self)

    def parry_measure(self) -> MealyHMM:
        """Measure of maximal entropy (Parry measure) as a labeled Mealy HMM.

        The unique measure attaining ``h_top`` (Parry 1964): built from the Perron
        data of a right-resolving presentation of this shift, with each edge's
        emitted symbol preserved. See
        :func:`~pensive.shifts.topological_anatomy.parry_measure_sofic`.
        """
        from pensive.shifts.topological_anatomy import parry_measure_sofic

        return parry_measure_sofic(self)

    def topological_anatomy(self) -> dict[str, float]:
        """Topological information anatomy: the MME split ``h_top = b_top + r_top``.

        Evaluates the metric information anatomy (James et al. 2013) of the
        observed symbol process at the measure of maximal entropy, returning
        ``{h_top, b_top, r_top, excess_entropy}``. Here ``r_top`` is the MME
        erasure entropy rate (Verdu & Weissman 2008). See
        :func:`~pensive.shifts.topological_anatomy.topological_anatomy`.
        """
        from pensive.shifts.topological_anatomy import topological_anatomy

        return topological_anatomy(self)

    def markov_order(self) -> int | float:
        """Markov order ``R`` when the presentation is right-resolving (unifilar)."""
        from pensive.generators.synchronization import graph_from_sofic_shift, markov_order_from_graph

        return markov_order_from_graph(graph_from_sofic_shift(self))

    def cryptic_order(self) -> int | float:
        """Cryptic order ``k_chi`` when the presentation is right-resolving."""
        from pensive.generators.synchronization import cryptic_order_from_graph, graph_from_sofic_shift

        return cryptic_order_from_graph(graph_from_sofic_shift(self))

    def is_exactly_synchronizable(self) -> bool:
        from pensive.generators.synchronization import graph_from_sofic_shift, is_exactly_synchronizable

        return is_exactly_synchronizable(graph_from_sofic_shift(self))
