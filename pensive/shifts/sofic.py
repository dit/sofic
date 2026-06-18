"""Sofic shift presentations."""

from __future__ import annotations

from pensive.shifts.base import SymbolicModel


class SoficShift(SymbolicModel):
    """Sofic subshift given by a labeled directed graph presentation."""

    def topological_entropy(self) -> float:
        from pensive.shifts.tmc_construction import topological_entropy

        return topological_entropy(self)

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
