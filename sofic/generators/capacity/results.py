"""Result type for channel-capacity computations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapacityBounds:
    """Bracket ``lower <= C <= upper`` on a channel capacity, in bits per use.

    ``feedback`` says which capacity is bounded. ``certified`` is ``False`` for
    numerical estimates (such as gridded value iteration) whose endpoints are
    not guaranteed bounds. ``details`` holds method-specific artifacts: optimal
    policies, Q-graphs, test distributions, solver residuals.
    """

    lower: float
    upper: float
    method: str
    feedback: bool
    certified: bool = True
    tol: float = 1e-4
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def gap(self) -> float:
        return self.upper - self.lower

    @property
    def value(self) -> float | None:
        """Return the midpoint when the bracket is narrower than ``tol``, else ``None``."""
        if self.gap < self.tol:
            return 0.5 * (self.lower + self.upper)
        return None
