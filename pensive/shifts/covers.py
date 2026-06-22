"""Fischer and Krieger covers of sofic shifts."""

from __future__ import annotations

from typing import Any

from pensive.shifts.sofic import SoficShift


class LeftFischerCover(SoficShift):
    """Left Fischer cover presentation."""

    @classmethod
    def from_sofic(cls, shift: SoficShift, **kwargs: Any) -> LeftFischerCover:
        from pensive.shifts.cover_construction import left_fischer_from_sofic

        return left_fischer_from_sofic(shift)


class RightFischerCover(SoficShift):
    """Right Fischer cover presentation."""

    @classmethod
    def from_sofic(cls, shift: SoficShift, **kwargs: Any) -> RightFischerCover:
        from pensive.shifts.cover_construction import right_fischer_from_sofic

        return right_fischer_from_sofic(shift)


class LeftKriegerCover(SoficShift):
    """Left Krieger cover presentation."""

    @classmethod
    def from_sofic(cls, shift: SoficShift, **kwargs: Any) -> LeftKriegerCover:
        from pensive.shifts.cover_construction import left_krieger_from_sofic

        return left_krieger_from_sofic(shift)


class RightKriegerCover(SoficShift):
    """Right Krieger cover presentation."""

    @classmethod
    def from_sofic(cls, shift: SoficShift, **kwargs: Any) -> RightKriegerCover:
        from pensive.shifts.cover_construction import right_krieger_from_sofic

        return right_krieger_from_sofic(shift)
