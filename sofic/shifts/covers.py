"""Fischer and Krieger covers of sofic shifts."""

from __future__ import annotations

from typing import Any

from sofic.shifts.sofic import SoficShift


class LeftFischerCover(SoficShift):
    """Left Fischer cover presentation."""

    @classmethod
    def from_sofic(cls, shift: SoficShift, **kwargs: Any) -> LeftFischerCover:
        from sofic.shifts.cover_construction import left_fischer_from_sofic

        return left_fischer_from_sofic(shift)


class RightFischerCover(SoficShift):
    """Right Fischer cover presentation."""

    @classmethod
    def from_sofic(cls, shift: SoficShift, **kwargs: Any) -> RightFischerCover:
        from sofic.shifts.cover_construction import right_fischer_from_sofic

        return right_fischer_from_sofic(shift)


class LeftKriegerCover(SoficShift):
    """Left Krieger cover presentation."""

    @classmethod
    def from_sofic(cls, shift: SoficShift, **kwargs: Any) -> LeftKriegerCover:
        from sofic.shifts.cover_construction import left_krieger_from_sofic

        return left_krieger_from_sofic(shift)


class RightKriegerCover(SoficShift):
    """Right Krieger cover presentation."""

    @classmethod
    def from_sofic(cls, shift: SoficShift, **kwargs: Any) -> RightKriegerCover:
        from sofic.shifts.cover_construction import right_krieger_from_sofic

        return right_krieger_from_sofic(shift)


class WheelerCover(SoficShift):
    """Wheeler presentation of a sofic shift, merged down.

    Unlike the Fischer and Krieger covers this one need not exist: Wheeler
    languages are star-free :cite:`Alanko2021`, and even for a shift that has
    one the presentation may need extra symbols of memory. See
    :func:`sofic.shifts.wheeler.wheeler_cover`.
    """

    @classmethod
    def from_sofic(cls, shift: SoficShift, **kwargs: Any) -> WheelerCover:
        from sofic.shifts.wheeler import wheeler_cover

        return wheeler_cover(shift, **kwargs)
