"""Regular-language algebra for automata constructions."""

from pensive.automata.languages.atoms import atoms, is_prime_atom, prime_atoms
from pensive.automata.languages.base import AutomatonLanguage, ExplicitLanguage, RegularLanguage
from pensive.automata.languages.operations import complement, intersection, reverse, union
from pensive.automata.languages.quotients import left_quotient, left_quotients, residuals, right_quotient
from pensive.automata.languages.residuals import is_composed_residual, prime_residuals

__all__ = [
    "AutomatonLanguage",
    "ExplicitLanguage",
    "RegularLanguage",
    "atoms",
    "complement",
    "intersection",
    "is_composed_residual",
    "is_prime_atom",
    "left_quotient",
    "left_quotients",
    "prime_atoms",
    "prime_residuals",
    "residuals",
    "reverse",
    "right_quotient",
    "union",
]
