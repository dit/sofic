"""Regular-language algebra for automata constructions."""

from sofic.automata.languages.atoms import atoms, is_prime_atom, prime_atoms
from sofic.automata.languages.base import AutomatonLanguage, ExplicitLanguage, RegularLanguage
from sofic.automata.languages.operations import (
    complement,
    concat,
    difference,
    intersection,
    kleene_star,
    product,
    reverse,
    union,
)
from sofic.automata.languages.quotients import left_quotient, left_quotients, residuals, right_quotient
from sofic.automata.languages.residuals import is_composed_residual, prime_residuals

__all__ = [
    "AutomatonLanguage",
    "ExplicitLanguage",
    "RegularLanguage",
    "atoms",
    "complement",
    "concat",
    "difference",
    "intersection",
    "is_composed_residual",
    "is_prime_atom",
    "kleene_star",
    "left_quotient",
    "left_quotients",
    "prime_atoms",
    "prime_residuals",
    "product",
    "residuals",
    "reverse",
    "right_quotient",
    "union",
]
