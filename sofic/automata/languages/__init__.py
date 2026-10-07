"""Regular-language algebra for automata constructions."""

from sofic.automata.languages.atoms import atoms, prime_atoms
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
from sofic.automata.languages.quotients import left_quotient, left_quotients, right_quotient
from sofic.automata.languages.quotients import residuals as _residuals
from sofic.automata.languages.residuals import is_composed_residual, prime_residuals

# Importing the ``residuals`` submodule rebinds the package attribute of that name.
residuals = _residuals

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
