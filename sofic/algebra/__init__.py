"""Algebraic automata theory: transformation semigroups and Krohn-Rhodes decompositions.

* :mod:`~sofic.algebra.transformation` -- transformations and the semigroups
  they generate.
* :mod:`~sofic.algebra.green` -- Green's relations, egg boxes, maximal
  subgroups.
* :mod:`~sofic.algebra.syntactic` -- syntactic monoids and star-freeness.
* :mod:`~sofic.algebra.holonomy` and :mod:`~sofic.algebra.cascade` -- the
  holonomy decomposition as a verifiable cascade product.
* :mod:`~sofic.algebra.complexity` -- Krohn-Rhodes complexity bounds and
  prime factors.
* :mod:`~sofic.algebra.information` -- the stationary state entropy split by
  holonomy level.

Group-theoretic outputs (holonomy and maximal subgroups as permutation
groups, composition factors) need the optional ``sympy`` dependency, which
the ``sofic[symbolic]`` extra installs.
"""

from sofic.algebra.adapters import semiautomaton, transformation_semigroup
from sofic.algebra.belief import mixed_state_semigroup, projective_semigroup
from sofic.algebra.cascade import CascadeProduct, LevelAction
from sofic.algebra.complexity import (
    ComplexityBounds,
    PrimeDecomposition,
    PrimeFactor,
    is_full_transformation_monoid,
    krohn_rhodes_complexity,
    prime_decomposition,
)
from sofic.algebra.green import DClass, GreenStructure, green_structure
from sofic.algebra.holonomy import EffectiveTheory, HolonomyDecomposition, HolonomyLevel, holonomy_decomposition
from sofic.algebra.information import HolonomyInformation, holonomy_information
from sofic.algebra.random_maps import (
    RandomMapRepresentation,
    WalkInformation,
    classify_elements,
    coupled_random_maps,
    element_walk,
    independent_random_maps,
    random_maps,
    walk_information,
)
from sofic.algebra.syntactic import factor_monoid, is_star_free, syntactic_monoid, transition_monoid
from sofic.algebra.transformation import SINK, Transformation, TransformationSemigroup
from sofic.algebra.varieties import (
    is_commutative_language,
    is_definite,
    is_group_language,
    is_j_trivial,
    is_l_trivial,
    is_locally_testable,
    is_piecewise_testable,
    is_r_trivial,
    is_reverse_definite,
)

__all__ = [
    "SINK",
    "CascadeProduct",
    "ComplexityBounds",
    "DClass",
    "EffectiveTheory",
    "GreenStructure",
    "HolonomyDecomposition",
    "HolonomyInformation",
    "HolonomyLevel",
    "LevelAction",
    "PrimeDecomposition",
    "PrimeFactor",
    "RandomMapRepresentation",
    "Transformation",
    "TransformationSemigroup",
    "WalkInformation",
    "classify_elements",
    "coupled_random_maps",
    "element_walk",
    "factor_monoid",
    "green_structure",
    "holonomy_decomposition",
    "holonomy_information",
    "independent_random_maps",
    "is_commutative_language",
    "is_definite",
    "is_full_transformation_monoid",
    "is_group_language",
    "is_j_trivial",
    "is_l_trivial",
    "is_locally_testable",
    "is_piecewise_testable",
    "is_r_trivial",
    "is_reverse_definite",
    "is_star_free",
    "krohn_rhodes_complexity",
    "mixed_state_semigroup",
    "prime_decomposition",
    "projective_semigroup",
    "random_maps",
    "semiautomaton",
    "syntactic_monoid",
    "transformation_semigroup",
    "transition_monoid",
    "walk_information",
]
