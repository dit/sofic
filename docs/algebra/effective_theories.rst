.. effective_theories.rst

******************
Effective Theories
******************

Each level of a holonomy cascade reads only the levels above it. So keeping
the top ``k`` levels gives a well-defined machine of its own, a
coarse-grained "effective theory" that tracks the large-scale logic of the
computation while ignoring the details below :cite:`DeDeo2011`. Its states
are *superstates*: sets of original states that agree on the top ``k``
coordinates of some lift. Superstates can overlap.

:meth:`~sofic.algebra.holonomy.HolonomyDecomposition.effective_theory`
returns the truncated machine as a DFA, together with its superstates and
the projection from full coordinates. The projection commutes with every
input. :meth:`~sofic.algebra.cascade.CascadeProduct.truncate` does the same
for a cascade.

The example is Fig. 4 of :cite:`DeDeo2011`: an eight-state machine with two
inputs, built from XOR gates on a three-bit shift register, here transcribed
from the paper's TeX source.

.. ipython::

   In [1]: from sofic.algebra import TransformationSemigroup, holonomy_decomposition

   In [2]: a = {"A": "B", "B": "E", "C": "F", "D": "A", "E": "H", "F": "C", "G": "D", "H": "G"}

   In [3]: b = {"A": "B", "B": "C", "C": "F", "D": "B", "E": "G", "F": "C", "G": "F", "H": "G"}

   In [4]: hd = holonomy_decomposition(TransformationSemigroup.from_maps({"a": a, "b": b}, states="ABCDEFGH"))

   @doctest
   In [5]: [level.order for level in hd.levels]
   Out[5]: [3, 2, 2]

   In [6]: top = hd.effective_theory(1)

   @doctest
   In [7]: sorted("".join(sorted(s)) for s in top.superstates.values())
   Out[7]: ['ACFH', 'BCFG', 'CDEF']

The top level is a counter modulo three on ``a`` and a reset on ``b``. Its
three superstates are AH, BG, and DE, and each one also contains C and F,
exactly as in the paper's Fig. 4b. The two lower levels are cyclic groups of
order two.

Given a stochastic machine, ``effective_theory`` also returns the stationary
mass of each superstate, grouping states by their canonical lift. It reports
whether that grouping is strongly lumpable
(:func:`~sofic.generators.lumping.is_lumpable`), that is, whether the
coarse-grained process is itself Markov on the superstates.

.. autoclass:: sofic.algebra.holonomy.EffectiveTheory
.. automethod:: sofic.algebra.holonomy.HolonomyDecomposition.effective_theory
   :noindex:
.. automethod:: sofic.algebra.cascade.CascadeProduct.truncate
   :noindex:
