.. noise.rst
.. py:module:: sofic.algebra.random_maps

********************
Noisy Automata
********************

A probabilistic automaton gives each input letter a row-stochastic matrix.
It can be rewritten as a random choice of a deterministic map of the states,
followed by applying that map :cite:`Maler1995`. The deterministic part is an
ordinary transformation semigroup, so it has a holonomy decomposition. How
the random choice is made is a *failure model* for the underlying mechanism
:cite:`DeDeo2011`:

* :func:`independent_random_maps` lets every state fail independently:
  ``pi(m) = prod_q P[q, m(q)]`` (Eq. 2 of :cite:`DeDeo2011`).
* :func:`coupled_random_maps` uses one shared random draw for every state,
  the random mapping representation of :cite:`LevinPeresWilmer2009`. Failures
  are then perfectly correlated.
* :func:`random_maps` takes any decomposition and checks it against the
  matrices.

The example is the two-bit XOR register of :cite:`DeDeo2011` with a 1%
bit-flip rate. Each state appends a 1 with the probability given in the list
``[eps, 1 - eps, 1 - eps, eps]``:

.. ipython::

   In [1]: import numpy as np

   In [2]: from sofic.algebra import independent_random_maps, coupled_random_maps

   In [3]: eps = 0.01

   In [4]: P = np.array([[1 - eps, eps, 0, 0], [0, 0, eps, 1 - eps], [eps, 1 - eps, 0, 0], [0, 0, 1 - eps, eps]])

   In [5]: independent, coupled = independent_random_maps(P), coupled_random_maps(P)

   @doctest
   In [6]: len(independent.maps("t")), len(coupled.maps("t"))
   Out[6]: (16, 2)

   @doctest
   In [7]: len(independent.semigroup()), len(coupled.semigroup())
   Out[7]: (256, 12)

Independent failures give all sixteen maps compatible with the register,
and together they generate the full transformation monoid on four states.
Correlated failures give just two maps, XOR and NOT-XOR, which generate the
alternating group ``A_4``.

Walks on the semigroup
======================

:func:`element_walk` returns the distribution of the composed random map
``M_t = m_1 ... m_t`` over semigroup elements at each time ``t``, a Markov
chain on the right Cayley graph. :func:`classify_elements` labels the
identity, the permutations, the constant maps (resets), and the minimal
ideal. Together they reproduce Figs. 6 and 8 of :cite:`DeDeo2011`. Under
independent noise the mass drains into the resets, which form the minimal
ideal, after a brief window in which other permutations outweigh them. Under
correlated noise no reset ever appears, and the walk spreads over ``A_4``.
Here it cycles with period three through the cosets of the Klein group, so
it is uniform on ``A_4`` only on average.

.. ipython::

   In [1]: from sofic.algebra import classify_elements, element_walk

   In [2]: sg = independent.semigroup(); resets = sorted(classify_elements(sg)["resets"])

   @doctest
   In [3]: round(float(element_walk(independent, 1000)[-1, resets].sum()), 3)
   Out[3]: 0.998

Erased versus dissipated information
====================================

DeDeo separates *irreversibility*, where a map deletes information, from
*dissipation*, where different histories become indistinguishable because
nobody knows which random operation ran. :func:`walk_information` measures
both. Take ``X_0`` uniform and independent of the noise, and let
``X_t = X_0 · M_t``. Then:

* ``irreversibility = H[X_0] - I[X_0 : X_t | M_t]`` is the information gone
  even if the random choices are known.
* ``dissipation = I[X_0 : X_t | M_t] - I[X_0 : X_t]`` is the information lost
  only because the random choices are unknown.

.. note:: Original, uncited: this information-theoretic split is not in
   :cite:`DeDeo2011`.

.. ipython::

   In [1]: from sofic.algebra import walk_information

   In [2]: ind, cpl = walk_information(independent, 1000), walk_information(coupled, 5000)

   @doctest
   In [3]: round(float(ind.irreversibility[-1]), 2), round(float(cpl.irreversibility[-1]), 2)
   Out[3]: (2.0, 0.0)

   @doctest
   In [4]: round(float(cpl.dissipation[-1]), 2)
   Out[4]: 2.0

Independent noise erases the two bits of the initial state. Correlated noise
erases nothing, but loses both bits to ignorance of the noise.

.. autoclass:: RandomMapRepresentation
   :members:
.. autofunction:: independent_random_maps
.. autofunction:: coupled_random_maps
.. autofunction:: random_maps
.. autofunction:: element_walk
.. autofunction:: classify_elements
.. autofunction:: walk_information
.. autoclass:: WalkInformation
