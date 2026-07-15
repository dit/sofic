.. synchronization.rst
.. py:module:: sofic.generators.synchronization

***************
Synchronization
***************

Topological synchronization theory :cite:`James2010` — Markov order :math:`R` and
cryptic order :math:`k_\chi` from ε-machine graph structure alone.

.. ipython::

   In [1]: from sofic.examples import golden_mean

   In [2]: eps = golden_mean(0.5)

   @doctest
   In [3]: eps.markov_order()
   Out[3]: 1

   @doctest
   In [4]: eps.cryptic_order()
   Out[4]: 1

API
===

.. autofunction:: sofic.generators.synchronization.markov_order_from_graph
.. autofunction:: sofic.generators.synchronization.cryptic_order_from_graph
.. autofunction:: sofic.generators.synchronization.is_exactly_synchronizable
.. autofunction:: sofic.generators.synchronization.graph_from_epsilon_machine
