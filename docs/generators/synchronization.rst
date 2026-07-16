.. synchronization.rst
.. py:module:: sofic.generators.synchronization

***************
Synchronization
***************

Topological synchronization theory :cite:`James2010` — Markov order :math:`R` and
cryptic order :math:`k_\chi` from ε-machine graph structure alone.

All of these quantities read off the *power automaton* (subset construction),
whose start is the full causal-state set and whose recurrent states are the
singletons (synchronized beliefs). Two extremal start-to-singleton paths give
two complementary observation-length scales:

* the **Markov order** :math:`R` is the *longest* prefix-free synchronizing
  path — the worst case number of symbols an observer might need before it is
  certain of the state;
* the **reset threshold** is the *shortest* synchronizing path — the best case,
  i.e. the length of the shortest synchronizing (reset) word. Its finiteness is
  the subject of the Černý conjecture for complete automata :cite:`Cerny1964`
  :cite:`Volkov2008`, though ε-machine presentations are generally partial and
  so that quadratic bound need not hold.

The Travers–Crutchfield synchronization predicates :cite:`Travers2010` classify
how an observer's state uncertainty behaves along almost every generated
sequence:

* :func:`is_exactly_synchronizable` — a finite synchronizing word exists
  (:math:`\Pr(\mathrm{SYN}) = 1`), equivalently the reset threshold is finite;
* :func:`is_asymptotically_synchronizable_from_graph` — the uncertainty merely
  vanishes as :math:`L \to \infty` (:math:`\Pr(\mathrm{WSYN}) = 1`), which holds
  for *every* finite-state ε-machine.

Exact synchronizability is strictly weaker than finite Markov order
(:func:`is_definite_from_graph`, the Perles–Rabin–Shamir notion of a definite
automaton): the butterfly process is exactly synchronizable with reset
threshold ``1`` yet has infinite Markov order.

.. ipython::

   In [1]: from sofic.examples import golden_mean

   In [2]: eps = golden_mean(0.5)

   @doctest
   In [3]: eps.markov_order()
   Out[3]: 1

   @doctest
   In [4]: eps.cryptic_order()
   Out[4]: 1

   @doctest
   In [5]: eps.reset_threshold()
   Out[5]: 1

   @doctest
   In [6]: eps.synchronizing_word()
   Out[6]: [0]

   @doctest
   In [7]: eps.is_exactly_synchronizable(), eps.is_definite()
   Out[7]: (True, True)

API
===

.. autofunction:: sofic.generators.synchronization.markov_order_from_graph
.. autofunction:: sofic.generators.synchronization.cryptic_order_from_graph
.. autofunction:: sofic.generators.synchronization.reset_threshold_from_graph
.. autofunction:: sofic.generators.synchronization.shortest_synchronizing_word_from_graph
.. autofunction:: sofic.generators.synchronization.is_exactly_synchronizable
.. autofunction:: sofic.generators.synchronization.is_asymptotically_synchronizable_from_graph
.. autofunction:: sofic.generators.synchronization.is_definite_from_graph
.. autofunction:: sofic.generators.synchronization.graph_from_epsilon_machine
