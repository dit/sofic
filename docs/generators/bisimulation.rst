.. bisimulation.rst

**************************
Probabilistic Bisimulation
**************************

:doc:`Lumping <lumping>` aggregates states along a *given* partition. The
coarsest partition that can be lumped is found automatically by computing
**probabilistic bisimulation** :cite:`LarsenSkou1991`: an equivalence relation
on states under which equivalent states send equal probability mass, for every
emitted symbol, into every equivalence class. For a
:class:`~sofic.generators.mealy.MealyHMM` with joint edge law
:math:`P(t, o \mid s)`, states :math:`s \sim s'` are bisimilar when

.. math::

   \sum_{t \in B} P(t, o \mid s) = \sum_{t \in B} P(t, o \mid s')
   \qquad \text{for every class } B \text{ and every symbol } o.

This is exactly the strong lumpability condition of Kemeny & Snell
:cite:`KemenySnell1976`, so the bisimulation classes form the coarsest strongly
lumpable partition: every lumpable partition refines it, and the lumped model
generates the same observed process. A :class:`~sofic.generators.moore.MooreHMM`
additionally requires equal state emission laws. A
:class:`~sofic.generators.markov.MarkovChain` has no edge labels, so its coarsest
lumpable partition is a single block unless an ``initial`` partition (for
example, the level sets of an observation function of the state) is supplied
to be refined.

:func:`~sofic.generators.lumping.bisimulation_partition` computes the classes
by splitter-driven partition refinement in the style of Hopcroft and
Paige--Tarjan: each splitter set divides every block by the per-symbol mass its
states send into the splitter, and once a block that has already served as a
splitter breaks up, all of its pieces but the largest are queued as new
splitters. Probabilities that are all exact (sympy expressions,
:class:`~fractions.Fraction`, or integers) are compared exactly; floating-point
masses are grouped when they lie within ``tol`` of their sorted neighbors.
:func:`~sofic.generators.lumping.coarsest_lumping` returns the quotient model,
and :func:`~sofic.generators.lumping.lump` with ``partition=None`` uses the
coarsest partition.

For a unifilar presentation whose states are all recurrent, bisimulation
minimization coincides with ε-machine minimization, so the quotient has as many
states as :meth:`EpsilonMachine.from_hmm
<sofic.generators.epsilon_machine.EpsilonMachine.from_hmm>`; an ε-machine is
already bisimulation-minimal. For non-unifilar presentations the quotient is
generally not unifilar and can be far smaller than the ε-machine.

.. ipython::

   In [1]: from sofic.generators.mealy import MealyHMM

   In [2]: from sofic.generators.lumping import bisimulation_partition, coarsest_lumping

   In [3]: hmm = MealyHMM(initial_distribution={0: 1.0})

   In [4]: _ = hmm.add_transition(0, 0, "a", 0.5), hmm.add_transition(0, 1, "b", 0.25), hmm.add_transition(0, 2, "b", 0.25)

   In [5]: _ = hmm.add_transition(1, 0, "a", 1.0), hmm.add_transition(2, 0, "a", 1.0)

   In [6]: bisimulation_partition(hmm)

   In [7]: sorted(coarsest_lumping(hmm).states(), key=str)

API
===

.. autofunction:: sofic.generators.lumping.bisimulation_partition
.. autofunction:: sofic.generators.lumping.coarsest_lumping
