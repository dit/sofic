.. lumping.rst
.. py:module:: sofic.generators.lumping

*******
Lumping
*******

*Lumping* aggregates the states of a Markov chain or hidden Markov model into a
coarser model by grouping states into the blocks of a partition. A partition is
**strongly lumpable** when the coarse-grained dynamics are again Markov for every
initial distribution :cite:`KemenySnell1976`.

For a Markov chain with transition matrix :math:`P` and partition
:math:`\{B_1, \dots, B_r\}`, strong lumpability requires that the total
probability of moving into a target block does not depend on which state of the
source block the chain currently occupies:

.. math::

   \sum_{t \in B_j} P(t \mid s) = \sum_{t \in B_j} P(t \mid s')
   \qquad \text{for all } s, s' \in B_i,\; \text{all } i, j.

The common value is the lumped transition probability
:math:`\hat P(B_j \mid B_i)`, and initial masses are summed within each block.

Hidden Markov models impose the condition per emitted symbol so that the lumped
model generates the same observed process. A
:class:`~sofic.generators.mealy.MealyHMM` with joint edge law
:math:`P(t, o \mid s)` is lumpable when

.. math::

   \sum_{t \in B_j} P(t, o \mid s) = \sum_{t \in B_j} P(t, o \mid s')
   \qquad \text{for all } s, s' \in B_i,\; \text{all symbols } o,

while a :class:`~sofic.generators.moore.MooreHMM` additionally requires the
state emission law :math:`P(o \mid s)` to be identical across each block.

:func:`is_lumpable` tests the condition and :func:`lump` builds the coarse model,
raising :class:`~sofic.exceptions.LumpabilityError` for a non-lumpable
partition unless ``check=False``. Because lumping can destroy unifilarity, an
:class:`~sofic.generators.epsilon_machine.EpsilonMachine` lumps to a plain
``MealyHMM``.

.. ipython::

   In [1]: from sofic import MarkovChain

   In [2]: chain = MarkovChain(initial_distribution={"A": 1.0})

   In [3]: _ = [chain.graph.add_state(s) for s in ("A", "B", "C")]

   In [4]: _ = chain.add_transition("A", "B", 0.5), chain.add_transition("A", "C", 0.5)

   In [5]: _ = chain.add_transition("B", "A", 0.5), chain.add_transition("B", "B", 0.25), chain.add_transition("B", "C", 0.25)

   In [6]: _ = chain.add_transition("C", "A", 0.5), chain.add_transition("C", "B", 0.5)

   In [7]: chain.is_lumpable([{"A"}, {"B", "C"}])

   In [8]: lumped = chain.lump([{"A"}, {"B", "C"}])

   In [9]: sorted(lumped.states())

   In [10]: lumped.initial_distribution

API
===

.. autofunction:: is_lumpable
.. autofunction:: lump
.. autofunction:: normalize_partition
