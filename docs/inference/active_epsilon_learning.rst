.. active_epsilon_learning.rst
.. py:module:: sofic.inference.active

****************************************
Active learning of ε-machines
****************************************

Sample-based reconstruction (:doc:`cssr`, :doc:`spectral`) estimates an
ε-machine :cite:`Shalizi2001` from data. Active learning instead asks a
*teacher* questions, in the manner of Angluin's L\* for regular languages
:cite:`Angluin1987` (see :func:`sofic.automata.learning.active.learn_dfa_lstar`).
The teacher answers

* **probability queries** ``oracle.word_probability(w)``: the stationary
  probability :math:`P(w)` of a finite word, and optionally
* **equivalence queries** ``oracle.equivalent(machine)``: ``None`` when
  ``machine`` generates the target process, otherwise a counterexample word
  whose probability differs.

:class:`ProcessOracle` answers both exactly from any finite HMM: probabilities
from the symbol-labeled matrices and the stationary state law (in sympy
arithmetic for symbolic models), equivalence with
:func:`~sofic.generators.process_equivalence.is_equal_process`.

.. code-block:: python

   from sofic.examples import nemo_process
   from sofic.inference.active import ProcessOracle, learn_epsilon_machine_active

   target = nemo_process(0.3, 0.6)
   oracle = ProcessOracle(target)
   machine = learn_epsilon_machine_active(oracle, ("0", "1"))
   len(list(machine.states()))                      # 3
   machine.is_equal_process(target)                 # True
   oracle.probability_queries, oracle.equivalence_queries   # (37, 2)

The observation table
=====================

Rows are histories :math:`h` (prefixes), columns are test suffixes :math:`s`, and
the entry is the conditional future probability :math:`P(s \mid h) = P(hs)/P(h)`.
A row is the finite-history *mixed state* :cite:`Ellison2009` seen through the
tests. Tests start as the single symbols, so every row carries its next-symbol
distribution. Rows match when their entries differ by at most ``tol``; answers
that are ``int``, :class:`~fractions.Fraction` or sympy are compared exactly.

The learner repeats:

1. **Close** the table: every positive-probability one-symbol extension
   :math:`hx` of a state row must match a state row; otherwise :math:`hx`
   becomes a state row. State rows stay pairwise distinct, and adding tests
   only refines rows, so the table is also *consistent*.
2. **Hypothesize**: one state per state row, the transition from :math:`h` on
   :math:`x` going to the row matching :math:`hx` with probability
   :math:`P(x \mid h)`. The hypothesis is unifilar by construction and starts at
   the row of the seed ``history`` (empty by default); when seeded, the oracle is
   asked ``equivalent(machine, history=history)`` and compares with the process
   conditioned on the seed.
3. **Query** equivalence. If the oracle has no ``equivalent``, every word of
   length at most ``max_counterexample_length`` is compared instead.
4. **Refine**: every suffix of the counterexample becomes a test.

The returned ε-machine is the hypothesis restricted to its closed class, started
at its stationary distribution; the transient rows (histories that have not yet
synchronized) are dropped.

Guarantees
==========

Suppose the answers and the equivalence oracle are exact, and let :math:`N` be
the number of distinct predictive distributions :math:`P(\cdot \mid h)` over the
positive-probability histories :math:`h` extending the seed ``history``. A closed
table whose tests include every suffix of a word :math:`w = x_0 \cdots x_{L-1}`
predicts :math:`P(w)` correctly, by the chain rule
:math:`P(w) = \prod_{t=0}^{L-1} P(x_t \mid x_{0:t})` and the row match of each
extension on the suffix :math:`x_{t+1:L}`. So each counterexample adds at least
one state row, and when :math:`N` is finite the learner stops after at most
:math:`N` equivalence queries with exactly :math:`N` hypothesis states. Rows are
needed for the :math:`N` state rows and their one-symbol extensions, answers are
cached, and so with an equivalence oracle the number of probability queries is
at most
:math:`2 N (|\mathcal{A}| + 1)(|\mathcal{A}| + \sum_k |w_k|)` for counterexamples
:math:`w_k` -- polynomial in :math:`N`, the alphabet size and the counterexample
lengths.

If the target is a stationary ergodic process whose ε-machine is finite and
exactly synchronizable :cite:`Travers2010`, every predictive state reaches a
causal state, so the closed class of the final hypothesis *is* the ε-machine:
the same number of states, the same process, and the same :math:`C_\mu` and
:math:`h_\mu`.

:math:`N` is finite when

* the mixed-state presentation from the stationary law is finite (golden mean,
  even, Nemo, butterfly, restricted golden mean, ...), or
* the seed ``history`` is a synchronizing word: every row is then a causal
  state, :math:`N` is the number of causal states, and the hypothesis is the
  ε-machine itself.

A finite ε-machine that is not exactly synchronizable has infinitely many mixed
states, and so can an exactly synchronizable one along a non-synchronizing
transient (for example a symbol that fixes two states with different
probabilities). The table then never closes, and the learner raises
:class:`~sofic.exceptions.MixedStateExplosionError` at ``max_states``; seed a
synchronizing ``history`` in that case. More than ``max_rounds`` equivalence
queries raises :class:`RuntimeError`.

With float answers the result is approximate. Matching within ``tol`` is not
transitive; converging transient mixed states are merged once they agree to
``tol``, which keeps the table finite at the cost of extra rows; and a
counterexample that separates no rows at ``tol`` ends learning with the current
hypothesis, which then merges causal states whose predictions differ by less than
``tol``.

API
===

.. autofunction:: learn_epsilon_machine_active

.. autoclass:: ProcessOracle
   :members: word_probability, equivalent

.. autoclass:: ProbabilityOracle
