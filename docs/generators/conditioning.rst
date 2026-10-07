.. conditioning.rst
.. py:module:: sofic.generators.conditioning

**************************************
Conditioning on a regular constraint
**************************************

:func:`condition_on_language` conditions a stationary HMM on its output staying
inside the language of a DFA forever: the limit, as :math:`n \to \infty`, of
the process conditioned on every prefix of :math:`X_{0:n}` being accepted.

Construction
============

With symbol matrices :math:`T^{(x)}` and DFA transition function
:math:`\delta`, the product of HMM states with *accepting* DFA states carries
the substochastic matrices

.. math::

   M^{(x)}_{(i, q), (j, \delta(q, x))} = T^{(x)}_{ij}
   \quad \text{whenever } \delta(q, x) \text{ is accepting.}

The DFA is completed first, so a missing transition is a move to a rejecting
trap. Only states reached from the stationary-positive HMM states paired with
the DFA's initial state are kept.

Surviving paths of length :math:`n` have mass of order :math:`\lambda^n`, where
:math:`\lambda` is the largest Perron root among the strongly connected
components of :math:`M = \sum_x M^{(x)}`, and they spend all but a bounded
number of steps in the component attaining it. That component must be unique;
otherwise the limit depends on how the constraint is entered and a
``ValueError`` is raised. On it, with right and left Perron eigenvectors
:math:`h` and :math:`u`, the conditioned process is the Doob
:math:`h`-transform

.. math::

   P'(x, j \mid i) = \frac{M^{(x)}_{ij} h_j}{\lambda h_i},
   \qquad \pi'_i \propto u_i h_i,

started in its stationary law :math:`\pi'`. For a 0-1 matrix this is Parry's
measure of maximal entropy :cite:`Parry1964` (:cite:`LindMarcus1995`, §13.3).
States of the result are pairs ``(hmm state, dfa state)``.

Which constraints
=================

The conditioning event is that the DFA run never leaves its accepting states,
i.e. every prefix is accepted. For a prefix-closed language this is just
:math:`X_{0:n} \in L`. For a factorial language, such as the language of a
shift of finite type or sofic shift, the event is also shift invariant, and the
result is the limit of the law of windows :math:`X_{k:k+m}` given
:math:`X_{0:n} \in L` with both :math:`k` and :math:`n - k` large (in the
Cesàro sense when the component is periodic). A language that is not
prefix-closed is effectively replaced by its largest prefix-closed sublanguage.

Example: the Parry measure
==========================

Conditioning the fair coin on the golden mean shift (forbid ``11``) gives the
golden mean process at :math:`p = 1/\varphi`, its measure of maximal entropy,
whose entropy rate is the topological entropy :math:`\log_2 \varphi`. The DFA of
the shift's language is obtained by determinizing its presentation with every
state initial and accepting.

.. ipython::

   In [1]: from sofic.automata.nfa import NFA

   In [2]: from sofic.examples import bernoulli

   In [3]: from sofic.generators.conditioning import condition_on_language

   In [4]: from sofic.shifts.sft import ShiftOfFiniteType

   In [5]: shift = ShiftOfFiniteType.from_forbidden_words({("1", "1")}, frozenset("01"))

   In [6]: states = frozenset(shift.states())

   In [7]: dfa = NFA(graph=shift.graph.copy(), input_alphabet=shift.symbol_alphabet,
      ...:           initial_states=states, accepting_states=states).determinize()

   @doctest float
   In [8]: condition_on_language(bernoulli(), dfa).entropy_rate()
   Out[8]: 0.6942419136306173

   @doctest float
   In [9]: shift.topological_entropy()
   Out[9]: 0.6942419136306162

API
===

.. autofunction:: condition_on_language
