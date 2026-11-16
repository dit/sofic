.. omega_probability.rst
.. py:module:: sofic.generators.omega_probability

*****************************************
Probabilities of regular properties
*****************************************

A deterministic automaton reading the output of an HMM is in a state that is a
function of the symbols read so far, so the pair (HMM state, automaton state) is
again a finite Markov chain. Both functions below work on this product chain
:cite:`BaierKatoen2008`. The HMM starts in its stationary law unless a
``start`` (a state, a state-to-mass mapping, or a vector) is given.

Finite words
============

:func:`regular_language_probability` returns
:math:`P(X_{0:n} \in L)` for the language :math:`L` of a DFA exactly, by
pushing one forward vector per DFA state through :math:`T^{(x)}` along
:math:`\delta(q, x)` for :math:`n` steps. For the fair coin and the golden mean
language, half of the 16 words of length 4 avoid ``11``:

.. ipython::

   In [1]: from sofic.automata.nfa import NFA

   In [2]: from sofic.examples import bernoulli, golden_mean

   In [3]: from sofic.generators.omega_probability import omega_probability, regular_language_probability

   In [4]: from sofic.shifts.sft import ShiftOfFiniteType

   In [5]: shift = ShiftOfFiniteType.from_forbidden_words({("1", "1")}, frozenset("01"))

   In [6]: states = frozenset(shift.states())

   In [7]: dfa = NFA(graph=shift.graph.copy(), input_alphabet=shift.symbol_alphabet,
      ...:           initial_states=states, accepting_states=states).determinize()

   @doctest float
   In [8]: regular_language_probability(bernoulli(), dfa, 4)
   Out[8]: 0.5

Infinite words
==============

:func:`omega_probability` returns the probability that the infinite output
:math:`X_0 X_1 \cdots` is accepted by a *deterministic* Büchi automaton (DBA).
A finite Markov chain almost surely enters a bottom strongly connected component
of the product and then visits each of its states infinitely often, so the run
is accepted iff that component contains an accepting automaton state. The
answer is the mass absorbed into such components, a linear solve on the
transient states :cite:`BaierKatoen2008`. A missing DBA transition moves to an
absorbing rejecting state.

DBAs recognize a strict subclass of the ω-regular languages. Safety properties
("never ``11``"), reachability ("eventually ``1``"), and recurrence
("infinitely many ``1``") are DBA-expressible; persistence ("eventually always
``0``", i.e. finitely many ``1``) is not, but its probability is one minus that
of its complement "infinitely many ``1``". A non-deterministic Büchi automaton
raises :class:`~sofic.exceptions.NonDeterministicError`.

.. ipython::

   In [9]: from sofic.automata.buchi import BuchiAutomaton

   In [10]: ones = BuchiAutomaton(input_alphabet=frozenset("01"), initial_states=frozenset({"a"}),
      ....:                       accepting_states=frozenset({"b"}))

   In [11]: ones.graph.add_state("a"); ones.graph.add_state("b")

   In [12]: for source, target, symbol in [("a", "a", "0"), ("a", "b", "1"), ("b", "a", "0"), ("b", "b", "1")]:
      ....:     ones.add_transition(source, target, symbol)

   @doctest float
   In [13]: omega_probability(golden_mean(0.4), ones)
   Out[13]: 1.0

API
===

.. autofunction:: regular_language_probability

.. autofunction:: omega_probability
