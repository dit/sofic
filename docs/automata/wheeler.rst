.. wheeler.rst
.. py:module:: sofic.automata.wheeler

********************************
Co-lexicographic (Wheeler) order
********************************

A labeled graph is *Wheeler* when its states admit a total order in which the
states with no incoming edges come first and, for edges ``(u, v)`` labeled
``a`` and ``(u', v')`` labeled ``a'``,

* ``a < a'`` implies ``v < v'``, and
* ``a == a'`` and ``u < u'`` imply ``v <= v'``

:cite:`Gagie2017`. Equivalently, the words reaching each state form an
interval of the co-lexicographically sorted prefixes of the language
:cite:`Alanko2020`. Co-lex order compares words from the last symbol backwards,
so a Wheeler presentation is one whose states partition history space into
*bands of recency* rather than arbitrary sets.

Wheeler orders are the width-one case of the co-lexicographic partial orders of
:cite:`CotumaccioPrezza2021`. :func:`colex_width` measures how far a machine is
from being Wheeler; the width bounds the cost of indexing it, storing it, and
determinizing it.

Presentations versus languages
==============================

:func:`is_wheeler` asks whether *this presentation* is Wheeler. For a
deterministic presentation that is a sorting question, answerable in polynomial
time; for a general labeled graph, recognizing Wheelerness is NP-complete
:cite:`GibneyThankachan2019`. Whether the *language* is Wheeler — whether any
equivalent automaton is — is harder still: ``O(mn)`` for a DFA
:cite:`Becker2023`, improving the first polynomial algorithm
:cite:`Alanko2021`, and PSPACE-complete for an NFA :cite:`DAgostino2023`.
:func:`~sofic.shifts.wheeler.wheeler_cover` and
:func:`~sofic.generators.wheeler_epsilon.wheeler_presentation` search over
presentations for the process-level question.

Wheeler is not a restatement of finite memory. The golden mean process is
Wheeler; so is
:func:`~sofic.examples.epsilon_machines.wheeler_infinite_order_process`, whose
Markov order is infinite. Conversely the even process is not Wheeler in any
presentation, because Wheeler languages are star-free
:cite:`ShyrThierrin1974` :cite:`Alanko2021` and the even process counts
``1``\ s modulo two.

.. ipython::

   In [1]: from sofic.examples import golden_mean, even_process, wheeler_infinite_order_process

   In [2]: golden_mean().wheeler_order().states

   In [3]: even_process().is_wheeler(), even_process().colex_width()

   In [4]: machine = wheeler_infinite_order_process(); machine.is_wheeler(), machine.markov_order()

Every graph-backed model inherits :meth:`~sofic.base.StateMachine.is_wheeler`,
:meth:`~sofic.base.StateMachine.wheeler_order`, and
:meth:`~sofic.base.StateMachine.colex_width`, so automata, shifts, and
ε-machines all answer the same questions.

What the order buys
===================

* A canonical state numbering, hence canonical transition matrices and
  ``O(m)`` equality by comparing Burrows-Wheeler strings
  (:func:`wheeler_canonical_form`, :func:`wheeler_isomorphic`).
* Path coherence: the states reachable from an interval on a given symbol are
  again an interval. :func:`~sofic.generators.synchronization.power_automaton`
  uses this to replace ``2^n`` subsets with ``n(n+1)/2`` intervals, which makes
  the Markov order, cryptic order, and reset threshold polynomial.
* Determinization to at most ``2n - 1 - |Sigma|`` states
  (:func:`wnfa_to_wdfa`) and a unique minimal WDFA (:func:`minimum_wdfa`),
  both of which fail for general automata.
* A succinct index — see below.

Order
=====

.. autoclass:: WheelerOrder
   :members: validate
.. autoclass:: LabeledGraph
.. autoexception:: WheelerError

.. autofunction:: labeled_graph
.. autofunction:: wheeler_order
.. autofunction:: wheeler_order_of_graph
.. autofunction:: is_wheeler
.. autofunction:: is_input_consistent
.. autofunction:: check_wheeler_axioms

Width
=====

.. autofunction:: colex_width
.. autofunction:: maximum_colex_relation
.. autofunction:: maximum_colex_relation_of_graph

Canonical forms and minimization
================================

.. autofunction:: minimum_wdfa
.. autofunction:: wnfa_to_wdfa
.. autofunction:: wheeler_canonical_form
.. autofunction:: wheeler_isomorphic
.. autofunction:: wheeler_state_index

Burrows-Wheeler index
=====================

.. py:currentmodule:: sofic.automata.wheeler_index

:class:`WheelerIndex` stores a Wheeler machine as the out-degree, in-degree,
and label arrays of :cite:`Gagie2017`. Because the Wheeler axioms make the
edges sorted by ``(label, source)`` coincide with the edges sorted by target,
following a symbol maps one node interval onto another — FM-index backward
search, generalized to labeled graphs. Rank is served by binary search over
per-symbol position arrays, which costs a logarithmic factor but adds no
dependency beyond numpy.

A shift presents its factor language with every state both initial and
accepting, the case :cite:`Gagie2017` Theorem 6 covers explicitly, so the same
index answers membership and enumeration queries for shifts.

.. ipython::

   In [1]: from sofic.examples import golden_mean

   In [2]: from sofic.automata.wheeler_index import WheelerIndex

   In [3]: index = WheelerIndex.from_model(golden_mean()); index.bits()

   In [4]: index.contains((0, 1, 0)), index.contains((1, 1))

   In [5]: list(index.words_of_length(3))

   In [6]: index.unrank_word(index.rank_word((0, 1, 0)), 3)

Words are listed in co-lexicographic order, so :meth:`WheelerIndex.rank_word`
and :meth:`WheelerIndex.unrank_word` invert one another and
:meth:`WheelerIndex.sample_word` draws uniformly from the words of a length
without enumerating them.

.. autoclass:: WheelerIndex
   :members: from_model, bits, step, forward_search, contains, count_states,
             states_reached, count_words, words_of_length, rank_word,
             unrank_word, sample_word

.. autofunction:: wheeler_index
