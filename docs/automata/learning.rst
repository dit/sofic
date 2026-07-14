.. learning.rst
.. py:module:: pensive.automata.learning

********
Learning
********

``pensive`` provides both **active** and **passive** automata learning.

Active learning (NL\*)
======================

Active learning of maximized prime átomatons via NL\* with a membership
teacher, following Angluin-style learning and its nondeterministic extension
:cite:`Angluin1987,Bollig2009`:

.. autofunction:: pensive.automata.learning.learn_maximized_prime_atomaton

Passive learning (RPNI)
=======================

Given labeled positive and negative example words, RPNI (Regular Positive and
Negative Inference) merges states of the prefix-tree acceptor to induce a DFA
consistent with the sample :cite:`Lang1998`:

.. code-block:: python

   from pensive.automata import learn_dfa_rpni

   dfa = learn_dfa_rpni(positive=["ab", "abab"], negative=["a", "b"])
   dfa.validate()

.. autofunction:: pensive.automata.rpni.learn_dfa_rpni

Passive learning (PAPNI)
========================

PAPNI extends passive inference to visibly pushdown languages. Words over a
:class:`~pensive.automata.papni.DyckAlphabet` are stack-encoded, a DFA is
induced over the encoding, and the result is decoded to a
:class:`~pensive.shifts.sofic_dyck.SoficDyckShift` :cite:`Muskardin2025`:

.. code-block:: python

   from pensive.automata import DyckAlphabet, learn_sofic_dyck_shift_papni

   alphabet = DyckAlphabet(
       call_alphabet=frozenset({"("}),
       return_alphabet=frozenset({")"}),
       internal_alphabet=frozenset(),
   )
   shift = learn_sofic_dyck_shift_papni(positive=["()", "(())"], negative=["("], alphabet=alphabet)

For fitting probabilities on the learned topology, see
:doc:`../generators/stack_inference`.

.. autoclass:: pensive.automata.papni.DyckAlphabet
   :members: classify, symbol_alphabet

.. autofunction:: pensive.automata.papni.learn_sofic_dyck_shift_papni
.. autofunction:: pensive.automata.papni.is_well_matched
.. autofunction:: pensive.automata.papni.papni_encode
.. autofunction:: pensive.automata.papni.papni_encode_samples
.. autofunction:: pensive.automata.papni.sofic_dyck_shift_from_papni_dfa
