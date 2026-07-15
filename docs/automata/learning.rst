.. learning.rst
.. py:module:: sofic.automata.learning

********
Learning
********

``sofic`` provides both **active** and **passive** automata learning.

Active learning (NL\*)
======================

Active learning of maximized prime átomatons via NL\* with a membership
teacher, following Angluin-style learning and its nondeterministic extension
:cite:`Angluin1987,Bollig2009`:

.. autofunction:: sofic.automata.learning.learn_maximized_prime_atomaton

Passive learning (RPNI)
=======================

Given labeled positive and negative example words, RPNI (Regular Positive and
Negative Inference) merges states of the prefix-tree acceptor to induce a DFA
consistent with the sample :cite:`Lang1998`:

.. code-block:: python

   from sofic.automata import learn_dfa_rpni

   dfa = learn_dfa_rpni(positive=["ab", "abab"], negative=["a", "b"])
   dfa.validate()

.. autofunction:: sofic.automata.rpni.learn_dfa_rpni

Passive learning (PAPNI)
========================

PAPNI extends passive inference to visibly pushdown languages. Words over a
:class:`~sofic.automata.papni.DyckAlphabet` are stack-encoded, a DFA is
induced over the encoding, and the result is decoded to a
:class:`~sofic.shifts.sofic_dyck.SoficDyckShift` :cite:`Muskardin2025`:

.. code-block:: python

   from sofic.automata import DyckAlphabet, learn_sofic_dyck_shift_papni

   alphabet = DyckAlphabet(
       call_alphabet=frozenset({"("}),
       return_alphabet=frozenset({")"}),
       internal_alphabet=frozenset(),
   )
   shift = learn_sofic_dyck_shift_papni(positive=["()", "(())"], negative=["("], alphabet=alphabet)

For fitting probabilities on the learned topology, see
:doc:`../generators/stack_inference`.

.. autoclass:: sofic.automata.papni.DyckAlphabet
   :members: classify, symbol_alphabet

.. autofunction:: sofic.automata.papni.learn_sofic_dyck_shift_papni
.. autofunction:: sofic.automata.papni.is_well_matched
.. autofunction:: sofic.automata.papni.papni_encode
.. autofunction:: sofic.automata.papni.papni_encode_samples
.. autofunction:: sofic.automata.papni.sofic_dyck_shift_from_papni_dfa
