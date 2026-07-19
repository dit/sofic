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

Active learning (L\*, TTT, Mealy)
=================================

:mod:`sofic.automata.active` learns an automaton from a *teacher* answering
**membership** and **equivalence** queries. It provides Angluin's L\*
:cite:`Angluin1987` and a redundancy-free **discrimination-tree** learner in the
TTT family :cite:`KearnsVazirani1994,Isberner2014` for
:class:`~sofic.automata.dfa.DFA`, plus the Mealy variant of L\*
:cite:`Shahbaz2009`; all use Rivest-Schapire counterexample analysis
:cite:`RivestSchapire1993`. Oracles adapt sofic models: a
:class:`~sofic.automata.active.LanguageMembershipOracle` wraps any model exposing
``recognizes`` / ``__contains__`` (a :class:`~sofic.automata.dfa.DFA`, NFA,
átomaton, or ``model.to_support_dfa()`` for a sofic shift or ε-machine), and the
equivalence oracles offer bounded-exhaustive or random-walk testing.

.. code-block:: python

   from sofic.automata import learn_dfa_from_language, learn_mealy_from_transducer

   learned = learn_dfa_from_language(target_dfa, {"a", "b"}, algorithm="ttt")
   mealy = learn_mealy_from_transducer(target_mealy)

For a fully black-box teacher, pair an oracle over a predicate with a random-walk
equivalence test:

.. code-block:: python

   from sofic.automata.active import (
       FunctionMembershipOracle,
       RandomWalkEquivalenceOracle,
       learn_dfa_lstar,
   )

   membership = FunctionMembershipOracle(lambda w: w.count("a") % 2 == 0)
   equivalence = RandomWalkEquivalenceOracle(membership, {"a", "b"}, rng=0)
   dfa = learn_dfa_lstar({"a", "b"}, membership, equivalence)

.. autofunction:: sofic.automata.active.learn_dfa_lstar
.. autofunction:: sofic.automata.active.learn_dfa_ttt
.. autofunction:: sofic.automata.active.learn_mealy_lstar
.. autofunction:: sofic.automata.active.learn_dfa_from_language
.. autofunction:: sofic.automata.active.learn_mealy_from_transducer

.. autoclass:: sofic.automata.active.MembershipOracle
   :members:
.. autoclass:: sofic.automata.active.EquivalenceOracle
   :members:
.. autoclass:: sofic.automata.active.LanguageMembershipOracle
.. autoclass:: sofic.automata.active.FunctionMembershipOracle
.. autoclass:: sofic.automata.active.ExhaustiveEquivalenceOracle
.. autoclass:: sofic.automata.active.RandomWalkEquivalenceOracle
.. autoclass:: sofic.automata.active.TransducerOutputOracle
.. autoclass:: sofic.automata.active.MealyExhaustiveEquivalenceOracle

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

Passive learning (EDSM / blue-fringe)
=====================================

EDSM (Evidence-Driven State Merging) upgrades RPNI's greedy merge order to the
red/blue **blue-fringe** strategy that won the Abbadingo One competition
:cite:`Lang1998`. It maintains confirmed *red* states and their *blue* fringe,
and at each step either promotes a blue state that cannot be merged with any red
state or commits the single highest-*evidence* merge -- the merge folding the
most identically-labeled state pairs. Like RPNI it returns a DFA consistent with
the sample, but the evidence heuristic typically recovers a much smaller
automaton:

.. code-block:: python

   from sofic.automata import learn_dfa_edsm

   dfa = learn_dfa_edsm(positive=["a", "aba", "ababa"], negative=["", "b", "ab"])
   dfa.validate()

.. autofunction:: sofic.automata.edsm.learn_dfa_edsm

Exact minimal DFA (SAT)
=======================

Where RPNI and EDSM are heuristics, :func:`sofic.automata.dfasat.learn_dfa_sat`
returns the **provably minimal** DFA consistent with the sample. Following Heule
& Verwer :cite:`HeuleVerwer2010`, it translates the augmented prefix-tree
acceptor into a graph-colouring SAT instance and searches the state count ``k``
upward -- from a lower bound to the EDSM upper bound -- returning the first
satisfiable ``k``. It requires the optional `python-sat
<https://pysathq.github.io/>`_ dependency (``pip install sofic[sat]``):

.. code-block:: python

   from sofic.automata import learn_dfa_sat

   dfa = learn_dfa_sat(positive=["a", "aba", "ababa"], negative=["", "b", "ab"])
   dfa.validate()

.. autofunction:: sofic.automata.dfasat.learn_dfa_sat

Probabilistic passive learning (ALERGIA)
========================================

ALERGIA learns a :class:`~sofic.generators.pfa.ProbabilisticFiniteAutomaton`
from **unlabeled** positive strings by merging states of a frequency
prefix-tree acceptor whenever a Hoeffding-bound test cannot distinguish their
transition statistics :cite:`Carrasco1994`. It is the stochastic, unlabeled
counterpart of RPNI/EDSM and a state-merging alternative to CSSR
(:func:`sofic.generators.epsilon_inference.cssr`). The compatibility threshold
``alpha`` trades off model size against fidelity: smaller ``alpha`` merges more
aggressively (fewer states); larger ``alpha`` is more conservative.

.. code-block:: python

   from sofic.automata import learn_pfa_alergia
   from sofic.examples import golden_mean

   samples = [golden_mean(0.3).sample(n)[0] for n in range(2, 14)]
   pfa = learn_pfa_alergia(samples, alpha=0.05)
   pfa.validate()

.. autofunction:: sofic.automata.alergia.learn_pfa_alergia

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
