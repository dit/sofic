.. learning.rst
.. py:module:: sofic.automata.learning.nlstar

********
Learning
********

``sofic`` provides both **active** and **passive** automata learning.

Active learning (NL\*)
======================

NL\* :cite:`Bollig2009` extends Angluin's L\* :cite:`Angluin1987` to
nondeterministic automata. It keeps an RFSA-closed, RFSA-consistent observation
table whose prime rows become the hypothesis states, and adds every suffix of a
counterexample as a new experiment. When the equivalence oracle accepts, the
hypothesis is the canonical RFSA of the target (:doc:`rfsa`). Running NL\* on
the reversed target and reversing the result learns the maximized prime
átomaton (:doc:`atomaton`).

:class:`~sofic.automata.learning.active.AutomatonEquivalenceOracle` answers equivalence
queries exactly against a target automaton, returning a shortest
counterexample.

.. autofunction:: sofic.automata.learning.nlstar.learn_rfsa_nlstar
.. autofunction:: sofic.automata.learning.nlstar.learn_prime_atomaton_nlstar
.. autofunction:: sofic.automata.learning.nlstar.learn_rfsa_from_language

Active learning (L\*, TTT, Mealy)
=================================

:mod:`sofic.automata.learning.active` learns an automaton from a *teacher* answering
**membership** and **equivalence** queries. It provides Angluin's L\*
:cite:`Angluin1987` and a redundancy-free **discrimination-tree** learner in the
TTT family :cite:`KearnsVazirani1994,Isberner2014` for
:class:`~sofic.automata.dfa.DFA`, plus the Mealy variant of L\*
:cite:`Shahbaz2009`; all use Rivest-Schapire counterexample analysis
:cite:`RivestSchapire1993`. Oracles adapt sofic models: a
:class:`~sofic.automata.learning.active.LanguageMembershipOracle` wraps any model exposing
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

   from sofic.automata.learning.active import (
       FunctionMembershipOracle,
       RandomWalkEquivalenceOracle,
       learn_dfa_lstar,
   )

   membership = FunctionMembershipOracle(lambda w: w.count("a") % 2 == 0)
   equivalence = RandomWalkEquivalenceOracle(membership, {"a", "b"}, rng=0)
   dfa = learn_dfa_lstar({"a", "b"}, membership, equivalence)

.. autofunction:: sofic.automata.learning.active.learn_dfa_lstar
.. autofunction:: sofic.automata.learning.active.learn_dfa_ttt
.. autofunction:: sofic.automata.learning.active.learn_mealy_lstar
.. autofunction:: sofic.automata.learning.active.learn_dfa_from_language
.. autofunction:: sofic.automata.learning.active.learn_mealy_from_transducer

.. autoclass:: sofic.automata.learning.active.MembershipOracle
   :members:
.. autoclass:: sofic.automata.learning.active.EquivalenceOracle
   :members:
.. autoclass:: sofic.automata.learning.active.LanguageMembershipOracle
.. autoclass:: sofic.automata.learning.active.FunctionMembershipOracle
.. autoclass:: sofic.automata.learning.active.AutomatonEquivalenceOracle
.. autoclass:: sofic.automata.learning.active.ExhaustiveEquivalenceOracle
.. autoclass:: sofic.automata.learning.active.RandomWalkEquivalenceOracle
.. autoclass:: sofic.automata.learning.active.TransducerOutputOracle
.. autoclass:: sofic.automata.learning.active.MealyExhaustiveEquivalenceOracle

Passive learning (RPNI)
=======================

Given labeled positive and negative example words, RPNI (Regular Positive and
Negative Inference) merges states of the prefix-tree acceptor to induce a DFA
consistent with the sample :cite:`Lang1998`:

.. code-block:: python

   from sofic.automata import learn_dfa_rpni

   dfa = learn_dfa_rpni(positive=["ab", "abab"], negative=["a", "b"])
   dfa.validate()

.. autofunction:: sofic.automata.learning.rpni.learn_dfa_rpni

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

.. autofunction:: sofic.automata.learning.edsm.learn_dfa_edsm

Exact minimal DFA (SAT)
=======================

Where RPNI and EDSM are heuristics, :func:`sofic.automata.learning.dfasat.learn_dfa_sat`
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

.. autofunction:: sofic.automata.learning.dfasat.learn_dfa_sat

Probabilistic passive learning (ALERGIA)
========================================

ALERGIA learns a :class:`~sofic.generators.pfa.ProbabilisticFiniteAutomaton`
from **unlabeled** positive strings by merging states of a frequency
prefix-tree acceptor whenever a Hoeffding-bound test cannot distinguish their
transition statistics :cite:`Carrasco1994`. It is the stochastic, unlabeled
counterpart of RPNI/EDSM and a state-merging alternative to CSSR
(:func:`sofic.inference.cssr.process.cssr`). The compatibility threshold
``alpha`` trades off model size against fidelity: smaller ``alpha`` merges more
aggressively (fewer states); larger ``alpha`` is more conservative.

.. code-block:: python

   from sofic.automata import learn_pfa_alergia
   from sofic.examples import golden_mean

   samples = [golden_mean(0.3).sample(n)[0] for n in range(2, 14)]
   pfa = learn_pfa_alergia(samples, alpha=0.05)
   pfa.validate()

.. autofunction:: sofic.automata.learning.alergia.learn_pfa_alergia

Passive learning (PAPNI)
========================

PAPNI extends passive inference to visibly pushdown languages. Words over a
:class:`~sofic.automata.learning.papni.DyckAlphabet` are stack-encoded, a DFA is
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
:doc:`../inference/stack_cssr`.

.. autoclass:: sofic.automata.learning.papni.DyckAlphabet
   :members: classify, symbol_alphabet

.. autofunction:: sofic.automata.learning.papni.learn_sofic_dyck_shift_papni
.. autofunction:: sofic.automata.learning.papni.is_well_matched
.. autofunction:: sofic.automata.learning.papni.encode_dyck_word
.. autofunction:: sofic.automata.learning.papni.encode_dyck_samples
.. autofunction:: sofic.automata.learning.papni.sofic_dyck_shift_from_papni_dfa
