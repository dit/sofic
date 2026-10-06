.. stack_cssr.rst
.. py:module:: sofic.inference.cssr.stack

*********************
Stack-HMM Inference
*********************

Inference routines that reconstruct a
:class:`~sofic.generators.stack_hmm.HiddenMarkovStackModel` from sequences
over a :class:`~sofic.automata.learning.papni.DyckAlphabet`. These extend the
finite-state inference of :doc:`cssr` with visibly pushdown stack
semantics :cite:`BealBlockeletDima2015`.

Two families are provided:

* **Topology known.** Given a
  :class:`~sofic.shifts.sofic_dyck.SoficDyckShift` presentation,
  :func:`learn_stack_hmm_mle` estimates smoothed maximum-likelihood transition
  weights from a sample.
* **Topology unknown.** :func:`learn_stack_hmm_cssr` and :func:`learn_stack_hmm_subtree`
  reconstruct both the control graph and its probabilities from a single long
  sequence by splitting stack-aware histories, in the spirit of Causal-State
  Splitting Reconstruction :cite:`Shalizi2004`.
  :func:`learn_stack_hmm_papni` learns the topology from labeled example words
  via PAPNI :cite:`Muskardin2025` (see :doc:`../automata/learning`) and then
  fits probabilities.

.. code-block:: python

   from sofic.automata import DyckAlphabet
   from sofic.inference.cssr import learn_stack_hmm_cssr

   alphabet = DyckAlphabet(
       call_alphabet=frozenset({"("}),
       return_alphabet=frozenset({")"}),
       internal_alphabet=frozenset({"a"}),
   )
   model = learn_stack_hmm_cssr(sequence, alphabet=alphabet, max_history=4, max_stack_depth=8)
   model.validate()

Histories are counted with a bounded stack depth, so ``max_stack_depth`` caps
the configurations considered during reconstruction.

Stack CSSR runs flat CSSR over ``(suffix, stack)`` configurations. Every observed
stack seeds its own root, and suffixes grow into the past with the stack fixed.
When morphs are compared, all return symbols count as one event: which return
can follow is decided by the stack top through matched call-return pairs, not by
the finite control. Return edges are matched only to calls observed to close
them.

``learn_stack_hmm_cssr`` accepts the same calibration options as
:func:`~sofic.inference.cssr.learn_epsilon_machine_cssr`: ``test="exact"``,
``correction="bonferroni"`` (over eligible configurations), and ``max_history="auto"``.
Stack processes generally have infinite Markov order, so the automatic depth is
a lower bound on the suffix length the data support.

API
===

.. autofunction:: learn_stack_hmm_cssr
.. autofunction:: learn_stack_hmm_subtree
.. autofunction:: learn_stack_hmm_mle
.. autofunction:: learn_stack_hmm_papni

.. autoclass:: StackSuffixCounts
   :members: from_sequence
