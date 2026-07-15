.. stack_inference.rst
.. py:module:: sofic.generators.stack_inference

*********************
Stack-HMM Inference
*********************

Inference routines that reconstruct a
:class:`~sofic.generators.stack_hmm.HiddenMarkovStackModel` from sequences
over a :class:`~sofic.automata.papni.DyckAlphabet`. These extend the
finite-state inference of :doc:`epsilon_inference` with visibly pushdown stack
semantics :cite:`BealBlockeletDima2015`.

Two families are provided:

* **Topology known.** Given a
  :class:`~sofic.shifts.sofic_dyck.SoficDyckShift` presentation,
  :func:`fit_stack_hmm_mle` estimates smoothed maximum-likelihood transition
  weights from a sample.
* **Topology unknown.** :func:`stack_cssr` and :func:`stack_subtree_merge`
  reconstruct both the control graph and its probabilities from a single long
  sequence by splitting stack-aware histories, in the spirit of Causal-State
  Splitting Reconstruction :cite:`Shalizi2004`.
  :func:`learn_stack_hmm_papni` learns the topology from labeled example words
  via PAPNI :cite:`Muskardin2025` (see :doc:`../automata/learning`) and then
  fits probabilities.

.. code-block:: python

   from sofic.automata import DyckAlphabet
   from sofic.generators import stack_cssr

   alphabet = DyckAlphabet(
       call_alphabet=frozenset({"("}),
       return_alphabet=frozenset({")"}),
       internal_alphabet=frozenset({"a"}),
   )
   model = stack_cssr(sequence, alphabet=alphabet, Lmax=4, max_stack_depth=8)
   model.validate()

Histories are counted with a bounded stack depth, so ``max_stack_depth`` caps
the configurations considered during reconstruction.

API
===

.. autofunction:: stack_cssr
.. autofunction:: stack_subtree_merge
.. autofunction:: fit_stack_hmm_mle
.. autofunction:: learn_stack_hmm_papni

.. autoclass:: StackSuffixCounts
   :members: from_sequence
