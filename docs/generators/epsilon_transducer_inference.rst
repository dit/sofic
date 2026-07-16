.. epsilon_transducer_inference.rst
.. py:module:: sofic.generators.epsilon_transducer_inference

*********************
transCSSR Inference
*********************

transCSSR reconstructs an :doc:`ε-transducer <epsilon_transducer>` from paired
input/output sample sequences, generalizing Causal-State Splitting
Reconstruction :cite:`Shalizi2004` from a single process to an input/output
channel :cite:`Barnett2015`.

Causal states are equivalence classes of joint ``(input, output)`` pasts that
induce the same conditional next-output law ``P(y | history, x)`` for every input
symbol ``x``. Rare histories inherit their parent's state (controlled by
``min_count``); the split decision uses a G-test at significance ``alpha``.

.. ipython::

   In [1]: import numpy as np

   In [2]: from sofic import EpsilonTransducer, MealyHMM

   In [3]: from sofic.examples.processes import Delay

   In [4]: from sofic.automata.transducer_operations import compose_tg

   In [5]: inp = MealyHMM(observation_alphabet=frozenset({'0', '1'}), initial_distribution={'S': 1.0})

   In [6]: inp.graph.add_state('S'); _ = inp.add_transition('S', 'S', '0', 0.5); _ = inp.add_transition('S', 'S', '1', 0.5); inp.validate()

   In [7]: joint = compose_tg(Delay(1), inp, joint=True)

   In [8]: obs, _ = joint.sample(8000, np.random.default_rng(0))

   In [9]: xs = [o[0] for o in obs]; ys = [o[1] for o in obs]

   @doctest
   In [10]: len(list(EpsilonTransducer.from_paired_sequences(xs, ys, input_alphabet=('0', '1'), output_alphabet=('0', '1')).states()))
   Out[10]: 2

API
===

.. autofunction:: transcssr

.. autoclass:: JointSuffixCounts
   :members: from_sequences, output_counts, state_morph
