.. mixed_state_presentation.rst
.. py:module:: sofic.generators.mixed_state

*************************
Mixed-State Presentation
*************************

A :class:`MixedStatePresentation` tracks belief-state dynamics: states are
probability simplices over hidden states of an HMM. The construction is the
mixed-state presentation used for nonunifilar hidden processes and
time-symmetric computational mechanics :cite:`Ellison2009`.

Use :meth:`MixedStatePresentation.to_recurrent` to drop transient belief states
and work only with the recurrent component. Pure recurrent components are
returned as :class:`~sofic.generators.epsilon_machine.EpsilonMachine`;
otherwise the result is a unifilar :class:`~sofic.generators.mealy.MealyHMM`.

.. ipython::

   In [1]: from sofic.examples import tent_map_misiurewicz_hmm

   In [2]: hmm = tent_map_misiurewicz_hmm()

   In [3]: msp = hmm.mixed_state_presentation()

   In [4]: msp.validate()

API
===

.. autoclass:: MixedState
.. autoclass:: MixedStatePresentation

.. autofunction:: sofic.generators.mixed_state_construction.build_mixed_state_presentation
