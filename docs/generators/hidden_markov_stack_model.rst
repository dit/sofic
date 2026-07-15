.. hidden_markov_stack_model.rst
.. py:module:: sofic.generators.stack_hmm

*************************
Hidden Markov Stack Model
*************************

A :class:`HiddenMarkovStackModel` is a stochastic visibly-pushdown generator.
Its hidden configuration consists of a finite control state and a stack of
pending call edges. Edges are partitioned into call, return, and internal
roles, and return edges are enabled by the same matched-edge relation used by
:class:`~sofic.shifts.sofic_dyck.SoficDyckShift`.

Edge probabilities are interpreted as weights over the transitions enabled by
the current stack configuration. Those enabled weights are normalized at each
step, so disabled return edges do not make a control-state row invalid.

The full stack process is generally infinite-state. The stationary API
therefore requires a finite ``max_stack_depth`` and returns the stationary law
of that truncated configuration chain, optionally marginalized to control
states.

API
===

.. autoclass:: HiddenMarkovStackModel
   :members:
