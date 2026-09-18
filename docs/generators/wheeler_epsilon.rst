.. wheeler_epsilon.rst
.. py:module:: sofic.generators.wheeler_epsilon

**********************************
Co-lexicographic epsilon-machines
**********************************

.. warning::

   Everything on this page is original and uncited. Wheeler automata are a
   purely topological theory :cite:`Gagie2017` :cite:`Alanko2020`; a literature
   search turned up no treatment of weighted, probabilistic, or
   information-theoretic Wheeler automata, so no canonical source exists for
   the quantities defined here. They are proposals, and every docstring says
   so.

Co-lexicographic order compares words from the last symbol backwards, which is
the *recency* order on pasts. A presentation is Wheeler exactly when its
partition of history space is an interval partition under recency: each state
owns a contiguous band of pasts rather than an arbitrary set. Two consequences
follow, one measure-theoretic and one information-theoretic.

The stationary distribution becomes a genuine cumulative distribution over
pasts. Under an arbitrary state numbering the partial sums of the stationary
vector mean nothing; in Wheeler order they sweep history space from the most
remote pasts to the most recent, which is what arithmetic coding over pasts
needs. :func:`colex_cdf` returns that sweep, :func:`cylinder_measure` the mass
of a rank interval, and :func:`word_cylinder_measure` the mass of the interval
a word selects — pairing directly with
:meth:`~sofic.automata.wheeler_index.WheelerIndex.forward_search`.

The constraint has a price in bits. A process whose causal states are not
recency intervals must split them to get one, and the split states cost
entropy. :func:`wheeler_statistical_complexity` measures the result and
:func:`wheeler_complexity_gap` the excess over :math:`C_\mu`.

.. math::

   C_W = \operatorname{H}[\text{Wheeler states}] \ge \max(C_\mu, \operatorname{H}[X_0]),

the first bound because every Wheeler presentation refines the causal-state
partition, with equality exactly when the ε-machine is already Wheeler; the
second because a Wheeler presentation is input consistent, so its state
determines the symbol that entered it. The fair coin is the extreme case of the
second bound: nothing at all to predict, :math:`C_\mu = 0`, yet sortability
costs a full bit of memory for a symbol the process will never reuse.

.. ipython::

   In [1]: from sofic.examples import golden_mean, fair_coin

   In [2]: from sofic.generators.wheeler_epsilon import colex_cdf, wheeler_complexity_gap

   In [3]: machine = golden_mean(); machine.is_wheeler()

   In [4]: colex_cdf(machine)

   In [5]: machine.wheeler_statistical_complexity(), machine.statistical_complexity()

   In [6]: coin = fair_coin(); coin.is_wheeler(), coin.statistical_complexity()

   In [7]: coin.wheeler_statistical_complexity(), wheeler_complexity_gap(coin)

Finding a presentation
======================

:func:`wheeler_presentation` returns the machine itself when it is already
Wheeler. Otherwise, when the Markov order ``R`` is finite, it returns the
order-``R`` de Bruijn presentation, whose states *are* the length-``R`` words
and so sort co-lexicographically by construction — every finite-order process
is therefore a Wheeler *language* even when its ε-machine is not a Wheeler
*presentation*. Failing that it tries edge-machine refinements, whose order-``k``
states are length-``k`` transition paths and are therefore entered on a single
symbol.

The search can fail for good reason: Wheeler languages are star-free
:cite:`Alanko2021`, so the even process has no Wheeler presentation at any
order and :func:`wheeler_presentation` raises
:class:`~sofic.automata.wheeler.WheelerError`.

The converse separation also holds.
:func:`~sofic.examples.epsilon_machines.wheeler_infinite_order_process` is a
five-state Wheeler ε-machine of infinite Markov order, so Wheelerness is not a
restatement of finite memory in either direction.

API
===

.. autofunction:: wheeler_presentation
.. autofunction:: debruijn_presentation
.. autofunction:: wheeler_statistical_complexity
.. autofunction:: wheeler_complexity_gap
.. autofunction:: colex_cdf
.. autofunction:: cylinder_measure
.. autofunction:: word_cylinder_measure

:class:`~sofic.generators.epsilon_machine.EpsilonMachine` also exposes
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.wheeler_presentation`
and
:meth:`~sofic.generators.epsilon_machine.EpsilonMachine.wheeler_statistical_complexity`,
alongside the :meth:`~sofic.base.StateMachine.is_wheeler`,
:meth:`~sofic.base.StateMachine.wheeler_order`, and
:meth:`~sofic.base.StateMachine.colex_width` methods every model inherits.
