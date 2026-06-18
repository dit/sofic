.. dit_bridge.rst
.. py:module:: pensive.dit_bridge

**********
dit Bridge
**********

:mod:`pensive.dit_bridge` connects generators to :mod:`dit` information measures.
Install with ``pip install pensive[measures]``.

Prefer model methods where available — several bridge functions are deprecated
in favor of methods on :class:`~pensive.generators.epsilon_machine.EpsilonMachine`
and :class:`~pensive.generators.bidirectional_epsilon_machine.BidirectionalEpsilonMachine`.

Distributions
=============

.. autofunction:: joint_block_distribution

Computational mechanics
=======================

.. autofunction:: excess_entropy_bidirectional
.. autofunction:: bidirectional_statistical_complexity
.. autofunction:: crypticity
.. autofunction:: excess_entropy

Information anatomy
===================

.. autofunction:: predicted_information
.. autofunction:: bound_information
.. autofunction:: ephemeral_information
.. autofunction:: information_anatomy

Deprecated
==========

.. autofunction:: entropy_rate
.. autofunction:: statistical_complexity

Quasistochastic measures
========================

.. autofunction:: collision_entropy
.. autofunction:: process_negativity
