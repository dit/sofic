.. state_machine.rst
.. py:module:: pensive.core

*************
StateMachine
*************

:class:`StateMachine` is the abstract base class for every graph-backed model in
``pensive``. Subclasses implement :meth:`~StateMachine.validate` to enforce
structural and semantic invariants.

Common operations
=================

Build a model, validate it, and inspect its graph:

.. ipython::

   In [1]: from pensive.examples import golden_mean

   In [2]: eps = golden_mean(0.5)

   In [3]: eps.validate()

   @doctest
   In [4]: repr(eps)
   Out[4]: 'EpsilonMachine(2 states, 3 transitions)'

   In [5]: g = eps.to_networkx()

   In [6]: eps2 = eps.copy()

Reverse the transition graph:

.. ipython::

   In [7]: from pensive.operations import reverse

   In [8]: rev = reverse(eps)

Serialize and reconstruct a model with YAML:

.. ipython::

   In [9]: yaml_text = eps.to_yaml()

   In [10]: eps3 = type(eps).from_yaml(yaml_text)

   In [11]: eps3.validate()

For HMM-style generators, emitted alphabets are inferred from graph emissions
when loading YAML. Initial distributions remain serialized because they are not
determined by the transition graph in general.

Jupyter notebooks display models as Graphviz SVG when ``pensive[viz]`` is
installed (see :doc:`../viz`).

API
===

.. autoclass:: StateMachine
   :members: validate, states, transitions, reindex, copy, reverse, to_networkx, from_networkx, to_yaml, write_yaml, from_yaml, read_yaml

.. autoclass:: StateIndex

.. autofunction:: pensive.operations.reverse
