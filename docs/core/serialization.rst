.. serialization.rst
.. py:module:: sofic.serialization

*************
Serialization
*************

Every :class:`~sofic.core.StateMachine` round-trips through a self-describing
YAML document. The schema records the concrete model class, the transition
graph (states, edges, and typed attributes), and any data not determined by the
graph alone — for example an HMM's initial distribution or a shift's alphabet.

.. ipython::

   In [1]: from sofic.examples import golden_mean

   In [2]: eps = golden_mean(0.5)

   In [3]: text = eps.to_yaml()

   In [4]: restored = type(eps).from_yaml(text)

   In [5]: restored.validate()

The instance methods :meth:`~sofic.core.StateMachine.to_yaml`,
:meth:`~sofic.core.StateMachine.write_yaml`,
:meth:`~sofic.core.StateMachine.from_yaml`, and
:meth:`~sofic.core.StateMachine.read_yaml` delegate to the module-level
functions below. The polymorphic :func:`model_from_yaml` / :func:`from_yaml`
readers reconstruct the correct subclass from the serialized class tag, so they
are convenient when the concrete type is not known in advance.

API
===

.. autofunction:: model_to_yaml
.. autofunction:: model_from_yaml
.. autofunction:: from_yaml
.. autofunction:: read_yaml
.. autofunction:: model_to_dict
.. autofunction:: model_from_dict

.. autodata:: SCHEMA
.. autodata:: VERSION
