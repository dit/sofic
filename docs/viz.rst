Usage
=====

Call :meth:`~pensive.base.StateMachine.draw` or :meth:`~pensive.base.StateMachine.to_graphviz`
on any model when Graphviz is installed:

.. code-block:: python

   from pensive.examples import golden_mean
   eps = golden_mean(0.5)
   eps.draw()  # requires pensive[viz] and system graphviz

Models implement Jupyter display via ``_repr_mimebundle_`` when Graphviz is
available.
