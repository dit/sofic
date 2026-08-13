.. viz.rst
.. py:module:: sofic.viz

*************
Visualization
*************

Every model can be rendered as a diagram, either through `Graphviz
<https://graphviz.org/>`_ (raster/SVG, great for notebooks) or as standalone
`TikZ <https://tikz.dev/>`_ / LaTeX source for publications.

Graphviz
========

Call :meth:`~sofic.core.StateMachine.draw` or
:meth:`~sofic.core.StateMachine.to_graphviz` on any model when Graphviz is
installed (``pip install sofic[viz]`` plus the system ``graphviz`` binary):

.. code-block:: python

   from sofic.examples import golden_mean

   eps = golden_mean(0.5)
   eps.draw()                 # write / view a rendered diagram
   dot = eps.to_graphviz()    # a graphviz.Digraph for further styling
   svg = sofic.viz.model_to_svg(eps)

Models implement Jupyter display via ``_repr_mimebundle_``, so simply
evaluating a model in a notebook shows its diagram when Graphviz is available.

By default, edges that share an emission (or an input/label symbol, on
automata and shifts with no emissions) are drawn in the same colour. Pass
``color_by_emission=False`` to ``draw`` / ``to_graphviz`` / ``to_tikz`` for
uncoloured edges. Visibly pushdown automata and Dyck shifts keep their
call / return / internal colours.

TikZ / LaTeX
============

For publication-quality figures, emit TikZ source or a compiled image:

.. code-block:: python

   from sofic.examples import golden_mean

   eps = golden_mean(0.5)
   tikz = eps.to_tikz()               # a LaTeX/TikZ fragment
   eps.draw_tikz("golden_mean.png")   # compile and write an image

The layout can be tuned (``layout``, ``radius``, ``bend_angle``, explicit
``positions`` / ``angles``); see :func:`model_to_tikz`.

API
===

.. autofunction:: model_to_graphviz
.. autofunction:: model_to_svg
.. autofunction:: draw
.. autofunction:: model_to_tikz
.. autofunction:: compile_tikz
.. autofunction:: model_to_tikz_image
.. autofunction:: draw_tikz
