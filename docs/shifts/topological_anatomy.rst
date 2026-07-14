.. topological_anatomy.rst
.. py:module:: pensive.shifts.topological_anatomy

*******************************
Topological Information Anatomy
*******************************

Topological entropy :math:`h_\mathrm{top}` is the measure-independent counterpart
of the metric entropy :math:`h_\mu`: by the variational principle it equals
:math:`\sup_\nu h_\nu`, attained by the **measure of maximal entropy** (MME) --
the Parry measure :cite:`Parry1964`.  Evaluating the information anatomy
:cite:`James2013` of the observed symbol process at that MME gives a topological
analogue of the :math:`h_\mu = b_\mu + r_\mu` split:

.. math::

   h_\mathrm{top} = b_\mathrm{top} + r_\mathrm{top}, \qquad
   b_\mathrm{top} = \I{X_0 : X_{1:\infty} \mid X_{-\infty:0}}, \qquad
   r_\mathrm{top} = \H{X_0 \mid X_{-\infty:0}, X_{1:\infty}}

all under the MME :math:`\mu^\*`.  Here :math:`h_\mathrm{top} = \log_2\lambda`
(the Perron eigenvalue of a right-resolving presentation's adjacency matrix),
:math:`b_\mathrm{top}` is the MME bound information, and :math:`r_\mathrm{top}`
is the MME **erasure entropy rate** :cite:`Verdu2008`.  Because the MME and its
observed process are intrinsic to the shift space, :math:`b_\mathrm{top}` and
:math:`r_\mathrm{top}` refine :math:`h_\mathrm{top}` into a
topological-conjugacy invariant of the *labeled* sofic shift.  They are exact:
the MME is a finite Markov chain, so the bidirectional-:math:`\varepsilon`-machine
anatomy is closed-form (requires ``dit``).

Pipeline: put the presentation in right-resolving (unifilar) form (Fischer cover
:cite:`Fischer1975` when needed), build the Parry chain from the Perron data,
minimize to the causal :class:`~pensive.generators.epsilon_machine.EpsilonMachine`,
and read its bidirectional anatomy.

Two shifts with the *same* :math:`h_\mathrm{top} = \log_2\varphi` can have
completely different anatomies -- the golden-mean SFT (forbid ``11``) splits its
entropy across both parts, while the sofic even shift is purely bound
(:math:`r_\mathrm{top} = 0`).

.. ipython::

   In [1]: from pensive.shifts import SoficShift

   In [2]: gm = SoficShift(symbol_alphabet=frozenset({0, 1}))

   In [3]: _ = [gm.graph.add_state(s) for s in ("A", "B")]

   In [4]: _ = gm.add_transition("A", "A", 0)

   In [5]: _ = gm.add_transition("A", "B", 1)

   In [6]: _ = gm.add_transition("B", "A", 0)

   In [7]: anatomy = gm.topological_anatomy()

   @doctest float
   In [8]: anatomy["h_top"]
   Out[8]: 0.6942419136306173

   @doctest float
   In [9]: anatomy["b_top"]
   Out[9]: 0.1414555091305152

   @doctest float
   In [10]: anatomy["r_top"]
   Out[10]: 0.5527864045001022

The parts add up to :math:`h_\mathrm{top}`, which equals
:meth:`~pensive.shifts.sofic.SoficShift.topological_entropy` divided by
:math:`\ln 2` on a right-resolving presentation.

API
===

.. autoclass:: pensive.shifts.sofic.SoficShift
   :members: parry_measure, topological_anatomy
   :noindex:

.. autofunction:: pensive.shifts.topological_anatomy.topological_anatomy

.. autofunction:: pensive.shifts.topological_anatomy.parry_measure_sofic
