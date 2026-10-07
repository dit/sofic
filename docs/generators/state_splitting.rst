.. state_splitting.rst
.. py:module:: sofic.generators.state_splitting

*******************************************
State splitting and amalgamation of HMMs
*******************************************

State splitting :cite:`LindMarcus1995` (§2.4; for edge shifts see
:mod:`sofic.shifts.state_splitting`) replaces a state by several copies,
partitioning either its out-edges or its in-edges. For a Mealy HMM the edge
probabilities must be redistributed so that the generated *process* is
unchanged; such splittings and their inverses are among the equivalence moves
between HMM presentations studied by :cite:`Upper1997`. Edges are named
``(source, target, key)`` (:func:`out_edges`, :func:`in_edges`), and splitting
state :math:`s` into :math:`m` copies names them ``(s, 0), ..., (s, m - 1)``.

Write :math:`p(e)` for the joint probability :math:`\Pr(\text{target}, \text{symbol}
\mid \text{source})` of edge :math:`e` and :math:`\pi` for the initial distribution.

Out-splitting
=============

Partition the out-edges of :math:`s` into :math:`E_0, \ldots, E_{m-1}` with masses
:math:`w_i = \sum_{e \in E_i} p(e)`. In the split model

* copy :math:`(s, i)` keeps the edges of :math:`E_i` with probabilities
  :math:`p(e) / w_i`;
* every in-edge :math:`e` of :math:`s` becomes :math:`m` edges, one into each copy,
  with probabilities :math:`p(e)\, w_i`;
* :math:`\pi(s, i) = \pi(s)\, w_i`.

By induction on :math:`t`, the forward vectors satisfy
:math:`\alpha_t(s, i) = w_i\, \alpha_t(s)`: inflow into a copy is the inflow into
:math:`s` scaled by :math:`w_i`, and outflow from :math:`(s, i)` is
:math:`w_i \alpha_t(s) \cdot p(e) / w_i = \alpha_t(s)\, p(e)` along each
:math:`e \in E_i`. Summing over copies, every word :math:`x_{0:n}` has the same
probability. If :math:`\pi` is stationary, so is the split :math:`\pi`.
Out-splitting generally breaks unifilarity, since an in-edge becomes :math:`m`
parallel edges with the same symbol.

In-splitting
============

Partition the in-edges of :math:`s` into :math:`F_0, \ldots, F_{m-1}`. Edge
:math:`e \in F_i` enters only copy :math:`(s, i)`, and every copy keeps all
out-edges of :math:`s` with their original probabilities. The copies have
identical futures, so the backward vectors agree,
:math:`\beta_t(s, i) = \beta_t(s)`, and :math:`\pi(s)` may be shared among the
copies in any proportion. It is shared in proportion to the stationary inflow
:math:`\sum_{e \in F_i} \mu(\operatorname{source}(e))\, p(e)` (uniformly when
:math:`s` has none), which keeps a stationary :math:`\pi` stationary. In-splitting
preserves unifilarity.

Amalgamation
============

:func:`amalgamate` merges the blocks of a state partition when one of two exact
rules applies:

* **identical futures** -- the partition is strongly lumpable and is delegated to
  :func:`~sofic.generators.lumping.lump` :cite:`KemenySnell1976`; this inverts
  in-splitting;
* **proportional inflow** -- for each block :math:`B` there is a weight vector
  :math:`w` on :math:`B` such that, for every source state and symbol, the edge
  mass into :math:`B` and the initial mass on :math:`B` are multiples of
  :math:`w`. The forward vector restricted to :math:`B` is then always a multiple
  of :math:`w`, so :math:`B` merges into a single state whose out-edges are the
  :math:`w`-mixture of its members' out-edges; this inverts out-splitting.

The whole partition is tried under each rule, then blocks are merged one at a
time, so a model that was both in- and out-split amalgamates in one call. A
block with neither inflow nor initial mass is never visited and is merged with
uniform weights: the process is unchanged, but its original out-split weights
cannot be recovered.

Splits and amalgamations leave every process quantity unchanged -- word
distributions, :math:`h_\mu`, :math:`\mathbf{E}`, and the ε-machine (hence
:math:`C_\mu`) -- because they change only the presentation, not the process.

.. ipython::

   In [1]: from sofic.examples import golden_mean

   In [2]: from sofic.generators.state_splitting import amalgamate, out_edges, split_state

   In [3]: machine = golden_mean(0.5)

   In [4]: edges = out_edges(machine, "A")

   In [5]: split = split_state(machine, "A", [edges[:1], edges[1:]])

   In [6]: sorted(split.states(), key=repr), split.is_equal_process(machine)

   In [7]: copies = frozenset({("A", 0), ("A", 1)})

   In [8]: merged = amalgamate(split, [copies, {"B"}], labels={copies: "A"})

   In [9]: sorted(merged.states()), merged.initial_distribution

API
===

.. autofunction:: split_state
.. autofunction:: amalgamate
.. autofunction:: out_edges
.. autofunction:: in_edges
