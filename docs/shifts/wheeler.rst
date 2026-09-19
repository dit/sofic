.. wheeler.rst
.. py:module:: sofic.shifts.wheeler

**********************
Wheeler presentations
**********************

A shift presents its factor language with every state both initial and
accepting, which is the "all states initial" case of :cite:`Gagie2017`
Theorem 6. The minimal right-resolving presentation is rarely Wheeler as it
stands — a state entered on two different symbols already breaks the axioms —
but remembering the last few symbols restores input consistency, and for many
shifts that is enough.

:func:`wheeler_cover` right-resolves the presentation, walks up the higher
block presentations until one is Wheeler, then merges the Wheeler-consecutive
states that share an incoming label and a follower language. The result is the
smallest Wheeler presentation *reachable that way*, not provably the smallest
Wheeler presentation of the language.

Not every sofic shift has one at any order. Wheeler languages are star-free
:cite:`ShyrThierrin1974` :cite:`Alanko2021`, so a shift whose syntactic monoid
contains a nontrivial group — the even shift being the standard example — is
excluded outright, and :func:`wheeler_cover` raises
:class:`~sofic.automata.wheeler.WheelerError`.

.. ipython::

   In [1]: from sofic.shifts.sofic import SoficShift

   In [2]: from sofic.shifts.wheeler import wheeler_cover, wheeler_order_of_shift

   In [3]: from sofic.graph import ATTR_SYMBOL

   In [4]: shift = SoficShift(symbol_alphabet=frozenset({0, 1}))

   In [5]: for source, symbol, target in ((0, 0, 0), (0, 1, 1), (1, 0, 0)):
      ...:     shift.graph.add_transition(source, target, **{ATTR_SYMBOL: symbol})

   In [6]: wheeler_order_of_shift(shift)

   In [7]: cover = wheeler_cover(shift); cover.wheeler_order().states

The cover is a :class:`~sofic.shifts.covers.WheelerCover`, a
:class:`~sofic.shifts.sofic.SoficShift` subclass that sits beside the Fischer
and Krieger covers. It does not fill the Krieger stubs in
:mod:`sofic.shifts.cover_construction`: a Krieger cover's states are *all*
sets of pasts closed under the follower relation, whereas a Wheeler cover
carries only those that happen to be recency intervals.

Once a shift has a Wheeler cover, :func:`wheeler_index_of_shift` gives
``O(|w| log |A|)`` factor-language membership in place of scanning
:meth:`~sofic.shifts.base.SymbolicModel.factor_language`.

API
===

.. autofunction:: wheeler_cover
.. autofunction:: is_wheeler_shift
.. autofunction:: wheeler_order_of_shift
.. autofunction:: wheeler_index_of_shift
.. autofunction:: higher_block_presentation
.. autofunction:: right_resolve

.. autoclass:: sofic.shifts.covers.WheelerCover
