.. pymc.rst
.. py:module:: pensive.inference.bayesian.pymc_backend

*************
PyMC Backends
*************

The exact conjugate calculations return point estimates and closed-form
evidences. When full posterior samples are needed — for credible intervals,
posterior-predictive checks, or hierarchical extensions — the posteriors expose
`PyMC <https://www.pymc.io/>`_ models through ``as_pymc_model()`` and the
builders below. These require the optional backend
(``pip install pensive[bayes]``, which installs PyMC and ArviZ).

.. code-block:: python

   from pensive.inference.bayesian import MarkovChainPosterior

   post = MarkovChainPosterior(alphabet=("0", "1"), data=data, order=1)
   model = post.as_pymc_model()          # a pymc.Model
   import pymc as pm
   with model:
       idata = pm.sample()

API
===

.. autofunction:: markov_chain_model
.. autofunction:: epsilon_machine_model
