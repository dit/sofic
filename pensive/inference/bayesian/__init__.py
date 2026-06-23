"""Bayesian inference for computational mechanics models.

The default implementations use exact conjugate Dirichlet calculations.  PyMC
support is available through ``as_pymc_model()`` methods and is imported only
when requested.
"""

from pensive.inference.bayesian.comparison import ModelComparisonEM, ModelComparisonMC, ModelComparisonMC2
from pensive.inference.bayesian.counts import (
    BayesianInferenceError,
    PathCountEM,
    WordCountsMC,
    pretty_symbol,
    pretty_word,
    split_word,
)
from pensive.inference.bayesian.epsilon import DirichletDistributionEM, EpsilonMachinePosterior, InferEM
from pensive.inference.bayesian.markov import DirichletPriorMC, InferMC, MarkovChainPosterior

BayesianMCException = BayesianInferenceError
BayesianEMException = BayesianInferenceError

__all__ = [
    "BayesianEMException",
    "BayesianMCException",
    "BayesianInferenceError",
    "DirichletDistributionEM",
    "DirichletPriorMC",
    "EpsilonMachinePosterior",
    "InferEM",
    "InferMC",
    "MarkovChainPosterior",
    "ModelComparisonEM",
    "ModelComparisonMC",
    "ModelComparisonMC2",
    "PathCountEM",
    "WordCountsMC",
    "pretty_symbol",
    "pretty_word",
    "split_word",
]
