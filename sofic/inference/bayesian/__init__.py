"""Bayesian inference for computational mechanics models.

The default implementations use exact conjugate Dirichlet calculations.  PyMC
support is available through ``as_pymc_model()`` methods and is imported only
when requested.
"""

from sofic.inference.bayesian.comparison import ModelComparisonEM, ModelComparisonMC, ModelComparisonMC2
from sofic.inference.bayesian.counts import (
    BayesianInferenceError,
    PathCountEM,
    WordCountsMC,
    pretty_symbol,
    pretty_word,
    split_word,
)
from sofic.inference.bayesian.diversity import (
    PosteriorDiversityResult,
    machine_diversity,
    posterior_mean_word_distribution,
    posterior_process_diversity,
    process_identification_word_length,
    word_distribution_to_pmf,
)
from sofic.inference.bayesian.epsilon import DirichletDistributionEM, EpsilonMachinePosterior, InferEM
from sofic.inference.bayesian.hdp_hmm import HDPHMMPosterior, infer_hdp_hmm
from sofic.inference.bayesian.markov import DirichletPriorMC, InferMC, MarkovChainPosterior
from sofic.inference.bayesian.stack_hmm import (
    DirichletDistributionStackHMM,
    ModelComparisonStackHMM,
    PathCountStackHMM,
    StackHMMPosterior,
)

BayesianMCException = BayesianInferenceError
BayesianEMException = BayesianInferenceError

__all__ = [
    "BayesianEMException",
    "BayesianMCException",
    "BayesianInferenceError",
    "DirichletDistributionEM",
    "DirichletDistributionStackHMM",
    "DirichletPriorMC",
    "EpsilonMachinePosterior",
    "HDPHMMPosterior",
    "infer_hdp_hmm",
    "InferEM",
    "InferMC",
    "MarkovChainPosterior",
    "ModelComparisonEM",
    "ModelComparisonMC",
    "ModelComparisonMC2",
    "PosteriorDiversityResult",
    "machine_diversity",
    "posterior_mean_word_distribution",
    "posterior_process_diversity",
    "process_identification_word_length",
    "word_distribution_to_pmf",
    "ModelComparisonStackHMM",
    "PathCountEM",
    "PathCountStackHMM",
    "StackHMMPosterior",
    "WordCountsMC",
    "pretty_symbol",
    "pretty_word",
    "split_word",
]
