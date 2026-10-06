"""Inference algorithms for stochastic generators.

The names ``cssr`` and ``spectral`` bound here are the learner functions; they
shadow the :mod:`sofic.inference.cssr` and :mod:`sofic.inference.spectral`
modules as attributes, so import from those modules with ``from ... import``.
"""

from sofic.inference import bayesian, hmm
from sofic.inference.cssr import (
    cssr,
    fit_stack_hmm_mle,
    learn_stack_hmm_papni,
    stack_cssr,
    stack_subtree_merge,
    subtree_merge,
    suggest_lmax,
    transcssr,
)
from sofic.inference.diagnostics import (
    GoodnessOfFit,
    StructureStability,
    goodness_of_fit,
    reconstruction_sweep,
    structure_stability,
    topology_key,
)
from sofic.inference.hmm import (
    backward,
    baum_welch,
    forward,
    free_parameter_labels,
    log_likelihood,
    observed_information,
    score,
    smooth,
    standard_errors,
    two_slice_marginals,
    viterbi,
)
from sofic.inference.model_selection import (
    ModelScores,
    WAICResult,
    compare_information_criteria,
    count_free_parameters,
    cross_validated_log_likelihood,
    information_criterion,
    rank_topological_epsilon_machines,
    score_model,
    waic,
    waic_epsilon_machine,
)
from sofic.inference.spectral import (
    SpectralInferenceError,
    hankel_matrices,
    learn_spectral_wfa,
    project_to_epsilon_machine,
    project_to_mealy,
    project_to_nmachine,
    spectral,
    spectral_singular_values,
)

__all__ = [
    "bayesian",
    "hmm",
    "cssr",
    "GoodnessOfFit",
    "StructureStability",
    "goodness_of_fit",
    "reconstruction_sweep",
    "structure_stability",
    "topology_key",
    "ModelScores",
    "WAICResult",
    "compare_information_criteria",
    "count_free_parameters",
    "cross_validated_log_likelihood",
    "information_criterion",
    "rank_topological_epsilon_machines",
    "score_model",
    "waic",
    "waic_epsilon_machine",
    "SpectralInferenceError",
    "hankel_matrices",
    "learn_spectral_wfa",
    "project_to_epsilon_machine",
    "project_to_mealy",
    "project_to_nmachine",
    "spectral",
    "spectral_singular_values",
    "subtree_merge",
    "suggest_lmax",
    "transcssr",
    "stack_cssr",
    "stack_subtree_merge",
    "fit_stack_hmm_mle",
    "learn_stack_hmm_papni",
    "baum_welch",
    "viterbi",
    "forward",
    "backward",
    "smooth",
    "two_slice_marginals",
    "log_likelihood",
    "score",
    "observed_information",
    "standard_errors",
    "free_parameter_labels",
]
