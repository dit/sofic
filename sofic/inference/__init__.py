"""Inference algorithms for stochastic generators."""

from sofic.inference import bayesian
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
    project_to_mealy,
    project_to_nmachine,
    spectral_singular_values,
)

__all__ = [
    "bayesian",
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
    "project_to_mealy",
    "project_to_nmachine",
    "spectral_singular_values",
]
