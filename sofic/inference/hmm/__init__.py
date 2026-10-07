"""Inference for hidden Markov models.

Forward/backward/Viterbi decoding plus the Cappe, Moulines & Ryden (2005)
toolbox: fixed-interval smoothing (one- and two-slice marginals), Baum-Welch EM
parameter re-estimation, and the score / observed information via the Fisher
and Louis identities.
"""

from sofic.inference.hmm.em import baum_welch
from sofic.inference.hmm.filtering import (
    backward,
    forward,
    log_likelihood,
    smooth,
    two_slice_marginals,
    viterbi,
)
from sofic.inference.hmm.information import (
    free_parameter_labels,
    observed_information,
    score,
    standard_errors,
)

__all__ = [
    "backward",
    "baum_welch",
    "forward",
    "free_parameter_labels",
    "log_likelihood",
    "observed_information",
    "score",
    "smooth",
    "standard_errors",
    "two_slice_marginals",
    "viterbi",
]
