"""Stochastic and quasiprobabilistic generators."""

from sofic.generators.base import HiddenMarkovModel, QuasiStochasticModel, StochasticModel
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.block_convergence import BlockConvergenceDiagram, BlockConvergenceEstimates
from sofic.generators.block_entropy import BlockEntropyDiagram, BlockEntropyEstimates
from sofic.generators.channel_measures import (
    channel_statistical_complexity,
    driven_entropy_rate,
)
from sofic.generators.complexity_dimension import StatisticalComplexityDimension, statistical_complexity_dimension
from sofic.generators.correlations import autocorrelation, mutual_information_function, power_spectrum
from sofic.generators.directional_flow import (
    directed_information,
    independent_pair_generator,
    intrinsic_information_flow,
    shared_information_flow,
    synergistic_information_flow,
    transfer_entropy,
)
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.epsilon_transducer import EpsilonTransducer
from sofic.generators.lumping import LumpabilityError, is_lumpable, lump, normalize_partition
from sofic.generators.markov import MarkovChain
from sofic.generators.mealy import MealyHMM
from sofic.generators.measures import EntropyRateEstimate, entropy_rate_blackwell, entropy_rate_bounds
from sofic.generators.minimal_generative_model import (
    FunctionalGenerativeModel,
    GacsKornerGenerativeModel,
    MinimalGenerativeModel,
    WynerGenerativeModel,
    functional_generative_model,
    gacs_korner_generative_model,
    minimal_generative_model,
    wyner_generative_model,
)
from sofic.generators.mixed_state import MixedState, MixedStatePresentation
from sofic.generators.moore import MooreHMM
from sofic.generators.nmachine import NMachine
from sofic.generators.pfa import ProbabilisticFiniteAutomaton
from sofic.generators.predictive_rd import PredictiveRateDistortionCurve, predictive_rate_distortion
from sofic.generators.quasi_realization import QuasiRealization
from sofic.generators.relative_entropy_rate import (
    RelativeEntropyRateBounds,
    relative_entropy_rate,
    relative_entropy_rate_bounds,
)
from sofic.generators.stack_hmm import HiddenMarkovStackModel
from sofic.generators.topological_epsilon_enumeration import (
    count_topological_epsilon_machines,
    epsilon_machine_to_idfa_string,
    idfa_string_to_epsilon_machine,
    is_canonical_topological_epsilon,
    is_minimal_idfa,
    is_strongly_connected_idfa,
    is_topological_epsilon_string,
    iter_topological_epsilon_machines,
    iter_topological_epsilon_strings,
)
from sofic.generators.wheeler_epsilon import (
    colex_cdf,
    cylinder_measure,
    debruijn_presentation,
    wheeler_complexity_gap,
    wheeler_presentation,
    wheeler_statistical_complexity,
    word_cylinder_measure,
)

__all__ = [
    "BidirectionalEpsilonMachine",
    "BlockEntropyDiagram",
    "BlockEntropyEstimates",
    "BlockConvergenceDiagram",
    "BlockConvergenceEstimates",
    "EpsilonMachine",
    "EpsilonTransducer",
    "LumpabilityError",
    "channel_statistical_complexity",
    "driven_entropy_rate",
    "is_lumpable",
    "lump",
    "normalize_partition",
    "directed_information",
    "independent_pair_generator",
    "intrinsic_information_flow",
    "shared_information_flow",
    "synergistic_information_flow",
    "transfer_entropy",
    "FunctionalGenerativeModel",
    "GacsKornerGenerativeModel",
    "HiddenMarkovModel",
    "HiddenMarkovStackModel",
    "MarkovChain",
    "MealyHMM",
    "MinimalGenerativeModel",
    "MixedState",
    "MixedStatePresentation",
    "MooreHMM",
    "NMachine",
    "ProbabilisticFiniteAutomaton",
    "QuasiRealization",
    "QuasiStochasticModel",
    "StochasticModel",
    "WynerGenerativeModel",
    "colex_cdf",
    "count_topological_epsilon_machines",
    "cylinder_measure",
    "debruijn_presentation",
    "epsilon_machine_to_idfa_string",
    "functional_generative_model",
    "gacs_korner_generative_model",
    "idfa_string_to_epsilon_machine",
    "is_canonical_topological_epsilon",
    "is_minimal_idfa",
    "is_strongly_connected_idfa",
    "is_topological_epsilon_string",
    "iter_topological_epsilon_machines",
    "iter_topological_epsilon_strings",
    "minimal_generative_model",
    "wheeler_complexity_gap",
    "wheeler_presentation",
    "wheeler_statistical_complexity",
    "word_cylinder_measure",
    "wyner_generative_model",
    "PredictiveRateDistortionCurve",
    "RelativeEntropyRateBounds",
    "autocorrelation",
    "mutual_information_function",
    "power_spectrum",
    "predictive_rate_distortion",
    "relative_entropy_rate",
    "relative_entropy_rate_bounds",
    "EntropyRateEstimate",
    "StatisticalComplexityDimension",
    "entropy_rate_blackwell",
    "entropy_rate_bounds",
    "statistical_complexity_dimension",
]
