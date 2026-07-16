"""Stochastic and quasiprobabilistic generators."""

from sofic.generators.base import HiddenMarkovModel, QuasiStochasticModel, StochasticModel
from sofic.generators.bidirectional_epsilon_machine import BidirectionalEpsilonMachine
from sofic.generators.block_convergence import BlockConvergenceDiagram, BlockConvergenceEstimates
from sofic.generators.block_entropy import BlockEntropyDiagram, BlockEntropyEstimates
from sofic.generators.channel_measures import (
    channel_statistical_complexity,
    driven_entropy_rate,
)
from sofic.generators.directional_flow import (
    directed_information,
    independent_pair_generator,
    intrinsic_information_flow,
    shared_information_flow,
    synergistic_information_flow,
    transfer_entropy,
)
from sofic.generators.epsilon_inference import cssr, subtree_merge
from sofic.generators.epsilon_machine import EpsilonMachine
from sofic.generators.epsilon_transducer import EpsilonTransducer
from sofic.generators.lumping import LumpabilityError, is_lumpable, lump, normalize_partition
from sofic.generators.markov import MarkovChain
from sofic.generators.mealy import MealyHMM
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
from sofic.generators.quasi_realization import QuasiRealization
from sofic.generators.stack_hmm import HiddenMarkovStackModel
from sofic.generators.stack_inference import (
    fit_stack_hmm_mle,
    learn_stack_hmm_papni,
    stack_cssr,
    stack_subtree_merge,
)
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
    "cssr",
    "is_lumpable",
    "lump",
    "normalize_partition",
    "subtree_merge",
    "fit_stack_hmm_mle",
    "learn_stack_hmm_papni",
    "stack_cssr",
    "stack_subtree_merge",
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
    "count_topological_epsilon_machines",
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
    "wyner_generative_model",
]
