"""Passive and active automaton learning algorithms."""

from sofic.automata.learning.active import (
    AutomatonEquivalenceOracle,
    EquivalenceOracle,
    ExhaustiveEquivalenceOracle,
    FunctionMealyOracle,
    FunctionMembershipOracle,
    LanguageMembershipOracle,
    MealyEquivalenceOracle,
    MealyExhaustiveEquivalenceOracle,
    MealyMembershipOracle,
    MembershipOracle,
    RandomWalkEquivalenceOracle,
    TransducerOutputOracle,
    learn_dfa_from_language,
    learn_dfa_lstar,
    learn_dfa_ttt,
    learn_mealy_from_transducer,
    learn_mealy_lstar,
)
from sofic.automata.learning.alergia import learn_pfa_alergia
from sofic.automata.learning.dfasat import learn_dfa_sat
from sofic.automata.learning.edsm import learn_dfa_edsm
from sofic.automata.learning.nlstar import learn_prime_atomaton_nlstar, learn_rfsa_from_language, learn_rfsa_nlstar
from sofic.automata.learning.observation import ObservationTable
from sofic.automata.learning.papni import (
    DyckAlphabet,
    encode_dyck_samples,
    encode_dyck_word,
    is_well_matched,
    learn_sofic_dyck_shift_papni,
    sofic_dyck_shift_from_papni_dfa,
)
from sofic.automata.learning.rpni import learn_dfa_rpni

__all__ = [
    "AutomatonEquivalenceOracle",
    "DyckAlphabet",
    "EquivalenceOracle",
    "ExhaustiveEquivalenceOracle",
    "FunctionMealyOracle",
    "FunctionMembershipOracle",
    "LanguageMembershipOracle",
    "MealyEquivalenceOracle",
    "MealyExhaustiveEquivalenceOracle",
    "MealyMembershipOracle",
    "MembershipOracle",
    "ObservationTable",
    "RandomWalkEquivalenceOracle",
    "TransducerOutputOracle",
    "is_well_matched",
    "learn_dfa_edsm",
    "learn_dfa_from_language",
    "learn_dfa_lstar",
    "learn_dfa_rpni",
    "learn_dfa_sat",
    "learn_dfa_ttt",
    "learn_mealy_from_transducer",
    "learn_mealy_lstar",
    "learn_pfa_alergia",
    "learn_prime_atomaton_nlstar",
    "learn_rfsa_from_language",
    "learn_rfsa_nlstar",
    "learn_sofic_dyck_shift_papni",
    "encode_dyck_word",
    "encode_dyck_samples",
    "sofic_dyck_shift_from_papni_dfa",
]
