"""Causal-State Splitting Reconstruction and its relatives.

Process CSSR follows Shalizi, Shalizi & Crutchfield (arXiv:cs/0210025); subtree
merging follows Crutchfield & Young (PRL 1989; PRE 1994); transCSSR generalizes
CSSR to ε-transducers (Barnett & Crutchfield, J. Stat. Phys. 161:2 (2015)); stack
CSSR lifts CSSR to (suffix, stack) configurations of hidden Markov stack models.
"""

from sofic.inference.cssr.counts import (
    ConfigurationHistory,
    History,
    JointHistory,
    JointSuffixCounts,
    StackSuffixCounts,
    SuffixCounts,
)
from sofic.inference.cssr.process import learn_epsilon_machine_cssr, suggest_max_history
from sofic.inference.cssr.significance import (
    MorphTest,
    TableTest,
    aggregates_differ,
    morph_test_score,
    morphs_differ,
)
from sofic.inference.cssr.stack import (
    learn_stack_hmm_cssr,
    learn_stack_hmm_mle,
    learn_stack_hmm_papni,
    learn_stack_hmm_subtree,
)
from sofic.inference.cssr.subtree import learn_epsilon_machine_subtree
from sofic.inference.cssr.transducer import learn_epsilon_transducer_cssr

__all__ = [
    "ConfigurationHistory",
    "History",
    "JointHistory",
    "JointSuffixCounts",
    "MorphTest",
    "StackSuffixCounts",
    "SuffixCounts",
    "TableTest",
    "aggregates_differ",
    "learn_epsilon_machine_cssr",
    "learn_stack_hmm_mle",
    "learn_stack_hmm_papni",
    "morph_test_score",
    "morphs_differ",
    "learn_stack_hmm_cssr",
    "learn_stack_hmm_subtree",
    "learn_epsilon_machine_subtree",
    "suggest_max_history",
    "learn_epsilon_transducer_cssr",
]
