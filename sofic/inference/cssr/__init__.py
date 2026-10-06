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
from sofic.inference.cssr.process import cssr, suggest_lmax
from sofic.inference.cssr.significance import (
    MorphTest,
    TableTest,
    aggregates_differ,
    morph_test_score,
    morphs_differ,
)
from sofic.inference.cssr.stack import (
    fit_stack_hmm_mle,
    learn_stack_hmm_papni,
    stack_cssr,
    stack_subtree_merge,
)
from sofic.inference.cssr.subtree import subtree_merge
from sofic.inference.cssr.transducer import transcssr

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
    "cssr",
    "fit_stack_hmm_mle",
    "learn_stack_hmm_papni",
    "morph_test_score",
    "morphs_differ",
    "stack_cssr",
    "stack_subtree_merge",
    "subtree_merge",
    "suggest_lmax",
    "transcssr",
]
