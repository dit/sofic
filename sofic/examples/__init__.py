"""Canonical examples from the computational mechanics literature."""

from importlib import import_module as _import_module

from sofic.examples.epsilon_machines import (
    TENT_MAP_MISIUREWICZ_PARTITIONS,
    alternating_biased_coins,
    bernoulli,
    butterfly_process,
    ellison_fig9_forward,
    ellison_fig9_reverse,
    ellison_fig15_bidirectional,
    even_process,
    fair_coin,
    golden_mean,
    golden_mean_bidirectional,
    golden_mean_forward,
    golden_mean_markov,
    golden_mean_reverse,
    golden_mean_shift_parry,
    nemo_process,
    noisy_random_phase_slip,
    restricted_golden_mean,
    tent_map_misiurewicz_a,
    tent_map_misiurewicz_bidirectional,
    tent_map_misiurewicz_forward,
    tent_map_misiurewicz_hmm,
    tent_map_misiurewicz_information_expected,
    tent_map_misiurewicz_partition_cuts,
    tent_map_misiurewicz_partition_forward,
    tent_map_misiurewicz_partition_information_expected,
    tent_map_misiurewicz_partition_symbol_matrices,
    tent_map_misiurewicz_reverse,
    wheeler_infinite_order_process,
)
from sofic.examples.processes import *
from sofic.examples.processes import __all__ as _process_all
from sofic.examples.shifts import (
    dyck_shift_order,
    motzkin_shift,
    sofic_dyck_fig1_shift,
    sofic_dyck_nondeterminizable_shift,
    sofic_dyck_zeta_example_shift,
)
from sofic.examples.tetris import (
    TETROMINOES,
    tetris_bag,
    tetris_gameboy,
    tetris_history,
    tetris_iid,
    tetris_nes,
    tetris_tgm,
    tetris_tgm2,
)

processes = _import_module("sofic.examples.processes")
shifts = _import_module("sofic.examples.shifts")
tetris = _import_module("sofic.examples.tetris")

__all__ = [
    "alternating_biased_coins",
    "bernoulli",
    "butterfly_process",
    "dyck_shift_order",
    "ellison_fig9_forward",
    "ellison_fig9_reverse",
    "ellison_fig15_bidirectional",
    "even_process",
    "fair_coin",
    "golden_mean",
    "golden_mean_forward",
    "golden_mean_markov",
    "golden_mean_reverse",
    "golden_mean_bidirectional",
    "golden_mean_shift_parry",
    "motzkin_shift",
    "nemo_process",
    "noisy_random_phase_slip",
    "restricted_golden_mean",
    "sofic_dyck_fig1_shift",
    "sofic_dyck_nondeterminizable_shift",
    "sofic_dyck_zeta_example_shift",
    "TENT_MAP_MISIUREWICZ_PARTITIONS",
    "tent_map_misiurewicz_a",
    "tent_map_misiurewicz_bidirectional",
    "tent_map_misiurewicz_forward",
    "tent_map_misiurewicz_hmm",
    "tent_map_misiurewicz_information_expected",
    "tent_map_misiurewicz_partition_cuts",
    "tent_map_misiurewicz_partition_forward",
    "tent_map_misiurewicz_partition_information_expected",
    "tent_map_misiurewicz_partition_symbol_matrices",
    "tent_map_misiurewicz_reverse",
    "TETROMINOES",
    "tetris_bag",
    "tetris_gameboy",
    "tetris_history",
    "tetris_iid",
    "tetris_nes",
    "tetris_tgm",
    "tetris_tgm2",
    "wheeler_infinite_order_process",
]
__all__ += _process_all
__all__ = [name for name in __all__ if name not in {"processes", "shifts", "tetris"}]
__all__.append("processes")
__all__.append("shifts")
__all__.append("tetris")
